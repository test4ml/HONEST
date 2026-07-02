#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLI-based QA consistency checking script (ConsisMatcher).
Use a pretrained NLI model to check the consistency of LLM answers to the original and mutated questions.

Workflow:
1. Extract the conclusion from the original answer (extractive conclusion generation via mBART)
2. Extract the conclusion from the mutated answer
3. Use the NLI model to judge whether the two conclusions are consistent

Advantages:
- No LLM API calls required; works offline
- The mBART model is specifically trained for conclusion extraction (81% exact match, 98% substring accuracy)
- Fast inference
- Deterministic results
"""

import os
import sys
import re
import ast
import argparse
import pandas as pd
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# NLI optimization for entity_rename: entity surface-form normalization
#
# Problem: entity_rename replaces entities with placeholders (e.g. Object1 / Object_A). After the model
#   correctly renames them in sync, the two conclusions are identical in predicate logic and differ only
#   in "entity name vs placeholder"; NLI does not recognize this equivalence -> false positive.
# Fix: before feeding NLI, replace both placeholders and real entity labels in the two conclusions with a
#   unified token (ENT), so that NLI compares only the predicate/logical structure, removing entity surface-form differences.
# ----------------------------------------------------------------------------

# Placeholder vocabulary (compatible with both old and new naming: numeric suffix Object1 and letter suffix Object_A / Corp_Z)
# Used only as a fallback; _extract_placeholders captures arbitrary placeholder naming via word-level diff
_PLACEHOLDER_RE = re.compile(
    r'\b(?:Person|People|Place|City|Town|Organization|Organisation|Org|'
    r'Company|Co|Corp|Corporation|Institute|Institution|'
    r'Object|Item|Thing|Entity)'
    r'(?:_[A-Za-z]+|\d+)\b'
)

# Stop words / punctuation ignored during word-level diff (cannot be entities or placeholders)
_DIFF_STOP = set("a an the of is are was were to in on at from and or that this it "
                 "for with by as be been being has have had do does did not no yes "
                 "next crossing upstream downstream first second".split())


def _parse_entity_labels(raw) -> List[str]:
    """Parse the list of entity-label strings from the entity_labels field.

    The field looks like ["Q1196669: Pont Amont | Parisian bridge...", "Q3397786: ..."];
    take the part after ':' and before the first '|' of each entry as the label.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = [raw]
    else:
        parsed = raw
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    labels = []
    for item in parsed:
        s = str(item)
        if ':' in s:
            s = s.split(':', 1)[1]
        label = s.split('|')[0].strip()
        if len(label) >= 2:
            labels.append(label)
    return labels


def _extract_placeholders(original_question: str, mutated_question: str) -> List[str]:
    """Extract placeholder surface forms from the word-level diff between the original and mutated questions.

    entity_rename only replaces entities, so a word that "appears in the mutated question but not in the original"
    is a placeholder (e.g. Bob / Corp_Z / Object_A / Person1; works for any naming scheme).
    """
    if not original_question or not mutated_question:
        return []
    orig_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(original_question)))
    mut_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(mutated_question)))
    diff = mut_tokens - orig_tokens
    # Filter out stop words and overly short tokens
    return [t for t in diff if len(t) >= 2 and t.lower() not in _DIFF_STOP]


def _normalize_entity_surface(text: str, surfaces: List[str]) -> str:
    """Replace all given surface forms (real entity labels + placeholders) with the unified token ENT.

    Replace in descending length order to avoid short strings matching first and breaking longer strings.
    """
    if not text:
        return text
    t = _PLACEHOLDER_RE.sub('ENT', text)
    for s in sorted(set(surfaces), key=len, reverse=True):
        if s and s != 'ENT':
            t = t.replace(s, 'ENT')
    return t


def _detect_mutation_type(csv_file_path: str) -> Optional[str]:
    """Infer the mutation operator type from the file name (entity_rename / body_augmentation / body_permutation)."""
    name = os.path.basename(csv_file_path).lower()
    for mut in ('entity_rename', 'body_augmentation', 'body_permutation'):
        if mut in name:
            return mut
    return None


# Strip the option-letter marker at the start of a conclusion, e.g. "A. xxx" / "**B.** xxx" / "Option C: xxx" -> "xxx"
# For the same conclusion the model may pick different letters (B vs A) but the content is identical; stripping the letter prevents NLI from misjudging inconsistency.
# Restricted to A-D (options rarely reach E) to avoid damaging species names like "E. coli"; the letter must be followed by a separator + whitespace/bold
_OPTION_LETTER_RE = re.compile(
    r'^\s*\**\s*(?:option\s+)?[A-Da-d]\s*[.):\-](?:\*+\s*|\s+)', re.IGNORECASE)


def _strip_option_letter(text: str) -> str:
    """Strip the option-letter marker at the start of a conclusion (A. / B) / Option C: etc.)."""
    if not text:
        return text
    # Strip at most twice (to handle residuals like "**A.** B. real text")
    for _ in range(2):
        new = _OPTION_LETTER_RE.sub('', text)
        if new == text:
            break
        text = new
    return text



def process_csv_file(csv_file_path: str, matcher, batch_size: int = 32,
                     mutation_type: Optional[str] = None) -> List[Dict]:
    """
    Process a single CSV file (batch version).

    Compared with the original row-by-row version, the behavior and output fields are identical; only the execution changes:
      1. First collect all non-empty answer pairs
      2. Use matcher.extractor.batch_extract to extract conclusions in one batch (answer1 / answer2)
      3. Use matcher.nli_checker.batch_check_consistency to compute NLI in one batch
    This avoids low GPU utilization caused by per-row calls.

    entity_rename optimization: if mutation_type=='entity_rename', normalize entity surface forms
    (placeholders and real entity labels -> ENT) on both conclusions before feeding NLI,
    to eliminate false positives caused by entity<->placeholder equivalence.
    """
    logger.info(f"Processing file: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file_path}: {e}")
        return []

    total_rows = len(df)
    logger.info(f"Checking consistency for {total_rows} answer pairs")

    # ---- First pass: collect each row's raw fields and identify empty-answer rows ----
    # rows_meta[i] = (row_dict, original_answer, mutated_answer)
    rows_meta = []
    for _, row in df.iterrows():
        # pandas NaN becomes 'nan' (non-empty) via str(); normalize to '' first so the empty-answer check can catch it
        _oa = row.get('original_llm_answer')
        _ma = row.get('mutated_llm_answer')
        original_answer = '' if pd.isna(_oa) else str(_oa).strip()
        mutated_answer = '' if pd.isna(_ma) else str(_ma).strip()
        rows_meta.append((row.to_dict(), original_answer, mutated_answer))

    # Global indices of non-empty rows
    nonempty_idxs = [i for i, (_, a1, a2) in enumerate(rows_meta) if a1 and a2]

    # ---- Second pass: batch extraction + batch NLI ----
    conc1_list = []  # One-to-one correspondence with nonempty_idxs
    conc2_list = []
    nli_list = []

    if nonempty_idxs:
        answers1 = [rows_meta[i][1] for i in nonempty_idxs]
        answers2 = [rows_meta[i][2] for i in nonempty_idxs]

        logger.info(f"Batch extracting conclusions for {len(nonempty_idxs)} non-empty pairs")
        # Batch conclusion extraction (real padding + one-shot generate)
        conc1_list = matcher.extractor.batch_extract(answers1, batch_size=batch_size)
        conc2_list = matcher.extractor.batch_extract(answers2, batch_size=batch_size)

        # Build conclusion pairs and run batch NLI inference (bidirectional is supported internally)
        # entity_rename: normalize entity surface forms before feeding NLI
        entnorm = (mutation_type == 'entity_rename')
        nli1_list = []  # Text actually fed to NLI (after normalization)
        nli2_list = []
        pairs = []
        for idx_ptr, (c1, c2) in enumerate(zip(conc1_list, conc2_list)):
            row_dict = rows_meta[nonempty_idxs[idx_ptr]][0]
            # First strip the option letter (any mutation may involve MCQ), then do entity normalization
            t1, t2 = _strip_option_letter(c1.conclusion), _strip_option_letter(c2.conclusion)
            if entnorm:
                labels = _parse_entity_labels(row_dict.get('entity_labels'))
                placeholders = _extract_placeholders(
                    row_dict.get('original_question'), row_dict.get('mutated_question'))
                surfaces = labels + placeholders
                t1 = _normalize_entity_surface(t1, surfaces)
                t2 = _normalize_entity_surface(t2, surfaces)
            nli1_list.append(t1)
            nli2_list.append(t2)
            pairs.append((t1, t2))
        if entnorm:
            logger.info("entity_rename: NLI is performed after normalizing entity surface forms in the conclusions")
        logger.info(f"Batch NLI inference for {len(pairs)} conclusion pairs")
        nli_list = matcher.nli_checker.batch_check_consistency(pairs)

    # ---- Third pass: assemble results in original order ----
    results = []
    ne_ptr = 0  # nonempty pointer
    for i, (row_dict, original_answer, mutated_answer) in enumerate(rows_meta):
        if not (original_answer and mutated_answer):
            # Empty answer: skip the consistency check (consistent with the original behavior)
            result = {
                **row_dict,
                'answers_consistent': False,
                'consistency_explanation': 'Empty answer(s)',
                'consistency_method': 'nli',
                'consistency_confidence': 0.0,
                'entailment_score': 0.0,
                'contradiction_score': 0.0,
                'neutral_score': 0.0,
                'extracted_original_conclusion': '',
                'extracted_mutated_conclusion': '',
                'extraction_method_original': 'empty',
                'extraction_method_mutated': 'empty',
                'nli_input_original': '',
                'nli_input_mutated': '',
            }
            results.append(result)
            continue

        try:
            extraction1 = conc1_list[ne_ptr]
            extraction2 = conc2_list[ne_ptr]
            consistency = nli_list[ne_ptr]
            nli_text1 = nli1_list[ne_ptr]
            nli_text2 = nli2_list[ne_ptr]
            ne_ptr += 1

            # consistency_method tags the NLI backend + extractor + whether entity normalization was applied
            ext_tag = 'llm' if getattr(matcher, 'extractor_type', 'mbart') == 'llm' else 'mbart'
            nli_backend = getattr(matcher, 'nli_method', 'cross-encoder')
            # cross-encoder -> nli_{ext}; llm -> nli_llm_{ext}
            method_tag = f'nli_{ext_tag}' if nli_backend == 'cross-encoder' else f'nli_llm_{ext_tag}'
            if entnorm:
                method_tag += '_entnorm'

            # Build the result record (fields are identical to the original row-by-row version, with NLI input text appended for review)
            result = {
                **row_dict,
                'answers_consistent': consistency.is_consistent,
                'consistency_explanation': consistency.explanation,
                'consistency_method': method_tag,
                'consistency_confidence': consistency.confidence,
                'entailment_score': consistency.entailment_score,
                'contradiction_score': consistency.contradiction_score,
                'neutral_score': consistency.neutral_score,
                'extracted_original_conclusion': extraction1.conclusion,
                'extracted_mutated_conclusion': extraction2.conclusion,
                'extraction_method_original': extraction1.method,
                'extraction_method_mutated': extraction2.method,
                'extraction_confidence_original': extraction1.confidence,
                'extraction_confidence_mutated': extraction2.confidence,
                'nli_input_original': nli_text1,
                'nli_input_mutated': nli_text2,
            }
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            ne_ptr += 1
            result = {
                **row_dict,
                'answers_consistent': False,
                'consistency_explanation': f'Processing error: {str(e)}',
                'consistency_method': 'error',
                'consistency_confidence': 0.0,
                'entailment_score': 0.0,
                'contradiction_score': 0.0,
                'neutral_score': 0.0,
                'extracted_original_conclusion': '',
                'extracted_mutated_conclusion': '',
                'extraction_method_original': 'error',
                'extraction_method_mutated': 'error',
                'nli_input_original': '',
                'nli_input_mutated': '',
            }
        results.append(result)

    return results


def save_results(results: List[Dict], output_file: str):
    """Save the consistency-check results."""
    if not results:
        logger.warning("No results to save")
        return

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Create the DataFrame
        df = pd.DataFrame(results)

        # Reorder columns: put the consistency-check related fields first
        consistency_fields = [
            'answers_consistent', 'consistency_explanation', 'consistency_method',
            'consistency_confidence', 'entailment_score', 'contradiction_score', 'neutral_score',
            'extracted_original_conclusion', 'extracted_mutated_conclusion',
            'extraction_method_original', 'extraction_method_mutated',
            'nli_input_original', 'nli_input_mutated',
            'original_llm_answer', 'mutated_llm_answer'
        ]

        # Get the fields that actually exist
        existing_consistency_fields = [field for field in consistency_fields if field in df.columns]
        other_fields = sorted([col for col in df.columns if col not in consistency_fields])

        # Reorder columns
        ordered_columns = existing_consistency_fields + other_fields
        df = df[ordered_columns]

        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')

        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description='NLI-based QA consistency checking script (ConsisMatcher)')

    # Model configuration
    parser.add_argument('--preset', default='accurate',
                        choices=['base', 'light', 'fast', 'accurate'],
                        help='Preset configuration (all presets use mBART for extraction): '
                             'accurate/base (DeBERTa NLI, recommended), light/fast (MiniLM NLI, faster)')
    parser.add_argument('--nli_model', default=None,
                        help='NLI model name (overrides the preset)')
    parser.add_argument('--device', default=None,
                        choices=['cuda', 'cpu'],
                        help='Run device (auto-detected by default)')

    # Input/output parameters
    parser.add_argument('--answers_dir', default='qa_answers',
                        help='Directory containing LLM answers')
    parser.add_argument('--output_dir', default='consistency_results_nli',
                        help='Output results directory')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--rules', type=str, default=None,
                        help='Process only specified rule numbers; supports ranges/comma lists, e.g. "1-63" or "1,3,5-10". '
                             'Used to split work across multiple GPUs (filtered by kg_rule_NNN in the path).')
    parser.add_argument('--mutations', type=str, nargs='+', default=None,
                        help='Process only files of the specified mutation operator(s), e.g. "body_permutation". '
                             'No filter by default (process all of entity_rename/body_augmentation/body_permutation).')

    # Conclusion extractor configuration
    parser.add_argument('--extractor', default='llm', choices=['llm', 'mbart'],
                        help='Conclusion extractor: "llm" calls a local small model via URL (default, e.g. Qwen3-8B, more robust extraction); '
                             '"mbart" uses the local mBART extractive model (offline, the old default)')
    parser.add_argument('--extractor_base_url', default='http://localhost:8001/v1',
                        help='LLM extractor service URL (only effective with --extractor llm)')
    parser.add_argument('--extractor_model', default='Qwen3-8B',
                        help='LLM extractor model name (only effective with --extractor llm)')
    parser.add_argument('--extractor_api_key', default='your-api-key',
                        help='LLM extractor API key (any value for a local service)')
    parser.add_argument('--extractor_protocol', default='openai',
                        help='LLM extractor protocol (use openai for local vLLM)')
    parser.add_argument('--extractor_concurrency', type=int, default=32,
                        help='LLM extractor concurrency (only effective with --extractor llm)')

    # NLI judging backend configuration (cross-encoder small model vs LLM)
    parser.add_argument('--nli_method', default='cross-encoder', choices=['cross-encoder', 'llm'],
                        help='NLI judging backend: "cross-encoder" local NLI small model (default); '
                             '"llm" calls an LLM via URL to judge and output JSON (e.g. DeepSeek-V4-Flash)')
    parser.add_argument('--nli_llm_base_url', default='https://api.deepseek.com/anthropic',
                        help='LLM NLI service URL (only effective with --nli_method llm)')
    parser.add_argument('--nli_llm_model', default='deepseek-v4-flash',
                        help='LLM NLI model name (only effective with --nli_method llm)')
    parser.add_argument('--nli_llm_protocol', default='anthropic',
                        help='LLM NLI protocol (use anthropic for DeepSeek)')
    parser.add_argument('--nli_llm_api_key', default=None,
                        help='LLM NLI API key (read from env DEEPSEEK_API_KEY by default)')
    parser.add_argument('--nli_llm_concurrency', type=int, default=20,
                        help='LLM NLI concurrency (only effective with --nli_method llm; keep conservative for remote APIs)')

    args = parser.parse_args()

    # Parse the set of rule numbers from --rules
    args.rule_set = None
    if args.rules:
        rule_set = set()
        for part in args.rules.split(','):
            part = part.strip()
            if '-' in part:
                lo, hi = part.split('-', 1)
                rule_set.update(range(int(lo), int(hi) + 1))
            elif part:
                rule_set.add(int(part))
        args.rule_set = rule_set
        logger.info(f"Rule filter: processing only {len(rule_set)} rules ({args.rules})")

    logger.info("=== NLI-based QA consistency check started ===")
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Extractor: {args.extractor}")
    logger.info(f"NLI method: {args.nli_method}")
    logger.info(f"Device: {args.device or 'auto'}")
    logger.info(f"Answers dir: {args.answers_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize ConsisMatcher
    logger.info("Initializing ConsisMatcher...")
    try:
        from honest.consismatcher import ConsisMatcher

        # Build the initialization kwargs
        init_kwargs = {
            'preset': args.preset,
            'device': args.device,
            'bidirectional': True,
            'extractor': args.extractor,
            'nli_method': args.nli_method,
        }

        # LLM extractor kwargs
        if args.extractor == 'llm':
            init_kwargs['extractor_kwargs'] = {
                'base_url': args.extractor_base_url,
                'api_key': args.extractor_api_key,
                'model_name': args.extractor_model,
                'protocol': args.extractor_protocol,
                'max_concurrent': args.extractor_concurrency,
                'rate_limit': 200.0,
            }

        # LLM NLI judging backend kwargs (only effective when nli_method='llm')
        if args.nli_method == 'llm':
            init_kwargs['nli_llm_kwargs'] = {
                'base_url': args.nli_llm_base_url,
                'api_key': args.nli_llm_api_key,
                'model_name': args.nli_llm_model,
                'protocol': args.nli_llm_protocol,
                'max_concurrent': args.nli_llm_concurrency,
                'rate_limit': float(args.nli_llm_concurrency),
            }

        # Override the preset (only effective for the cross-encoder backend)
        if args.nli_model and args.nli_method == 'cross-encoder':
            init_kwargs['nli_model'] = args.nli_model

        matcher = ConsisMatcher(**init_kwargs)
        logger.info("ConsisMatcher initialized successfully")

    except ImportError as e:
        logger.error(f"Failed to import ConsisMatcher: {e}")
        logger.error("Please make sure sentence-transformers and transformers are installed:")
        logger.error("  pip install sentence-transformers transformers torch")
        return
    except Exception as e:
        logger.error(f"Failed to initialize ConsisMatcher: {e}")
        return

    # Test the model
    logger.info("Testing model with sample input...")
    try:
        test_result = matcher.check_consistency_simple(
            "The answer is yes.",
            "Yes, that is correct."
        )
        logger.info(f"Test result: consistent={test_result[0]}, explanation={test_result[1]}")
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return

    # Process all CSV files
    answer_files = []
    for root, _, files in os.walk(args.answers_dir):
        for file in files:
            if file.endswith('.csv'):
                answer_files.append(os.path.join(root, file))

    # Filter by --rules (split for multi-GPU parallelism)
    if args.rule_set:
        import re as _re
        filtered = []
        for fp in answer_files:
            m = _re.search(r'kg_rule_(\d+)', fp)
            if m and int(m.group(1)) in args.rule_set:
                filtered.append(fp)
        logger.info(f"Rule filter: {len(answer_files)} -> {len(filtered)} files")
        answer_files = filtered

    # Filter by --mutations (process only files of the specified mutation operator)
    if args.mutations:
        mut_set = set(args.mutations)
        filtered = [fp for fp in answer_files
                    if any(f"{m}_" in os.path.basename(fp) for m in mut_set)]
        logger.info(f"Mutation filter: {len(answer_files)} -> {len(filtered)} files (mutations={args.mutations})")
        answer_files = filtered

    if args.max_files:
        answer_files = answer_files[:args.max_files]
        logger.info(f"Limited to processing the first {args.max_files} files")

    logger.info(f"Found {len(answer_files)} answer files")

    if not answer_files:
        logger.warning(f"No CSV files found in {args.answers_dir}")
        return

    total_results = []
    processed_files = 0

    for file_path in tqdm(answer_files, desc="Processing files"):
        try:
            # Determine the output file path
            relative_path = os.path.relpath(file_path, args.answers_dir)
            output_file = os.path.join(args.output_dir, relative_path)

            # Check whether the output file already exists
            if os.path.exists(output_file):
                logger.info(f"Output file already exists, skipping: {output_file}")
                # Read the existing results and append them to the overall results
                try:
                    existing_df = pd.read_csv(output_file)
                    existing_results = existing_df.to_dict('records')
                    total_results.extend(existing_results)
                    processed_files += 1
                except Exception as e:
                    logger.warning(f"Failed to read existing file {output_file}: {e}, will reprocess")
                    mutation_type = _detect_mutation_type(file_path)
                    results = process_csv_file(file_path, matcher, args.batch_size,
                                               mutation_type=mutation_type)
                    total_results.extend(results)
                    processed_files += 1
                    save_results(results, output_file)
                continue

            # Process the file (detect mutation type, used for entity_rename NLI normalization)
            mutation_type = _detect_mutation_type(file_path)
            if mutation_type == 'entity_rename':
                logger.info(f"Detected entity_rename file, enabling entity normalization: {file_path}")
            results = process_csv_file(file_path, matcher, args.batch_size,
                                       mutation_type=mutation_type)
            total_results.extend(results)
            processed_files += 1

            # Save results per file
            save_results(results, output_file)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue

    # Save the aggregated results
    if total_results:
        summary_file = os.path.join(args.output_dir, 'all_consistency_summary.csv')
        save_results(total_results, summary_file)

        # Statistics
        total_checks = len(total_results)
        consistent_checks = sum(1 for r in total_results if r.get('answers_consistent', False))
        consistency_rate = consistent_checks / total_checks if total_checks > 0 else 0

        # Compute the average confidence
        confidences = [r.get('consistency_confidence', 0) for r in total_results if r.get('consistency_method') == 'nli']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        logger.info("=== Consistency check completed ===")
        logger.info(f"Files processed: {processed_files}")
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Consistent cases: {consistent_checks}")
        logger.info(f"Consistency rate: {consistency_rate:.2%}")
        logger.info(f"Average confidence: {avg_confidence:.3f}")

        # Statistics by NLI score distribution
        entailment_scores = [r.get('entailment_score', 0) for r in total_results if r.get('consistency_method') == 'nli']
        contradiction_scores = [r.get('contradiction_score', 0) for r in total_results if r.get('consistency_method') == 'nli']

        if entailment_scores:
            logger.info(f"Average entailment score: {sum(entailment_scores)/len(entailment_scores):.3f}")
        if contradiction_scores:
            logger.info(f"Average contradiction score: {sum(contradiction_scores)/len(contradiction_scores):.3f}")

        # Statistics by extraction method
        extraction_methods = {}
        for result in total_results:
            method = result.get('extraction_method_original', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1

        logger.info("Extraction method statistics:")
        for method, count in sorted(extraction_methods.items(), key=lambda x: -x[1]):
            logger.info(f"  {method}: {count} ({count/total_checks:.1%})")

    else:
        logger.warning("No consistency-check results were generated")


if __name__ == '__main__':
    main()
