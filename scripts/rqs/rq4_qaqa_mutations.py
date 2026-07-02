#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 QAQA Mutation Generator Script (Updated)

This script reads golden dataset from data/examples/golden_dataset,
extracts context from questions, applies QAQA mutations, and outputs to
data/examples/golden_dataset_qaqa_mutation.

QAQA provides five metamorphic relations (MRs):
- EC (Extra Context): Add similar sentences to context
- EQ (Extra Question): Add similar sentences to question
- EQC (Extra Question + Context): Add similar sentences to both
- ETI (Extra Two Inputs): Add redundant QA pairs as context
- TI (Two Inputs): Combine two QA pairs

Key Updates:
- Extract context from questions BEFORE applying QAQA mutations
- This allows QAQA to work with KG-based QA datasets that don't have separate passages
- Filter training data by kg_rule to ensure semantic domain consistency
  Similar sentences are only selected from questions with the same kg_rule,
  preventing cross-domain contamination (e.g., German bridges vs Japanese tunnels)

Based on: "Natural Test Generation for Precise Testing of Question Answering Software"
https://github.com/yichuan-cs/QAQA

Usage:
    # Generate mutations for golden dataset
    python scripts/rqs/rq4_qaqa_mutations.py

    # Generate only specific mutation types
    python scripts/rqs/rq4_qaqa_mutations.py --no-ec --no-ti

    # Show detailed output
    python scripts/rqs/rq4_qaqa_mutations.py --verbose --limit 5
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.qaqa import QAQA, QA, QAList, MutationResult

# Define paths
GOLDEN_DATASET_DIR = Path("data/examples/golden_dataset")
OUTPUT_DIR = Path("data/examples/golden_dataset_qaqa_mutation")

# Random seed
RANDOM_SEED = 42


def extract_context_question(question: str, question_type: str) -> tuple[str, str]:
    """
    Extract context and question from a combined question string.

    For KG-based QA datasets, the question often contains both context (premises)
    and the actual question. We need to split them for QAQA mutations.

    Examples:
    - "Given that A is B, is it true that C is D?" -> context="Given that A is B", question="is it true that C is D?"
    - "Based on X, what is Y?" -> context="Based on X", question="what is Y?"

    Args:
        question: The full question string
        question_type: Type of question (yes_no, wh_question, true_false, multiple_choice)

    Returns:
        Tuple of (context, question)
    """
    question_lower = question.lower()

    # Define question start patterns to look for
    question_patterns = [
        'is it true that',
        'is it correct that',
        'is it the case that',
        'what is',
        'where is',
        'when is',
        'who is',
        'which is',
        'how many',
        'how much',
        'how does',
        'why does',
    ]

    # Try to find a pattern that marks the start of the actual question
    for pattern in question_patterns:
        idx = question_lower.find(pattern)
        if idx > 0:
            # Found the question start
            context = question[:idx].strip()
            quest = question[idx:].strip()
            return context, quest

    # If no pattern found, try alternative extraction for wh_questions
    if question_type == 'wh_question':
        # For wh-questions, the question part usually starts with wh-words
        # Look for the last comma or period as separator
        for separator in [', and', ',']:
            if separator in question:
                parts = question.split(separator)
                if len(parts) >= 2:
                    # Take everything before the last separator as context
                    context = separator.join(parts[:-1]).strip()
                    quest = parts[-1].strip()
                    # Ensure context is not too long (should be premises)
                    if len(context) > 50:  # Minimum reasonable context length
                        return context, quest

    # Multiple choice questions usually don't have separate context
    if question_type == 'multiple_choice':
        return '', question

    # Last resort: no clear separation found
    return '', question


def load_csv_files(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all CSV files from golden dataset directory"""
    csv_files = {}

    for csv_file in input_dir.rglob("*.csv"):
        if "summary" in csv_file.name:
            continue

        try:
            df = pd.read_csv(csv_file)
            rel_path = csv_file.relative_to(input_dir)
            csv_files[str(rel_path)] = df
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")

    return csv_files


def mutation_result_to_dict(mut: MutationResult) -> Dict[str, Any]:
    """Convert MutationResult to a serializable dictionary"""
    return {
        'original_question': mut.original_question,
        'mutated_question': mut.mutated_question,
        'mutation_type': mut.mutation_type,
        'original_context': mut.original_context,
        'mutated_context': mut.mutated_context,
        'original_answer': mut.original_answer,
        'mutated_answer': mut.mutated_answer,
        'is_violation': mut.is_violation,
        'metadata': mut.metadata or {},
    }


def build_training_qalist(df: pd.DataFrame, current_idx: int, kg_rule: Optional[str] = None) -> QAList:
    """
    Build a QAList from the dataset for similarity search.

    For KG-based QA datasets, we need to extract context from questions
    since there's no separate passage/context field.

    Key improvement: Filter by kg_rule to ensure semantic domain consistency.
    Only questions with the same kg_rule will be used for similarity search.

    Args:
        df: DataFrame containing the questions
        current_idx: Index of current question (to exclude from training)
        kg_rule: KG rule identifier (e.g., 'kg_rule_1', 'kg_rule_10')
                 If provided, only include questions with matching kg_rule

    Returns:
        QAList with questions from the same semantic domain
    """
    qalist = QAList()

    for idx, row in df.iterrows():
        if idx == current_idx:
            continue

        # Filter by kg_rule to ensure semantic domain consistency
        if kg_rule is not None:
            row_kg_rule = row.get('kg_rule', '')
            if row_kg_rule != kg_rule:
                continue

        question = row.get('mutated_question', row.get('original_question', ''))
        answer = row.get('mutated_correct_answer', row.get('original_correct_answer', ''))

        # Get question type for context extraction
        question_type = row.get('original_question_type', row.get('mutated_question_type', ''))

        # Skip multiple choice questions
        if question_type == 'multiple_choice':
            continue

        # Try to get context from various sources
        context = row.get('extracted_context', '')  # Pre-extracted context (preferred)
        if not context:
            context = row.get('passage', row.get('context', ''))  # Traditional passage/context
        if not context and question:
            # Extract context from question for KG-based QA
            context, _ = extract_context_question(question, question_type)

        if question:
            qa = QA(q=question, a=answer, c=context)
            qalist.append_(qa)

    return qalist


def apply_qaqa_mutations(
    qaqa: QAQA,
    question: str,
    context: str,
    answer: Optional[str] = None,
    training_qa_list: Optional[QAList] = None,
    apply_ec: bool = True,
    apply_eq: bool = True,
    apply_eqc: bool = True,
    apply_eti: bool = True,
    apply_ti: bool = True,
    is_boolq: bool = False,
    verbose: bool = False
) -> Dict[str, List[Dict]]:
    """
    Apply QAQA mutations to a single question

    Args:
        qaqa: QAQA instance
        question: The actual question (without context)
        context: The extracted context (premises)
        answer: Answer (optional but recommended)
        training_qa_list: Training QA list for similarity search
        apply_ec: Whether to apply EC mutations
        apply_eq: Whether to apply EQ mutations
        apply_eqc: Whether to apply EQC mutations
        apply_eti: Whether to apply ETI mutations
        apply_ti: Whether to apply TI mutations
        is_boolq: Whether this is a boolean question dataset
        verbose: Whether to show detailed output

    Returns:
        Dictionary with mutation results for each type
    """
    results = {
        'EC': [],
        'EQ': [],
        'EQC': [],
        'ETI': [],
        'TI': []
    }

    if not question:
        return results

    try:
        # Create QA object for current question
        qa = QA(q=question, a=answer or "", c=context)

        # Determine which attack mods to apply
        attack_mods = []
        if apply_ec:
            attack_mods.append('EC')
        if apply_eq:
            attack_mods.append('EQ')
        if apply_eqc:
            attack_mods.append('EQC')
        if apply_eti:
            attack_mods.append('ETI')
        if apply_ti and not is_boolq:
            attack_mods.append('TI')

        if not attack_mods:
            return results

        # Generate mutations
        # For each attack type, generate separately to get all types
        for attack_mod in attack_mods:
            mut_results = qaqa.generate_mutations(
                qa,
                training_qa_list=training_qa_list,
                attack_mods=[attack_mod],
                extra_sent_num2context=1,
                max_attack_num=1,
                combined_context_num=1,
                is_boolq=is_boolq,
            )

            # Convert to dict and store
            for mut in mut_results:
                results[attack_mod].append(mutation_result_to_dict(mut))

        if verbose:
            for mut_type, muts in results.items():
                if muts:
                    print(f"  {mut_type}: Generated {len(muts)} mutations")

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")

    return results


def process_dataframe(
    df: pd.DataFrame,
    qaqa: QAQA,
    apply_ec: bool = True,
    apply_eq: bool = True,
    apply_eqc: bool = True,
    apply_eti: bool = True,
    apply_ti: bool = True,
    is_boolq: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Process a single DataFrame, applying QAQA mutations

    Key: First extract context from questions, then apply QAQA mutations
    """
    result_df = df.copy()

    # Initialize mutation columns
    result_df['qaqa_ec_mutations'] = None
    result_df['qaqa_eq_mutations'] = None
    result_df['qaqa_eqc_mutations'] = None
    result_df['qaqa_eti_mutations'] = None
    result_df['qaqa_ti_mutations'] = None
    # Also store simplified question lists
    result_df['qaqa_ec_questions'] = None
    result_df['qaqa_eq_questions'] = None
    result_df['qaqa_eqc_questions'] = None
    result_df['qaqa_eti_questions'] = None
    result_df['qaqa_ti_questions'] = None

    # Store extracted context and question for reference
    result_df['extracted_context'] = None
    result_df['extracted_question'] = None

    # Determine which question column to use
    if 'mutated_question' in df.columns:
        question_col = 'mutated_question'
    elif 'original_question' in df.columns:
        question_col = 'original_question'

    if 'mutated_correct_answer' in df.columns:
        answer_col = 'mutated_correct_answer'
    elif 'original_correct_answer' in df.columns:
        answer_col = 'original_correct_answer'

    if question_col is None:
        print("Warning: No question column found, skipping mutations")
        return result_df

    # Process each row
    for idx, row in df.iterrows():
        question = row.get(question_col, '')
        answer = row.get(answer_col, None) if answer_col else None

        if not question or pd.isna(question):
            continue

        # Get question type
        question_type = row.get('original_question_type', row.get('mutated_question_type', ''))

        # Skip multiple choice questions (no suitable QAQA mutation)
        if question_type == 'multiple_choice':
            if verbose:
                print(f"\nSkipping multiple choice question: {question[:60]}...")
            continue

        if verbose:
            print(f"\nProcessing: {question[:60]}...")

        # Step 1: Extract context and question
        context, pure_question = extract_context_question(question, question_type)

        # Store extracted context and question for reference
        result_df.at[idx, 'extracted_context'] = context
        result_df.at[idx, 'extracted_question'] = pure_question

        # Step 1.5: Build training QAList filtered by kg_rule for semantic domain consistency
        # This ensures that similar sentences come from the same semantic domain (e.g., same KG rule)
        kg_rule = row.get('kg_rule', None)
        training_qa_list = build_training_qalist(df, idx, kg_rule=kg_rule)

        # Step 2: Apply QAQA mutations
        mutations = apply_qaqa_mutations(
            qaqa,
            pure_question,  # Use extracted question (without context)
            context,        # Use extracted context
            str(answer) if answer and not pd.isna(answer) else None,
            training_qa_list=training_qa_list,
            apply_ec=apply_ec,
            apply_eq=apply_eq,
            apply_eqc=apply_eqc,
            apply_eti=apply_eti,
            apply_ti=apply_ti,
            is_boolq=is_boolq,
            verbose=verbose
        )

        # Step 3: Store results with full context+question reconstruction
        if mutations['EC']:
            # Reconstruct the full question with mutated context
            ec_results = []
            for mut in mutations['EC']:
                # Create mutated full question by combining mutated context with pure question
                if mut['mutated_context']:
                    full_question = f"{mut['mutated_context']} {pure_question}"
                else:
                    full_question = question
                mut_copy = mut.copy()
                mut_copy['original_full_question'] = question
                mut_copy['mutated_full_question'] = full_question
                ec_results.append(mut_copy)
            result_df.at[idx, 'qaqa_ec_mutations'] = json.dumps(ec_results, ensure_ascii=False)
            result_df.at[idx, 'qaqa_ec_questions'] = json.dumps(
                [m['mutated_full_question'] for m in ec_results], ensure_ascii=False
            )

        if mutations['EQ']:
            eq_results = []
            for mut in mutations['EQ']:
                # EQ mutates the question itself
                # Need to add context back to the mutated question
                if context:
                    full_question = f"{context} {mut['mutated_question']}"
                else:
                    full_question = mut['mutated_question']
                mut_copy = mut.copy()
                mut_copy['original_full_question'] = question
                mut_copy['mutated_full_question'] = full_question
                eq_results.append(mut_copy)
            result_df.at[idx, 'qaqa_eq_mutations'] = json.dumps(eq_results, ensure_ascii=False)
            result_df.at[idx, 'qaqa_eq_questions'] = json.dumps(
                [m['mutated_full_question'] for m in eq_results], ensure_ascii=False
            )

        if mutations['EQC']:
            eqc_results = []
            for mut in mutations['EQC']:
                # EQC mutates both
                if context and mut['mutated_context']:
                    full_question = f"{mut['mutated_context']} {mut['mutated_question']}"
                elif context:
                    full_question = f"{context} {mut['mutated_question']}"
                else:
                    full_question = mut['mutated_question']
                mut_copy = mut.copy()
                mut_copy['original_full_question'] = question
                mut_copy['mutated_full_question'] = full_question
                eqc_results.append(mut_copy)
            result_df.at[idx, 'qaqa_eqc_mutations'] = json.dumps(eqc_results, ensure_ascii=False)
            result_df.at[idx, 'qaqa_eqc_questions'] = json.dumps(
                [m['mutated_full_question'] for m in eqc_results], ensure_ascii=False
            )

        if mutations['ETI']:
            result_df.at[idx, 'qaqa_eti_mutations'] = json.dumps(mutations['ETI'], ensure_ascii=False)
            result_df.at[idx, 'qaqa_eti_questions'] = json.dumps(
                [m['mutated_question'] for m in mutations['ETI']], ensure_ascii=False
            )

        if mutations['TI']:
            result_df.at[idx, 'qaqa_ti_mutations'] = json.dumps(mutations['TI'], ensure_ascii=False)
            result_df.at[idx, 'qaqa_ti_questions'] = json.dumps(
                [m['mutated_question'] for m in mutations['TI']], ensure_ascii=False
            )

    return result_df


def save_mutations(
    csv_files: Dict[str, pd.DataFrame],
    output_dir: Path,
    verbose: bool = False,
    quiet: bool = False
):
    """Save mutation results to output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'EC': 0,
        'EQ': 0,
        'EQC': 0,
        'ETI': 0,
        'TI': 0,
        'EC_questions': 0,
        'EQ_questions': 0,
        'EQC_questions': 0,
        'ETI_questions': 0,
        'TI_questions': 0,
        'total_files': 0
    }

    # Save each file
    for rel_path, df in csv_files.items():
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)

        # Update statistics
        for mutation_type in ['EC', 'EQ', 'EQC', 'ETI', 'TI']:
            col_name = f'qaqa_{mutation_type.lower()}_mutations'
            if col_name in df.columns:
                count = df[col_name].notna().sum()
                stats[mutation_type] += count
                # Count total questions
                for val in df[f'qaqa_{mutation_type.lower()}_questions'].dropna():
                    try:
                        questions = json.loads(val)
                        stats[f'{mutation_type}_questions'] += len(questions)
                    except (json.JSONDecodeError, TypeError):
                        stats[f'{mutation_type}_questions'] += 1

        stats['total_files'] += 1

    # Save statistics summary
    summary_file = output_dir / "mutation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("QAQA Mutation Summary (with Context Extraction & Semantic Domain Filtering)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Total files processed: {stats['total_files']}\n\n")
        f.write("Mutation statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  EC (Extra Context):     {stats['EC']} source questions, {stats['EC_questions']} generated questions\n")
        f.write(f"  EQ (Extra Question):    {stats['EQ']} source questions, {stats['EQ_questions']} generated questions\n")
        f.write(f"  EQC (Extra Q+Context):  {stats['EQC']} source questions, {stats['EQC_questions']} generated questions\n")
        f.write(f"  ETI (Extra Two Inputs):  {stats['ETI']} source questions, {stats['ETI_questions']} generated questions\n")
        f.write(f"  TI (Two Inputs):        {stats['TI']} source questions, {stats['TI_questions']} generated questions\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total source questions with mutations: {stats['EC'] + stats['EQ'] + stats['EQC'] + stats['ETI'] + stats['TI']}\n")
        f.write(f"  Total generated questions: {stats['EC_questions'] + stats['EQ_questions'] + stats['EQC_questions'] + stats['ETI_questions'] + stats['TI_questions']}\n")
        f.write("\n")
        f.write("Key Features:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Context Extraction:\n")
        f.write("   Questions are split into context (premises) and question\n")
        f.write("   using pattern matching on common question starters.\n")
        f.write("   This allows QAQA mutations to work with KG-based QA datasets.\n")
        f.write("\n")
        f.write("2. Semantic Domain Filtering (NEW):\n")
        f.write("   Similar sentences are only selected from questions with the same kg_rule.\n")
        f.write("   This prevents cross-domain contamination (e.g., mixing German bridges\n")
        f.write("   with Japanese tunnels or Soviet military ranks), ensuring mutations\n")
        f.write("   are semantically relevant and answerable.\n")

    # Save JSON format statistics
    def convert(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    stats_json = {k: convert(v) for k, v in stats.items()}
    stats_file = output_dir / "mutation_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_json, f, indent=2)

    if not quiet:
        print(f"\nMutation results saved to: {output_dir}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  EC:  {stats['EC']} sources -> {stats['EC_questions']} questions")
        print(f"  EQ:  {stats['EQ']} sources -> {stats['EQ_questions']} questions")
        print(f"  EQC: {stats['EQC']} sources -> {stats['EQC_questions']} questions")
        print(f"  ETI: {stats['ETI']} sources -> {stats['ETI_questions']} questions")
        print(f"  TI:  {stats['TI']} sources -> {stats['TI_questions']} questions")
        print(f"  Total mutations: {stats['EC_questions'] + stats['EQ_questions'] + stats['EQC_questions'] + stats['ETI_questions'] + stats['TI_questions']}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate QAQA mutations for RQ4 golden dataset (with context extraction)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all mutations
    python scripts/rqs/rq4_qaqa_mutations.py

    # Only EC and EQ
    python scripts/rqs/rq4_qaqa_mutations.py --no-eqc --no-eti --no-ti

    # Verbose output
    python scripts/rqs/rq4_qaqa_mutations.py --verbose --limit 5

    # Quiet mode (no progress bars)
    python scripts/rqs/rq4_qaqa_mutations.py --quiet

Key Updates:
    1. Extract context from questions before applying QAQA mutations
    2. Filter training data by kg_rule to ensure semantic domain consistency

Context Extraction:
    "Given that A is B, is it true that C is D?"
    -> context="Given that A is B", question="is it true that C is D?"

Semantic Domain Filtering:
    Similar sentences are only selected from questions with the same kg_rule,
    preventing cross-domain contamination (e.g., German bridges vs Japanese tunnels).
    This ensures mutations are semantically relevant and can be answered correctly.
        """
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(GOLDEN_DATASET_DIR),
        help=f'Input golden dataset directory (default: {GOLDEN_DATASET_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help=f'Output directory for mutations (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--no-ec',
        action='store_true',
        help='Disable EC (Extra Context) mutations'
    )
    parser.add_argument(
        '--no-eq',
        action='store_true',
        help='Disable EQ (Extra Question) mutations'
    )
    parser.add_argument(
        '--no-eqc',
        action='store_true',
        help='Disable EQC (Extra Question + Context) mutations'
    )
    parser.add_argument(
        '--no-eti',
        action='store_true',
        help='Disable ETI (Extra Two Inputs) mutations'
    )
    parser.add_argument(
        '--no-ti',
        action='store_true',
        help='Disable TI (Two Inputs) mutations'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output during processing'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all progress bars and non-essential output'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    if not args.quiet:
        print("=" * 70)
        print("RQ4 QAQA Mutation Generator (with Context Extraction)")
        print("=" * 70)
        print(f"\nInput directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Random seed: {args.seed}")
        print(f"\nMutation settings:")
        print(f"  EC (Extra Context):     {'ENABLED' if not args.no_ec else 'disabled'}")
        print(f"  EQ (Extra Question):    {'ENABLED' if not args.no_eq else 'disabled'}")
        print(f"  EQC (Extra Q+Context):  {'ENABLED' if not args.no_eqc else 'disabled'}")
        print(f"  ETI (Extra Two Inputs):  {'ENABLED' if not args.no_eti else 'disabled'}")
        print(f"  TI (Two Inputs):        {'ENABLED' if not args.no_ti else 'disabled'}")

        # Load CSV files
        print(f"\nLoading CSV files from {input_dir}...")

    csv_files = load_csv_files(input_dir)

    if not args.quiet:
        print(f"Loaded {len(csv_files)} CSV files")

    if not csv_files:
        print("Error: No CSV files found in input directory")
        sys.exit(1)

    # Initialize QAQA
    if not args.quiet:
        print("\nInitializing QAQA...")
    qaqa = QAQA(random_seed=args.seed)

    # Process each file
    print("\nProcessing files...")
    processed_files = {}

    file_list = list(csv_files.items())
    if args.limit:
        file_list = file_list[:args.limit]

    for rel_path, df in tqdm(file_list, desc="Generating mutations", disable=args.quiet or not args.verbose):
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {rel_path}")
            print(f"  Rows: {len(df)}")

        # Apply mutations
        processed_df = process_dataframe(
            df,
            qaqa,
            apply_ec=not args.no_ec,
            apply_eq=not args.no_eq,
            apply_eqc=not args.no_eqc,
            apply_eti=not args.no_eti,
            apply_ti=not args.no_ti,
            is_boolq=False,
            verbose=args.verbose
        )
        processed_files[rel_path] = processed_df

    # Save results
    if not args.quiet:
        print("\nSaving results...")
    save_mutations(processed_files, output_dir, verbose=args.verbose, quiet=args.quiet)

    if not args.quiet:
        print("\n" + "=" * 70)
        print("Mutation generation complete!")
        print("=" * 70)


if __name__ == '__main__':
    main()
