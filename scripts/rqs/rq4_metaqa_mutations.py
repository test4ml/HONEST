#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 MetaQA Pipeline Script

Implements the full MetaQA pipeline (paper-faithful):
  1. Get LLM's concise answer to the original question (base response)
  2. Generate N synonym mutations of the base response
  3. Generate M antonym mutations of the base response
  4. Verify each mutation (ask LLM if factual → Yes/No/Not Sure)
  5. Calculate hallucination score

Output: data/examples/golden_dataset_metaqa_answer_<model>/

Usage:
    conda activate karma
    # Default run with local LLM
    python scripts/rqs/rq4_metaqa_mutations.py

    # Specific model
    python scripts/rqs/rq4_metaqa_mutations.py --models deepseek_v4_flash

    # Custom mutation counts
    python scripts/rqs/rq4_metaqa_mutations.py --n-synonym 3 --n-antonym 3

    # Resume from interruption
    python scripts/rqs/rq4_metaqa_mutations.py --skip-processed

    # Limit for testing
    python scripts/rqs/rq4_metaqa_mutations.py --limit 3
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from openai import APIConnectionError, APITimeoutError
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import get_config
from honest.llm import SUPPORTED_PROTOCOLS, create_llm_client
from baselines.metaqa import MetaQA, MetaQAMutationType, MetaQAMutation

# Paths
GOLDEN_DATASET_PATH = Path("data/examples/golden_dataset/golden_dataset_full.csv")
OUTPUT_BASE_DIR = Path("data/examples")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model config aliases (shared with other rq4 scripts)
MODEL_CONFIG_ALIASES = {
    "deepseek-v4-flash": "deepseek_v4_flash",
    "deepseek_v4_flash": "deepseek_v4_flash",
    "glm-5-turbo": "glm_5_turbo",
    "GLM-5-Turbo": "glm_5_turbo",
    "glm_5_turbo": "glm_5_turbo",
}


# ---------------------------------------------------------------------------
# Model config resolution (same pattern as DrHall/KONTEST)
# ---------------------------------------------------------------------------

def resolve_model_config(
    model_config: str, args
) -> Tuple[Optional[str], Optional[str], str, str]:
    config_key = MODEL_CONFIG_ALIASES.get(model_config, model_config)
    if config_key == "local":
        llm_config = get_config("llm.local", {})
        return (
            args.base_url,
            args.api_key,
            args.model or llm_config.get("default_model", "Qwen2.5-7B-Instruct"),
            args.protocol or llm_config.get("protocol", "openai"),
        )
    if config_key == "deepseek":
        llm_config = get_config("llm.deepseek", {})
        return (
            args.base_url or llm_config.get("base_url"),
            args.api_key or llm_config.get("api_key"),
            args.model or llm_config.get("default_model", "deepseek-chat"),
            args.protocol or llm_config.get("protocol", "openai"),
        )
    if config_key in {"deepseek_v4_flash", "glm_5_turbo", "anthropic", "gemini"}:
        llm_config = get_config(f"llm.{config_key}", {})
        return (
            args.base_url or llm_config.get("base_url"),
            args.api_key or llm_config.get("api_key"),
            args.model or llm_config.get("default_model", model_config),
            args.protocol or llm_config.get("protocol", "openai"),
        )
    return (
        args.base_url,
        args.api_key,
        args.model or model_config,
        args.protocol or "openai",
    )


# ---------------------------------------------------------------------------
# Answer re-use helpers
# ---------------------------------------------------------------------------

def parse_existing_json(value: Any) -> Optional[List[Dict]]:
    """Parse a JSON list from a CSV cell."""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def is_complete_answer(value: Any) -> bool:
    """Check if a base response cell is filled and not an error."""
    if not value or pd.isna(value):
        return False
    return not str(value).startswith("ERROR:")


def is_complete_mutation_list(value: Any) -> bool:
    """Check if a mutation list is filled with valid entries."""
    parsed = parse_existing_json(value)
    if not parsed:
        return False
    return all(
        isinstance(m, dict) and m.get("mutation_text") and m.get("verification_result")
        for m in parsed
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

async def get_concise_answer(
    llm_client,
    question: str,
) -> str:
    """Step 1: Get a concise answer from the LLM."""
    prompt = (
        "Answer the following question concisely with just the factual answer. "
        "Do not add explanations.\n\n"
        f"Question: {question}"
    )
    return await llm_client.generate_answer(prompt)


async def generate_mutations(
    llm_client,
    mr_type: str,
    question: str,
    answer: str,
    n: int,
) -> List[Dict[str, Any]]:
    """Step 2: Generate synonym or antonym mutations."""
    metaqa = MetaQA()
    if mr_type == MetaQAMutationType.SYNONYM.value:
        prompt = metaqa.build_synonym_prompt(question, answer)
    else:
        prompt = metaqa.build_antonym_prompt(question, answer)

    # Ask the LLM to generate a list of mutations
    raw_response = await llm_client.generate_answer(prompt)
    items = metaqa.parse_numbered_list(raw_response)

    # Take the first n
    items = items[:n]

    mutations = []
    for item in items:
        mutations.append({
            "mr_type": mr_type,
            "mutation_text": item,
            "verification_result": "",
            "hallucination_contribution": 0.0,
        })
    return mutations


async def verify_mutation(
    llm_client,
    mutation_text: str,
    mr_type: Optional[str] = None,
) -> Optional[str]:
    """Step 3: Verify a single mutation -> Yes/No/Not Sure (or None).

    ``mr_type`` ("Synonym"/"Antonym") selects the MR-type-aware verification
    prompt per Algorithm 1 ``VerifyFactByLLM(m, r)``. Returns ``None`` when the
    LLM reply cannot be parsed (observability; counted as unparseable).
    """
    metaqa = MetaQA()
    prompt = metaqa.build_verification_prompt(mutation_text, mr_type)
    raw_response = await llm_client.generate_answer(prompt)
    return metaqa.parse_verification(raw_response)


async def process_row(
    row_idx: int,
    row: pd.Series,
    llm_client,
    model_name: str,
    n_synonym: int,
    n_antonym: int,
    existing_row: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Process a single row through the full MetaQA pipeline.
    """
    result = {}
    question = str(row.get("original_question", "")).strip()
    if not question:
        return result

    kg_rule = str(row.get("kg_rule", ""))
    mutation_type = str(row.get("mutation_type", ""))
    question_type = str(row.get("original_question_type", ""))
    correct_answer = row.get("original_correct_answer", "")

    base_response_key = f"metaqa_base_response_{model_name}"
    syn_key = f"metaqa_synonym_mutations_{model_name}"
    ant_key = f"metaqa_antonym_mutations_{model_name}"
    score_key = f"metaqa_hallucination_score_{model_name}"

    # --- Step 1: Get concise answer ---
    base_response = ""
    if existing_row and base_response_key in existing_row:
        existing_val = existing_row[base_response_key]
        if is_complete_answer(existing_val):
            base_response = str(existing_val)

    if not base_response:
        try:
            base_response = await get_concise_answer(llm_client, question)
        except Exception as e:
            logger.error(f"Error getting base response for row {row_idx}: {e}")
            base_response = f"ERROR: {e}"

    result[base_response_key] = base_response

    if base_response.startswith("ERROR:"):
        result[syn_key] = None
        result[ant_key] = None
        result[score_key] = None
        return result

    # --- Step 2 & 3: Generate and verify synonym mutations ---
    syn_mutations = None
    if existing_row and syn_key in existing_row:
        existing = existing_row[syn_key]
        if is_complete_mutation_list(existing):
            syn_mutations = parse_existing_json(existing)

    if syn_mutations is None:
        try:
            raw_syn = await generate_mutations(
                llm_client, MetaQAMutationType.SYNONYM.value,
                question, base_response, n_synonym,
            )
            # Verify each mutation
            for mut in raw_syn:
                try:
                    ver = await verify_mutation(
                        llm_client, mut["mutation_text"],
                        mr_type=MetaQAMutationType.SYNONYM.value,
                    )
                    mut["verification_result"] = ver if ver is not None else "Not Sure"
                    mut["hallucination_contribution"] = {"Yes": 0.0, "No": 1.0, "Not Sure": 0.5}.get(ver, 0.5)
                except Exception as e:
                    logger.warning(f"Synonym verification failed for row {row_idx}: {e}")
                    mut["verification_result"] = "Not Sure"
                    mut["hallucination_contribution"] = 0.5

            syn_mutations = raw_syn if raw_syn else None
        except Exception as e:
            logger.error(f"Error generating synonym mutations for row {row_idx}: {e}")
            syn_mutations = None

    result[syn_key] = json.dumps(syn_mutations, ensure_ascii=False) if syn_mutations else None

    # --- Step 2 & 3: Generate and verify antonym mutations ---
    ant_mutations = None
    if existing_row and ant_key in existing_row:
        existing = existing_row[ant_key]
        if is_complete_mutation_list(existing):
            ant_mutations = parse_existing_json(existing)

    if ant_mutations is None:
        try:
            raw_ant = await generate_mutations(
                llm_client, MetaQAMutationType.ANTONYM.value,
                question, base_response, n_antonym,
            )
            for mut in raw_ant:
                try:
                    ver = await verify_mutation(
                        llm_client, mut["mutation_text"],
                        mr_type=MetaQAMutationType.ANTONYM.value,
                    )
                    mut["verification_result"] = ver if ver is not None else "Not Sure"
                    mut["hallucination_contribution"] = {"Yes": 1.0, "No": 0.0, "Not Sure": 0.5}.get(ver, 0.5)
                except Exception as e:
                    logger.warning(f"Antonym verification failed for row {row_idx}: {e}")
                    mut["verification_result"] = "Not Sure"
                    mut["hallucination_contribution"] = 0.5

            ant_mutations = raw_ant if raw_ant else None
        except Exception as e:
            logger.error(f"Error generating antonym mutations for row {row_idx}: {e}")
            ant_mutations = None

    result[ant_key] = json.dumps(ant_mutations, ensure_ascii=False) if ant_mutations else None

    # --- Step 4: Calculate hallucination score ---
    syn_results = [m["verification_result"] for m in syn_mutations if m.get("verification_result")] if syn_mutations else []
    ant_results = [m["verification_result"] for m in ant_mutations if m.get("verification_result")] if ant_mutations else []

    metaqa = MetaQA()
    detail = metaqa.calculate_hallucination_detail(syn_results, ant_results)
    score = detail["score"]
    result[score_key] = score
    # Binary hallucination label (paper Section 3.4: S_QB >= theta).
    result[f"metaqa_is_hallucination_{model_name}"] = bool(detail["is_hallucination"])

    # Store metadata
    result[f"metaqa_metadata_{model_name}"] = json.dumps({
        "kg_rule": kg_rule,
        "mutation_type": mutation_type,
        "question_type": question_type,
        "original_question": question,
        "base_response": base_response,
        "n_synonym": len(syn_results),
        "n_antonym": len(ant_results),
        "hallucination_score": score,
        "threshold": detail["threshold"],
        "is_hallucination": detail["is_hallucination"],
        "unparseable": detail["unparseable"],
    }, ensure_ascii=False)

    if debug and row_idx < 3:
        logger.info(f"Row {row_idx}: score={score:.3f}, syn={syn_results}, ant={ant_results}")

    return result


async def process_dataframe(
    df: pd.DataFrame,
    llm_client,
    model_name: str,
    n_synonym: int,
    n_antonym: int,
    debug: bool = False,
    existing_df: Optional[pd.DataFrame] = None,
    checkpoint_every: int = 0,
    checkpoint_fn: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """Process entire DataFrame through the MetaQA pipeline.

    Checkpoint large files against interruption: persist every ``checkpoint_every``
    rows (``checkpoint_fn`` writes to the same file as the final output), and do
    one final save in the ``finally`` block on any abnormal exit (network drop
    ``APIConnectionError`` / ctrl-C / other exceptions). Combined with per-row
    reuse under ``--skip-processed``, reruns only redo the unfinished rows instead
    of losing the whole file when it is not fully written.
    """
    result_df = df.copy()

    # Initialize columns
    result_df[f"metaqa_base_response_{model_name}"] = None
    result_df[f"metaqa_synonym_mutations_{model_name}"] = None
    result_df[f"metaqa_antonym_mutations_{model_name}"] = None
    result_df[f"metaqa_hallucination_score_{model_name}"] = None
    result_df[f"metaqa_metadata_{model_name}"] = None

    tasks = []
    for idx, row in df.iterrows():
        row_existing = None
        if existing_df is not None and idx in existing_df.index:
            row_existing = {}
            for key in [
                f"metaqa_base_response_{model_name}",
                f"metaqa_synonym_mutations_{model_name}",
                f"metaqa_antonym_mutations_{model_name}",
                f"metaqa_hallucination_score_{model_name}",
            ]:
                if key in existing_df.columns:
                    row_existing[key] = existing_df.at[idx, key]

        task = process_row(
            idx, row, llm_client, model_name,
            n_synonym, n_antonym, row_existing, debug,
        )
        tasks.append((idx, task))

    total = len(tasks)
    completed = 0
    aborted = True
    try:
        for idx, task in async_tqdm(tasks, total=total, desc=f"MetaQA pipeline ({model_name})"):
            try:
                row_result = await task
                for key, value in row_result.items():
                    result_df.at[idx, key] = value
            except (APIConnectionError, APITimeoutError):
                raise
            except Exception as exc:
                logger.error(f"Unexpected error processing row {idx}: {exc}")
                result_df.at[idx, f"metaqa_base_response_{model_name}"] = f"ERROR: {exc}"
            completed += 1
            # Periodic checkpoint: persist every checkpoint_every rows.
            if (checkpoint_fn is not None and checkpoint_every > 0
                    and completed % checkpoint_every == 0):
                try:
                    checkpoint_fn(result_df)
                except Exception as ce:  # noqa: BLE001
                    logger.warning(f"Checkpoint save failed at {completed}/{total}: {ce}")
        aborted = False
    finally:
        # On abnormal exit, persist the completed rows (completed only counts
        # rows fully written back to result_df).
        if checkpoint_fn is not None and aborted and completed > 0:
            try:
                checkpoint_fn(result_df)
                logger.info(f"Checkpoint saved on abort: {completed}/{total} rows persisted.")
            except Exception as ce:  # noqa: BLE001
                logger.warning(f"Final checkpoint save failed: {ce}")

    return result_df


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_csv_files(input_path: Path) -> Dict[str, pd.DataFrame]:
    """Load CSV files from a directory or a single file."""
    csv_files = {}
    if input_path.is_file():
        try:
            csv_files[input_path.name] = pd.read_csv(input_path)
        except Exception as exc:
            logger.warning(f"Failed to read {input_path}: {exc}")
    else:
        for csv_file in input_path.rglob("*.csv"):
            if "summary" in csv_file.name:
                continue
            try:
                csv_files[str(csv_file.relative_to(input_path))] = pd.read_csv(csv_file)
            except Exception as exc:
                logger.warning(f"Failed to read {csv_file}: {exc}")
    return csv_files


def _atomic_to_csv(df: pd.DataFrame, output_file: Path) -> None:
    """Write ``df`` to ``output_file`` atomically (temp file + ``os.replace``).

    Prevent a checkpoint write from being interrupted halfway (network drop /
    ctrl-C) and corrupting the output file: write to a temp file first, then
    atomically rename, so on disk there is always either the previous complete
    file or the new complete file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    df.to_csv(tmp_file, index=False)
    os.replace(tmp_file, output_file)


def save_single_file(df: pd.DataFrame, output_dir: Path, rel_path: str, model_name: str) -> Dict:
    output_file = output_dir / rel_path
    _atomic_to_csv(df, output_file)

    score_col = f"metaqa_hallucination_score_{model_name}"
    return {
        "file": rel_path,
        "rows": len(df),
        "rows_with_score": int(df[score_col].notna().sum()) if score_col in df.columns else 0,
    }


def save_summary(stats_list: List[Dict], output_dir: Path, model_name: str) -> None:
    total_rows = sum(s["rows"] for s in stats_list)
    total_scored = sum(s["rows_with_score"] for s in stats_list)

    summary_path = output_dir / "metaqa_pipeline_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MetaQA Pipeline Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total files: {len(stats_list)}\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Rows with hallucination score: {total_scored}\n\n")
        for s in stats_list:
            f.write(f"  {s['file']}: {s['rows']} rows, {s['rows_with_score']} scored\n")
    logger.info(f"Summary saved to {summary_path}")


def check_file_already_processed(output_dir: Path, rel_path: str, model_name: str) -> str:
    """Classify an output file as not / partially / fully processed.

    Key fix: the old logic marked a file as fully_processed as long as
    base_response had no ERROR, which misclassified files whose checkpoint had
    only written some rows as "done" and skipped them, leaving the remaining rows
    forever unprocessed. Now we use the score column (the last pipeline step) to
    check whether rows are fully written: scored < total also counts as
    partially_processed, triggering per-row resume.
    """
    output_file = output_dir / rel_path
    if not output_file.exists():
        return "not_processed"
    try:
        df = pd.read_csv(output_file)
        score_col = f"metaqa_hallucination_score_{model_name}"
        base_col = f"metaqa_base_response_{model_name}"
        if score_col not in df.columns or df[score_col].notna().sum() == 0:
            return "not_processed"
        total = len(df)
        scored = int(df[score_col].notna().sum())
        has_errors = any(
            "ERROR:" in str(v) for v in df.get(base_col, pd.Series(dtype=object)).dropna()
        )
        # Unfilled or with ERROR both count as "partially processed" to drive
        # per-row reuse and resume.
        if has_errors or scored < total:
            return "partially_processed"
        return "fully_processed"
    except Exception:
        return "not_processed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="MetaQA pipeline: answer → mutate → verify → score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=str(GOLDEN_DATASET_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE_DIR))
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model config names")
    parser.add_argument("--model", type=str, default=None, help="Model name sent to the API (single model override)")
    parser.add_argument("--model-name", type=str, default=None, help="Display/alias name for output folder and column naming; defaults to --model")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--protocol", type=str, choices=sorted(SUPPORTED_PROTOCOLS), default=None)
    parser.add_argument("--n-synonym", type=int, default=5, help="Number of synonym mutations")
    parser.add_argument("--n-antonym", type=int, default=5, help="Number of antonym mutations")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--rate-limit", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-processed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Periodically checkpoint the in-progress file every N rows so an "
             "interruption (network drop / ctrl-C) only loses the last <N rows. "
             "0 disables. Combined with --skip-processed this gives per-row resume. "
             "(default: 50)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 70)
    print("RQ4 MetaQA Pipeline (Paper-Faithful)")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Synonym mutations per question: {args.n_synonym}")
    print(f"Antonym mutations per question: {args.n_antonym}")

    # Load golden dataset
    csv_files = load_csv_files(input_path)
    print(f"Loaded {len(csv_files)} CSV files")
    if not csv_files:
        logger.error("No CSV files found")
        sys.exit(1)

    if args.limit:
        # For a single file, take only the first limit rows
        limited = {}
        for rel_path, df in csv_files.items():
            limited[rel_path] = df.head(args.limit)
        csv_files = limited
        print(f"Limited to {args.limit} rows per file")

    # Resolve model configs
    if args.models:
        model_configs = args.models.split(",")
    elif args.model:
        model_configs = [args.model]
    else:
        model_configs = ["local"]

    llm_clients = {}
    for mc in model_configs:
        mc = mc.strip()
        base_url, api_key, model_name, protocol = resolve_model_config(mc, args)
        column_model_name = (args.model_name or model_name).replace("/", "_").replace("-", "_")
        try:
            llm_clients[column_model_name] = create_llm_client(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                protocol=protocol,
                max_concurrent=args.max_concurrent,
                rate_limit=args.rate_limit,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"  {column_model_name}: {model_name} ({protocol})")
        except ValueError as exc:
            logger.warning(f"Failed to initialize {mc}: {exc}")

    if not llm_clients:
        logger.error("No LLM clients initialized")
        sys.exit(1)

    # Process with each model
    for model_key, llm_client in llm_clients.items():
        print(f"\n{'=' * 70}")
        print(f"Processing with model: {model_key}")
        print(f"{'=' * 70}")

        model_output_dir = output_dir / f"golden_dataset_metaqa_answer_{model_key}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        stats_list = []
        skipped_count = 0
        skip_processed = args.skip_processed and not args.force

        for rel_path, df in csv_files.items():
            try:
                existing_df = None
                if skip_processed:
                    status = check_file_already_processed(model_output_dir, rel_path, model_key)
                    if status == "fully_processed":
                        logger.info(f"Skipping {rel_path} (fully processed)")
                        skipped_count += 1
                        continue
                    elif status == "partially_processed":
                        try:
                            existing_df = pd.read_csv(model_output_dir / rel_path)
                            logger.info(f"Resuming {rel_path}")
                        except Exception as exc:
                            logger.warning(f"Failed to load existing: {exc}")

                logger.info(f"Processing {rel_path} with {model_key}...")
                # Checkpoint to the same file as the final output; on resume,
                # check_file_already_processed classifies it as partially_processed
                # and reuses rows per-row.
                output_file = model_output_dir / rel_path

                def _checkpoint(frame: pd.DataFrame, _of: Path = output_file) -> None:
                    _atomic_to_csv(frame, _of)

                processed_df = await process_dataframe(
                    df.copy(), llm_client, model_key,
                    args.n_synonym, args.n_antonym,
                    debug=args.debug,
                    existing_df=existing_df,
                    checkpoint_every=args.checkpoint_every,
                    checkpoint_fn=_checkpoint,
                )

                stats = save_single_file(processed_df, model_output_dir, rel_path, model_key)
                stats_list.append(stats)
                logger.info(f"Saved: {rel_path} ({stats['rows']} rows, {stats['rows_with_score']} scored)")

            except (APIConnectionError, APITimeoutError) as exc:
                logger.error(f"Critical error: {exc}")
                if stats_list:
                    save_summary(stats_list, model_output_dir, model_key)
                break
            except Exception as exc:
                logger.error(f"Error processing {rel_path}: {exc}")
                traceback.print_exc()

        if stats_list:
            save_summary(stats_list, model_output_dir, model_key)
        if skipped_count:
            logger.info(f"Skipped {skipped_count} already processed files")
        print(f"Completed {model_key}: {llm_client.request_count} requests made")

    for llm_client in llm_clients.values():
        await llm_client.close()

    print(f"\n{'=' * 70}")
    print("MetaQA pipeline complete!")
    print(f"{'=' * 70}")
    for model_key in llm_clients:
        print(f"  - {output_dir / f'golden_dataset_metaqa_answer_{model_key}'}")


if __name__ == "__main__":
    asyncio.run(main())
