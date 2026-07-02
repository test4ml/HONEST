#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 KONTEST-inspired LLM Answering Script

This script answers KONTEST-style atomic Yes/No paraphrase pairs with the
configured LLM clients and writes per-model answer directories.
"""

import argparse
import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import APIConnectionError, APITimeoutError
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import get_config
from honest.llm import LLMClient, SUPPORTED_PROTOCOLS, create_llm_client

MUTATION_DIR = Path("data/examples/golden_dataset_kontest_mutation")
OUTPUT_BASE_DIR = Path("data/examples")
MUTATION_COLUMNS = ["kontest_atomic_mutations"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_CONFIG_ALIASES = {
    "deepseek-v4-flash": "deepseek_v4_flash",
    "deepseek_v4_flash": "deepseek_v4_flash",
    "glm-5-turbo": "glm_5_turbo",
    "GLM-5-Turbo": "glm_5_turbo",
    "glm_5_turbo": "glm_5_turbo",
}


def resolve_model_config(model_config: str, args) -> Tuple[Optional[str], Optional[str], str, str]:
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

    return args.base_url, args.api_key, args.model or model_config, args.protocol or "openai"


def load_csv_files(input_dir: Path) -> Dict[str, pd.DataFrame]:
    csv_files = {}
    for csv_file in input_dir.rglob("*.csv"):
        if "summary" in csv_file.name or "mutation_summary" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file)
            csv_files[str(csv_file.relative_to(input_dir))] = df
        except Exception as exc:
            logger.warning(f"Failed to read {csv_file}: {exc}")
    return csv_files


def parse_mutations(mutations_str: str) -> List[Dict]:
    if pd.isna(mutations_str) or not mutations_str:
        return []
    try:
        mutations = json.loads(str(mutations_str).strip())
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(mutations, list):
        return []
    return [mutation for mutation in mutations if isinstance(mutation, dict)]


def parse_existing_answers(value) -> Optional[List[Dict]]:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def answer_is_reusable(answer_obj) -> bool:
    if not isinstance(answer_obj, dict):
        return False
    q1_answer = str(answer_obj.get("q1_answer", ""))
    q2_answer = str(answer_obj.get("q2_answer", ""))
    return bool(q1_answer and q2_answer and not q1_answer.startswith("ERROR:") and not q2_answer.startswith("ERROR:"))


def build_yes_no_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Answer with exactly \"Yes\" or \"No\" as the first word. "
        "You may add a short explanation after that."
    )


async def process_row_with_llm(
    row_idx: int,
    row: pd.Series,
    llm_client: LLMClient,
    model_name: str,
    df_columns: List[str],
    debug: bool = False,
    existing_answers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    result = {}

    for mutation_col in MUTATION_COLUMNS:
        if mutation_col not in df_columns:
            continue

        mutations = parse_mutations(row.get(mutation_col, ""))
        answer_col = f"{mutation_col}_{model_name}_answers"
        if not mutations:
            result[answer_col] = None
            continue

        existing_answer_list = None
        if existing_answers and answer_col in existing_answers:
            existing_answer_list = parse_existing_answers(existing_answers[answer_col])

        answers = []
        for idx, mutation in enumerate(mutations):
            if existing_answer_list and idx < len(existing_answer_list) and answer_is_reusable(existing_answer_list[idx]):
                answers.append(existing_answer_list[idx])
                continue

            q1 = mutation.get("original_question", "")
            q2 = mutation.get("mutated_question", "")
            answer_obj = {"q1": q1, "q2": q2, "q1_answer": "", "q2_answer": ""}

            try:
                if debug and row_idx < 3:
                    logger.info(f"Row {row_idx}: answering KONTEST pair {idx + 1}/{len(mutations)}")
                answer_obj["q1_answer"] = await llm_client.generate_answer(build_yes_no_prompt(q1))
                answer_obj["q2_answer"] = await llm_client.generate_answer(build_yes_no_prompt(q2))
            except Exception as exc:
                logger.error(f"Error answering KONTEST pair in row {row_idx}: {exc}")
                if not answer_obj["q1_answer"]:
                    answer_obj["q1_answer"] = f"ERROR: {exc}"
                if not answer_obj["q2_answer"]:
                    answer_obj["q2_answer"] = f"ERROR: {exc}"

            answers.append(answer_obj)

        result[answer_col] = json.dumps(answers, ensure_ascii=False) if answers else None

    return result


async def process_dataframe(
    df: pd.DataFrame,
    llm_client: LLMClient,
    model_name: str,
    debug: bool = False,
    existing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    result_df = df.copy()
    for mutation_col in MUTATION_COLUMNS:
        if mutation_col in df.columns:
            result_df[f"{mutation_col}_{model_name}_answers"] = None

    df_columns = df.columns.tolist()
    tasks = []
    for idx, row in df.iterrows():
        row_existing_answers = None
        if existing_df is not None and idx in existing_df.index:
            row_existing_answers = {}
            for mutation_col in MUTATION_COLUMNS:
                answer_col = f"{mutation_col}_{model_name}_answers"
                if answer_col in existing_df.columns:
                    row_existing_answers[answer_col] = existing_df.at[idx, answer_col]
        task = process_row_with_llm(idx, row, llm_client, model_name, df_columns, debug, row_existing_answers)
        tasks.append((idx, task))

    for idx, task in async_tqdm(tasks, total=len(tasks), desc=f"Answering with {model_name}"):
        try:
            row_result = await task
            for key, value in row_result.items():
                result_df.at[idx, key] = value
        except (APIConnectionError, APITimeoutError):
            raise
        except Exception as exc:
            logger.error(f"Unexpected error processing row {idx}: {exc}")
            for mutation_col in MUTATION_COLUMNS:
                if mutation_col in df.columns:
                    result_df.at[idx, f"{mutation_col}_{model_name}_answers"] = f"ERROR: {exc}"

    return result_df


def save_single_file(df: pd.DataFrame, output_dir: Path, rel_path: str, model_name: str) -> Dict:
    output_file = output_dir / rel_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    answer_col = f"kontest_atomic_mutations_{model_name}_answers"
    return {
        "file": rel_path,
        "rows": len(df),
        "answer_rows": int(df[answer_col].notna().sum()) if answer_col in df.columns else 0,
    }


def save_summary(stats_list: List[Dict], output_dir: Path, model_name: str) -> None:
    total_files = len(stats_list)
    total_rows = sum(item["rows"] for item in stats_list)
    total_answer_rows = sum(item["answer_rows"] for item in stats_list)

    summary_path = output_dir / "answering_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("KONTEST-inspired LLM Answering Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Total rows processed: {total_rows}\n")
        f.write(f"Rows with answers: {total_answer_rows}\n\n")
        f.write("File details:\n")
        f.write("-" * 50 + "\n")
        for item in stats_list:
            f.write(f"  {item['file']}: {item['rows']} rows, {item['answer_rows']} rows with answers\n")

    logger.info(f"Summary saved to: {summary_path}")


def check_file_already_processed(output_dir: Path, rel_path: str, model_name: str) -> str:
    output_file = output_dir / rel_path
    if not output_file.exists():
        return "not_processed"
    try:
        df = pd.read_csv(output_file)
        answer_col = f"kontest_atomic_mutations_{model_name}_answers"
        if answer_col not in df.columns or df[answer_col].notna().sum() == 0:
            return "partially_processed"
        has_errors = any("ERROR:" in str(value) for value in df[answer_col].dropna())
        return "partially_processed" if has_errors else "fully_processed"
    except Exception:
        return "not_processed"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Answer KONTEST-inspired mutations with multiple LLMs")
    parser.add_argument("--input-dir", type=str, default=str(MUTATION_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE_DIR))
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--rate-limit", type=float, default=10.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None, help="Display/alias name for output folder and column naming; defaults to --model")
    parser.add_argument("--protocol", type=str, choices=sorted(SUPPORTED_PROTOCOLS), default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-processed", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)

    print("=" * 70)
    print("RQ4 KONTEST-inspired LLM Answering")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    csv_files = load_csv_files(input_dir)
    print(f"Loaded {len(csv_files)} CSV files")
    if not csv_files:
        logger.error("No CSV files found in input directory")
        sys.exit(1)
    if args.limit:
        csv_files = dict(list(csv_files.items())[:args.limit])
        print(f"Limiting to {args.limit} files")

    if args.models:
        model_configs = args.models.split(",")
    elif args.model:
        model_configs = [args.model]
    else:
        model_configs = ["local"]

    llm_clients = {}
    for model_config in model_configs:
        model_config = model_config.strip()
        base_url, api_key, model_name, protocol = resolve_model_config(model_config, args)
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
            logger.warning(f"Failed to initialize {model_config}: {exc}")

    if not llm_clients:
        logger.error("No LLM clients initialized. Please configure API keys.")
        sys.exit(1)

    for model_key, llm_client in llm_clients.items():
        print(f"\n{'=' * 70}")
        print(f"Processing with model: {model_key}")
        print(f"{'=' * 70}")
        model_output_dir = output_dir / f"golden_dataset_kontest_answer_{model_key}"
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
                    if status == "partially_processed":
                        try:
                            existing_df = pd.read_csv(model_output_dir / rel_path)
                            logger.info(f"Resuming {rel_path}")
                        except Exception as exc:
                            logger.warning(f"Failed to load existing results for {rel_path}: {exc}")

                logger.info(f"Processing {rel_path} with {model_key}...")
                processed_df = await process_dataframe(df.copy(), llm_client, model_key, args.debug, existing_df)
                stats = save_single_file(processed_df, model_output_dir, rel_path, model_key)
                stats_list.append(stats)
                logger.info(f"Saved: {rel_path} ({stats['answer_rows']} rows with answers)")
            except (APIConnectionError, APITimeoutError) as exc:
                logger.error(f"Critical error processing {rel_path}: {exc}")
                if stats_list:
                    save_summary(stats_list, model_output_dir, model_key)
                break
            except Exception as exc:
                logger.error(f"Error processing {rel_path}: {exc}")
                traceback.print_exc()
                save_single_file(df, model_output_dir, rel_path, model_key)

        if stats_list:
            save_summary(stats_list, model_output_dir, model_key)
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already processed files")
        print(f"Completed {model_key}: {llm_client.request_count} requests made")

    for llm_client in llm_clients.values():
        await llm_client.close()

    print("\nAnswering complete!")
    for model_key in llm_clients.keys():
        print(f"  - {output_dir / f'golden_dataset_kontest_answer_{model_key}'}")


if __name__ == "__main__":
    asyncio.run(main())
