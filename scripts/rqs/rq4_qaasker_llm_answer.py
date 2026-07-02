#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 QAAskeR LLM Answering Script

This script reads QAAskeR mutations from data/examples/golden_dataset_qaasker_mutation,
generates LLM answers for the mutated questions, and outputs to
data/examples/golden_dataset_answer_qaasker.

The script:
1. Reads CSV files with QAAskeR mutations (MR1, MR2, MR3)
2. Calls multiple LLMs to answer each mutated question
3. Maintains the original directory structure and CSV format
4. Adds new columns for LLM answers
5. SAVES EACH FILE IMMEDIATELY after processing (incremental saving)
6. Supports row-level resume: detects ERROR entries in existing output and retries only failed mutations

Usage:
    # Answer with default LLM configuration
    python scripts/rqs/rq4_qaasker_llm_answer.py

    # Skip already processed files (resume from interruption, with row-level retry)
    python scripts/rqs/rq4_qaasker_llm_answer.py --skip-processed

    # Force reprocess all files
    python scripts/rqs/rq4_qaasker_llm_answer.py --force

    # Answer with specific models
    python scripts/rqs/rq4_qaasker_llm_answer.py --models Qwen2.5-7B-Instruct,deepseek-v3

    # Limit processing for testing
    python scripts/rqs/rq4_qaasker_llm_answer.py --limit 3
"""

import os
import sys
import argparse
import asyncio
import traceback
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm.asyncio import tqdm as async_tqdm
from openai import APIConnectionError, APITimeoutError
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import get_config
from honest.llm import LLMClient, SUPPORTED_PROTOCOLS, create_llm_client

# Define paths
MUTATION_DIR = Path("data/examples/golden_dataset_qaasker_mutation")
OUTPUT_BASE_DIR = Path("data/examples")

# Random seed
RANDOM_SEED = 42

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_CONFIG_ALIASES = {
    'deepseek-v4-flash': 'deepseek_v4_flash',
    'deepseek_v4_flash': 'deepseek_v4_flash',
    'glm-5-turbo': 'glm_5_turbo',
    'GLM-5-Turbo': 'glm_5_turbo',
    'glm_5_turbo': 'glm_5_turbo',
}


def resolve_model_config(model_config: str, args) -> Tuple[Optional[str], Optional[str], str, str]:
    config_key = MODEL_CONFIG_ALIASES.get(model_config, model_config)

    if config_key == 'local':
        llm_config = get_config('llm.local', {})
        return (
            args.base_url,
            args.api_key,
            args.model or llm_config.get('default_model', 'Qwen2.5-7B-Instruct'),
            args.protocol or llm_config.get('protocol', 'openai')
        )
    if config_key == 'deepseek':
        llm_config = get_config('llm.deepseek', {})
        return (
            args.base_url or llm_config.get('base_url'),
            args.api_key or llm_config.get('api_key'),
            args.model or llm_config.get('default_model', 'deepseek-chat'),
            args.protocol or llm_config.get('protocol', 'openai')
        )
    if config_key in {'deepseek_v4_flash', 'glm_5_turbo', 'anthropic', 'gemini'}:
        llm_config = get_config(f'llm.{config_key}', {})
        return (
            args.base_url or llm_config.get('base_url'),
            args.api_key or llm_config.get('api_key'),
            args.model or llm_config.get('default_model', model_config),
            args.protocol or llm_config.get('protocol', 'openai')
        )

    return args.base_url, args.api_key, args.model or model_config, args.protocol or 'openai'


def load_csv_files(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from mutation directory

    Args:
        input_dir: Mutation directory path

    Returns:
        Dictionary with relative paths as keys and DataFrames as values
    """
    csv_files = {}

    for csv_file in input_dir.rglob("*.csv"):
        # Skip summary files
        if "summary" in csv_file.name or "mutation_summary" in csv_file.name:
            continue

        try:
            df = pd.read_csv(csv_file)
            rel_path = csv_file.relative_to(input_dir)
            csv_files[str(rel_path)] = df
        except Exception as e:
            logger.warning(f"Failed to read {csv_file}: {e}")

    return csv_files


def parse_mutations(mutations_str: str) -> List[Dict]:
    """
    Parse mutations from a JSON array string.

    Args:
        mutations_str: JSON array string [{"mutated_question": "...", "mutation_type": "...", "target_answer": "..."}, {...}]

    Returns:
        List of mutation dictionaries with keys: question, mutation_type, target_answer, statement, original_answer
    """
    if pd.isna(mutations_str) or not mutations_str:
        return []

    mutations_str = str(mutations_str).strip()

    try:
        mutations = json.loads(mutations_str)
        result = []
        for m in mutations:
            if m.get('mutated_question'):
                result.append({
                    'question': m.get('mutated_question', ''),
                    'mutation_type': m.get('mutation_type', ''),
                    'target_answer': m.get('target_answer', ''),
                    'statement': m.get('statement', ''),
                    'original_answer': m.get('answer', '')
                })
        return result
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


async def process_row_with_llm(
    row_idx: int,
    row: pd.Series,
    llm_client: LLMClient,
    model_name: str,
    mr_columns: List[str],
    df_columns: List[str],
    debug: bool = False,
    force_mr2_yes_no: bool = True,
    existing_answers: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Process a single row with LLM

    Args:
        row_idx: Row index
        row: DataFrame row (pd.Series)
        llm_client: LLM client
        model_name: Model name for column naming
        mr_columns: List of MR columns to process
        df_columns: List of all columns in the DataFrame
        debug: Enable debug logging
        force_mr2_yes_no: Force MR2 questions to answer with yes/no prefix (default True)
        existing_answers: Optional dict of existing answer column values for row-level resume.
            Keys are answer column names, values are JSON array strings of answers.
            Mutations with valid (non-ERROR, non-null) existing answers will be skipped.

    Returns:
        Dictionary with LLM answers
    """
    result = {}
    total_mutations = 0
    total_skipped = 0

    for mr_col in mr_columns:
        # Check if column exists in DataFrame (row is Series, use df_columns)
        if mr_col not in df_columns:
            if debug and row_idx == 0:
                logger.info(f"  Column '{mr_col}' not found in DataFrame")
            continue

        mutations_str = row.get(mr_col, '')
        mutations = parse_mutations(mutations_str)

        if not mutations:
            result[f'{mr_col}_{model_name}_answers'] = None
            continue

        total_mutations += len(mutations)

        # Parse existing answers for row-level resume
        answer_key = f'{mr_col}_{model_name}_answers'
        existing_answer_list = None
        if existing_answers and answer_key in existing_answers:
            try:
                existing_val = existing_answers[answer_key]
                if pd.notna(existing_val) and existing_val:
                    existing_answer_list = json.loads(str(existing_val))
            except (json.JSONDecodeError, TypeError):
                existing_answer_list = None

        if debug and row_idx < 3:
            logger.info(f"  Row {row_idx}: Processing {len(mutations)} mutations from '{mr_col}'")
            if existing_answer_list:
                logger.info(f"    Found {len(existing_answer_list)} existing answers for resume check")
            for i, mut in enumerate(mutations[:2]):  # Show first 2
                logger.info(f"    Mutation {i+1}: {mut.get('question', mut)[:100] if isinstance(mut, dict) else mut[:100]}...")

        # Answer each mutation
        answers = []
        for i, mutation in enumerate(mutations):
            # Check existing answer: if present and not ERROR, reuse it directly
            if existing_answer_list and i < len(existing_answer_list):
                existing_ans = existing_answer_list[i]
                if isinstance(existing_ans, str) and not existing_ans.startswith("ERROR:"):
                    answers.append(existing_ans)
                    total_skipped += 1
                    if debug and row_idx < 3:
                        logger.info(f"  Row {row_idx}: Reusing existing answer for mutation {i+1}/{len(mutations)}")
                    continue

            try:
                if debug and row_idx < 3:
                    logger.info(f"  Row {row_idx}: Calling LLM for mutation {i+1}/{len(mutations)}...")

                # Handle both old string format and new dict format
                if isinstance(mutation, dict):
                    question = mutation['question']
                    mutation_type = mutation.get('mutation_type', '')
                    target_answer = mutation.get('target_answer', '')
                else:
                    # Backward compatibility: old string format
                    question = mutation
                    mutation_type = ''
                    target_answer = ''

                # Special handling for MR2: Force yes/no answer
                if force_mr2_yes_no and 'MR2' in mutation_type and 'Wh_to_General' in mutation_type:
                    # Add explicit instruction for yes/no questions
                    prompt_with_instruction = (
                        f"{question}\n\n"
                        f"Please answer with 'yes' or 'no' at the beginning of your response, "
                        f"followed by a brief explanation if needed."
                    )
                    answer = await llm_client.generate_answer(prompt_with_instruction)
                    if debug and row_idx < 3:
                        logger.info(f"  Row {row_idx} [MR2]: Got answer: {answer[:100]}...")
                else:
                    answer = await llm_client.generate_answer(question)
                    if debug and row_idx < 3:
                        logger.info(f"  Row {row_idx}: Got answer: {answer[:100]}...")

                answers.append(answer)
            except Exception as e:
                logger.error(f"Error answering mutation in row {row_idx}: {e}")
                answers.append(f"ERROR: {str(e)}")

        # Store answers as JSON array
        result[f'{mr_col}_{model_name}_answers'] = json.dumps(answers, ensure_ascii=False) if answers else None

    if debug and total_mutations > 0 and row_idx < 3:
        logger.info(f"  Row {row_idx}: Generated answers for {total_mutations} mutations (skipped {total_skipped} existing)")

    return result


async def process_dataframe(
    df: pd.DataFrame,
    llm_client: LLMClient,
    model_name: str,
    mr_columns: List[str],
    debug: bool = False,
    force_mr2_yes_no: bool = True,
    existing_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Process a DataFrame with LLM

    Args:
        df: Input DataFrame
        llm_client: LLM client
        model_name: Model name for column naming
        mr_columns: List of MR columns to process
        debug: Enable debug logging
        force_mr2_yes_no: Force MR2 questions to answer with yes/no prefix
        existing_df: Optional existing output DataFrame for row-level resume.
            Rows with valid (non-ERROR) answers will be reused without calling LLM.

    Returns:
        DataFrame with LLM answers
    """
    result_df = df.copy()

    if debug:
        logger.info(f"  DataFrame shape: {df.shape}")
        logger.info(f"  DataFrame columns: {list(df.columns)}")
        logger.info(f"  MR columns to process: {mr_columns}")
        if existing_df is not None:
            logger.info(f"  Existing results loaded: {existing_df.shape}")
        logger.info(f"  Checking which MR columns exist in DataFrame...")
        for mr_col in mr_columns:
            exists = mr_col in df.columns
            if exists:
                count = df[mr_col].notna().sum()
                logger.info(f"    '{mr_col}': EXISTS, {count} non-null values")
            else:
                logger.info(f"    '{mr_col}': NOT FOUND")

    # Initialize answer columns
    for mr_col in mr_columns:
        if mr_col in df.columns:
            result_df[f'{mr_col}_{model_name}_answers'] = None

    # Get DataFrame columns for checking
    df_columns = df.columns.tolist()

    # Process each row
    tasks = []
    for idx, row in df.iterrows():
        # Extract this row's answers from existing results for row-level resume
        row_existing_answers = None
        if existing_df is not None and idx in existing_df.index:
            row_existing_answers = {}
            for mr_col in mr_columns:
                answer_key = f'{mr_col}_{model_name}_answers'
                if answer_key in existing_df.columns:
                    row_existing_answers[answer_key] = existing_df.at[idx, answer_key]

        task = process_row_with_llm(
            idx, row, llm_client, model_name, mr_columns, df_columns,
            debug, force_mr2_yes_no, row_existing_answers
        )
        tasks.append((idx, task))

    # Process rows concurrently
    for idx, task in async_tqdm(tasks, total=len(tasks), desc=f"Answering with {model_name}"):
        try:
            result = await task

            # Update result DataFrame
            for key, value in result.items():
                result_df.at[idx, key] = value

        except (APIConnectionError, APITimeoutError) as e:
            logger.error(f"Critical error at row {idx}: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error processing row {idx}: {e}")
            for mr_col in mr_columns:
                if mr_col in df.columns:
                    result_df.at[idx, f'{mr_col}_{model_name}_answers'] = f"ERROR: {str(e)}"

    return result_df


def save_single_file(
    df: pd.DataFrame,
    output_dir: Path,
    rel_path: str,
    model_name: str
):
    """
    Save a single processed CSV file to output directory

    Args:
        df: Processed DataFrame
        output_dir: Output directory path
        rel_path: Relative path of the file
        model_name: Name of the model

    Returns:
        Dictionary with statistics for this file
    """
    output_file = output_dir / rel_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    # Count answers for this file
    stats = {
        'file': rel_path,
        'rows': len(df),
        'answers': 0
    }

    answer_col = f'qaasker_mr1_mutations_{model_name}_answers'
    if answer_col in df.columns:
        stats['answers'] += df[answer_col].notna().sum()
    answer_col = f'qaasker_mr2_mutations_{model_name}_answers'
    if answer_col in df.columns:
        stats['answers'] += df[answer_col].notna().sum()
    answer_col = f'qaasker_mr3_mutations_{model_name}_answers'
    if answer_col in df.columns:
        stats['answers'] += df[answer_col].notna().sum()

    return stats


def save_summary(
    stats_list: List[Dict],
    output_dir: Path,
    model_name: str,
    summary_file: str = "answering_summary.txt"
):
    """
    Save summary statistics after processing all files

    Args:
        stats_list: List of file statistics
        output_dir: Output directory path
        model_name: Name of the model
        summary_file: Summary file name
    """
    # Aggregate statistics
    total_files = len(stats_list)
    total_rows = sum(s['rows'] for s in stats_list)
    total_answers = sum(s['answers'] for s in stats_list)

    # Save summary
    summary_path = output_dir / summary_file
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("QAAskeR LLM Answering Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Total rows processed: {total_rows}\n")
        f.write(f"Total answers generated: {total_answers}\n\n")
        f.write("File details:\n")
        f.write("-" * 50 + "\n")
        for s in stats_list:
            f.write(f"  {s['file']}: {s['rows']} rows, {s['answers']} answers\n")

    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Total answers: {total_answers}")


def check_file_already_processed(
    output_dir: Path,
    rel_path: str,
    model_name: str,
    mr_columns: List[str]
) -> str:
    """
    Check if a file has already been processed (has all answer columns without errors)

    Args:
        output_dir: Output directory path
        rel_path: Relative path of the file
        model_name: Name of the model
        mr_columns: List of MR columns to check

    Returns:
        "fully_processed" if file is complete with no errors,
        "partially_processed" if file exists but has ERROR or missing answers,
        "not_processed" if file doesn't exist or can't be read
    """
    output_file = output_dir / rel_path
    if not output_file.exists():
        return "not_processed"

    try:
        df = pd.read_csv(output_file)
        has_all_columns = True
        has_errors = False

        for mr_col in mr_columns:
            answer_col = f'{mr_col}_{model_name}_answers'
            if answer_col not in df.columns:
                has_all_columns = False
                continue
            # Check if at least one answer is generated
            if df[answer_col].notna().sum() == 0:
                has_all_columns = False
                continue
            # Check for ERROR values
            for val in df[answer_col].dropna():
                val_str = str(val)
                if 'ERROR:' in val_str:
                    has_errors = True
                    break
            if has_errors:
                break

        if has_all_columns and not has_errors:
            return "fully_processed"
        elif has_all_columns or output_file.exists():
            return "partially_processed"
        else:
            return "not_processed"
    except Exception:
        return "not_processed"


async def main():
    parser = argparse.ArgumentParser(
        description="Answer QAAskeR mutations with multiple LLMs"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(MUTATION_DIR),
        help=f'Input mutation directory (default: {MUTATION_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_BASE_DIR),
        help=f'Base output directory for answers (default: {OUTPUT_BASE_DIR})'
    )
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help='Comma-separated list of model configurations (e.g., "local,deepseek"). Default: uses config from configs/default.yaml'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum concurrent requests per model (default: 10)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=10.0,
        help='Maximum requests per second per model (default: 10.0)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Request timeout in seconds (default: 600)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Maximum tokens for LLM response (default: 2048)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for LLM generation (default: 0.0)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM (overrides config)'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='Base URL for LLM API (overrides config)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name sent to the API (overrides config)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Display/alias name for output folder and column naming; defaults to --model'
    )
    parser.add_argument(
        '--protocol',
        type=str,
        choices=sorted(SUPPORTED_PROTOCOLS),
        default=None,
        help='LLM protocol override (default: from config, usually openai)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--skip-processed',
        action='store_true',
        help='Skip fully processed files and resume partially processed files with row-level retry (default: False)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocess all files, even if already processed (default: False)'
    )
    parser.add_argument(
        '--force-mr2-yes-no',
        action='store_true',
        default=True,
        help='Force MR2 questions to answer with yes/no prefix (default: True)'
    )
    parser.add_argument(
        '--no-force-mr2-yes-no',
        dest='force_mr2_yes_no',
        action='store_false',
        help='Disable forcing MR2 questions to answer with yes/no prefix'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)

    print("=" * 70)
    print("RQ4 QAAskeR LLM Answering")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load CSV files
    print(f"\nLoading CSV files from {input_dir}...")
    csv_files = load_csv_files(input_dir)
    print(f"Loaded {len(csv_files)} CSV files")

    if not csv_files:
        logger.error("No CSV files found in input directory")
        sys.exit(1)

    # Apply limit
    if args.limit:
        limited_files = dict(list(csv_files.items())[:args.limit])
        print(f"Limiting to {args.limit} files")
        csv_files = limited_files

    # Initialize LLM clients
    print("\nInitializing LLM clients...")

    # MR columns to process
    mr_columns = ['qaasker_mr1_mutations', 'qaasker_mr2_mutations', 'qaasker_mr3_mutations']

    # Determine models to use
    if args.models:
        model_configs = args.models.split(',')
    elif args.model:
        # If --model is specified, use it as the model name
        model_configs = [args.model]
    else:
        # Default: use 'local' model from config
        model_configs = ['local']

    llm_clients = {}

    for model_config in model_configs:
        model_config = model_config.strip()

        base_url, api_key, model_name, protocol = resolve_model_config(model_config, args)

        # Create column-safe model name
        column_model_name = (args.model_name or model_name).replace('/', '_').replace('-', '_')

        try:
            # Create LLM client using unified function
            llm_client = create_llm_client(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                protocol=protocol,
                max_concurrent=args.max_concurrent,
                rate_limit=args.rate_limit,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

            llm_clients[column_model_name] = llm_client
            print(f"  {column_model_name}: {model_name} ({protocol})")

        except ValueError as e:
            logger.warning(f"Failed to initialize {model_config}: {e}")
            logger.info(f"  Skipping {model_config}")
            continue

    if not llm_clients:
        logger.error("No LLM clients initialized. Please configure API keys.")
        sys.exit(1)

    # Process files for each model separately
    print("\nProcessing files...")

    for model_key, llm_client in llm_clients.items():
        print(f"\n{'='*70}")
        print(f"Processing with model: {model_key}")
        print(f"{'='*70}")

        # Create output directory for this model
        model_output_dir = output_dir / f"golden_dataset_qaasker_answer_{model_key}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics for this model
        stats_list = []
        skipped_count = 0

        # Determine if we should skip processed files
        skip_processed = args.skip_processed and not args.force

        # Process each file and save immediately
        for rel_path, df in csv_files.items():
            try:
                existing_df = None

                # Row-level resume: load existing results, detect ERROR and missing entries
                if skip_processed:
                    status = check_file_already_processed(model_output_dir, rel_path, model_key, mr_columns)
                    if status == "fully_processed":
                        logger.info(f"Skipping {rel_path} (fully processed, no errors)")
                        skipped_count += 1
                        continue
                    elif status == "partially_processed":
                        # Load existing results for row-level resume
                        try:
                            existing_df = pd.read_csv(model_output_dir / rel_path)
                            # Count entries that need to be reprocessed
                            error_count = 0
                            for mr_col in mr_columns:
                                answer_col = f'{mr_col}_{model_key}_answers'
                                if answer_col in existing_df.columns:
                                    for val in existing_df[answer_col].dropna():
                                        if 'ERROR:' in str(val):
                                            error_count += 1
                            logger.info(f"Resuming {rel_path} (found {error_count} error entries, will retry)")
                        except Exception as e:
                            logger.warning(f"Failed to load existing results for {rel_path}: {e}")
                            existing_df = None

                logger.info(f"Processing {rel_path} with {model_key}...")

                # Process with single model, pass existing results for row-level resume
                processed_df = await process_dataframe(
                    df.copy(), llm_client, model_key, mr_columns,
                    args.debug, args.force_mr2_yes_no, existing_df
                )

                # Save immediately after processing
                stats = save_single_file(processed_df, model_output_dir, rel_path, model_key)
                stats_list.append(stats)

                logger.info(f"  Saved: {rel_path} ({stats['answers']} answers)")

            except (APIConnectionError, APITimeoutError) as e:
                logger.error(f"Critical error processing {rel_path}: {e}")
                logger.error("Saving partial results and exiting...")
                # Save summary for files processed so far
                if stats_list:
                    save_summary(stats_list, model_output_dir, model_key)
                break

            except Exception as e:
                logger.error(f"Error processing {rel_path}: {e}")
                traceback.print_exc()
                # Save original file on error
                save_single_file(df, model_output_dir, rel_path, model_key)

        # Save final summary
        if stats_list:
            print(f"\nSaving summary for {model_key}...")
            save_summary(stats_list, model_output_dir, model_key)

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already processed files")

        print(f"Completed {model_key}: {llm_client.request_count} requests made")

    # Close clients
    for llm_client in llm_clients.values():
        await llm_client.close()

    # Print final statistics
    print("\n" + "=" * 70)
    print("Answering Complete!")
    print("=" * 70)
    print(f"Output directories created:")
    for model_key in llm_clients.keys():
        model_output_dir = output_dir / f"golden_dataset_qaasker_answer_{model_key}"
        print(f"  - {model_output_dir}")


if __name__ == '__main__':
    asyncio.run(main())
