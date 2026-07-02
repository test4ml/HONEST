#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Answer Generation Script

This script:
1. Reads questions (both original and mutated) from CSV files
2. Generates answers using LLM for both question types
3. Outputs CSV files with LLM raw answers for downstream analysis

Note: This script ONLY generates LLM answers. It does NOT:
- Extract conclusions
- Evaluate correctness
- Perform consistency checking
These tasks should be done by separate downstream scripts.

Error Handling:
- Critical errors (APIConnectionError, APITimeoutError) will cause the script to exit immediately
- Partial results are saved before exit to ensure data integrity
- Use --resume flag to continue from where it stopped
- Non-critical errors are logged but allow processing to continue
"""

import os
import sys
import argparse
import traceback

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import asyncio
from typing import List, Dict
import logging
from tqdm.asyncio import tqdm as async_tqdm
from openai import APIConnectionError, APITimeoutError

# Import configuration management
from configs import get_config
from honest.llm import LLMClient, SUPPORTED_PROTOCOLS, create_llm_client, DEFAULT_THINKING_PROMPT

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM Answer Generation and Conclusion Extraction Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # LLM configuration arguments
    parser.add_argument('--base-url', type=str, default=None,
                       help='LLM API base URL (overrides config)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='LLM API key (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model name (overrides config)')

    # Processing arguments
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory containing CSV files (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (overrides config)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of CSV files to process (None for no limit)')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum number of rows to process per file (None for no limit)')
    parser.add_argument('--rules', type=str, default=None,
                       help='Comma-separated list of rule numbers to process (e.g., "1,2,3"). If not specified, all rules will be processed.')
    parser.add_argument('--summary-file', type=str, default='conclusion_extraction_summary.txt',
                       help='Summary report file name')

    # LLM parameters
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='Maximum tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for LLM generation')
    parser.add_argument('--request-delay', type=float, default=0.1,
                       help='Delay between LLM requests in seconds (deprecated, use --rate-limit)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Request timeout in seconds (default: 600, recommended: 600-1200)')
    parser.add_argument('--stream', action='store_true',
                       help='Use streaming mode to avoid timeout on long responses (deprecated, no longer needed)')
    parser.add_argument('--protocol', type=str, choices=sorted(SUPPORTED_PROTOCOLS),
                       default=None,
                       help='LLM protocol to use (default: openai). Use "anthropic" for Claude-style APIs.')
    parser.add_argument('--enable-thinking', action='store_true',
                       help='Enable thinking/reasoning mode for reasoning models (DeepSeek-V4, Qwen3). '
                            'By default thinking is DISABLED for these models to match non-reasoning models.')

    # Async concurrency parameters
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum number of concurrent requests')
    parser.add_argument('--rate-limit', type=float, default=10.0,
                       help='Maximum requests per second (e.g., 10.0 for 10 req/s)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous interrupted run (row-level checkpoint)')
    parser.add_argument('--retry-errors', action='store_true', default=True,
                       help='Retry rows with ERROR answers when resuming (default: True)')

    return parser.parse_args()


# AsyncLLMAnswerGenerator has been replaced by honest.llm.LLMClient
# The type alias is kept for compatibility with the process_single_row / process_csv_file signatures
AsyncLLMAnswerGenerator = LLMClient


async def process_single_row(row_idx: int, row: pd.Series,
                            answer_generator: AsyncLLMAnswerGenerator,
                            stats: Dict) -> Dict:
    """Process a single row asynchronously

    Raises:
        APIConnectionError: When connection to LLM API fails (propagated from generate_answer)
        APITimeoutError: When request times out (propagated from generate_answer)
    """
    # Get original question - safely handle NaN and None values
    original_question_raw = row.get('original_question', '')
    # Multiple safety checks: ensure it is not NaN, not None, and convert to string
    if pd.isna(original_question_raw) or original_question_raw is None:
        original_question = ''
    else:
        original_question = str(original_question_raw).strip()

    # Get mutated question - safely handle NaN and None values
    mutated_question_raw = row.get('mutated_question', '')
    # Multiple safety checks: ensure it is not NaN, not None, and convert to string
    if pd.isna(mutated_question_raw) or mutated_question_raw is None:
        mutated_question = ''
    else:
        mutated_question = str(mutated_question_raw).strip()

    if not original_question or not mutated_question:
        logger.warning(f"Empty question at row {row_idx}")
        return None

    stats["total"] += 1

    # === Generate ORIGINAL and MUTATED answers in parallel ===
    # Note: We don't use return_exceptions=True here so that critical errors
    # (APIConnectionError, APITimeoutError) will propagate and stop the entire process
    try:
        original_llm_answer, mutated_llm_answer = await asyncio.gather(
            answer_generator.generate_answer(original_question),
            answer_generator.generate_answer(mutated_question)
        )

        # Check for ERROR responses (from non-critical exceptions)
        if isinstance(original_llm_answer, str) and original_llm_answer.startswith("ERROR:"):
            stats["errors"] += 1
        if isinstance(mutated_llm_answer, str) and mutated_llm_answer.startswith("ERROR:"):
            stats["errors"] += 1

    except (APIConnectionError, APITimeoutError) as e:
        # Critical errors - propagate to stop the entire process
        logger.error(f"Critical error at row {row_idx}: {e}")
        raise

    except Exception as e:
        # Other unexpected errors - log and mark as error
        logger.error(f"Unexpected error processing row {row_idx}: {e}")
        original_llm_answer = f"ERROR: {str(e)}"
        mutated_llm_answer = f"ERROR: {str(e)}"
        stats["errors"] += 2

    # Build result record (combining original data with LLM answers)
    result = {
        # Keep all original columns
        **row.to_dict(),

        # Add LLM answers (RAW outputs only, no evaluation)
        'original_llm_answer': original_llm_answer,
        'mutated_llm_answer': mutated_llm_answer,
    }

    return result


async def process_csv_file(csv_file_path: str, answer_generator: AsyncLLMAnswerGenerator,
                     input_dir: str, output_dir: str,
                     max_rows: int = None, resume: bool = False,
                     retry_errors: bool = True) -> Dict:
    """Process a single CSV file and generate LLM answers (Async version with row-level checkpointing)

    This function ONLY generates LLM answers. It does NOT:
    - Extract conclusions
    - Evaluate correctness
    - Perform any downstream analysis

    Processes BOTH original and mutated questions for consistency checking.
    Outputs CSV format for easy integration with downstream analysis scripts.

    Args:
        csv_file_path: Path to input CSV file
        answer_generator: Async LLM answer generator instance
        input_dir: Input directory for calculating relative paths
        output_dir: Output directory for saving results
        max_rows: Maximum rows to process (for testing)
        resume: If True, resume from checkpoint (existing output file)

    Raises:
        APIConnectionError: When connection to LLM API fails (critical error)
        APITimeoutError: When request times out (critical error)
    """
    logger.info(f"Processing file: {csv_file_path}")

    # Calculate relative path from input directory
    relative_path = os.path.relpath(csv_file_path, input_dir)
    # Get directory path and filename
    relative_dir = os.path.dirname(relative_path)
    filename = os.path.basename(csv_file_path).replace('.csv', '_llm_answers.csv')

    # Create output directory structure
    output_subdir = os.path.join(output_dir, relative_dir)
    os.makedirs(output_subdir, exist_ok=True)

    # Save results to CSV file
    output_file = os.path.join(output_subdir, filename)

    # Check for existing results (row-level checkpoint)
    processed_rows = set()
    if os.path.exists(output_file):
        if not resume:
            logger.info(f"Results already exist for {csv_file_path}, skipping...")
            return {"total": 0, "errors": 0}
        else:
            # Load existing results to resume
            try:
                existing_df = pd.read_csv(output_file)
                # Use a unique identifier to track processed rows
                # Assuming each row has original_question + mutated_question as unique key
                retry_count = 0
                for _, row in existing_df.iterrows():
                    key = f"{row.get('original_question', '')}||{row.get('mutated_question', '')}"

                    # Check if this row has ERROR / missing answers and should be retried
                    if retry_errors:
                        original_ans = str(row.get('original_llm_answer', ''))
                        mutated_ans = str(row.get('mutated_llm_answer', ''))
                        has_error = original_ans.startswith('ERROR:') or mutated_ans.startswith('ERROR:')
                        # Also retry rows with missing/empty answers (pandas NaN becomes 'nan' via str(), so check explicitly)
                        missing_answer = (
                            bool(pd.isna(row.get('original_llm_answer')))
                            or bool(pd.isna(row.get('mutated_llm_answer')))
                            or original_ans in ('', 'nan', 'None')
                            or mutated_ans in ('', 'nan', 'None')
                        )
                        if has_error or missing_answer:
                            retry_count += 1
                            continue  # Don't add to processed_rows, will retry

                    processed_rows.add(key)

                logger.info(f"Resuming from checkpoint: {len(processed_rows)} rows successfully processed")
                if retry_count > 0:
                    logger.info(f"Will retry {retry_count} rows with ERROR / missing answers")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                processed_rows = set()

    stats = {
        "total": 0,
        "errors": 0
    }

    try:
        df = pd.read_csv(csv_file_path)

        if max_rows:
            df = df.head(max_rows)

        # Filter out already processed rows if resuming
        rows_to_process = []
        for idx, row in df.iterrows():
            key = f"{row.get('original_question', '')}||{row.get('mutated_question', '')}"
            if key not in processed_rows:
                rows_to_process.append((idx, row))

        if not rows_to_process:
            logger.info(f"All rows already processed for {csv_file_path}")
            return {"total": 0, "errors": 0}

        logger.info(f"Processing {len(rows_to_process)} rows asynchronously...")

        # Process all rows concurrently
        tasks = [
            process_single_row(idx, row, answer_generator, stats)
            for idx, row in rows_to_process
        ]

        # Use asyncio.gather to process all rows concurrently
        # Use tqdm for progress tracking
        results = []
        try:
            for coro in async_tqdm.as_completed(tasks, total=len(tasks),
                                               desc=f"Processing {os.path.basename(csv_file_path)}"):
                result = await coro
                if result is not None:
                    results.append(result)

                    # Incremental save every 100 rows (for checkpoint)
                    if len(results) % 100 == 0:
                        await save_results_checkpoint(results, output_file, processed_rows, resume)

        except (APIConnectionError, APITimeoutError) as e:
            # Critical error occurred - save what we have and propagate the error
            logger.error(f"Critical error occurred, saving {len(results)} processed rows before exit")
            if results:
                await save_results_checkpoint(results, output_file, processed_rows, resume)
            raise

        # Final save
        if results:
            await save_results_checkpoint(results, output_file, processed_rows, resume)

        logger.info(f"Results saved to {output_file}")

    except (APIConnectionError, APITimeoutError):
        # Re-raise critical errors to stop the entire process
        raise

    except Exception as e:
        logger.error(f"Error processing file {csv_file_path}: {e}")

    return stats


async def save_results_checkpoint(results: List[Dict], output_file: str,
                                  processed_rows: set, resume: bool):
    """Save results checkpoint incrementally"""
    try:
        df_new = pd.DataFrame(results)

        # If resuming, merge with existing file (update in place to preserve order)
        if resume and os.path.exists(output_file):
            df_existing = pd.read_csv(output_file)

            # Create a dict for fast lookup of new results
            new_data_dict = {}
            for _, row in df_new.iterrows():
                key = str(row.get('original_question', '')) + '||' + str(row.get('mutated_question', ''))
                new_data_dict[key] = row.to_dict()

            # Update existing rows or keep them, preserving order
            updated_rows = []
            existing_keys = set()
            for _, row in df_existing.iterrows():
                key = str(row.get('original_question', '')) + '||' + str(row.get('mutated_question', ''))
                existing_keys.add(key)
                if key in new_data_dict:
                    # Update with new data (retry result)
                    updated_rows.append(new_data_dict[key])
                else:
                    # Keep existing data
                    updated_rows.append(row.to_dict())

            # Append truly new rows (not in existing file)
            for key, row_dict in new_data_dict.items():
                if key not in existing_keys:
                    updated_rows.append(row_dict)

            df_combined = pd.DataFrame(updated_rows)
        else:
            df_combined = df_new

        # Reorder columns: put LLM answer columns at the front
        llm_columns = ['original_llm_answer', 'mutated_llm_answer']
        other_columns = [col for col in df_combined.columns if col not in llm_columns]
        ordered_columns = llm_columns + other_columns
        df_combined = df_combined[ordered_columns]

        df_combined.to_csv(output_file, index=False, encoding='utf-8')

    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def save_summary_report(all_stats: List[Dict], output_file: str):
    """Save summary report - simplified version without evaluation metrics"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM Answer Generation Report\n")
            f.write("(This script ONLY generates LLM answers, no evaluation)\n")
            f.write("=" * 80 + "\n\n")

            total_questions = sum(s["total"] for s in all_stats)
            total_errors = sum(s["errors"] for s in all_stats)

            f.write(f"Total Question Pairs Processed: {total_questions}\n")
            f.write(f"  - Original questions: {total_questions}\n")
            f.write(f"  - Mutated questions: {total_questions}\n")
            f.write(f"  - Total LLM requests: {total_questions * 2}\n\n")

            f.write(f"Errors: {total_errors}\n")
            if total_questions > 0:
                error_rate = (total_errors / (total_questions * 2)) * 100
                f.write(f"Error Rate: {error_rate:.2f}%\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("Per-File Statistics:\n")
            f.write("-" * 80 + "\n\n")

            for i, stats in enumerate(all_stats, 1):
                f.write(f"File {i}:\n")
                f.write(f"  Question Pairs: {stats['total']}\n")
                f.write(f"  LLM Requests: {stats['total'] * 2}\n")
                f.write(f"  Errors: {stats['errors']}\n")
                f.write("\n")

        logger.info(f"Summary report saved to {output_file}")

    except Exception as e:
        logger.error(f"Error saving summary report: {e}")


async def main():
    """Main function (Async version)"""
    # Parse command line arguments
    args = parse_arguments()

    # Configuration - use configuration system
    llm_config = get_config('llm.local', {})
    base_url = args.base_url or llm_config.get('base_url', 'http://localhost:8000/v1')
    api_key = args.api_key or llm_config.get('api_key', 'your-api-key-here')
    model_name = args.model or llm_config.get('default_model', 'Qwen2.5-7B-Instruct')

    # Path configuration
    test_questions_dir = args.input_dir or get_config('paths.examples.questions', 'data/examples/questions')
    output_dir = args.output_dir or get_config('paths.results.llm', 'data/results/llm')
    summary_report_file = args.summary_file
    max_files = args.max_files
    max_rows_per_file = args.max_rows

    logger.info("=" * 80)
    logger.info("LLM Answer Generation (Async with Rate Limiting)")
    logger.info("=" * 80)
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Test questions dir: {test_questions_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max files: {max_files}")
    logger.info(f"Max rows per file: {max_rows_per_file}")
    logger.info(f"Rules filter: {args.rules if args.rules else 'All rules'}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max concurrent requests: {args.max_concurrent}")
    logger.info(f"Rate limit: {args.rate_limit} req/s")
    logger.info(f"Request timeout: {args.timeout}s")
    logger.info(f"Streaming mode: {args.stream} (deprecated)")
    logger.info(f"Protocol: {args.protocol or 'openai'}")
    logger.info(f"Resume mode: {args.resume}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Async LLM answer generator via honest.llm (supports multiple protocols)
    if args.stream:
        logger.warning("--stream flag is deprecated and no longer has any effect. "
                       "The unified LLM client handles timeouts via retry logic.")

    answer_generator = create_llm_client(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        protocol=args.protocol or 'openai',
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        rate_limit=args.rate_limit,
        timeout=args.timeout,
        prompt_template=DEFAULT_THINKING_PROMPT,
        enable_thinking=True if args.enable_thinking else None,
    )

    try:
        # Test LLM connection
        logger.info("Testing LLM connection...")
        test_answer = await answer_generator.generate_answer("What is 2+2?")
        if test_answer.startswith("ERROR:"):
            logger.error("Cannot connect to LLM, please check configuration")
            return
        logger.info(f"LLM connection successful, test answer: {test_answer}")

        # Collect CSV files
        csv_files = []
        for root, _, files in os.walk(test_questions_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        # Sort by rule number
        def extract_rule_number(path):
            try:
                parts = path.split('/')
                for part in parts:
                    # Support both kg_rule_X and memgraph_rule_X formats
                    if part.startswith('kg_rule_'):
                        rule_num = int(part.replace('kg_rule_', ''))
                        return rule_num
                    elif part.startswith('memgraph_rule_'):
                        rule_num = int(part.replace('memgraph_rule_', ''))
                        return rule_num
                return 99999
            except:
                return 99999

        csv_files.sort(key=extract_rule_number)

        # Filter by specified rules if --rules parameter is provided
        if args.rules:
            try:
                # Parse comma-separated rule numbers
                specified_rules = set(int(r.strip()) for r in args.rules.split(','))
                logger.info(f"Filtering by specified rules: {sorted(specified_rules)}")

                # Filter CSV files to only include specified rules
                filtered_files = []
                for csv_file in csv_files:
                    rule_num = extract_rule_number(csv_file)
                    if rule_num in specified_rules:
                        filtered_files.append(csv_file)

                csv_files = filtered_files
                logger.info(f"After filtering by rules: {len(csv_files)} CSV files match")
            except ValueError as e:
                logger.error(f"Invalid --rules parameter format: {args.rules}. Must be comma-separated integers (e.g., '1,2,3')")
                return

        if max_files is not None:
            logger.info(f"Found {len(csv_files)} CSV files, processing first {max_files}")
            csv_files = csv_files[:max_files]
        else:
            logger.info(f"Found {len(csv_files)} CSV files, processing all files")

        # Process all files
        all_stats = []
        try:
            for csv_file in csv_files:
                stats = await process_csv_file(csv_file, answer_generator,
                                        test_questions_dir, output_dir, max_rows_per_file,
                                        resume=args.resume, retry_errors=args.retry_errors)
                all_stats.append(stats)

        except (APIConnectionError, APITimeoutError) as e:
            # Critical error - log and exit gracefully
            logger.error("=" * 80)
            logger.error("CRITICAL ERROR: Connection to LLM API failed")
            logger.error(f"Error: {e}")
            logger.error("=" * 80)
            logger.error("Process stopped to ensure data integrity.")
            logger.error("All processed results have been saved to disk.")
            logger.error(f"Processed {len(all_stats)} files before error occurred.")
            logger.error("You can resume processing by running the script with --resume flag.")
            logger.error("=" * 80)

            # Save partial summary report
            if all_stats:
                partial_summary_file = summary_report_file.replace('.txt', '_partial.txt')
                save_summary_report(all_stats, partial_summary_file)
                logger.info(f"Partial summary report saved to {partial_summary_file}")

            # Exit with error code
            sys.exit(1)

        # Save summary report
        save_summary_report(all_stats, summary_report_file)

        # Final statistics
        logger.info("=" * 80)
        logger.info("Processing Completed Successfully")
        logger.info("=" * 80)
        total_questions = sum(s["total"] for s in all_stats)
        total_errors = sum(s["errors"] for s in all_stats)

        logger.info(f"Total question pairs processed: {total_questions}")
        logger.info(f"Total LLM requests made: {answer_generator.request_count}")
        logger.info(f"Total errors: {total_errors}")

        if answer_generator.request_count > 0:
            error_rate = (total_errors / answer_generator.request_count) * 100
            logger.info(f"Error rate: {error_rate:.2f}%")

    except (APIConnectionError, APITimeoutError):
        # This is already handled above, but if it somehow bubbles up here, exit
        sys.exit(1)

    finally:
        # Close async client
        await answer_generator.close()


if __name__ == '__main__':
    asyncio.run(main())
