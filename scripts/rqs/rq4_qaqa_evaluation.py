#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 QAQA Evaluation Script

This script evaluates QAQA mutations by checking answer consistency.

QAQA Evaluation Logic:
1. Get the LLM answer to the ORIGINAL question
2. Get the LLM answer to the MUTATED question
3. Use QAQA's is_same_answer() to check if answers are consistent
4. If answers are NOT consistent -> violation detected (bug found)

Error Detection Rate = Number of violations / Total number of mutations

Based on: "Natural Test Generation for Precise Testing of Question Answering Software"
https://github.com/yichuan-cs/QAQA

The key difference from QAAskeR:
- QAAskeR checks if LLM answer == target_answer (specific expected answer)
- QAQA checks if is_same_answer(LLM_mutated_answer, LLM_original_answer) (consistency)

Usage:
    # Evaluate QAQA mutations with LLM answers
    python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_Qwen2.5_7B_Instruct

    # Evaluate specific mutation types
    python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_Qwen2.5_7B_Instruct --mutation-types EC EQ
"""

import argparse
import os
import sys
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.qaqa.wrapper import is_same_answer

# Paths
INPUT_DIR = Path("data/examples/golden_dataset_qaqa_answer")
OUTPUT_DIR = Path("data/examples/rq4_results")


def parse_answer(answer_str: str) -> str:
    """
    Parse answer from various formats

    Args:
        answer_str: Answer string (could be direct answer or JSON array)

    Returns:
        Parsed answer string
    """
    if pd.isna(answer_str) or not answer_str:
        return ""

    answer_str = str(answer_str).strip()

    # Try to parse as JSON array
    try:
        answers = json.loads(answer_str)
        if isinstance(answers, list) and len(answers) > 0:
            return str(answers[0]).strip()
    except (json.JSONDecodeError, TypeError):
        pass

    return answer_str


def get_original_answer(row: pd.Series, model_name: str, use_golden_truth: bool = False) -> str:
    """
    Get the original answer to compare against.

    For QAQA evaluation (CORRECT approach per QAQA paper):
    - Compare LLM answer to ORIGINAL question vs LLM answer to MUTATED question
    - This checks if the QA system is consistent under metamorphic transformations

    Priority:
    1. LLM answer to original question (original_question_{model_name}_answer) - RECOMMENDED
    2. Golden truth answer (original_correct_answer) - FALLBACK

    Args:
        row: DataFrame row
        model_name: Name of the LLM model
        use_golden_truth: If True, force use golden truth instead of LLM answer

    Returns:
        Original answer string
    """
    # Priority 1: Use LLM answer to original question (QAQA's correct approach)
    if not use_golden_truth:
        llm_original_col = f'original_question_{model_name}_answer'
        if llm_original_col in row.index and pd.notna(row.get(llm_original_col)):
            answer = str(row[llm_original_col]).strip()
            if answer and not answer.startswith('ERROR:'):
                return answer

    # Priority 2: Fallback to golden truth
    if pd.notna(row.get('original_correct_answer')):
        return str(row['original_correct_answer']).strip()
    elif pd.notna(row.get('mutated_correct_answer')):
        return str(row['mutated_correct_answer']).strip()

    return ""


def evaluate_mutation_consistency(
    original_answer: str,
    mutated_answer: str,
    mutation_type: str,
    is_boolq: bool = False
) -> Tuple[bool, str, float]:
    """
    Evaluate if the mutated answer is consistent with the original answer.

    Uses QAQA's is_same_answer() function which:
    - For boolean questions: Uses special boolean logic
    - For other questions: Uses phrase-bert semantic similarity (>= 0.76 = same)
    - Falls back to string matching if phrase-bert is unavailable

    Args:
        original_answer: The original (correct) answer
        mutated_answer: The LLM answer to the mutated question
        mutation_type: The type of mutation (EC, EQ, EQC, ETI, TI)
        is_boolq: Whether this is a boolean question

    Returns:
        Tuple of (is_violation, label, confidence)
        - is_violation: True if answers are inconsistent (bug detected)
        - label: 'same' or 'different'
        - confidence: Confidence score (0.0-1.0)
    """
    if not original_answer or not mutated_answer:
        return False, 'unknown', 0.0

    # Use QAQA's is_same_answer function
    is_consistent = is_same_answer(mutated_answer, original_answer, is_bool=is_boolq)

    if is_consistent:
        return False, 'same', 1.0  # No violation, answers are consistent
    else:
        return True, 'different', 0.0  # Violation! Answers are inconsistent


def process_file(
    df: pd.DataFrame,
    model_name: str,
    mutation_types: List[str],
    use_golden_truth: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process a single file and evaluate QAQA mutations.

    Args:
        df: DataFrame with mutations and LLM answers
        model_name: Name of the LLM model (e.g., 'Qwen2.5_7B_Instruct')
        mutation_types: List of mutation types to evaluate (EC, EQ, EQC, ETI, TI)
        use_golden_truth: Whether to use golden truth as original answer

    Returns:
        Tuple of (results DataFrame, statistics dictionary)
    """
    results = []
    stats = {
        'total': 0,
        'violations': 0,
        'by_type': {mt: {'total': 0, 'violations': 0} for mt in mutation_types}
    }

    for idx, row in df.iterrows():
        # Get original answer
        original_answer = get_original_answer(row, model_name, use_golden_truth)

        if not original_answer:
            continue

        # Check question type
        question_type = row.get('original_question_type', row.get('mutated_question_type', ''))
        is_boolq = question_type in ['yes_no', 'true_false']

        # Evaluate each mutation type
        for mut_type in mutation_types:
            mutations_col = f'qaqa_{mut_type.lower()}_mutations'
            answers_col = f'qaqa_{mut_type.lower()}_mutations_{model_name}_answers'

            if mutations_col not in df.columns or answers_col not in df.columns:
                continue

            mutations_str = row.get(mutations_col, '')
            answers_str = row.get(answers_col, '')

            if pd.isna(mutations_str) or pd.isna(answers_str):
                continue

            # Parse mutations and answers
            try:
                mutations = json.loads(mutations_str)
                answers = json.loads(answers_str)

                if not mutations or not answers:
                    continue

                # Evaluate each mutation-answer pair
                for i, (mutation, answer) in enumerate(zip(mutations, answers)):
                    if not answer or answer.startswith('ERROR:'):
                        continue

                    stats['total'] += 1
                    stats['by_type'][mut_type]['total'] += 1

                    # Determine expected answer for comparison
                    # For TI mutation: compare against the COMBINED answer (not original answer)
                    # This matches original QAQA: combine2input computes new_answer = 'yes' if both are 'yes' else 'no'
                    if mut_type == 'TI':
                        # TI uses the mutated_answer (combined answer) as the expected answer
                        expected_answer = mutation.get('mutated_answer', original_answer)
                    else:
                        # EC, EQ, EQC, ETI: compare against original answer
                        expected_answer = original_answer

                    # Evaluate consistency
                    is_violation, label, confidence = evaluate_mutation_consistency(
                        expected_answer,
                        answer,
                        mut_type,
                        is_boolq
                    )

                    if is_violation:
                        stats['violations'] += 1
                        stats['by_type'][mut_type]['violations'] += 1

                    result = {
                        'index': idx,
                        'mutation_type': mut_type,
                        'mutation_index': i,
                        'original_question': mutation.get('original_question', row.get('original_question', '')),
                        'mutated_question': mutation.get('mutated_question', ''),
                        'original_context': mutation.get('original_context', ''),
                        'mutated_context': mutation.get('mutated_context', ''),
                        'original_answer': original_answer,
                        'expected_answer': expected_answer,  # For TI, this is the combined answer
                        'llm_answer': answer,
                        'is_violation': is_violation,
                        'label': label,
                        'confidence': confidence,
                        'question_type': question_type,
                        'kg_rule': row.get('kg_rule', ''),
                        'mutation_type_base': row.get('mutation_type', ''),
                    }
                    results.append(result)

            except (json.JSONDecodeError, TypeError) as e:
                continue

    results_df = pd.DataFrame(results)

    # Calculate error detection rate
    error_detection_rate = stats['violations'] / stats['total'] if stats['total'] > 0 else 0

    return results_df, {**stats, 'error_detection_rate': error_detection_rate}


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate QAQA mutations for RQ4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate with default settings
    python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_Qwen2.5_7B_Instruct

    # Evaluate specific mutation types
    python scripts/rqs/rq4_qaqa_evaluation.py --input-dir data/examples/golden_dataset_qaqa_answer_Qwen2.5_7B_Instruct --mutation-types EC EQ

QAQA Evaluation:
    QAQA evaluates consistency between LLM answers to original vs mutated questions.
    If is_same_answer(llm_mutated_answer, llm_original_answer) == False -> VIOLATION (bug detected)

    The is_same_answer() function uses semantic similarity (phrase-bert >= 0.76)
    instead of exact string matching.
        """
    )
    parser.add_argument(
        '--input-dir', '--input_dir',
        type=str,
        required=True,
        dest='input_dir',
        help='Directory containing LLM-answered QAQA mutations'
    )
    parser.add_argument(
        '--output-dir', '--output_dir',
        type=str,
        default=str(OUTPUT_DIR),
        dest='output_dir',
        help=f'Directory to save evaluation results (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--model-name', '--model_name',
        type=str,
        default=None,
        dest='model_name',
        help='Model name (e.g., Qwen2.5_7B_Instruct). If not specified, extracted from input directory name'
    )
    parser.add_argument(
        '--mutation-types', '--mutation_types',
        type=str,
        default='EC,EQ,EQC,ETI,TI',
        dest='mutation_types',
        help='Comma-separated list of mutation types to evaluate (default: EC,EQ,EQC,ETI,TI)'
    )
    parser.add_argument(
        '--use-golden-truth', '--use_golden_truth',
        action='store_true',
        dest='use_golden_truth',
        help='Use golden truth as original answer instead of LLM answer to original question'
    )
    parser.add_argument(
        '--is-boolq', '--is_boolq',
        action='store_true',
        dest='is_boolq',
        help='Treat all questions as boolean questions (uses special boolean comparison logic)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    # Extract model name from input directory if not specified
    if args.model_name is None:
        dir_name = input_dir.name
        if 'answer_' in dir_name:
            args.model_name = dir_name.split('answer_')[-1]
        else:
            args.model_name = 'model'

    # Clean model name for column matching
    model_name = args.model_name.replace('/', '_').replace('-', '_')

    print("=" * 70)
    print("RQ4 QAQA Evaluation")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {model_name}")
    print(f"Mutation types: {args.mutation_types}")
    print(f"Use golden truth: {args.use_golden_truth}")
    print(f"Is BoolQ: {args.is_boolq}")

    # Parse mutation types
    mutation_types = [mt.strip().upper() for mt in args.mutation_types.split(',')]

    # Load all CSV files
    print(f"\nLoading CSV files from {input_dir}...")
    all_results = []
    all_stats = []

    # Track if we have LLM original answers
    has_llm_original = False
    llm_original_col = f'original_question_{model_name}_answer'

    for csv_file in input_dir.rglob("*.csv"):
        if "summary" in csv_file.name:
            continue

        try:
            df = pd.read_csv(csv_file)
            rel_path = csv_file.relative_to(input_dir)

            # Check if LLM original answer column exists
            if llm_original_col in df.columns and df[llm_original_col].notna().any():
                has_llm_original = True

            print(f"  Processing {rel_path} ({len(df)} rows)...")

            results_df, stats = process_file(
                df,
                model_name,
                mutation_types,
                args.use_golden_truth
            )

            if len(results_df) > 0:
                results_df['source_file'] = str(rel_path)
                all_results.append(results_df)
                stats['source_file'] = str(rel_path)
                all_stats.append(stats)

        except Exception as e:
            print(f"  Warning: Failed to process {csv_file}: {e}")

    if not all_results:
        print("\nNo results to process!")
        sys.exit(0)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Calculate overall statistics
    total_count = len(combined_results)
    violation_count = combined_results['is_violation'].sum()
    error_detection_rate = violation_count / total_count if total_count > 0 else 0

    # Statistics by mutation type
    by_type = {}
    for mut_type in mutation_types:
        type_df = combined_results[combined_results['mutation_type'] == mut_type]
        type_total = len(type_df)
        type_violations = type_df['is_violation'].sum()
        by_type[mut_type] = {
            'total': type_total,
            'violations': type_violations,
            'error_rate': type_violations / type_total if type_total > 0 else 0
        }

    overall_stats = {
        'model_name': model_name,
        'total_count': total_count,
        'violation_count': violation_count,
        'error_detection_rate': error_detection_rate,
        'by_mutation_type': by_type,
        'use_golden_truth': args.use_golden_truth,
        'has_llm_original_answers': has_llm_original,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_dir / f'qaqa_{model_name}_evaluation_results.csv'
    combined_results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")

    # Save statistics
    stats_file = output_dir / f'qaqa_{model_name}_statistics.json'
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(overall_stats), f, indent=2)
    print(f"Statistics saved to: {stats_file}")

    # Determine evaluation mode for display
    if has_llm_original and not args.use_golden_truth:
        eval_mode = "LLM-to-LLM (QAQA correct approach)"
    elif args.use_golden_truth:
        eval_mode = "Golden Truth (LLM vs expected answer)"
    else:
        eval_mode = "Golden Truth (LLM original answers not available)"

    # Print summary
    print("\n" + "=" * 70)
    print("QAQA Evaluation Summary")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Evaluation Mode: {eval_mode}")
    if not has_llm_original and not args.use_golden_truth:
        print(f"WARNING: LLM original answers not found. Using golden truth as fallback.")
        print(f"         For correct QAQA evaluation, re-run with: python scripts/rqs/rq4_qaqa_llm_answer.py --force")
    print(f"Total mutations evaluated: {total_count}")
    print(f"Violations detected: {violation_count}")
    print(f"Error Detection Rate: {error_detection_rate:.4f}")
    print("\nBy Mutation Type:")
    for mut_type, type_stats in by_type.items():
        print(f"  {mut_type}:")
        print(f"    Total: {type_stats['total']}")
        print(f"    Violations: {type_stats['violations']}")
        print(f"    Error Rate: {type_stats['error_rate']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
