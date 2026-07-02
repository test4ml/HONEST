#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 QAAskeR Evaluation Script

This script evaluates the consistency of LLM answers on QAAskeR mutations
and calculates the metamorphic violation rate.

The script:
1. Loads QAAskeR mutation files with LLM answers
2. Evaluates each mutation for consistency using the wrapper's check methods
3. Calculates violation rates by MR type and overall
4. Outputs detailed results and statistics

Based on: "Testing Your Question Answering Software via Asking Recursively"
https://github.com/imcsq/ASE21-QAAskeR

Input: CSV files with qaasker mutations and LLM answers
Output: Violation rate statistics and detailed results

Usage:
    # Evaluate all files for a specific model
    python scripts/rqs/rq4_qaasker_evaluation.py \\
        --input-dir data/examples/golden_dataset_qaasker_answer_Qwen2.5_7B_Instruct \\
        --output-dir data/examples/rq4_evaluation_Qwen2.5_7B_Instruct \\
        --model-name Qwen2.5_7B_Instruct

    # Evaluate single file with verbose output
    python scripts/rqs/rq4_qaasker_evaluation.py \\
        --input-file data/examples/golden_dataset_qaasker_answer_Qwen2.5_7B_Instruct/kg_rule_1/entity_rename.csv \\
        --output-dir data/examples/rq4_evaluation \\
        --model-name Qwen2.5_7B_Instruct \\
        --verbose
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.qaasker import QAAskeR

# Define paths
DEFAULT_INPUT_DIR = Path("data/examples/golden_dataset_qaasker_answer_Qwen2.5_7B_Instruct")
DEFAULT_OUTPUT_DIR = Path("data/examples/rq4_results")


@dataclass
class EvaluationResult:
    """Result of evaluating a single mutation"""
    file_path: str
    row_index: int
    mr_type: str
    mutation_index: int
    original_question: str
    mutated_question: str
    target_answer: str
    original_answer: str
    llm_answer: str
    is_violation: bool
    is_consistent: bool
    similarity: float
    method: str
    details: Dict
    metadata: Dict


class QAAskeREvaluator:
    """Evaluate QAAskeR mutations for metamorphic violation detection"""

    def __init__(self, model_name: str = "Unknown"):
        """
        Initialize the evaluator

        Args:
            model_name: Name of the LLM being evaluated
        """
        self.model_name = model_name
        self.qaasker = QAAskeR(random_seed=42)

    def parse_mutations(self, mutations_str: str) -> List[Dict]:
        """
        Parse JSON string of mutations

        Args:
            mutations_str: JSON string containing mutations

        Returns:
            List of mutation dictionaries
        """
        if pd.isna(mutations_str) or not mutations_str:
            return []
        try:
            return json.loads(mutations_str)
        except json.JSONDecodeError:
            return []

    def parse_answers(self, answers_str: str) -> List[str]:
        """
        Parse LLM answers from JSON array string.

        Args:
            answers_str: JSON array string of answers

        Returns:
            List of answer strings
        """
        if pd.isna(answers_str) or not answers_str:
            return []

        answers_str = str(answers_str).strip()

        try:
            return json.loads(answers_str)
        except json.JSONDecodeError:
            return []

    def evaluate_row(
        self,
        row: pd.Series,
        file_path: str,
        mr_column: str,
        answer_column: str,
        mr_type: str
    ) -> List[EvaluationResult]:
        """
        Evaluate a single row for mutations of a specific MR type

        Args:
            row: DataFrame row
            file_path: Path to the source file
            mr_column: Column name for mutations (e.g., 'qaasker_mr1_mutations')
            answer_column: Column name for LLM answers
            mr_type: Type of MR ('MR1', 'MR2', 'MR3')

        Returns:
            List of EvaluationResult objects
        """
        results = []

        # Parse mutations and answers
        mutations = self.parse_mutations(row.get(mr_column, ''))
        answers = self.parse_answers(row.get(answer_column, ''))

        if not mutations:
            return results

        # Evaluate each mutation
        for i, mutation in enumerate(mutations):
            # Get corresponding answer if available
            llm_answer = answers[i] if i < len(answers) else ""

            # Evaluate using wrapper's method
            eval_result = self.qaasker.evaluate_mr_violation(mutation, llm_answer)

            result = EvaluationResult(
                file_path=file_path,
                row_index=row.name if hasattr(row, 'name') else -1,
                mr_type=mr_type,
                mutation_index=i,
                original_question=mutation.get('original_question', ''),
                mutated_question=mutation.get('mutated_question', ''),
                target_answer=mutation.get('target_answer', ''),
                original_answer=mutation.get('answer', ''),
                llm_answer=llm_answer,
                is_violation=eval_result['is_violation'],
                is_consistent=eval_result['is_consistent'],
                similarity=eval_result['similarity'],
                method=eval_result['method'],
                details=eval_result.get('details', {}),
                metadata={
                    'statement': mutation.get('statement', ''),
                    'rouge_1_p': mutation.get('metadata', {}).get('rouge_1_p', 0),
                    'rouge_1_r': mutation.get('metadata', {}).get('rouge_1_r', 0),
                }
            )
            results.append(result)

        return results

    def evaluate_file(
        self,
        file_path: str,
        verbose: bool = False
    ) -> Tuple[List[EvaluationResult], Dict]:
        """
        Evaluate a single CSV file

        Args:
            file_path: Path to the CSV file
            verbose: Whether to show verbose output

        Returns:
            Tuple of (list of results, statistics dictionary)
        """
        df = pd.read_csv(file_path)
        all_results = []

        # Define MR columns
        mr_configs = [
            ('MR1', 'qaasker_mr1_mutations', 'qaasker_mr1_mutations_{}_answers'.format(self.model_name.replace('.', '_').replace('-', '_'))),
            ('MR2', 'qaasker_mr2_mutations', 'qaasker_mr2_mutations_{}_answers'.format(self.model_name.replace('.', '_').replace('-', '_'))),
            ('MR3', 'qaasker_mr3_mutations', 'qaasker_mr3_mutations_{}_answers'.format(self.model_name.replace('.', '_').replace('-', '_'))),
        ]

        # Try exact model name first, then fall back to patterns
        answer_columns = [col for col in df.columns if 'answers' in col and self.model_name.split('_')[-1] in col]

        for mr_type, mr_col, answer_col_pattern in mr_configs:
            # Find the actual answer column
            answer_col = None
            if answer_col_pattern in df.columns:
                answer_col = answer_col_pattern
            else:
                # Try to find matching column
                for col in df.columns:
                    if mr_col in col.replace('_mutations', '_mutations_{}'.format(self.model_name.replace('.', '_').replace('-', '_'))):
                        answer_col = col
                        break
                    if col.endswith(f'_{self.model_name.replace(".", "_")}_answers') or col.endswith(f'_{self.model_name}_answers'):
                        if mr_col.split('_')[2] in col:  # Match MR1/MR2/MR3
                            answer_col = col
                            break

            if mr_col not in df.columns:
                continue

            # Get answer column
            actual_answer_col = None
            for col in df.columns:
                if 'answers' in col and mr_type.lower() in col.lower():
                    actual_answer_col = col
                    break

            if actual_answer_col is None:
                # Try exact pattern match
                pattern = f'qaasker_{mr_type.lower()}_mutations_{self.model_name.replace(".", "_")}_answers'
                if pattern in df.columns:
                    actual_answer_col = pattern

            if actual_answer_col is None:
                if verbose:
                    print(f"  Warning: Could not find answer column for {mr_type}")
                continue

            # Evaluate each row
            for idx, row in df.iterrows():
                results = self.evaluate_row(
                    row,
                    file_path,
                    mr_col,
                    actual_answer_col,
                    mr_type
                )
                all_results.extend(results)

                if verbose and len(results) > 0:
                    for r in results:
                        if r.is_violation:
                            print(f"    VIOLATION: {mr_type} mutation {r.mutation_index} - similarity={r.similarity:.3f}")

        # Calculate statistics
        stats = self._calculate_statistics(all_results)

        return all_results, stats

    def _calculate_statistics(self, results: List[EvaluationResult]) -> Dict:
        """
        Calculate statistics from evaluation results

        Args:
            results: List of EvaluationResult objects

        Returns:
            Statistics dictionary
        """
        total = len(results)
        if total == 0:
            return {
                'total_mutations': 0,
                'violations': 0,
                'consistent': 0,
                'violation_rate': 0.0,
                'by_mr_type': {}
            }

        violations = sum(1 for r in results if r.is_violation)
        consistent = sum(1 for r in results if r.is_consistent)

        # Statistics by MR type
        by_mr = {}
        for mr_type in ['MR1', 'MR2', 'MR3']:
            mr_results = [r for r in results if r.mr_type == mr_type]
            mr_total = len(mr_results)
            mr_violations = sum(1 for r in mr_results if r.is_violation)
            mr_consistent = sum(1 for r in mr_results if r.is_consistent)

            by_mr[mr_type] = {
                'total': mr_total,
                'violations': mr_violations,
                'consistent': mr_consistent,
                'violation_rate': mr_violations / mr_total if mr_total > 0 else 0.0,
                'consistency_rate': mr_consistent / mr_total if mr_total > 0 else 0.0,
            }

        return {
            'total_mutations': total,
            'violations': violations,
            'consistent': consistent,
            'violation_rate': violations / total,
            'consistency_rate': consistent / total,
            'by_mr_type': by_mr,
        }

    def evaluate_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate all CSV files in a directory

        Args:
            input_dir: Directory containing CSV files
            output_dir: Directory to save results
            verbose: Whether to show verbose output

        Returns:
            Combined statistics dictionary
        """
        csv_files = list(input_dir.rglob("*.csv"))
        # Filter out summary files
        csv_files = [f for f in csv_files if "summary" not in f.name and "evaluation" not in f.name]

        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return {}

        all_results = []
        all_stats = {}

        for file_path in tqdm(csv_files, desc="Evaluating files"):
            if verbose:
                print(f"\nEvaluating {file_path.relative_to(input_dir)}...")

            results, stats = self.evaluate_file(str(file_path), verbose)

            # Add file info
            rel_path = str(file_path.relative_to(input_dir))
            stats['file_path'] = rel_path
            all_stats[rel_path] = stats

            all_results.extend(results)

        # Calculate combined statistics
        combined_stats = self._calculate_statistics(all_results)
        combined_stats['by_file'] = {
            k: {
                'total': v['total_mutations'],
                'violations': v['violations'],
                'violation_rate': v['violation_rate'],
            }
            for k, v in all_stats.items()
        }

        # Save results
        self._save_results(all_results, combined_stats, output_dir)

        return combined_stats

    def _save_results(
        self,
        results: List[EvaluationResult],
        stats: Dict,
        output_dir: Path
    ):
        """
        Save evaluation results to files

        Args:
            results: List of EvaluationResult objects
            stats: Statistics dictionary
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_df = pd.DataFrame([asdict(r) for r in results])
        results_file = output_dir / f"qaasker_{self.model_name}_evaluation_results.csv"
        results_df.to_csv(results_file, index=False)

        # Save statistics
        stats_file = output_dir / f"qaasker_{self.model_name}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print(f"\n{'='*70}")
        print(f"QAAskeR Evaluation Results for {self.model_name}")
        print(f"{'='*70}")
        print(f"Total mutations evaluated: {stats['total_mutations']}")
        print(f"Violations: {stats['violations']} ({stats['violation_rate']:.2%})")
        print(f"Consistent: {stats['consistent']} ({stats['consistency_rate']:.2%})")
        print(f"\nBy MR Type:")
        for mr_type, mr_stats in stats.get('by_mr_type', {}).items():
            print(f"  {mr_type}:")
            print(f"    Total: {mr_stats['total']}")
            print(f"    Violations: {mr_stats['violations']} ({mr_stats['violation_rate']:.2%})")
            print(f"    Consistent: {mr_stats['consistent']} ({mr_stats['consistency_rate']:.2%})")
        print(f"{'='*70}\n")

        print(f"Results saved to:")
        print(f"  - {results_file}")
        print(f"  - {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate QAAskeR mutations for RQ4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate all files for a model
    python scripts/rqs/rq4_qaasker_evaluation.py \\
        --input-dir data/examples/golden_dataset_qaasker_answer_Qwen2.5_7B_Instruct \\
        --model-name Qwen2.5_7B_Instruct

    # Evaluate single file
    python scripts/rqs/rq4_qaasker_evaluation.py \\
        --input-file data/examples/golden_dataset_qaasker_answer_Qwen2.5_7B_Instruct/kg_rule_1/entity_rename.csv \\
        --model-name Qwen2.5_7B_Instruct \\
        --verbose
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Input directory containing CSV files with mutations and LLM answers'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Single input CSV file to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Name of the model being evaluated (auto-detected from input-dir if not specified)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output during evaluation'
    )

    args = parser.parse_args()

    # Determine input path
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file {input_path} does not exist")
            sys.exit(1)
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"Error: Input directory {input_path} does not exist")
            sys.exit(1)
    else:
        input_path = DEFAULT_INPUT_DIR
        if not input_path.exists():
            print(f"Error: Default input directory {input_path} does not exist")
            print("Please specify --input-dir or --input-file")
            sys.exit(1)

    # Auto-detect model name from input directory if not specified
    model_name = args.model_name
    if model_name is None:
        # Try to extract model name from input directory path
        # Expected format: .../golden_dataset_qaasker_answer_{model_name}
        dir_name = input_path.name
        prefix = "golden_dataset_qaasker_answer_"
        if dir_name.startswith(prefix):
            model_name = dir_name[len(prefix):]
            print(f"Auto-detected model name: {model_name}")
        else:
            model_name = "Qwen2.5_7B_Instruct"
            print(f"Warning: Could not auto-detect model name, using default: {model_name}")

    output_dir = Path(args.output_dir)

    print("="*70)
    print("RQ4 QAAskeR Evaluation")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Initialize evaluator
    evaluator = QAAskeREvaluator(model_name=model_name)

    # Evaluate
    if input_path.is_file():
        # Single file
        results, stats = evaluator.evaluate_file(str(input_path), verbose=args.verbose)

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator._save_results(results, stats, output_dir)
    else:
        # Directory
        evaluator.evaluate_directory(input_path, output_dir, verbose=args.verbose)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
