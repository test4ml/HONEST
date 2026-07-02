#!/usr/bin/env python3
"""
Question Type Consistency Statistics Summarizer

This script analyzes consistency test results grouped by question types
and generates a summary CSV with statistics including:
- Question type
- Total test count
- Consistent count
- Inconsistent count
- Consistency rate
- Inconsistency rate
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


def parse_consistency_csv(csv_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse a consistency result CSV file and count consistent/inconsistent answers by question type.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary mapping question_type to (consistent_count, inconsistent_count)
    """
    question_type_stats = defaultdict(lambda: [0, 0])  # [consistent, inconsistent]

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Get the original question type
                question_type = row.get('original_question_type', 'unknown')

                # Check if answers are consistent
                is_consistent = row.get('answers_consistent', '').lower() == 'true'

                if is_consistent:
                    question_type_stats[question_type][0] += 1  # consistent
                else:
                    question_type_stats[question_type][1] += 1  # inconsistent

    except Exception as e:
        print(f"Warning: Error reading {csv_path}: {e}")
        return {}

    return dict(question_type_stats)


def analyze_rule_directory(rule_dir: Path) -> Dict[str, Tuple[int, int]]:
    """
    Analyze all CSV files in a rule directory and aggregate by question type.

    Args:
        rule_dir: Path to the rule directory

    Returns:
        Dictionary mapping question_type to (consistent_count, inconsistent_count)
    """
    aggregated_stats = defaultdict(lambda: [0, 0])

    # Look for CSV files in the directory
    for csv_file in rule_dir.glob("*.csv"):
        question_stats = parse_consistency_csv(str(csv_file))

        # Aggregate the statistics
        for question_type, (consistent, inconsistent) in question_stats.items():
            aggregated_stats[question_type][0] += consistent
            aggregated_stats[question_type][1] += inconsistent

    return dict(aggregated_stats)


def calculate_rates(total: int, consistent: int, inconsistent: int) -> Tuple[float, float]:
    """
    Calculate consistency and inconsistency rates.

    Args:
        total: Total number of tests
        consistent: Number of consistent tests
        inconsistent: Number of inconsistent tests

    Returns:
        Tuple of (consistency_rate, inconsistency_rate) as percentages
    """
    if total == 0:
        return 0.0, 0.0

    consistency_rate = (consistent / total) * 100
    inconsistency_rate = (inconsistent / total) * 100

    return consistency_rate, inconsistency_rate


def summarize_by_question_type(input_dir: str, output_csv: str):
    """
    Summarize consistency test results grouped by question type.

    Args:
        input_dir: Directory containing rule subdirectories with consistency results
        output_csv: Path to the output CSV file
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Aggregate statistics across all rules by question type
    global_question_stats = defaultdict(lambda: [0, 0])

    # Iterate through rule directories
    for rule_dir in sorted(input_path.iterdir()):
        if not rule_dir.is_dir():
            continue

        # Analyze the rule directory
        question_stats = analyze_rule_directory(rule_dir)

        # Aggregate into global statistics
        for question_type, (consistent, inconsistent) in question_stats.items():
            global_question_stats[question_type][0] += consistent
            global_question_stats[question_type][1] += inconsistent

    # Prepare output statistics
    all_stats = []

    for question_type in sorted(global_question_stats.keys()):
        consistent, inconsistent = global_question_stats[question_type]
        total = consistent + inconsistent
        consistency_rate, inconsistency_rate = calculate_rates(total, consistent, inconsistent)

        all_stats.append({
            'question_type': question_type,
            'total_tests': total,
            'consistent_count': consistent,
            'inconsistent_count': inconsistent,
            'consistency_rate': f"{consistency_rate:.2f}%",
            'inconsistency_rate': f"{inconsistency_rate:.2f}%"
        })

    # Calculate overall statistics
    total_all = sum(stat['total_tests'] for stat in all_stats)
    consistent_all = sum(stat['consistent_count'] for stat in all_stats)
    inconsistent_all = sum(stat['inconsistent_count'] for stat in all_stats)
    consistency_rate_all, inconsistency_rate_all = calculate_rates(
        total_all, consistent_all, inconsistent_all
    )

    all_stats.append({
        'question_type': 'OVERALL',
        'total_tests': total_all,
        'consistent_count': consistent_all,
        'inconsistent_count': inconsistent_all,
        'consistency_rate': f"{consistency_rate_all:.2f}%",
        'inconsistency_rate': f"{inconsistency_rate_all:.2f}%"
    })

    # Write to CSV
    if not all_stats:
        print("Warning: No statistics collected. No output file will be generated.")
        return

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'question_type',
            'total_tests',
            'consistent_count',
            'inconsistent_count',
            'consistency_rate',
            'inconsistency_rate'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    print(f"Question type statistics summary written to: {output_csv}")
    print(f"Total question types analyzed: {len(all_stats) - 1}")  # -1 for OVERALL
    print(f"Total tests: {total_all}")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize consistency test results grouped by question type',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results and generate summary CSV grouped by question type
  python summarize_by_question_type.py -i data/examples/consistency_results_Qwen2.5-7B-Instruct -o question_type_summary.csv

  # Using full paths
  python summarize_by_question_type.py \\
    --input /path/to/consistency_results \\
    --output /path/to/output.csv
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing rule subdirectories with consistency results'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path for the question type statistics summary'
    )

    args = parser.parse_args()

    summarize_by_question_type(args.input, args.output)


if __name__ == '__main__':
    main()
