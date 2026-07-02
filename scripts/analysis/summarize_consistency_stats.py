#!/usr/bin/env python3
"""
Consistency Results Statistics Summarizer

This script analyzes consistency test results from different metamorphic testing rules
and generates a summary CSV with statistics including:
- Rule name
- Mutation type
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


def parse_consistency_csv(csv_path: str) -> Tuple[int, int, int]:
    """
    Parse a consistency result CSV file and count consistent/inconsistent answers.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (total_count, consistent_count, inconsistent_count)
    """
    total_count = 0
    consistent_count = 0
    inconsistent_count = 0

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                # Check the answers_consistent field
                if row.get('answers_consistent', '').lower() == 'true':
                    consistent_count += 1
                else:
                    inconsistent_count += 1
    except Exception as e:
        print(f"Warning: Error reading {csv_path}: {e}")
        return 0, 0, 0

    return total_count, consistent_count, inconsistent_count


def analyze_rule_directory(rule_dir: Path) -> Dict[str, Tuple[int, int, int]]:
    """
    Analyze all CSV files in a rule directory.

    Args:
        rule_dir: Path to the rule directory

    Returns:
        Dictionary mapping mutation type to (total, consistent, inconsistent) counts
    """
    results = {}

    # Look for CSV files in the directory
    for csv_file in rule_dir.glob("*.csv"):
        mutation_type = csv_file.stem.replace('_llm_answers', '')
        total, consistent, inconsistent = parse_consistency_csv(str(csv_file))

        if total > 0:
            results[mutation_type] = (total, consistent, inconsistent)

    return results


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


def summarize_consistency_results(input_dir: str, output_csv: str):
    """
    Summarize consistency test results from all rules in the input directory.

    Args:
        input_dir: Directory containing rule subdirectories with consistency results
        output_csv: Path to the output CSV file
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Collect all statistics
    all_stats = []

    # Iterate through rule directories
    for rule_dir in sorted(input_path.iterdir()):
        if not rule_dir.is_dir():
            continue

        rule_name = rule_dir.name

        # Analyze the rule directory
        mutation_results = analyze_rule_directory(rule_dir)

        # Add statistics for each mutation type
        for mutation_type, (total, consistent, inconsistent) in mutation_results.items():
            consistency_rate, inconsistency_rate = calculate_rates(total, consistent, inconsistent)

            all_stats.append({
                'rule_name': rule_name,
                'mutation_type': mutation_type,
                'total_tests': total,
                'consistent_count': consistent,
                'inconsistent_count': inconsistent,
                'consistency_rate': f"{consistency_rate:.2f}%",
                'inconsistency_rate': f"{inconsistency_rate:.2f}%"
            })

    # Write to CSV
    if not all_stats:
        print("Warning: No statistics collected. No output file will be generated.")
        return

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'rule_name',
            'mutation_type',
            'total_tests',
            'consistent_count',
            'inconsistent_count',
            'consistency_rate',
            'inconsistency_rate'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    print(f"Statistics summary written to: {output_csv}")
    print(f"Total rules analyzed: {len(set(stat['rule_name'] for stat in all_stats))}")
    print(f"Total entries: {len(all_stats)}")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize consistency test results from metamorphic testing rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results and generate summary CSV
  python summarize_consistency_stats.py -i data/examples/consistency_results_Qwen2.5-7B-Instruct -o consistency_summary.csv

  # Using full paths
  python summarize_consistency_stats.py \\
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
        help='Output CSV file path for the statistics summary'
    )

    args = parser.parse_args()

    summarize_consistency_results(args.input, args.output)


if __name__ == '__main__':
    main()
