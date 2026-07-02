#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 Golden Dataset Sampling Script

Sample golden dataset from generated questions for RQ4.
The script samples from data/examples/questions directory and outputs
to data/examples/golden_dataset directory.

Sampling Strategy:
1. Stratified sampling across all kg_rule directories
2. Ensure all mutation types are covered
3. Stratify by question type (yes_no, wh_question, true_false, multiple_choice)
4. Fixed random seed for reproducibility

Usage:
    # Sample golden dataset with default settings
    python scripts/rqs/rq4_golden_dataset_sampling.py

    # Specify samples per rule per mutation type
    python scripts/rqs/rq4_golden_dataset_sampling.py --samples-per-combination 3

    # Analyze data distribution only
    python scripts/rqs/rq4_golden_dataset_sampling.py --analyze-only
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
import numpy as np

# Define the base data path
DATA_BASE_PATH = Path("data/examples")
QUESTIONS_DIR = DATA_BASE_PATH / "questions"
OUTPUT_DIR = DATA_BASE_PATH / "golden_dataset"

# Random seed
RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_all_rules() -> List[str]:
    """Get all rule directories."""
    if not QUESTIONS_DIR.exists():
        return []

    return sorted([
        d for d in os.listdir(QUESTIONS_DIR)
        if d.startswith("kg_rule_") and (QUESTIONS_DIR / d).is_dir()
    ])


def get_mutation_files(rule_dir: str) -> List[str]:
    """Get all mutation files under the given rule directory."""
    rule_path = QUESTIONS_DIR / rule_dir
    if not rule_path.exists():
        return []

    return sorted([
        f for f in os.listdir(rule_path)
        if f.endswith('.csv')
    ])


def load_question_csv(rule_dir: str, mutation_file: str) -> Optional[pd.DataFrame]:
    """Load a single CSV file from the questions directory."""
    file_path = QUESTIONS_DIR / rule_dir / mutation_file
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        # Add metadata columns
        df['kg_rule'] = rule_dir
        df['mutation_type'] = mutation_file.replace('.csv', '')
        return df
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return None


def analyze_data_distribution() -> Dict:
    """Analyze the data distribution."""
    print("=" * 70)
    print("Data Distribution Analysis")
    print("=" * 70)

    rules = get_all_rules()
    print(f"\nTotal rule directories: {len(rules)}")

    mutation_stats = defaultdict(lambda: {'count': 0, 'question_types': defaultdict(int)})
    question_type_stats = defaultdict(int)

    total_samples = 0

    for rule in rules:
        mutation_files = get_mutation_files(rule)
        for mutation_file in mutation_files:
            df = load_question_csv(rule, mutation_file)
            if df is None or df.empty:
                continue

            mutation_type = mutation_file.replace('.csv', '')
            mutation_stats[mutation_type]['count'] += len(df)

            # Tally question types
            if 'original_question_type' in df.columns:
                for qtype in df['original_question_type'].value_counts().index:
                    count = df['original_question_type'].value_counts()[qtype]
                    mutation_stats[mutation_type]['question_types'][qtype] += count
                    question_type_stats[qtype] += count

            total_samples += len(df)

    print(f"\nSamples by mutation type:")
    for mutation_type in sorted(mutation_stats.keys()):
        stats = mutation_stats[mutation_type]
        print(f"  {mutation_type}: {stats['count']} samples")
        for qtype, count in sorted(stats['question_types'].items()):
            print(f"    - {qtype}: {count}")

    print(f"\nSamples by question type:")
    for qtype in sorted(question_type_stats.keys()):
        print(f"  {qtype}: {question_type_stats[qtype]}")

    print(f"\nTotal samples: {total_samples}")

    return {
        'total_rules': len(rules),
        'mutation_stats': dict(mutation_stats),
        'question_type_stats': dict(question_type_stats),
        'total_samples': total_samples,
    }


def sample_golden_dataset(
    samples_per_combination: int = 2,
    max_rules: Optional[int] = None,
    stratify_by_question_type: bool = True
) -> pd.DataFrame:
    """
    Sample the golden dataset.

    Args:
        samples_per_combination: number of samples per rule x mutation type combination
        max_rules: maximum number of rules (None means all)
        stratify_by_question_type: whether to stratify sampling by question type

    Returns:
        Golden dataset dataframe
    """
    set_random_seed(RANDOM_SEED)

    rules = get_all_rules()

    if max_rules:
        rules = sorted(rules)[:max_rules]

    print(f"Sampling from {len(rules)} rules...")

    all_samples = []
    sampling_stats = defaultdict(lambda: defaultdict(int))

    for rule in rules:
        mutation_files = get_mutation_files(rule)

        for mutation_file in mutation_files:
            df = load_question_csv(rule, mutation_file)

            if df is None or df.empty:
                continue

            mutation_type = mutation_file.replace('.csv', '')

            if stratify_by_question_type and 'original_question_type' in df.columns:
                # Stratified sampling by question type
                sampled_dfs = []
                question_types = df['original_question_type'].unique()

                # Compute how many samples to draw per question type
                samples_per_type = max(1, samples_per_combination // len(question_types))

                for qtype in question_types:
                    qtype_df = df[df['original_question_type'] == qtype]
                    n_samples = min(samples_per_type, len(qtype_df))

                    if n_samples > 0:
                        sampled = qtype_df.sample(n=n_samples, random_state=RANDOM_SEED)
                        sampled_dfs.append(sampled)
                        sampling_stats[rule][mutation_type] += n_samples

                if sampled_dfs:
                    sampled = pd.concat(sampled_dfs, ignore_index=True)
                else:
                    continue
            else:
                # Random sampling
                n_samples = min(samples_per_combination, len(df))
                if n_samples > 0:
                    sampled = df.sample(n=n_samples, random_state=RANDOM_SEED)
                    sampling_stats[rule][mutation_type] += n_samples
                else:
                    continue

            all_samples.append(sampled)

    if not all_samples:
        print("Warning: No samples collected!")
        return pd.DataFrame()

    golden_df = pd.concat(all_samples, ignore_index=True)

    # Print statistics
    print("\n" + "=" * 70)
    print("Sampling Statistics")
    print("=" * 70)

    total_by_mutation = defaultdict(int)
    for rule in sorted(sampling_stats.keys()):
        print(f"\n{rule}:")
        for mutation_type in sorted(sampling_stats[rule].keys()):
            count = sampling_stats[rule][mutation_type]
            total_by_mutation[mutation_type] += count
            print(f"  {mutation_type}: {count} samples")

    print(f"\nTotal by mutation type:")
    for mutation_type in sorted(total_by_mutation.keys()):
        print(f"  {mutation_type}: {total_by_mutation[mutation_type]} samples")

    print(f"\nGolden dataset total samples: {len(golden_df)}")

    return golden_df


def save_golden_dataset(df: pd.DataFrame, output_dir: Path):
    """Save the golden dataset to multiple CSV files, organized by rule and mutation type."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the full dataset
    output_file = output_dir / "golden_dataset_full.csv"
    df.to_csv(output_file, index=False)
    print(f"\nFull golden dataset saved to: {output_file}")

    # Save separately per rule directory
    for rule in df['kg_rule'].unique():
        rule_df = df[df['kg_rule'] == rule]
        rule_output_dir = output_dir / rule
        rule_output_dir.mkdir(parents=True, exist_ok=True)

        for mutation_type in rule_df['mutation_type'].unique():
            mutation_df = rule_df[rule_df['mutation_type'] == mutation_type]
            mutation_file = rule_output_dir / f"{mutation_type}.csv"

            # Drop metadata columns to keep the original format
            columns_to_drop = ['kg_rule', 'mutation_type']
            output_df = mutation_df.drop(columns=[c for c in columns_to_drop if c in mutation_df.columns])
            output_df.to_csv(mutation_file, index=False)

    print(f"Golden dataset saved by rule and mutation type to: {output_dir}")

    # Save the summary statistics
    summary_file = output_dir / "golden_dataset_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Golden Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("Samples by rule:\n")
        for rule in sorted(df['kg_rule'].unique()):
            count = sum(1 for x in df['kg_rule'] if x == rule)
            f.write(f"  {rule}: {count}\n")

        f.write("\nSamples by mutation type:\n")
        for mutation in sorted(df['mutation_type'].unique()):
            count = sum(1 for x in df['mutation_type'] if x == mutation)
            f.write(f"  {mutation}: {count}\n")

        if 'original_question_type' in df.columns:
            f.write("\nSamples by question type:\n")
            for qtype in sorted(df['original_question_type'].unique()):
                count = sum(1 for x in df['original_question_type'] if x == qtype)
                f.write(f"  {qtype}: {count}\n")

    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ4: Sample golden dataset from generated questions"
    )
    parser.add_argument(
        "--samples-per-combination", "-n",
        type=int,
        default=2,
        help="Number of samples per rule and mutation type combination (default: 2)"
    )
    parser.add_argument(
        "--max-rules", "-r",
        type=int,
        default=None,
        help="Maximum number of rules to sample from (default: all)"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Do not stratify by question type during sampling"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze data distribution, do not perform sampling"
    )
    args = parser.parse_args()

    # First analyze the data distribution
    analyze_data_distribution()

    if args.analyze_only:
        return

    # Sample the golden dataset
    print("\n" + "=" * 70)
    print("Sampling Golden Dataset")
    print("=" * 70)

    golden_df = sample_golden_dataset(
        samples_per_combination=args.samples_per_combination,
        max_rules=args.max_rules,
        stratify_by_question_type=not args.no_stratify
    )

    if not golden_df.empty:
        # Save the golden dataset
        save_golden_dataset(golden_df, OUTPUT_DIR)

        print(f"\nDataset columns: {list(golden_df.columns)}")
        print(f"\nFirst few rows preview:")
        print(golden_df.head())

    print("\n" + "=" * 70)
    print("Sampling Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
