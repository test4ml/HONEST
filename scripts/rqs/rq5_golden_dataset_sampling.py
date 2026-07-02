#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5 Golden Dataset Sampling Script

Research Question 5: Can fine-tuning help fix logical errors in LLM responses?

This script samples a golden dataset from the questions directory for RQ5.
The sampling strategy:
1. Sample from all mutation methods across all rules
2. Use fixed random seed for reproducibility
3. Balance between consistent and inconsistent samples (based on NLI consistency)
4. Output dataset for train/val/test splitting

Usage:
    # Sample golden dataset with default parameters
    python scripts/rqs/rq5_golden_dataset_sampling.py

    # Sample with custom parameters
    python scripts/rqs/rq5_golden_dataset_sampling.py --samples-per-rule 3 --max-rules 50

    # Analyze data distribution only
    python scripts/rqs/rq5_golden_dataset_sampling.py --analyze-only
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# Define base data path
DATA_BASE_PATH = Path("$PROJECT_ROOT/data/examples")

# Define source model for NLI consistency results (qwen-7b baseline)
SAMPLE_SOURCE_MODEL = "consistency_results_nli_Qwen2.5-7B-Instruct"

# Define mutation method files
MUTATION_METHODS = {
    "body_augmentation_llm_answers.csv": "body_augmentation",
    "entity_rename_llm_answers.csv": "entity_rename",
    "body_permutation_llm_answers.csv": "body_permutation",
}


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)


def get_all_rules() -> List[str]:
    """Get all rule directories"""
    model_path = DATA_BASE_PATH / SAMPLE_SOURCE_MODEL
    if not model_path.exists():
        return []

    return sorted([d for d in os.listdir(model_path)
                   if d.startswith("kg_rule_") and (model_path / d).is_dir()])


def load_consistency_csv(rule_dir: str, mutation_file: str,
                         model_dir: str = SAMPLE_SOURCE_MODEL) -> pd.DataFrame:
    """Load a single CSV file from consistency results directory"""
    file_path = DATA_BASE_PATH / model_dir / rule_dir / mutation_file
    if not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Warning: Cannot read {file_path}: {e}")
        return None


def analyze_data_distribution():
    """Analyze data distribution"""
    print("=" * 70)
    print("Data Distribution Analysis (NLI Consistency Results - Qwen2.5-7B-Instruct)")
    print("=" * 70)

    rules = get_all_rules()
    print(f"\nTotal rule directories: {len(rules)}")

    method_counts = defaultdict(int)
    inconsistent_counts = defaultdict(int)

    total_samples = 0
    total_inconsistent = 0

    for rule in rules:
        for mutation_file, method_name in MUTATION_METHODS.items():
            df = load_consistency_csv(rule, mutation_file)
            if df is None or df.empty:
                continue

            if 'answers_consistent' not in df.columns:
                continue

            method_counts[method_name] += len(df)

            # Count inconsistent samples
            inconsistent = sum(1 for x in df['answers_consistent'] if x == False)
            inconsistent_counts[method_name] += inconsistent

            total_samples += len(df)
            total_inconsistent += inconsistent

    print(f"\nSamples per method:")
    for method_name, count in sorted(method_counts.items()):
        inc = inconsistent_counts.get(method_name, 0)
        print(f"  {method_name}: {count} samples, {inc} inconsistent ({inc/count:.4f})")

    print(f"\nTotal:")
    print(f"  Total samples: {total_samples}")
    print(f"  Inconsistent samples: {total_inconsistent}")
    print(f"  Overall inconsistency rate: {total_inconsistent/total_samples:.4f}" if total_samples > 0 else "N/A")

    return {
        'total_rules': len(rules),
        'method_counts': dict(method_counts),
        'inconsistent_counts': dict(inconsistent_counts),
        'total_samples': total_samples,
        'total_inconsistent': total_inconsistent,
    }


def sample_golden_dataset(
    samples_per_rule: int = 3,
    max_rules: int = None,
    balance_methods: bool = True,
    ensure_inconsistent: bool = True
) -> pd.DataFrame:
    """
    Sample golden dataset for RQ5

    Args:
        samples_per_rule: Number of samples per rule per method
        max_rules: Maximum number of rules (None means all)
        balance_methods: Whether to balance sampling across methods
        ensure_inconsistent: Whether to prioritize inconsistent samples

    Returns:
        Golden dataset dataframe
    """
    set_random_seed(42)

    rules = get_all_rules()

    if max_rules:
        rules = sorted(rules)[:max_rules]

    print(f"Sampling from {len(rules)} rules...")

    all_samples = []
    method_stats = defaultdict(lambda: {'total': 0, 'inconsistent': 0})

    for rule in rules:
        for mutation_file, method_name in MUTATION_METHODS.items():
            df = load_consistency_csv(rule, mutation_file)

            if df is None or df.empty:
                continue

            if 'answers_consistent' not in df.columns:
                continue

            # Separate consistent and inconsistent samples
            consistent_df = df[df['answers_consistent'] == True]
            inconsistent_df = df[df['answers_consistent'] == False]

            # Sampling strategy: prioritize inconsistent samples
            if ensure_inconsistent and len(inconsistent_df) > 0:
                # Sample inconsistent samples first
                n_inconsistent = min(samples_per_rule // 2, len(inconsistent_df))
                n_consistent = min(samples_per_rule - n_inconsistent, len(consistent_df))

                inconsistent_samples = inconsistent_df.sample(n=n_inconsistent, random_state=42) if n_inconsistent > 0 else pd.DataFrame()
                consistent_samples = consistent_df.sample(n=n_consistent, random_state=42) if n_consistent > 0 else pd.DataFrame()

                sampled = pd.concat([inconsistent_samples, consistent_samples], ignore_index=True)
            else:
                # Random sampling
                n_samples = min(samples_per_rule, len(df))
                if n_samples > 0:
                    sampled = df.sample(n=n_samples, random_state=42)
                else:
                    continue

            # Add metadata
            sampled = sampled.copy()
            sampled['rule_dir'] = rule
            sampled['method_name'] = method_name
            sampled['mutation_file'] = mutation_file

            all_samples.append(sampled)

            # Update statistics
            method_stats[method_name]['total'] += len(sampled)
            method_stats[method_name]['inconsistent'] += sum(
                1 for x in sampled['answers_consistent'] if x == False
            )

    if not all_samples:
        print("Warning: No samples collected!")
        return pd.DataFrame()

    golden_df = pd.concat(all_samples, ignore_index=True)

    # Print statistics
    print("\n" + "=" * 70)
    print("Sampling Statistics")
    print("=" * 70)

    for method_name in sorted(method_stats.keys()):
        stats = method_stats[method_name]
        inc_rate = stats['inconsistent'] / stats['total'] if stats['total'] > 0 else 0
        print(f"\n{method_name}:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Inconsistent samples: {stats['inconsistent']}")
        print(f"  Inconsistency rate: {inc_rate:.4f}")

    print(f"\nGolden dataset total samples: {len(golden_df)}")

    return golden_df


def save_golden_dataset(df: pd.DataFrame, output_dir: Path):
    """Save golden dataset"""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "rq5_golden_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\nGolden dataset saved to: {output_file}")

    # Save summary
    summary_file = output_dir / "rq5_golden_dataset_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("RQ5 Golden Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(df)}\n\n")

        if 'method_name' in df.columns:
            f.write("Samples by method:\n")
            for method in df['method_name'].unique():
                count = sum(1 for x in df['method_name'] if x == method)
                f.write(f"  {method}: {count}\n")

        if 'answers_consistent' in df.columns:
            total_inconsistent = sum(1 for x in df['answers_consistent'] if x == False)
            total_consistent = sum(1 for x in df['answers_consistent'] if x == True)
            f.write(f"\nConsistent samples: {total_consistent}/{len(df)}\n")
            f.write(f"Inconsistent samples: {total_inconsistent}/{len(df)}\n")

    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Sample golden dataset for fine-tuning experiment"
    )
    parser.add_argument(
        "--samples-per-rule", "-n",
        type=int,
        default=3,
        help="Number of samples per rule per method (default: 3)"
    )
    parser.add_argument(
        "--max-rules", "-r",
        type=int,
        default=None,
        help="Maximum number of rules (default: all)"
    )
    parser.add_argument(
        "--no-ensure-inconsistent",
        action="store_true",
        help="Do not prioritize inconsistent samples"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/examples/rq5_results",
        help="Output directory"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze data distribution, do not sample"
    )
    args = parser.parse_args()

    # First analyze data distribution
    analyze_data_distribution()

    if args.analyze_only:
        return

    # Sample golden dataset
    print("\n" + "=" * 70)
    print("Sampling Golden Dataset for RQ5")
    print("=" * 70)

    golden_df = sample_golden_dataset(
        samples_per_rule=args.samples_per_rule,
        max_rules=args.max_rules,
        ensure_inconsistent=not args.no_ensure_inconsistent
    )

    if not golden_df.empty:
        # Save golden dataset
        output_dir = Path(args.output_dir)
        save_golden_dataset(golden_df, output_dir)

        print(f"\nDataset columns: {list(golden_df.columns)}")
        print(f"\nFirst few rows preview:")
        print(golden_df.head())

    print("\n" + "=" * 70)
    print("Sampling completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
