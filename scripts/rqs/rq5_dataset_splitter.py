#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5 Dataset Splitter Script

Split the golden dataset into train, validation, and test sets.
Stratified splitting ensures consistent distribution of:
- Mutation methods
- Question types
- Consistency labels

Usage:
    # Split with default ratios (70/15/15)
    python scripts/rqs/rq5_dataset_splitter.py

    # Split with custom ratios
    python scripts/rqs/rq5_dataset_splitter.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Define base data path
DATA_BASE_PATH = Path("$PROJECT_ROOT/data/examples")


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)


def load_golden_dataset(dataset_path: str) -> pd.DataFrame:
    """Load golden dataset from CSV"""
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples from {dataset_path}")
    return df


def analyze_dataset(df: pd.DataFrame) -> Dict:
    """Analyze dataset distribution"""
    analysis = {}

    # By method
    if 'method_name' in df.columns:
        analysis['method_distribution'] = df['method_name'].value_counts().to_dict()

    # By question type
    if 'original_question_type' in df.columns:
        analysis['question_type_distribution'] = df['original_question_type'].value_counts().to_dict()

    # By consistency
    if 'answers_consistent' in df.columns:
        analysis['consistent_count'] = sum(1 for x in df['answers_consistent'] if x == True)
        analysis['inconsistent_count'] = sum(1 for x in df['answers_consistent'] if x == False)

    # By rule
    if 'rule_dir' in df.columns:
        analysis['rule_count'] = df['rule_dir'].nunique()

    return analysis


def print_analysis(analysis: Dict, title: str = "Dataset Analysis"):
    """Print analysis results"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if 'rule_count' in analysis:
        print(f"Number of unique rules: {analysis['rule_count']}")

    if 'method_distribution' in analysis:
        print(f"\nMethod distribution:")
        for method, count in sorted(analysis['method_distribution'].items()):
            print(f"  {method}: {count}")

    if 'question_type_distribution' in analysis:
        print(f"\nQuestion type distribution:")
        for qtype, count in sorted(analysis['question_type_distribution'].items()):
            print(f"  {qtype}: {count}")

    if 'consistent_count' in analysis:
        total = analysis['consistent_count'] + analysis['inconsistent_count']
        print(f"\nConsistency distribution:")
        print(f"  Consistent: {analysis['consistent_count']} ({analysis['consistent_count']/total:.2%})")
        print(f"  Inconsistent: {analysis['inconsistent_count']} ({analysis['inconsistent_count']/total:.2%})")


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets with stratification.

    Stratification is done based on:
    - method_name (mutation type)
    - original_question_type

    Args:
        df: Input dataframe
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    set_random_seed(random_seed)

    # Create stratification key
    df = df.copy()
    if 'method_name' in df.columns and 'original_question_type' in df.columns:
        df['strata_key'] = df['method_name'].astype(str) + '_' + df['original_question_type'].astype(str)
    elif 'method_name' in df.columns:
        df['strata_key'] = df['method_name'].astype(str)
    else:
        df['strata_key'] = 'default'

    # Get unique strata
    unique_strata = df['strata_key'].unique()

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for stratum in unique_strata:
        stratum_df = df[df['strata_key'] == stratum]
        n_samples = len(stratum_df)

        # Calculate split sizes
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val  # Remaining samples go to test

        if n_test < 0:
            # Adjust if ratios don't sum to 1
            n_train = int(n_samples * train_ratio / (train_ratio + val_ratio))
            n_val = n_samples - n_train
            n_test = 0

        # Shuffle within stratum
        stratum_df = stratum_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Split
        train_dfs.append(stratum_df.iloc[:n_train])
        val_dfs.append(stratum_df.iloc[n_train:n_train + n_val])
        if n_test > 0:
            test_dfs.append(stratum_df.iloc[n_train + n_val:])

    # Concatenate all strata
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    if test_dfs:
        test_df = pd.concat(test_dfs, ignore_index=True)
    else:
        test_df = pd.DataFrame()

    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    if not test_df.empty:
        test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
):
    """Save train/val/test splits in separate directories"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create separate directories for each split
    train_dir = output_dir / "train"
    val_dir = output_dir / "valid"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    train_df.to_csv(train_dir / "train.csv", index=False)
    val_df.to_csv(val_dir / "valid.csv", index=False)
    test_df.to_csv(test_dir / "test.csv", index=False)

    print(f"\nSplits saved to {output_dir}:")
    print(f"  train/train.csv: {len(train_df)} samples")
    print(f"  valid/valid.csv: {len(val_df)} samples")
    print(f"  test/test.csv: {len(test_df)} samples")

    # Save split summary
    summary_file = output_dir / "split_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("RQ5 Dataset Split Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}\n\n")
        f.write(f"Train: {len(train_df)} ({len(train_df)/(len(train_df) + len(val_df) + len(test_df)):.2%})\n")
        f.write(f"Validation: {len(val_df)} ({len(val_df)/(len(train_df) + len(val_df) + len(test_df)):.2%})\n")
        f.write(f"Test: {len(test_df)} ({len(test_df)/(len(train_df) + len(val_df) + len(test_df)):.2%})\n\n")

        # Method distribution in each split
        if 'method_name' in train_df.columns:
            f.write("Method distribution:\n")
            f.write("\nTrain:\n")
            for method in sorted(train_df['method_name'].unique()):
                count = sum(1 for x in train_df['method_name'] if x == method)
                f.write(f"  {method}: {count}\n")

            f.write("\nValidation:\n")
            for method in sorted(val_df['method_name'].unique()):
                count = sum(1 for x in val_df['method_name'] if x == method)
                f.write(f"  {method}: {count}\n")

            if not test_df.empty and 'method_name' in test_df.columns:
                f.write("\nTest:\n")
                for method in sorted(test_df['method_name'].unique()):
                    count = sum(1 for x in test_df['method_name'] if x == method)
                    f.write(f"  {method}: {count}\n")

        # Consistency distribution in each split
        if 'answers_consistent' in train_df.columns:
            f.write("\nConsistency distribution:\n")

            train_cons = sum(1 for x in train_df['answers_consistent'] if x == True)
            train_inc = sum(1 for x in train_df['answers_consistent'] if x == False)
            f.write(f"\nTrain: {train_cons} consistent, {train_inc} inconsistent\n")

            val_cons = sum(1 for x in val_df['answers_consistent'] if x == True)
            val_inc = sum(1 for x in val_df['answers_consistent'] if x == False)
            f.write(f"Validation: {val_cons} consistent, {val_inc} inconsistent\n")

            if not test_df.empty:
                test_cons = sum(1 for x in test_df['answers_consistent'] if x == True)
                test_inc = sum(1 for x in test_df['answers_consistent'] if x == False)
                f.write(f"Test: {test_cons} consistent, {test_inc} inconsistent\n")

    print(f"Split summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Split golden dataset into train/val/test sets"
    )
    parser.add_argument(
        "--dataset-path", "-i",
        type=str,
        default="data/examples/golden_dataset/golden_dataset_full.csv",
        help="Path to golden dataset CSV"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/examples/rq5_results/splits",
        help="Output directory for split files"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Ratio for training set (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio for test set (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)"
    )
    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        print("Error: Train, val, and test ratios must sum to 1.0")
        return

    print("=" * 70)
    print("RQ5 Dataset Splitter")
    print("=" * 70)
    print(f"Input dataset: {args.dataset_path}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.seed}")

    # Load golden dataset
    df = load_golden_dataset(args.dataset_path)

    # Analyze original dataset
    analysis = analyze_dataset(df)
    print_analysis(analysis, "Original Dataset Analysis")

    # Split dataset
    print("\n" + "=" * 70)
    print("Splitting dataset...")
    print("=" * 70)

    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # Analyze splits
    print_analysis(analyze_dataset(train_df), "Train Set Analysis")
    print_analysis(analyze_dataset(val_df), "Validation Set Analysis")
    print_analysis(analyze_dataset(test_df), "Test Set Analysis")

    # Save splits
    save_splits(train_df, val_df, test_df, Path(args.output_dir))

    print("\n" + "=" * 70)
    print("Splitting completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
