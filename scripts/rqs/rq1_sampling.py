#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ1 Sampling: Sample Questions for Manual Annotation

This script samples questions from data/examples/questions for manual annotation.
The sampled questions will be manually inspected for:
- Syntax errors
- Logic errors
- Other quality issues

Directory Structure:
    data/examples/questions/
    ├── kg_rule_1/
    │   ├── body_augmentation.csv
    │   └── entity_rename.csv
    ├── kg_rule_2/
    │   ├── body_augmentation.csv
    │   └── entity_rename.csv
    ...

Sampling Strategy:
1. Stratified sampling by kg_rule (ensure coverage across different rules)
2. Stratified sampling by mutation type
3. Configurable sample size
"""

import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# Configuration
QUESTIONS_BASE_PATH = Path("data/examples/questions")

# Mutation types to sample
MUTATION_TYPES = [
    "body_augmentation",
    "body_permutation",
    "entity_rename",
]

# Sampling configuration
DEFAULT_TOTAL_SAMPLES = 500  # Total samples to draw
DEFAULT_SAMPLES_PER_RULE = 5  # Samples per rule (if not specified)
RANDOM_SEED = 42


def get_all_rule_dirs(questions_path: Path) -> List[str]:
    """Get all kg_rule directories."""
    rule_dirs = sorted([d for d in os.listdir(questions_path)
                       if d.startswith("kg_rule_") and (questions_path / d).is_dir()])
    return rule_dirs


def load_question_file(rule_dir: str, mutation_type: str, questions_path: Path) -> pd.DataFrame:
    """Load a question CSV file."""
    file_path = questions_path / rule_dir / f"{mutation_type}.csv"
    if not file_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        # Add metadata columns
        df["kg_rule"] = rule_dir
        df["mutation_type"] = mutation_type
        return df
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return pd.DataFrame()


def load_all_questions(questions_path: Path,
                      rule_dirs: List[str],
                      mutation_types: List[str]) -> pd.DataFrame:
    """Load all questions from all rule directories and mutation types."""
    all_data = []

    for rule_dir in rule_dirs:
        for mutation_type in mutation_types:
            df = load_question_file(rule_dir, mutation_type, questions_path)
            if not df.empty:
                all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def stratified_sample(df: pd.DataFrame,
                     total_samples: int,
                     samples_per_rule: int,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Perform stratified sampling by kg_rule and mutation type.

    Strategy:
    1. First, uniformly sample from each (kg_rule, mutation_type) combination
    2. If still less than total_samples, randomly sample from remaining data
    3. Ensure no duplicates in random sampling

    Args:
        df: DataFrame with all questions
        total_samples: Total number of samples to draw
        samples_per_rule: Samples per rule (used if total_samples is None)
        seed: Random seed

    Returns:
        Sampled DataFrame
    """
    random.seed(seed)

    # If total_samples is specified, use it; otherwise use samples_per_rule approach
    if total_samples is not None:
        target_samples = total_samples
    else:
        n_rules = df["kg_rule"].nunique()
        target_samples = samples_per_rule * n_rules

    # Get all unique (kg_rule, mutation_type) combinations
    combinations = df.groupby(["kg_rule", "mutation_type"]).size().reset_index(name="count")
    n_combinations = len(combinations)

    # Calculate base samples per combination
    base_samples_per_combination = max(1, target_samples // n_combinations)

    sampled_dfs = []
    sampled_indices = set()

    # Phase 1: Uniformly sample from each combination
    print(f"  Phase 1: Uniformly sampling from {n_combinations} combinations...")
    for _, row in combinations.iterrows():
        rule_dir = row["kg_rule"]
        mutation_type = row["mutation_type"]
        available = row["count"]

        # Get data for this combination
        mask = (df["kg_rule"] == rule_dir) & (df["mutation_type"] == mutation_type)
        mt_df = df[mask]

        # Determine how many to sample
        n_samples = min(base_samples_per_combination, available)

        if n_samples > 0:
            if available <= n_samples:
                # Take all
                sampled_dfs.append(mt_df)
                sampled_indices.update(mt_df.index.tolist())
            else:
                # Random sample
                sampled = mt_df.sample(n=n_samples, random_state=seed)
                sampled_dfs.append(sampled)
                sampled_indices.update(sampled.index.tolist())

    # Check if we need more samples
    current_count = len(sampled_indices)
    remaining_needed = target_samples - current_count

    if remaining_needed > 0:
        # Phase 2: Random sample from remaining data
        print(f"  Phase 2: Need {remaining_needed} more samples, randomly sampling from remaining data...")
        available_indices = set(df.index) - sampled_indices

        if len(available_indices) >= remaining_needed:
            # Sample exactly what we need
            additional_indices = random.sample(list(available_indices), remaining_needed)
            additional_df = df.loc[additional_indices]
            sampled_dfs.append(additional_df)
            sampled_indices.update(additional_indices)
            print(f"  Sampled {len(additional_indices)} additional samples")
        else:
            # Take all remaining
            remaining_df = df.loc[list(available_indices)]
            sampled_dfs.append(remaining_df)
            sampled_indices.update(available_indices)
            print(f"  Took all {len(available_indices)} remaining samples")

    result = pd.concat(sampled_dfs, ignore_index=True)
    print(f"  Total sampled: {len(result)} samples")

    return result


def create_annotation_template(sampled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an annotation template with empty columns for manual annotation.

    Adds columns for:
    - syntax_error: Yes/No
    - logic_error: Yes/No
    - other_issues: Free text
    - notes: Free text
    """
    sampled_df = sampled_df.copy()

    # Add annotation columns at the beginning
    annotation_columns = {
        "syntax_error": "",
        "logic_error": "",
        "other_issues": "",
        "notes": "",
    }

    for col, value in annotation_columns.items():
        sampled_df.insert(0, col, value)

    return sampled_df


def save_sampled_data(sampled_df: pd.DataFrame, output_dir: Path) -> None:
    """Save sampled data to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save complete sampled data
    output_file = output_dir / "rq1_sampled_questions.csv"
    sampled_df.to_csv(output_file, index=False)
    print(f"Saved {len(sampled_df)} sampled questions to {output_file}")

    # Save annotation template
    annotation_df = create_annotation_template(sampled_df)
    annotation_file = output_dir / "rq1_annotation_template.csv"
    annotation_df.to_csv(annotation_file, index=False)
    print(f"Saved annotation template to {annotation_file}")

    # Create summary
    create_sampling_summary(sampled_df, output_dir)


def create_sampling_summary(sampled_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a summary file for the sampled data."""
    summary_lines = [
        "RQ1 Sampling Summary",
        "=" * 60,
        "",
        f"Total samples: {len(sampled_df)}",
        "",
        "Samples by kg_rule:",
    ]

    for rule_dir in sorted(sampled_df["kg_rule"].unique()):
        rule_df = sampled_df[sampled_df["kg_rule"] == rule_dir]
        summary_lines.append(f"  {rule_dir}: {len(rule_df)} samples")

        for mutation_type in sorted(rule_df["mutation_type"].unique()):
            mt_df = rule_df[rule_df["mutation_type"] == mutation_type]
            summary_lines.append(f"    - {mutation_type}: {len(mt_df)}")

        summary_lines.append("")

    # Add mutation type summary
    summary_lines.extend([
        "",
        "Samples by mutation type:",
    ])

    for mutation_type in sorted(sampled_df["mutation_type"].unique()):
        mt_df = sampled_df[sampled_df["mutation_type"] == mutation_type]
        summary_lines.append(f"  {mutation_type}: {len(mt_df)} samples")

    summary_path = output_dir / "sampling_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ1: Sample questions for manual annotation"
    )
    parser.add_argument("--questions-dir", "-q",
                       type=str,
                       default=str(QUESTIONS_BASE_PATH),
                       help="Directory containing questions")
    parser.add_argument("--output-dir", "-o",
                       type=str,
                       default="data/examples/rq1_sampled_questions",
                       help="Output directory for sampled questions")
    parser.add_argument("--total-samples", "-n",
                       type=int,
                       default=DEFAULT_TOTAL_SAMPLES,
                       help="Total number of samples to draw")
    parser.add_argument("--samples-per-rule", "-r",
                       type=int,
                       default=DEFAULT_SAMPLES_PER_RULE,
                       help="Number of samples per rule (overrides --total-samples)")
    parser.add_argument("--seed", "-s",
                       type=int,
                       default=RANDOM_SEED,
                       help="Random seed for reproducibility")
    parser.add_argument("--mutation-types", "-m",
                       nargs="+",
                       default=MUTATION_TYPES,
                       help="Mutation types to sample (default: all mutation types)")
    parser.add_argument("--rules",
                       nargs="+",
                       default=None,
                       help="Specific rules to sample (default: all rules)")

    args = parser.parse_args()

    questions_path = Path(args.questions_dir)
    output_dir = Path(args.output_dir)

    # Get rule directories
    if args.rules:
        rule_dirs = [f"kg_rule_{r}" if not r.startswith("kg_rule_") else r
                    for r in args.rules]
    else:
        rule_dirs = get_all_rule_dirs(questions_path)

    print("=" * 60)
    print("RQ1: Question Sampling for Manual Annotation")
    print("=" * 60)
    print(f"Questions directory: {questions_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of rules: {len(rule_dirs)}")
    print(f"Mutation types: {args.mutation_types}")
    print(f"Random seed: {args.seed}")

    # Load all questions
    print(f"\nLoading questions from {len(rule_dirs)} rule directories...")
    all_df = load_all_questions(questions_path, rule_dirs, args.mutation_types)

    if all_df.empty:
        print("Error: No questions found!")
        return

    print(f"Loaded {len(all_df)} total questions")

    # Perform stratified sampling
    print(f"\nSampling questions...")
    # Use samples_per_rule if explicitly set via flag, otherwise use total_samples
    if args.samples_per_rule != DEFAULT_SAMPLES_PER_RULE:
        sampled_df = stratified_sample(all_df, None, args.samples_per_rule, args.seed)
    else:
        sampled_df = stratified_sample(all_df, args.total_samples, None, args.seed)

    print(f"Sampled {len(sampled_df)} questions")

    # Display distribution
    print("\nSampling distribution:")
    for rule_dir in sorted(sampled_df["kg_rule"].unique()):
        rule_count = len(sampled_df[sampled_df["kg_rule"] == rule_dir])
        print(f"  {rule_dir}: {rule_count}")

    # Save sampled data
    print(f"\nSaving sampled data...")
    save_sampled_data(sampled_df, output_dir)

    print("\n" + "=" * 60)
    print("Sampling complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
