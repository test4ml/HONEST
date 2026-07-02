#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 QAAskeR Mutation Generator Script

This script reads golden dataset from data/examples/golden_dataset,
applies QAAskeR mutations, and outputs to data/examples/golden_dataset_qaasker_mutation.

QAAskeR provides three metamorphic relations (MRs):
- MR1 (Wh→New Wh): Q2S → S2W → UniLM → new Wh-question
- MR2 (Wh→General): Q2S → S2G → General question (Yes/No)
- MR3 (General→Wh): GA2S → S2W → UniLM → new Wh-question

Complete Flow:
- MR1: Wh-question + Answer -> Q2S (declarative statement) -> S2W (extract targets) -> generate new Wh-questions
- MR2: Wh-question + Answer -> Q2S (declarative statement) -> S2G (general question)
- MR3: General/Alternative + Answer -> GA2S (declarative statement) -> S2W -> generate Wh-questions

Based on: "Testing Your Question Answering Software via Asking Recursively"
https://github.com/imcsq/ASE21-QAAskeR

Usage:
    # Generate mutations for golden dataset
    python scripts/rqs/rq4_qaasker_mutations.py

    # Generate only specific MR types
    python scripts/rqs/rq4_qaasker_mutations.py --no-mr1 --no-mr3

    # Show detailed output
    python scripts/rqs/rq4_qaasker_mutations.py --verbose
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.qaasker import QAAskeR, MutationResult

# Define paths
GOLDEN_DATASET_DIR = Path("data/examples/golden_dataset")
OUTPUT_DIR = Path("data/examples/golden_dataset_qaasker_mutation")

# Random seed
RANDOM_SEED = 42


def load_csv_files(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the golden dataset directory

    Args:
        input_dir: Path to the golden dataset directory

    Returns:
        Dictionary mapping relative paths to DataFrames
    """
    csv_files = {}

    # Traverse all subdirectories
    for csv_file in input_dir.rglob("*.csv"):
        # Skip summary files
        if "summary" in csv_file.name:
            continue

        try:
            df = pd.read_csv(csv_file)
            # Use the path relative to input_dir as the key
            rel_path = csv_file.relative_to(input_dir)
            csv_files[str(rel_path)] = df
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")

    return csv_files


def mutation_result_to_dict(mut: MutationResult) -> Dict[str, Any]:
    """
    Convert a MutationResult into a serializable dictionary

    Args:
        mut: MutationResult object

    Returns:
        Mutation result in dictionary form
    """
    return {
        'original_question': mut.original_question,
        'mutated_question': mut.mutated_question,
        'mutation_type': mut.mutation_type,
        'target_answer': mut.target_answer,
        'statement': mut.statement,
        'premise': mut.premise,  # Add premise field
        'is_consistent': mut.is_consistent,
        'answer': mut.answer,
        'metadata': mut.metadata or {},
    }


def apply_qaasker_mutations(
    qaasker: QAAskeR,
    question: str,
    answer: Optional[str] = None,
    question_type_label: Optional[str] = None,
    apply_mr1: bool = True,
    apply_mr2: bool = True,
    apply_mr3: bool = True,
    verbose: bool = False
) -> Dict[str, List[Dict]]:
    """
    Apply QAAskeR mutations to a single question

    Fully implements three metamorphic relations:
    - MR1: Wh-question -> Q2S -> S2W -> new Wh-questions
    - MR2: Wh-question -> Q2S -> S2G -> General question
    - MR3: General/Alternative -> GA2S -> S2W -> Wh-questions

    MR application rules (based on the QAAskeR paper):
    - MR1/MR2: only applied to wh_question type
    - MR3: only applied to yes_no, true_false, alternative types
    - multiple_choice: skipped (no MR is applied)

    Args:
        qaasker: QAAskeR instance
        question: Original question
        answer: Answer (optional but strongly recommended)
        question_type_label: Question type label from the dataset (wh_question, yes_no, true_false, multiple_choice, etc.)
                           If provided, this label takes precedence over auto-detection
        apply_mr1: Whether to apply MR1
        apply_mr2: Whether to apply MR2
        apply_mr3: Whether to apply MR3
        verbose: Whether to show detailed output

    Returns:
        Dictionary containing mutation results for each MR type
    """
    results = {
        'MR1': [],
        'MR2': [],
        'MR3': []
    }

    if not question or pd.isna(question):
        return results

    try:
        # Determine the question type: prefer the dataset label, fall back to auto-detection
        if question_type_label:
            # Use the question type label from the dataset
            if question_type_label == 'wh_question':
                q_type = 'wh'
                wh_type = None
            elif question_type_label in ['yes_no', 'true_false']:
                q_type = 'boolean'
                wh_type = None
            elif question_type_label == 'multiple_choice':
                if verbose:
                    print(f"  Skipping multiple_choice question (by label)")
                return results
            else:
                # Unknown type, fall back to auto-detection
                q_type, wh_type = qaasker.detect_question_type(question)
        else:
            # Auto-detect the question type
            q_type, wh_type = qaasker.detect_question_type(question)

        if verbose:
            label_info = f" (label: {question_type_label})" if question_type_label else ""
            print(f"  Question type: {q_type}, WH type: {wh_type}{label_info}")

        # Apply the corresponding MR based on question type
        # MR1 and MR2 only apply to wh_question type
        if q_type == 'wh' and answer:
            # MR1: Wh → New Wh (full pipeline: Q2S → S2W → generate questions)
            if apply_mr1:
                mr1_results = qaasker.mr1_wh_to_new_wh(question, answer)
                for mut in mr1_results:
                    results['MR1'].append(mutation_result_to_dict(mut))
                if verbose and mr1_results:
                    print(f"  MR1: Generated {len(mr1_results)} mutations")
                    for mut in mr1_results[:2]:  # show only the first 2
                        print(f"    - {mut.mutated_question[:50]}... (target: {mut.target_answer})")

            # MR2: Wh → General (full pipeline: Q2S → S2G)
            if apply_mr2:
                mr2_results = qaasker.mr2_wh_to_general(question, answer)
                for mut in mr2_results:
                    results['MR2'].append(mutation_result_to_dict(mut))
                if verbose and mr2_results:
                    print(f"  MR2: Generated {len(mr2_results)} mutations")
                    for mut in mr2_results:
                        print(f"    - {mut.mutated_question[:60]}...")

        # MR3 only applies to boolean/alternative types (yes_no, true_false)
        elif q_type in ['boolean', 'alternative'] and answer:
            # MR3: General/Alternative → Wh (full pipeline: GA2S → S2W → generate questions)
            if apply_mr3:
                mr3_results = qaasker.mr3_general_to_wh(question, answer)
                for mut in mr3_results:
                    results['MR3'].append(mutation_result_to_dict(mut))
                if verbose and mr3_results:
                    print(f"  MR3: Generated {len(mr3_results)} mutations")
                    for mut in mr3_results[:2]:
                        print(f"    - {mut.mutated_question[:50]}... (target: {mut.target_answer})")

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")

    return results


def process_dataframe(
    df: pd.DataFrame,
    qaasker: QAAskeR,
    apply_mr1: bool = True,
    apply_mr2: bool = True,
    apply_mr3: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Process a single DataFrame, applying QAAskeR mutations

    Args:
        df: Original DataFrame
        qaasker: QAAskeR instance
        apply_mr1: Whether to apply MR1
        apply_mr2: Whether to apply MR2
        apply_mr3: Whether to apply MR3
        verbose: Whether to show detailed output

    Returns:
        New DataFrame containing the mutation results
    """
    # Create a copy to avoid modifying the original data
    result_df = df.copy()

    # Initialize mutation columns - store full mutation info (in JSON)
    result_df['qaasker_mr1_mutations'] = None
    result_df['qaasker_mr2_mutations'] = None
    result_df['qaasker_mr3_mutations'] = None
    # Additionally store a simplified list of mutated questions (|||-separated)
    result_df['qaasker_mr1_questions'] = None
    result_df['qaasker_mr2_questions'] = None
    result_df['qaasker_mr3_questions'] = None
    # Store the intermediate state (statement)
    result_df['qaasker_statement'] = None

    # Determine which question column to use for mutations
    question_col = None
    answer_col = None

    if 'mutated_question' in df.columns:
        question_col = 'mutated_question'
    elif 'original_question' in df.columns:
        question_col = 'original_question'

    if 'mutated_correct_answer' in df.columns:
        answer_col = 'mutated_correct_answer'
    elif 'original_correct_answer' in df.columns:
        answer_col = 'original_correct_answer'

    if question_col is None:
        print("Warning: No question column found, skipping mutations")
        return result_df

    # Process each row
    for idx, row in df.iterrows():
        question = row.get(question_col, '')
        answer = row.get(answer_col, None) if answer_col else None

        if not question or pd.isna(question):
            continue

        # Get question type from dataset (more reliable than auto-detection)
        # Prefer original_question_type; fall back to mutated_question_type if absent
        question_type = row.get('original_question_type', row.get('mutated_question_type', ''))

        # Skip multiple choice questions (no MR applies)
        if question_type == 'multiple_choice':
            if verbose:
                print(f"\nSkipping multiple choice question: {question[:60]}...")
            continue

        if verbose:
            print(f"\nProcessing: {question[:60]}...")

        # Apply mutations, passing the question_type label to ensure correct MR application
        mutations = apply_qaasker_mutations(
            qaasker,
            str(question),
            str(answer) if answer and not pd.isna(answer) else None,
            question_type_label=question_type,
            apply_mr1=apply_mr1,
            apply_mr2=apply_mr2,
            apply_mr3=apply_mr3,
            verbose=verbose
        )

        # Store the full mutation info (in JSON)
        if mutations['MR1']:
            result_df.at[idx, 'qaasker_mr1_mutations'] = json.dumps(mutations['MR1'], ensure_ascii=False)
            result_df.at[idx, 'qaasker_mr1_questions'] = json.dumps(
                [m['mutated_question'] for m in mutations['MR1']], ensure_ascii=False
            )
            # Store the statement (taken from the first MR1 result)
            if mutations['MR1'][0].get('statement'):
                result_df.at[idx, 'qaasker_statement'] = mutations['MR1'][0]['statement']

        if mutations['MR2']:
            result_df.at[idx, 'qaasker_mr2_mutations'] = json.dumps(mutations['MR2'], ensure_ascii=False)
            result_df.at[idx, 'qaasker_mr2_questions'] = json.dumps(
                [m['mutated_question'] for m in mutations['MR2']], ensure_ascii=False
            )
            # Store the statement (taken from MR2 if MR1 did not provide one)
            if not result_df.at[idx, 'qaasker_statement'] and mutations['MR2'][0].get('statement'):
                result_df.at[idx, 'qaasker_statement'] = mutations['MR2'][0]['statement']

        if mutations['MR3']:
            result_df.at[idx, 'qaasker_mr3_mutations'] = json.dumps(mutations['MR3'], ensure_ascii=False)
            result_df.at[idx, 'qaasker_mr3_questions'] = json.dumps(
                [m['mutated_question'] for m in mutations['MR3']], ensure_ascii=False
            )
            # Store the statement (taken from MR3)
            if not result_df.at[idx, 'qaasker_statement'] and mutations['MR3'][0].get('statement'):
                result_df.at[idx, 'qaasker_statement'] = mutations['MR3'][0]['statement']

    return result_df


def save_mutations(
    csv_files: Dict[str, pd.DataFrame],
    output_dir: Path,
    verbose: bool = False
):
    """
    Save mutation results to the output directory

    Args:
        csv_files: Dictionary mapping relative paths to DataFrames containing mutation results
        output_dir: Output directory path
        verbose: Whether to show detailed output
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'MR1': 0,
        'MR2': 0,
        'MR3': 0,
        'MR1_questions': 0,
        'MR2_questions': 0,
        'MR3_questions': 0,
        'total_files': 0,
        'statements_generated': 0
    }

    # Save each file
    for rel_path, df in csv_files.items():
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)

        # Update statistics
        if 'qaasker_mr1_mutations' in df.columns:
            mr1_count = df['qaasker_mr1_mutations'].notna().sum()
            stats['MR1'] += mr1_count
            # Count total questions (in JSON)
            for val in df['qaasker_mr1_questions'].dropna():
                try:
                    questions = json.loads(val)
                    stats['MR1_questions'] += len(questions)
                except (json.JSONDecodeError, TypeError):
                    stats['MR1_questions'] += 1

        if 'qaasker_mr2_mutations' in df.columns:
            mr2_count = df['qaasker_mr2_mutations'].notna().sum()
            stats['MR2'] += mr2_count
            for val in df['qaasker_mr2_questions'].dropna():
                try:
                    questions = json.loads(val)
                    stats['MR2_questions'] += len(questions)
                except (json.JSONDecodeError, TypeError):
                    stats['MR2_questions'] += 1

        if 'qaasker_mr3_mutations' in df.columns:
            mr3_count = df['qaasker_mr3_mutations'].notna().sum()
            stats['MR3'] += mr3_count
            for val in df['qaasker_mr3_questions'].dropna():
                try:
                    questions = json.loads(val)
                    stats['MR3_questions'] += len(questions)
                except (json.JSONDecodeError, TypeError):
                    stats['MR3_questions'] += 1

        if 'qaasker_statement' in df.columns:
            stats['statements_generated'] += df['qaasker_statement'].notna().sum()

        stats['total_files'] += 1

    # Save the summary statistics
    summary_file = output_dir / "mutation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("QAAskeR Mutation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Total files processed: {stats['total_files']}\n\n")
        f.write("Mutation statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  MR1 (Wh→New Wh): {stats['MR1']} source questions, {stats['MR1_questions']} generated questions\n")
        f.write(f"  MR2 (Wh→General): {stats['MR2']} source questions, {stats['MR2_questions']} generated questions\n")
        f.write(f"  MR3 (General→Wh): {stats['MR3']} source questions, {stats['MR3_questions']} generated questions\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total source questions with mutations: {stats['MR1'] + stats['MR2'] + stats['MR3']}\n")
        f.write(f"  Total generated questions: {stats['MR1_questions'] + stats['MR2_questions'] + stats['MR3_questions']}\n")
        f.write(f"  Total statements generated (Q2S/GA2S): {stats['statements_generated']}\n")
        f.write("\n")
        f.write("MR Flow Details:\n")
        f.write("-" * 60 + "\n")
        f.write("  MR1: Wh-question + Answer → Q2S → S2W → New Wh-questions\n")
        f.write("  MR2: Wh-question + Answer → Q2S → S2G → General question (Yes/No)\n")
        f.write("  MR3: General/Alt + Answer → GA2S → S2W → New Wh-questions\n")

    # Save statistics in JSON format
    # Convert numpy int64 to Python int for JSON serialization
    stats_json = {k: int(v) for k, v in stats.items()}
    stats_file = output_dir / "mutation_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_json, f, indent=2)

    print(f"\nMutation results saved to: {output_dir}")
    print(f"  Total files: {stats['total_files']}")
    print(f"  MR1: {stats['MR1']} sources → {stats['MR1_questions']} questions")
    print(f"  MR2: {stats['MR2']} sources → {stats['MR2_questions']} questions")
    print(f"  MR3: {stats['MR3']} sources → {stats['MR3_questions']} questions")
    print(f"  Statements generated: {stats['statements_generated']}")
    print(f"  Total mutations: {stats['MR1_questions'] + stats['MR2_questions'] + stats['MR3_questions']}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate QAAskeR mutations for RQ4 golden dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all mutations
    python scripts/rqs/rq4_qaasker_mutations.py

    # Only MR2 (Wh→General)
    python scripts/rqs/rq4_qaasker_mutations.py --no-mr1 --no-mr3

    # Verbose output
    python scripts/rqs/rq4_qaasker_mutations.py --verbose --limit 5

MR Flow Details:
    MR1: Wh-question + Answer → Q2S → S2W → New Wh-questions
    MR2: Wh-question + Answer → Q2S → S2G → General question (Yes/No)
    MR3: General/Alt + Answer → GA2S → S2W → New Wh-questions
        """
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(GOLDEN_DATASET_DIR),
        help=f'Input golden dataset directory (default: {GOLDEN_DATASET_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help=f'Output directory for mutations (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--no-mr1',
        action='store_true',
        help='Disable MR1 (Wh→New Wh) mutations'
    )
    parser.add_argument(
        '--no-mr2',
        action='store_true',
        help='Disable MR2 (Wh→General) mutations'
    )
    parser.add_argument(
        '--no-mr3',
        action='store_true',
        help='Disable MR3 (General→Wh) mutations'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output during processing'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    print("=" * 70)
    print("RQ4 QAAskeR Mutation Generator")
    print("=" * 70)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"\nMutation settings:")
    print(f"  MR1 (Wh→New Wh):    {'ENABLED' if not args.no_mr1 else 'disabled'}")
    print(f"  MR2 (Wh→General):   {'ENABLED' if not args.no_mr2 else 'disabled'}")
    print(f"  MR3 (General→Wh):   {'ENABLED' if not args.no_mr3 else 'disabled'}")

    # Load CSV files
    print(f"\nLoading CSV files from {input_dir}...")
    csv_files = load_csv_files(input_dir)
    print(f"Loaded {len(csv_files)} CSV files")

    if not csv_files:
        print("Error: No CSV files found in input directory")
        sys.exit(1)

    # Initialize QAAskeR
    print("\nInitializing QAAskeR...")
    qaasker = QAAskeR(random_seed=args.seed)

    # Show component status
    status = qaasker.get_status()
    print(f"  NLP (spaCy): {'available' if status['nlp_available'] else 'NOT available'}")
    print(f"  Pattern lib: {'available' if status['pattern_available'] else 'NOT available'}")
    print(f"  Benepar:     {'available' if status['benepar_available'] else 'NOT available'}")
    print(f"  WH Rules:    {'available' if status['wh_rules_available'] else 'NOT available'}")
    print(f"  ROUGE:       {'available' if status['rouge_available'] else 'NOT available'}")

    if not status['nlp_available']:
        print("\nWarning: NLP components not available. Using fallback implementations.")

    # Process each file
    print("\nProcessing files...")
    processed_files = {}

    file_list = list(csv_files.items())
    if args.limit:
        file_list = file_list[:args.limit]

    for rel_path, df in tqdm(file_list, desc="Generating mutations"):
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {rel_path}")
            print(f"  Rows: {len(df)}")

        # Apply mutations
        processed_df = process_dataframe(
            df,
            qaasker,
            apply_mr1=not args.no_mr1,
            apply_mr2=not args.no_mr2,
            apply_mr3=not args.no_mr3,
            verbose=args.verbose
        )
        processed_files[rel_path] = processed_df

    # Save results
    print("\nSaving results...")
    save_mutations(processed_files, output_dir, verbose=args.verbose)

    print("\n" + "=" * 70)
    print("Mutation generation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
