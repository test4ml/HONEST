#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 KONTEST-inspired Mutation Generator Script

This script reads the HONEST golden dataset, extracts concrete KG facts from
rule instances, and generates atomic KONTEST-style Yes/No paraphrase pairs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.kontest import KONTEST, KontestMutation

GOLDEN_DATASET_DIR = Path("data/examples/golden_dataset")
OUTPUT_DIR = Path("data/examples/golden_dataset_kontest_mutation")


def load_csv_files(input_dir: Path) -> Dict[str, pd.DataFrame]:
    csv_files = {}
    for csv_file in input_dir.rglob("*.csv"):
        if "summary" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file)
            csv_files[str(csv_file.relative_to(input_dir))] = df
        except Exception as exc:
            print(f"Warning: Failed to read {csv_file}: {exc}")
    return csv_files


def mutation_to_dict(mutation: KontestMutation) -> Dict[str, Any]:
    return KONTEST.mutation_to_dict(mutation)


def process_dataframe(
    df: pd.DataFrame,
    kontest: KONTEST,
    instance_column: str = "original_instance",
    verbose: bool = False,
) -> pd.DataFrame:
    result_df = df.copy()
    result_df["kontest_atomic_mutations"] = None
    result_df["kontest_atomic_questions"] = None
    result_df["kontest_num_pairs"] = 0
    result_df["kontest_extraction_errors"] = None

    if instance_column not in df.columns:
        result_df["kontest_extraction_errors"] = f"missing column {instance_column}"
        return result_df

    for idx, row in df.iterrows():
        mutations, errors = kontest.generate_atomic_mutations(row, instance_column=instance_column)
        mutation_dicts = [mutation_to_dict(mutation) for mutation in mutations]

        if mutation_dicts:
            result_df.at[idx, "kontest_atomic_mutations"] = json.dumps(mutation_dicts, ensure_ascii=False)
            result_df.at[idx, "kontest_atomic_questions"] = json.dumps(
                [
                    {
                        "original_question": mutation["original_question"],
                        "mutated_question": mutation["mutated_question"],
                    }
                    for mutation in mutation_dicts
                ],
                ensure_ascii=False,
            )
            result_df.at[idx, "kontest_num_pairs"] = len(mutation_dicts)

        if errors:
            result_df.at[idx, "kontest_extraction_errors"] = json.dumps(errors, ensure_ascii=False)
            if verbose:
                print(f"Row {idx}: {errors}")

    return result_df


def _count_json_list(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        return 0
    return 0


def save_mutations(csv_files: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_files": 0,
        "total_rows": 0,
        "rows_with_pairs": 0,
        "atomic_pairs": 0,
        "rows_with_errors": 0,
        "by_source_part": {},
        "by_predicate": {},
    }

    for rel_path, df in csv_files.items():
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        stats["total_files"] += 1
        stats["total_rows"] += len(df)
        if "kontest_num_pairs" in df.columns:
            stats["rows_with_pairs"] += int((df["kontest_num_pairs"].fillna(0) > 0).sum())
            stats["atomic_pairs"] += int(df["kontest_num_pairs"].fillna(0).sum())
        if "kontest_extraction_errors" in df.columns:
            stats["rows_with_errors"] += int(df["kontest_extraction_errors"].notna().sum())

        for value in df.get("kontest_atomic_mutations", pd.Series(dtype=object)).dropna():
            try:
                mutations = json.loads(value)
            except Exception:
                continue
            for mutation in mutations:
                source_part = mutation.get("source_part", "unknown") or "unknown"
                predicate = mutation.get("predicate", "unknown") or "unknown"
                stats["by_source_part"][source_part] = stats["by_source_part"].get(source_part, 0) + 1
                stats["by_predicate"][predicate] = stats["by_predicate"].get(predicate, 0) + 1

    summary_file = output_dir / "mutation_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("KONTEST-inspired Mutation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total files processed: {stats['total_files']}\n")
        f.write(f"Total rows processed: {stats['total_rows']}\n")
        f.write(f"Rows with atomic pairs: {stats['rows_with_pairs']}\n")
        f.write(f"Generated atomic pairs: {stats['atomic_pairs']}\n")
        f.write(f"Rows with extraction errors: {stats['rows_with_errors']}\n\n")
        f.write("Template pair:\n")
        f.write("  Does {subject_label} have a {object_label}?\n")
        f.write("  Is there a {object_label} in {subject_label}?\n")

    stats_file = output_dir / "mutation_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nMutation results saved to: {output_dir}")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Rows with pairs: {stats['rows_with_pairs']}")
    print(f"  Atomic pairs: {stats['atomic_pairs']}")
    print(f"  Rows with errors: {stats['rows_with_errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate KONTEST-inspired atomic mutations for the RQ4 golden dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    conda activate karma
    python scripts/rqs/rq4_kontest_mutations.py
    python scripts/rqs/rq4_kontest_mutations.py --limit 5 --verbose
        """,
    )
    parser.add_argument("--input-dir", type=str, default=str(GOLDEN_DATASET_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--instance-column", type=str, default="original_instance")
    parser.add_argument("--label-mode", choices=["full", "short"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    print("=" * 70)
    print("RQ4 KONTEST-inspired Mutation Generator")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Instance column: {args.instance_column}")
    print(f"Label mode: {args.label_mode}")

    csv_files = load_csv_files(input_dir)
    print(f"Loaded {len(csv_files)} CSV files")
    if not csv_files:
        print("Error: No CSV files found in input directory")
        sys.exit(1)

    kontest = KONTEST(label_mode=args.label_mode)
    file_list = list(csv_files.items())
    if args.limit:
        file_list = file_list[:args.limit]

    processed_files = {}
    for rel_path, df in tqdm(file_list, desc="Generating KONTEST mutations"):
        if args.verbose:
            print(f"\nProcessing: {rel_path} ({len(df)} rows)")
        processed_files[rel_path] = process_dataframe(
            df,
            kontest,
            instance_column=args.instance_column,
            verbose=args.verbose,
        )

    save_mutations(processed_files, output_dir)
    print("\nMutation generation complete!")


if __name__ == "__main__":
    main()
