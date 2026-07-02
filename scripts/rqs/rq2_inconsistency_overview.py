#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ2: Inconsistency Overview Analysis

Generate comprehensive tables showing inconsistency rates across:
- 5 models (DeepSeek-V3, DeepSeek-V4-Flash, GLM-5-Turbo, GPT-5.5, Qwen2.5-7B)
- All rules (127 rules)
- Question types (multiple_choice, true_false, wh_question, yes_no)
- Metamorphic rules (Body Augmentation, Body Permutation, Entity Rename)

Output:
- CSV and LaTeX tables to data/examples/rq2_inconsistency_overview/

Usage:
    conda activate karma
    python scripts/rqs/rq2_inconsistency_overview.py
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Configuration
DATA_BASE_PATH = Path("$PROJECT_ROOT/data/examples")
OUTPUT_DIR = DATA_BASE_PATH / "rq2_inconsistency_overview"

# Models configuration
MODEL_DIRS = [
    "consistency_results_nli_deepseek-v3",
    "consistency_results_nli_deepseek-v4-flash",
    "consistency_results_nli_GLM-5-Turbo",
    "consistency_results_nli_gpt-5.5",
    "consistency_results_nli_Qwen2.5-7B-Instruct",
]

MODEL_DISPLAY_NAMES = {
    "consistency_results_nli_deepseek-v3": "DeepSeek-V3",
    "consistency_results_nli_deepseek-v4-flash": "DeepSeek-V4-Flash",
    "consistency_results_nli_GLM-5-Turbo": "GLM-5-Turbo",
    "consistency_results_nli_gpt-5.5": "GPT-5.5",
    "consistency_results_nli_Qwen2.5-7B-Instruct": "Qwen2.5-7B",
}

# Metamorphic mutation types
MUTATION_FILES = {
    "body_augmentation_llm_answers.csv": "Body Aug.",
    "body_permutation_llm_answers.csv": "Body Perm.",
    "entity_rename_llm_answers.csv": "Entity Rename",
}

# Question types (will be detected dynamically)
# Note: true_false and yes_no are merged into "Yes/No" category
QUESTION_TYPES = ["multiple_choice", "wh_question", "yes_no"]  # yes_no includes both true_false and yes_no

QUESTION_TYPE_DISPLAY = {
    "multiple_choice": "Multi-Choice",
    "wh_question": "WH-Question",
    "yes_no": "Yes/No",  # Merged category for both true_false and yes_no
}

# Original question types in data (for filtering)
ORIGINAL_QTYPES_YESNO = ["true_false", "yes_no"]


def load_consistency_data(model_dir: str) -> pd.DataFrame:
    """Load all consistency data for a model into a single DataFrame."""
    model_path = DATA_BASE_PATH / model_dir
    if not model_path.exists():
        print(f"Warning: Model directory {model_path} does not exist")
        return pd.DataFrame()

    all_data = []

    # Get all rule directories
    rule_dirs = sorted([
        d for d in os.listdir(model_path)
        if d.startswith("kg_rule_") and (model_path / d).is_dir()
    ])

    for rule_dir in rule_dirs:
        rule_num = int(rule_dir.replace("kg_rule_", ""))
        rule_path = model_path / rule_dir

        for mutation_file in MUTATION_FILES.keys():
            file_path = rule_path / mutation_file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df["rule_id"] = rule_num
                    df["mutation_type"] = MUTATION_FILES[mutation_file]
                    all_data.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def compute_inconsistency_stats(df: pd.DataFrame) -> Dict:
    """
    Compute inconsistency statistics for a DataFrame.

    Returns dict with:
    - total: total samples
    - inconsistent: count of inconsistent samples
    - rate: inconsistency rate
    """
    if df.empty:
        return {"total": 0, "inconsistent": 0, "rate": 0.0}

    total = len(df)
    inconsistent = (df["answers_consistent"] == False).sum()
    rate = inconsistent / total if total > 0 else 0.0

    return {"total": total, "inconsistent": inconsistent, "rate": rate}


def analyze_model(model_dir: str) -> Dict:
    """
    Analyze inconsistency statistics for a single model.

    Returns comprehensive statistics:
    - overall stats
    - by mutation type
    - by question type
    """
    print(f"  Loading data for {model_dir}...")
    df = load_consistency_data(model_dir)

    if df.empty:
        print(f"  Warning: No data found for {model_dir}")
        return None

    model_name = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)

    # Overall stats
    overall = compute_inconsistency_stats(df)

    # By mutation type
    by_mutation = {}
    for mutation in MUTATION_FILES.values():
        mutation_df = df[df["mutation_type"] == mutation]
        by_mutation[mutation] = compute_inconsistency_stats(mutation_df)

    # By question type (using original_question_type)
    # Merge true_false and yes_no into "yes_no" category
    by_qtype = {}
    for qtype in QUESTION_TYPES:
        if qtype == "yes_no":
            # Merge both true_false and yes_no
            qtype_df = df[df["original_question_type"].isin(ORIGINAL_QTYPES_YESNO)]
        else:
            qtype_df = df[df["original_question_type"] == qtype]
        by_qtype[qtype] = compute_inconsistency_stats(qtype_df)

    return {
        "model": model_name,
        "model_dir": model_dir,
        "overall": overall,
        "by_mutation": by_mutation,
        "by_qtype": by_qtype,
    }


def generate_main_table(results: List[Dict]) -> pd.DataFrame:
    """
    Generate main overview table with:
    - Model name
    - Overall statistics
    - By mutation type rates
    - By question type rates
    """
    rows = []

    for r in results:
        if r is None:
            continue

        row = {
            "Model": r["model"],
            "Total": r["overall"]["total"],
            "Inconsistent": r["overall"]["inconsistent"],
            "Inc. Rate (%)": f"{r['overall']['rate'] * 100:.2f}",
        }

        # Add mutation type rates
        for mutation in MUTATION_FILES.values():
            stats = r["by_mutation"].get(mutation, {"rate": 0})
            row[f"{mutation} (%)"] = f"{stats['rate'] * 100:.2f}"

        # Add question type rates
        for qtype in QUESTION_TYPES:
            display_name = QUESTION_TYPE_DISPLAY.get(qtype, qtype)
            stats = r["by_qtype"].get(qtype, {"rate": 0})
            row[f"{display_name} (%)"] = f"{stats['rate'] * 100:.2f}"

        rows.append(row)

    # Define column order
    columns = ["Model", "Total", "Inconsistent", "Inc. Rate (%)"]
    columns += [f"{m} (%)" for m in MUTATION_FILES.values()]
    columns += [f"{QUESTION_TYPE_DISPLAY[q]} (%)" for q in QUESTION_TYPES]

    df = pd.DataFrame(rows, columns=columns)
    return df


def generate_detailed_csv(results: List[Dict]) -> pd.DataFrame:
    """
    Generate detailed table with counts and rates.
    Format: "Count (Rate%)" for each cell.
    """
    rows = []

    for r in results:
        if r is None:
            continue

        row = {
            "Model": r["model"],
            "Total": r["overall"]["total"],
            "Inconsistent": f"{r['overall']['inconsistent']} ({r['overall']['rate']*100:.2f}%)",
        }

        # Add mutation type with counts
        for mutation in MUTATION_FILES.values():
            stats = r["by_mutation"].get(mutation, {"total": 0, "inconsistent": 0, "rate": 0})
            row[mutation] = f"{stats['inconsistent']}/{stats['total']} ({stats['rate']*100:.2f}%)"

        # Add question type with counts
        for qtype in QUESTION_TYPES:
            display_name = QUESTION_TYPE_DISPLAY.get(qtype, qtype)
            stats = r["by_qtype"].get(qtype, {"total": 0, "inconsistent": 0, "rate": 0})
            row[display_name] = f"{stats['inconsistent']}/{stats['total']} ({stats['rate']*100:.2f}%)"

        rows.append(row)

    columns = ["Model", "Total", "Inconsistent"]
    columns += list(MUTATION_FILES.values())
    columns += [QUESTION_TYPE_DISPLAY[q] for q in QUESTION_TYPES]

    return pd.DataFrame(rows, columns=columns)


def generate_latex_table(results: List[Dict]) -> str:
    """
    Generate LaTeX table for the paper.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall Inconsistency Rate Across Models, Metamorphic Rules, and Question Types}")
    lines.append(r"\label{tab:rq2_overview}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & & & \multicolumn{3}{c}{\textbf{Metamorphic Rule}} & \multicolumn{3}{c}{\textbf{Question Type}} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}")
    lines.append(r"\textbf{Model} & \textbf{Total} & \textbf{Inc. Rate} & "
                r"\textbf{Body Aug.} & \textbf{Body Perm.} & \textbf{Entity Rename} & "
                r"\textbf{Multi-Choice} & \textbf{WH-Question} & \textbf{Yes/No} \\")
    lines.append(r"\midrule")

    for r in results:
        if r is None:
            continue

        model = r["model"]
        total = r["overall"]["total"]
        rate = f"{r['overall']['rate']*100:.2f}"

        # Mutation rates
        ba = r["by_mutation"].get("Body Aug.", {"rate": 0})["rate"] * 100
        bp = r["by_mutation"].get("Body Perm.", {"rate": 0})["rate"] * 100
        er = r["by_mutation"].get("Entity Rename", {"rate": 0})["rate"] * 100

        # Question type rates
        mc = r["by_qtype"].get("multiple_choice", {"rate": 0})["rate"] * 100
        wh = r["by_qtype"].get("wh_question", {"rate": 0})["rate"] * 100
        yn = r["by_qtype"].get("yes_no", {"rate": 0})["rate"] * 100

        lines.append(f"{model} & {total} & {rate}\\% & "
                    f"{ba:.2f}\\% & {bp:.2f}\\% & {er:.2f}\\% & "
                    f"{mc:.2f}\\% & {wh:.2f}\\% & {yn:.2f}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latex_table_compact(results: List[Dict]) -> str:
    """
    Generate a more compact LaTeX table (only rates).
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inconsistency Rates by Model and Category (\%)}")
    lines.append(r"\label{tab:rq2_overview_compact}")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Overall} & "
                r"\textbf{Aug.} & \textbf{Perm.} & \textbf{Rename} & "
                r"\textbf{MC} & \textbf{WH} & \textbf{YN} \\")
    lines.append(r"\midrule")

    for r in results:
        if r is None:
            continue

        model = r["model"]
        rate = r["overall"]["rate"] * 100
        ba = r["by_mutation"].get("Body Aug.", {"rate": 0})["rate"] * 100
        bp = r["by_mutation"].get("Body Perm.", {"rate": 0})["rate"] * 100
        er = r["by_mutation"].get("Entity Rename", {"rate": 0})["rate"] * 100
        mc = r["by_qtype"].get("multiple_choice", {"rate": 0})["rate"] * 100
        wh = r["by_qtype"].get("wh_question", {"rate": 0})["rate"] * 100
        yn = r["by_qtype"].get("yes_no", {"rate": 0})["rate"] * 100

        lines.append(f"{model} & {rate:.2f} & {ba:.2f} & {bp:.2f} & {er:.2f} & "
                    f"{mc:.2f} & {wh:.2f} & {yn:.2f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="RQ2: Generate inconsistency overview tables"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for result tables"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=MODEL_DIRS,
        help="Models to analyze (default: all models)"
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RQ2: Inconsistency Overview Analysis")
    print("=" * 70)

    # Analyze each model
    results = []
    for model_dir in args.models:
        print(f"\nAnalyzing {model_dir}...")
        result = analyze_model(model_dir)
        if result:
            results.append(result)
            print(f"  Total: {result['overall']['total']}, "
                  f"Inconsistent: {result['overall']['inconsistent']} "
                  f"({result['overall']['rate']*100:.2f}%)")

    if not results:
        print("No results to analyze!")
        return

    # Generate tables
    print("\n" + "=" * 70)
    print("Generating tables...")
    print("=" * 70)

    # Main overview table (CSV)
    main_table = generate_main_table(results)
    main_csv_path = output_dir / "rq2_overview_table.csv"
    main_table.to_csv(main_csv_path, index=False)
    print(f"\nMain table saved to: {main_csv_path}")
    print(main_table.to_string(index=False))

    # Detailed table (CSV)
    detailed_table = generate_detailed_csv(results)
    detailed_csv_path = output_dir / "rq2_detailed_table.csv"
    detailed_table.to_csv(detailed_csv_path, index=False)
    print(f"\nDetailed table saved to: {detailed_csv_path}")

    # LaTeX tables
    latex_table = generate_latex_table(results)
    latex_path = output_dir / "rq2_overview_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: {latex_path}")

    latex_compact = generate_latex_table_compact(results)
    latex_compact_path = output_dir / "rq2_overview_table_compact.tex"
    with open(latex_compact_path, "w") as f:
        f.write(latex_compact)
    print(f"Compact LaTeX table saved to: {latex_compact_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  Overall: {r['overall']['inconsistent']}/{r['overall']['total']} "
              f"({r['overall']['rate']*100:.2f}%)")
        print("  By Mutation:")
        for m, stats in r["by_mutation"].items():
            print(f"    {m}: {stats['inconsistent']}/{stats['total']} ({stats['rate']*100:.2f}%)")
        print("  By Question Type:")
        for q, stats in r["by_qtype"].items():
            print(f"    {q}: {stats['inconsistent']}/{stats['total']} ({stats['rate']*100:.2f}%)")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
