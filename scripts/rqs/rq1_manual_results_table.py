#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ1 Manual Results Table Generator

Generate statistics tables for manual inspection results, including:
- Statistics for original questions, mutated questions, and all questions
- Categories: valid, syntax error, logic error
- Cohen's Kappa agreement coefficient (computed between labelers A and B per row)

Usage:
    python scripts/rqs/rq1_manual_results_table.py
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# Configuration
MANUAL_RESULTS_DIR = Path("data/examples/rq1_manual_results")
OUTPUT_DIR = Path("data/examples/rq1_manual_results")

LABELER_A_FILE = "labeled_rq1_sampled_questions_labeler_A.csv"
LABELER_B_FILE = "labeled_rq1_sampled_questions_labeler_B.csv"
FINAL_FILE = "labeled_rq1_sampled_questions_final.csv"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three annotation files."""
    df_a = pd.read_csv(MANUAL_RESULTS_DIR / LABELER_A_FILE)
    df_b = pd.read_csv(MANUAL_RESULTS_DIR / LABELER_B_FILE)
    df_final = pd.read_csv(MANUAL_RESULTS_DIR / FINAL_FILE)
    return df_a, df_b, df_final


def compute_error_stats(df: pd.DataFrame, column: str) -> dict:
    """Compute error-type statistics."""
    total = len(df)
    correct = (df[column] == "correct").sum()
    syntax_error = (df[column] == "syntax_error").sum()
    logic_error = (df[column] == "logic_error").sum()

    return {
        "correct": correct,
        "syntax_error": syntax_error,
        "logic_error": logic_error,
        "total": total,
    }


def compute_all_stats(df: pd.DataFrame) -> dict:
    """Compute statistics for original, mutated, and all questions."""
    stats_original = compute_error_stats(df, "original_error_type")
    stats_mutated = compute_error_stats(df, "mutated_error_type")
    stats_all = {
        "correct": stats_original["correct"] + stats_mutated["correct"],
        "syntax_error": stats_original["syntax_error"] + stats_mutated["syntax_error"],
        "logic_error": stats_original["logic_error"] + stats_mutated["logic_error"],
        "total": stats_original["total"] + stats_mutated["total"],
    }
    return {
        "original": stats_original,
        "mutated": stats_mutated,
        "all": stats_all,
    }


def compute_kappa_by_category(df_a: pd.DataFrame, df_b: pd.DataFrame,
                               column: str, category: str) -> float:
    """
    Compute Cohen's Kappa for a given category (binarized).

    Args:
        df_a: data from labeler A
        df_b: data from labeler B
        column: column name (original_error_type or mutated_error_type)
        category: category (valid/syntax/logic)

    Returns:
        The binarized Cohen's Kappa coefficient.
    """
    # Map category to error type
    if category == "valid":
        error_type = "correct"
    elif category == "syntax":
        error_type = "syntax_error"
    elif category == "logic":
        error_type = "logic_error"
    else:
        raise ValueError(f"Unknown category: {category}")

    labels_a = (df_a[column] == error_type).astype(int).values
    labels_b = (df_b[column] == error_type).astype(int).values

    kappa = cohen_kappa_score(labels_a, labels_b)
    return kappa


def compute_all_kappas(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """
    Compute Cohen's Kappa coefficients for all categories.

    Returns:
        dict: {
            'original': {'valid': x, 'syntax': x, 'logic': x, 'overall': x},
            'mutated': {'valid': x, 'syntax': x, 'logic': x, 'overall': x},
            'all': {'valid': x, 'syntax': x, 'logic': x, 'overall': x},
        }
    """
    kappas = {}

    for qtype, col in [('original', 'original_error_type'), ('mutated', 'mutated_error_type')]:
        kappas[qtype] = {}
        for cat in ['valid', 'syntax', 'logic']:
            kappas[qtype][cat] = compute_kappa_by_category(df_a, df_b, col, cat)
        # Compute the overall multi-class kappa
        kappas[qtype]['overall'] = cohen_kappa_score(df_a[col], df_b[col])

    # All (combined original + mutated)
    kappas['all'] = {}
    for cat in ['valid', 'syntax', 'logic']:
        error_type = 'correct' if cat == 'valid' else cat + '_error'
        if cat == 'valid':
            error_type = 'correct'
        elif cat == 'syntax':
            error_type = 'syntax_error'
        else:
            error_type = 'logic_error'

        labels_a_orig = (df_a["original_error_type"] == error_type).astype(int).values
        labels_a_mut = (df_a["mutated_error_type"] == error_type).astype(int).values
        labels_a_all = np.concatenate([labels_a_orig, labels_a_mut])

        labels_b_orig = (df_b["original_error_type"] == error_type).astype(int).values
        labels_b_mut = (df_b["mutated_error_type"] == error_type).astype(int).values
        labels_b_all = np.concatenate([labels_b_orig, labels_b_mut])

        kappas['all'][cat] = cohen_kappa_score(labels_a_all, labels_b_all)

    # Compute the overall multi-class kappa across all samples
    labels_a_all_orig = df_a["original_error_type"].values
    labels_a_all_mut = df_a["mutated_error_type"].values
    labels_a_combined = np.concatenate([labels_a_all_orig, labels_a_all_mut])

    labels_b_all_orig = df_b["original_error_type"].values
    labels_b_all_mut = df_b["mutated_error_type"].values
    labels_b_combined = np.concatenate([labels_b_all_orig, labels_b_all_mut])

    kappas['all']['overall'] = cohen_kappa_score(labels_a_combined, labels_b_combined)

    return kappas


def generate_latex_table_with_final(stats_a: dict, stats_b: dict, stats_final: dict,
                                    kappas: dict) -> str:
    """Generate a LaTeX table with Labeler A, Labeler B, Final, Cohen's κ columns."""
    # Labeler A data
    a_orig = stats_a['original']
    a_mut = stats_a['mutated']
    a_all = stats_a['all']

    # Labeler B data
    b_orig = stats_b['original']
    b_mut = stats_b['mutated']
    b_all = stats_b['all']

    # Final data
    f_orig = stats_final['original']
    f_mut = stats_final['mutated']
    f_all = stats_final['all']

    latex = r"""% Requires: \usepackage{multirow} in preamble
\begin{table}[htbp]
\centering
\caption{Manual Inspection Results of Question Quality}
\label{tab:rq1_manual_results}
\begin{tabular}{llcccc}
\toprule
\multicolumn{2}{l}{\textbf{Category}} & \textbf{Labeler A} & \textbf{Labeler B} & \textbf{Final} & \textbf{Cohen's} $\boldsymbol{\kappa}$ \\
\midrule
\multirow{3}{*}{\textbf{Original}}
 & Valid & """ + f"{a_orig['correct']} ({a_orig['correct']/a_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_orig['correct']} ({b_orig['correct']/b_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_orig['correct']} ({f_orig['correct']/f_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['original']['valid']:.3f}" + r""" \\
 & Syntax Error & """ + f"{a_orig['syntax_error']} ({a_orig['syntax_error']/a_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_orig['syntax_error']} ({b_orig['syntax_error']/b_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_orig['syntax_error']} ({f_orig['syntax_error']/f_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['original']['syntax']:.3f}" + r""" \\
 & Logic Error & """ + f"{a_orig['logic_error']} ({a_orig['logic_error']/a_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_orig['logic_error']} ({b_orig['logic_error']/b_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_orig['logic_error']} ({f_orig['logic_error']/f_orig['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['original']['logic']:.3f}" + r""" \\
\midrule
\multirow{3}{*}{\textbf{Mutated}}
 & Valid & """ + f"{a_mut['correct']} ({a_mut['correct']/a_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_mut['correct']} ({b_mut['correct']/b_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_mut['correct']} ({f_mut['correct']/f_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['mutated']['valid']:.3f}" + r""" \\
 & Syntax Error & """ + f"{a_mut['syntax_error']} ({a_mut['syntax_error']/a_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_mut['syntax_error']} ({b_mut['syntax_error']/b_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_mut['syntax_error']} ({f_mut['syntax_error']/f_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['mutated']['syntax']:.3f}" + r""" \\
 & Logic Error & """ + f"{a_mut['logic_error']} ({a_mut['logic_error']/a_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_mut['logic_error']} ({b_mut['logic_error']/b_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_mut['logic_error']} ({f_mut['logic_error']/f_mut['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['mutated']['logic']:.3f}" + r""" \\
\midrule
\multirow{3}{*}{\textbf{All}}
 & Valid & """ + f"{a_all['correct']} ({a_all['correct']/a_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_all['correct']} ({b_all['correct']/b_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_all['correct']} ({f_all['correct']/f_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['all']['valid']:.3f}" + r""" \\
 & Syntax Error & """ + f"{a_all['syntax_error']} ({a_all['syntax_error']/a_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_all['syntax_error']} ({b_all['syntax_error']/b_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_all['syntax_error']} ({f_all['syntax_error']/f_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['all']['syntax']:.3f}" + r""" \\
 & Logic Error & """ + f"{a_all['logic_error']} ({a_all['logic_error']/a_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{b_all['logic_error']} ({b_all['logic_error']/b_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{f_all['logic_error']} ({f_all['logic_error']/f_all['total']*100:.1f}\\%)" + r""" & """ + \
f"{kappas['all']['logic']:.3f}" + r""" \\
\midrule
\multicolumn{2}{l}{\textbf{Total}} & """ + f"{a_all['total']}" + r""" & """ + f"{b_all['total']}" + r""" & """ + f"{f_all['total']}" + r""" & \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table_with_final(stats_a: dict, stats_b: dict, stats_final: dict,
                                       kappas: dict) -> str:
    """Generate a Markdown table with Labeler A, Labeler B, Final, Cohen's κ columns."""
    # Labeler A, B, Final data
    a_orig = stats_a['original']
    a_mut = stats_a['mutated']
    a_all = stats_a['all']
    b_orig = stats_b['original']
    b_mut = stats_b['mutated']
    b_all = stats_b['all']
    f_orig = stats_final['original']
    f_mut = stats_final['mutated']
    f_all = stats_final['all']

    md = """# Manual Inspection Results of Question Quality

| Category | | Labeler A | Labeler B | Final | Cohen's κ |
|:---------|:---------|:---------:|:---------:|:-----:|:---------:|
| **Original** | Valid | """ + f"{a_orig['correct']} ({a_orig['correct']/a_orig['total']*100:.1f}%)" + """ | """ + \
f"{b_orig['correct']} ({b_orig['correct']/b_orig['total']*100:.1f}%)" + """ | """ + \
f"{f_orig['correct']} ({f_orig['correct']/f_orig['total']*100:.1f}%)" + """ | """ + \
f"{kappas['original']['valid']:.3f}" + """ |
| | Syntax Error | """ + f"{a_orig['syntax_error']} ({a_orig['syntax_error']/a_orig['total']*100:.1f}%)" + """ | """ + \
f"{b_orig['syntax_error']} ({b_orig['syntax_error']/b_orig['total']*100:.1f}%)" + """ | """ + \
f"{f_orig['syntax_error']} ({f_orig['syntax_error']/f_orig['total']*100:.1f}%)" + """ | """ + \
f"{kappas['original']['syntax']:.3f}" + """ |
| | Logic Error | """ + f"{a_orig['logic_error']} ({a_orig['logic_error']/a_orig['total']*100:.1f}%)" + """ | """ + \
f"{b_orig['logic_error']} ({b_orig['logic_error']/b_orig['total']*100:.1f}%)" + """ | """ + \
f"{f_orig['logic_error']} ({f_orig['logic_error']/f_orig['total']*100:.1f}%)" + """ | """ + \
f"{kappas['original']['logic']:.3f}" + """ |
| **Mutated** | Valid | """ + f"{a_mut['correct']} ({a_mut['correct']/a_mut['total']*100:.1f}%)" + """ | """ + \
f"{b_mut['correct']} ({b_mut['correct']/b_mut['total']*100:.1f}%)" + """ | """ + \
f"{f_mut['correct']} ({f_mut['correct']/f_mut['total']*100:.1f}%)" + """ | """ + \
f"{kappas['mutated']['valid']:.3f}" + """ |
| | Syntax Error | """ + f"{a_mut['syntax_error']} ({a_mut['syntax_error']/a_mut['total']*100:.1f}%)" + """ | """ + \
f"{b_mut['syntax_error']} ({b_mut['syntax_error']/b_mut['total']*100:.1f}%)" + """ | """ + \
f"{f_mut['syntax_error']} ({f_mut['syntax_error']/f_mut['total']*100:.1f}%)" + """ | """ + \
f"{kappas['mutated']['syntax']:.3f}" + """ |
| | Logic Error | """ + f"{a_mut['logic_error']} ({a_mut['logic_error']/a_mut['total']*100:.1f}%)" + """ | """ + \
f"{b_mut['logic_error']} ({b_mut['logic_error']/b_mut['total']*100:.1f}%)" + """ | """ + \
f"{f_mut['logic_error']} ({f_mut['logic_error']/f_mut['total']*100:.1f}%)" + """ | """ + \
f"{kappas['mutated']['logic']:.3f}" + """ |
| **All** | Valid | """ + f"{a_all['correct']} ({a_all['correct']/a_all['total']*100:.1f}%)" + """ | """ + \
f"{b_all['correct']} ({b_all['correct']/b_all['total']*100:.1f}%)" + """ | """ + \
f"{f_all['correct']} ({f_all['correct']/f_all['total']*100:.1f}%)" + """ | """ + \
f"{kappas['all']['valid']:.3f}" + """ |
| | Syntax Error | """ + f"{a_all['syntax_error']} ({a_all['syntax_error']/a_all['total']*100:.1f}%)" + """ | """ + \
f"{b_all['syntax_error']} ({b_all['syntax_error']/b_all['total']*100:.1f}%)" + """ | """ + \
f"{f_all['syntax_error']} ({f_all['syntax_error']/f_all['total']*100:.1f}%)" + """ | """ + \
f"{kappas['all']['syntax']:.3f}" + """ |
| | Logic Error | """ + f"{a_all['logic_error']} ({a_all['logic_error']/a_all['total']*100:.1f}%)" + """ | """ + \
f"{b_all['logic_error']} ({b_all['logic_error']/b_all['total']*100:.1f}%)" + """ | """ + \
f"{f_all['logic_error']} ({f_all['logic_error']/f_all['total']*100:.1f}%)" + """ | """ + \
f"{kappas['all']['logic']:.3f}" + """ |
| **Total** | | **""" + f"{a_all['total']}" + """** | **""" + f"{b_all['total']}" + """** | **""" + f"{f_all['total']}" + """** | |
"""

    return md


def main():
    """Main function."""
    print("Loading data...")
    df_a, df_b, df_final = load_data()

    # Compute per-labeler statistics
    print("Computing statistics...")
    stats_a = compute_all_stats(df_a)
    stats_b = compute_all_stats(df_b)
    stats_final = compute_all_stats(df_final)

    # Compute Cohen's Kappa for all categories
    print("Computing Cohen's Kappa...")
    kappas = compute_all_kappas(df_a, df_b)

    # Print results
    print("\n" + "="*60)
    print("MANUAL INSPECTION RESULTS")
    print("="*60)

    for name, stats in [("Labeler A", stats_a), ("Labeler B", stats_b), ("Final", stats_final)]:
        print(f"\n{name}:")
        for qtype in ["original", "mutated", "all"]:
            s = stats[qtype]
            print(f"  {qtype.capitalize()} (n={s['total']}): Valid={s['correct']}, Syntax={s['syntax_error']}, Logic={s['logic_error']}")

    print(f"\nCohen's Kappa:")
    for qtype in ["original", "mutated", "all"]:
        print(f"  {qtype.capitalize()}:")
        for cat in ["valid", "syntax", "logic"]:
            print(f"    - {cat}: {kappas[qtype][cat]:.3f}")
        print(f"    - **Overall (multi-class)**: {kappas[qtype]['overall']:.3f}")

    # Generate and save the tables
    print("\nGenerating tables...")

    # Full table (includes Labeler A, B, Final, Cohen's κ)
    latex_table_full = generate_latex_table_with_final(stats_a, stats_b, stats_final, kappas)
    latex_path_full = OUTPUT_DIR / "rq1_manual_results_table_full.tex"
    with open(latex_path_full, "w") as f:
        f.write(latex_table_full)
    print(f"Full LaTeX table saved to: {latex_path_full}")

    md_table_full = generate_markdown_table_with_final(stats_a, stats_b, stats_final, kappas)
    md_path_full = OUTPUT_DIR / "rq1_manual_results_table_full.md"
    with open(md_path_full, "w") as f:
        f.write(md_table_full)
    print(f"Full Markdown table saved to: {md_path_full}")

    print("\nDone!")


if __name__ == "__main__":
    main()
