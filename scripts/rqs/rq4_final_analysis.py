#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 Final Analysis Script

Generate final comparison tables and overlap Venn diagrams for RQ4 analysis.
Uses evaluation results for QAQA, QAASKER, KONTEST, and HONEST.

Outputs:
1. Main comparison table (CSV + LaTeX)
2. Overall comparison table (CSV + LaTeX)
3. Overlap Venn diagrams for each model (PNG + PDF)

Usage:
    conda activate karma
    python scripts/rqs/rq4_final_analysis.py
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration
DATA_BASE_PATH = Path("$PROJECT_ROOT/data/examples")
NLI_RESULTS_DIR = DATA_BASE_PATH / "rq4_results_nli"
KONTEST_RESULTS_DIR = DATA_BASE_PATH / "rq4_results"
HONEST_RESULTS_BASE = DATA_BASE_PATH
OUTPUT_DIR = DATA_BASE_PATH / "rq4_summary"
FIGURES_DIR = Path("$PROJECT_ROOT/figures/rq4")

# Model names mapping
MODEL_DISPLAY_NAMES = {
    "deepseek_v3": "DeepSeek-V3",
    "deepseek-v3": "DeepSeek-V3",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "GLM_5_Turbo": "GLM-5-Turbo",
    "GLM-5-Turbo": "GLM-5-Turbo",
    "glm_5_turbo": "GLM-5-Turbo",
    "glm-5-turbo": "GLM-5-Turbo",
    "gpt_5.5": "GPT-5.5",
    "gpt-5.5": "GPT-5.5",
    "Qwen2.5_7B_Instruct": "Qwen2.5-7B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
}

MODEL_ORDER = [
    "DeepSeek-V3",
    "DeepSeek-V4-Flash",
    "GLM-5-Turbo",
    "GPT-5.5",
    "Qwen2.5-7B",
]

# Internal model name mapping for file lookup
INTERNAL_MODEL_NAMES = {
    "DeepSeek-V3": "deepseek_v3",
    "DeepSeek-V4-Flash": "deepseek_v4_flash",
    "GLM-5-Turbo": "GLM_5_Turbo",
    "GPT-5.5": "gpt_5.5",
    "Qwen2.5-7B": "Qwen2.5_7B_Instruct",
}

# HONEST directory mapping
HONEST_DIR_MAPPING = {
    "DeepSeek-V3": "consistency_results_nli_deepseek-v3",
    "DeepSeek-V4-Flash": "consistency_results_nli_deepseek-v4-flash",
    "GLM-5-Turbo": "consistency_results_nli_GLM-5-Turbo",
    "GPT-5.5": "consistency_results_nli_gpt-5.5",
    "Qwen2.5-7B": "consistency_results_nli_Qwen2.5-7B-Instruct",
}

# Mutation types
QAQA_MUTATION_TYPES = ["EQ", "EC", "EQC"]  # Reordered to match paper: MR1, MR2, MR3
QAQA_MUTATION_DISPLAY = {
    "EQ": "Equiv. Question",      # MR1: Add similar sentences to question
    "EC": "Equiv. Context",       # MR2: Add similar sentences to context
    "EQC": "Equiv. Context+",     # MR3: Add to both (Test Integration)
}

QAASKER_MR_TYPES = ["MR1", "MR2", "MR3"]
QAASKER_MR_DISPLAY = {
    "MR1": "Wh→Wh",        # Wh-question → new Wh-question
    "MR2": "Wh→General",   # Wh-question → General question
    "MR3": "General→Wh",   # General/Alternative → Wh-question
}

HONEST_MUTATION_TYPES = ["body_permutation", "entity_rename", "body_augmentation"]
HONEST_MUTATION_DISPLAY = {
    "body_permutation": "Body Permutation",
    "entity_rename": "Entity Rename",
    "body_augmentation": "Body Augmentation",
}

KONTEST_MUTATION_TYPES = ["atomic"]
KONTEST_MUTATION_DISPLAY = {
    "atomic": "Atomic Yes/No Paraphrase",
}

# NOTE: QMR4 is intentionally excluded from DrHall experiments (see
# rq4_drhall_evaluation.py), so it is not listed here — otherwise the table
# would show an all "--" row.
DRHALL_MR_TYPES = ["QMR1", "QMR2", "AMR1"]
DRHALL_MR_DISPLAY = {
    "QMR1": "CoT (QMR1)",
    "QMR2": "Multilingual (QMR2)",
    "AMR1": "Negation (AMR1)",
}

# MetaQA: synonym / antonym answer-substitution metamorphic relations
METAQA_MUTATION_TYPES = ["Synonym", "Antonym"]
METAQA_MUTATION_DISPLAY = {
    "Synonym": "Synonym Substitution",
    "Antonym": "Antonym Substitution",
}


def normalize_model_name(model_name: str) -> str:
    """Normalize model name to display name."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_qaqa_nli_results(input_dir: Path) -> Dict[str, Dict]:
    """Load QAQA NLI statistics files for all models."""
    results = {}

    # Load regular statistics files (these have correct NLI evaluation with full scores)
    # The *_nli_statistics.json files have a bug (only used entailment_score)
    stats_files = sorted(input_dir.glob("qaqa_*_statistics.json"))
    for stats_file in stats_files:
        if "_nli_" in stats_file.name:
            continue  # Skip buggy NLI-specific files

        try:
            with open(stats_file, "r") as f:
                data = json.load(f)

            model_name = data.get("model_name", "")
            display_name = normalize_model_name(model_name)
            data["display_name"] = display_name
            # Use error_detection_rate which is NLI-based (from correct evaluation)
            data["error_detection_rate"] = data.get("error_detection_rate", 0)
            results[display_name] = data
            print(f"  [QAQA] Loaded: {display_name} (NLI rate: {data.get('error_detection_rate', 0)*100:.2f}%)")
        except Exception as e:
            print(f"  [QAQA] Warning: Could not load {stats_file}: {e}")

    return results


def load_qaasker_nli_results(input_dir: Path) -> Dict[str, Dict]:
    """Load QAASKER NLI statistics files for all models."""
    results = {}

    stats_files = sorted(input_dir.glob("qaasker_*_statistics.json"))
    for stats_file in stats_files:
        try:
            with open(stats_file, "r") as f:
                data = json.load(f)

            model_name = data.get("model_name", "")
            if not model_name:
                parts = stats_file.stem.split("_")
                if len(parts) > 1:
                    model_name = "_".join(parts[1:-1])

            display_name = normalize_model_name(model_name)
            data["display_name"] = display_name
            data["violation_rate"] = data.get("violation_rate", 0)
            results[display_name] = data
            print(f"  [QAASKER] Loaded: {display_name} (NLI rate: {data.get('violation_rate', 0)*100:.2f}%)")
        except Exception as e:
            print(f"  [QAASKER] Warning: Could not load {stats_file}: {e}")

    return results


def load_kontest_results(input_dir: Path) -> Dict[str, Dict]:
    """Load KONTEST-inspired exact-match statistics files for all models."""
    results = {}

    stats_files = sorted(input_dir.glob("kontest_*_statistics.json"))
    for stats_file in stats_files:
        try:
            with open(stats_file, "r") as f:
                data = json.load(f)

            model_name = data.get("model_name", "")
            if not model_name:
                stem = stats_file.stem
                prefix = "kontest_"
                suffix = "_statistics"
                if stem.startswith(prefix) and stem.endswith(suffix):
                    model_name = stem[len(prefix):-len(suffix)]

            display_name = normalize_model_name(model_name)
            valid_pairs = data.get("valid_pairs", data.get("total_pairs", 0))
            violations = data.get("violations", 0)
            violation_rate = data.get("violation_rate", violations / valid_pairs if valid_pairs else 0)

            data["display_name"] = display_name
            data["error_detection_rate"] = violation_rate
            data["total_count"] = valid_pairs
            data["violation_count"] = violations
            data.setdefault("by_mutation_type", {
                "atomic": {
                    "total": valid_pairs,
                    "violations": violations,
                    "error_rate": violation_rate,
                    "violation_rate": violation_rate,
                }
            })
            results[display_name] = data
            print(f"  [KONTEST] Loaded: {display_name} (exact rate: {violation_rate*100:.2f}%, valid pairs: {valid_pairs})")
        except Exception as e:
            print(f"  [KONTEST] Warning: Could not load {stats_file}: {e}")

    return results


def load_honest_results(base_path: Path) -> Dict[str, Dict]:
    """Load HONEST consistency results for all models."""
    results = {}

    for display_name, honest_dir_name in HONEST_DIR_MAPPING.items():
        honest_dir = base_path / honest_dir_name
        summary_file = honest_dir / "all_consistency_summary.csv"

        if not summary_file.exists():
            print(f"  [HONEST] Warning: {summary_file} not found")
            continue

        try:
            df = pd.read_csv(summary_file)
            total = len(df)
            inconsistent = (~df['answers_consistent']).sum()
            inconsistency_rate = inconsistent / total if total > 0 else 0

            # Group by mutation type
            by_mutation_type = defaultdict(lambda: {"total": 0, "violations": 0})

            for rule_dir in honest_dir.glob("kg_rule_*"):
                for mutation_type in HONEST_MUTATION_TYPES:
                    csv_file = rule_dir / f"{mutation_type}_llm_answers.csv"
                    if csv_file.exists():
                        try:
                            mut_df = pd.read_csv(csv_file)
                            mut_total = len(mut_df)
                            mut_inconsistent = (~mut_df['answers_consistent']).sum()
                            by_mutation_type[mutation_type]["total"] += mut_total
                            by_mutation_type[mutation_type]["violations"] += mut_inconsistent
                        except:
                            pass

            mutation_stats = {}
            for mt, stats in by_mutation_type.items():
                rate = stats["violations"] / stats["total"] if stats["total"] > 0 else 0
                mutation_stats[mt] = {
                    "total": stats["total"],
                    "violations": stats["violations"],
                    "error_rate": rate
                }

            results[display_name] = {
                "display_name": display_name,
                "total_count": total,
                "violation_count": int(inconsistent),
                "error_detection_rate": inconsistency_rate,
                "by_mutation_type": mutation_stats
            }
            print(f"  [HONEST] Loaded: {display_name} (rate: {inconsistency_rate*100:.2f}%)")
        except Exception as e:
            print(f"  [HONEST] Warning: Could not load {honest_dir}: {e}")

    return results


def load_drhall_results(input_dir: Path) -> Dict[str, Dict]:
    """Load DrHall statistics files for all models."""
    results = {}

    stats_files = sorted(input_dir.glob("drhall_*_statistics.json"))
    for stats_file in stats_files:
        if "_nli_" in stats_file.name:
            continue  # prefer non-NLI file (same content anyway)
        try:
            with open(stats_file, "r") as f:
                data = json.load(f)

            model_name = data.get("model_name", "")
            display_name = normalize_model_name(model_name)
            data["display_name"] = display_name
            results[display_name] = data
            print(f"  [DrHall] Loaded: {display_name} (rate: {data.get('error_detection_rate', 0)*100:.2f}%)")
        except Exception as e:
            print(f"  [DrHall] Warning: Could not load {stats_file}: {e}")

    return results


def load_metaqa_results(input_dir: Path) -> Dict[str, Dict]:
    """Load MetaQA statistics files for all models.

    Reads metaqa_<model>_statistics.json produced by rq4_metaqa_evaluation.py.
    Each file exposes ``error_detection_rate`` and ``by_mutation_type`` keyed by
    the MetaQA MR names ("Synonym" / "Antonym").
    """
    results = {}

    stats_files = sorted(input_dir.glob("metaqa_*_statistics.json"))
    for stats_file in stats_files:
        if "_nli_" in stats_file.name:
            continue  # NLI variant duplicates the regular file
        try:
            with open(stats_file, "r") as f:
                data = json.load(f)

            model_name = data.get("model_name", "")
            if not model_name:
                # Fall back to filename: metaqa_<model>_statistics.json
                stem = stats_file.stem
                prefix, suffix = "metaqa_", "_statistics"
                if stem.startswith(prefix) and stem.endswith(suffix):
                    model_name = stem[len(prefix):-len(suffix)]

            display_name = normalize_model_name(model_name)
            data["display_name"] = display_name
            results[display_name] = data
            print(f"  [MetaQA] Loaded: {display_name} (rate: {data.get('error_detection_rate', 0)*100:.2f}%)")
        except Exception as e:
            print(f"  [MetaQA] Warning: Could not load {stats_file}: {e}")

    return results


# ============================================================================
# Table Generation Functions
# ============================================================================

def format_rate(rate: Optional[float]) -> str:
    """Format rate as percentage string."""
    if rate is None:
        return "N/A"
    return f"{rate * 100:.2f}%"


def generate_main_comparison_table(
    qaqa_results: Dict[str, Dict],
    qaasker_results: Dict[str, Dict],
    kontest_results: Dict[str, Dict],
    honest_results: Dict[str, Dict],
    drhall_results: Dict[str, Dict] = None,
    metaqa_results: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """Generate main comparison table: Method × Mutation Operator × Model."""
    rows = []

    # QAQA rows
    for mutation_type in QAQA_MUTATION_TYPES:
        row = {"Method": "QAQA", "Mutation Operator": QAQA_MUTATION_DISPLAY.get(mutation_type, mutation_type)}
        rates = []
        for model in MODEL_ORDER:
            if model in qaqa_results:
                by_mut = qaqa_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"error_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats["error_rate"]
                    row[model] = rate
                    rates.append(rate)
                else:
                    row[model] = None
            else:
                row[model] = None
        row["Avg."] = sum(rates) / len(rates) if rates else None
        rows.append(row)

    # QAASKER rows
    for mr_type in QAASKER_MR_TYPES:
        row = {"Method": "QAASKER", "Mutation Operator": QAASKER_MR_DISPLAY.get(mr_type, mr_type)}
        rates = []
        for model in MODEL_ORDER:
            if model in qaasker_results:
                by_mr = qaasker_results[model].get("by_mr_type", {})
                stats = by_mr.get(mr_type, {"violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats["violation_rate"]
                    row[model] = rate
                    rates.append(rate)
                else:
                    row[model] = None
            else:
                row[model] = None
        row["Avg."] = sum(rates) / len(rates) if rates else None
        rows.append(row)

    # KONTEST rows
    for mutation_type in KONTEST_MUTATION_TYPES:
        row = {"Method": "KONTEST", "Mutation Operator": KONTEST_MUTATION_DISPLAY.get(mutation_type, mutation_type)}
        rates = []
        for model in MODEL_ORDER:
            if model in kontest_results:
                by_mut = kontest_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"error_rate": 0, "violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats.get("error_rate", stats.get("violation_rate", 0))
                    row[model] = rate
                    rates.append(rate)
                else:
                    row[model] = None
            else:
                row[model] = None
        row["Avg."] = sum(rates) / len(rates) if rates else None
        rows.append(row)

    # HONEST rows
    for mutation_type in HONEST_MUTATION_TYPES:
        row = {"Method": "HONEST", "Mutation Operator": HONEST_MUTATION_DISPLAY.get(mutation_type, mutation_type)}
        rates = []
        for model in MODEL_ORDER:
            if model in honest_results:
                by_mut = honest_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"error_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats["error_rate"]
                    row[model] = rate
                    rates.append(rate)
                else:
                    row[model] = None
            else:
                row[model] = None
        row["Avg."] = sum(rates) / len(rates) if rates else None
        rows.append(row)

    # DrHall rows
    drhall_results = drhall_results or {}
    for mr_type in DRHALL_MR_TYPES:
        row = {"Method": "DrHall", "Mutation Operator": DRHALL_MR_DISPLAY.get(mr_type, mr_type)}
        rates = []
        for model in MODEL_ORDER:
            if model in drhall_results:
                by_mr = drhall_results[model].get("by_mutation_type", {})
                stats = by_mr.get(mr_type, {"violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats.get("violation_rate", 0)
                    row[model] = rate
                    rates.append(rate)
                else:
                    row[model] = None
            else:
                row[model] = None
        row["Avg."] = sum(rates) / len(rates) if rates else None
        rows.append(row)

    # MetaQA rows
    metaqa_results = metaqa_results or {}
    for mutation_type in METAQA_MUTATION_TYPES:
        row = {"Method": "MetaQA", "Mutation Operator": METAQA_MUTATION_DISPLAY.get(mutation_type, mutation_type)}
        rates = []
        for model in MODEL_ORDER:
            if model in metaqa_results:
                by_mut = metaqa_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats.get("violation_rate", 0)
                    row[model] = rate
                    rates.append(rate)
                else:
                    row[model] = None
            else:
                row[model] = None
        row["Avg."] = sum(rates) / len(rates) if rates else None
        rows.append(row)

    columns = ["Method", "Mutation Operator"] + MODEL_ORDER + ["Avg."]
    return pd.DataFrame(rows, columns=columns)


def generate_overall_table(
    qaqa_results: Dict[str, Dict],
    qaasker_results: Dict[str, Dict],
    kontest_results: Dict[str, Dict],
    honest_results: Dict[str, Dict],
    drhall_results: Dict[str, Dict] = None,
    metaqa_results: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """Generate overall comparison table."""
    rows = []

    for method_name, method_results in [
        ("QAQA", qaqa_results),
        ("QAASKER", qaasker_results),
        ("KONTEST", kontest_results),
        ("HONEST", honest_results),
        ("DrHall", drhall_results or {}),
        ("MetaQA", metaqa_results or {}),
    ]:
        row = {"Method": method_name}
        rates = []
        for model in MODEL_ORDER:
            if method_results and model in method_results:
                r = method_results[model]
                if "error_detection_rate" in r:
                    rate = r["error_detection_rate"]
                elif "violation_rate" in r:
                    rate = r["violation_rate"]
                else:
                    rate = 0
                row[model] = rate
                rates.append(rate)
            else:
                row[model] = None
        row["Avg."] = sum(rates) / len(rates) if rates else None
        rows.append(row)

    columns = ["Method"] + MODEL_ORDER + ["Avg."]
    return pd.DataFrame(rows, columns=columns)


def generate_latex_main_table(
    qaqa_results: Dict[str, Dict],
    qaasker_results: Dict[str, Dict],
    kontest_results: Dict[str, Dict],
    honest_results: Dict[str, Dict],
    drhall_results: Dict[str, Dict] = None,
    metaqa_results: Dict[str, Dict] = None,
) -> str:
    """Generate LaTeX table for main comparison."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inconsistency Rates by Method and Mutation Operator Across Models (\%)}")
    lines.append(r"\label{tab:rq4_main}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "r" * len(MODEL_ORDER) + "r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Mutation Operator} & " +
                " & ".join([r"\textbf{" + m + r"}" for m in MODEL_ORDER]) +
                r" & \textbf{Avg.} \\")
    lines.append(r"\midrule")

    # QAQA rows
    for i, mutation_type in enumerate(QAQA_MUTATION_TYPES):
        rates, cells = [], []
        for model in MODEL_ORDER:
            if model in qaqa_results:
                by_mut = qaqa_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"error_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats["error_rate"]
                    cells.append(f"{rate*100:.2f}")
                    rates.append(rate)
                else:
                    cells.append("--")
            else:
                cells.append("--")
        cells.append(f"{sum(rates)/len(rates)*100:.2f}" if rates else "--")
        display_name = QAQA_MUTATION_DISPLAY.get(mutation_type, mutation_type)
        if i == 0:
            lines.append(f"\\multirow{{3}}{{*}}{{QAQA}} & {display_name} & " + " & ".join(cells) + r" \\")
        else:
            lines.append(f" & {display_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # QAASKER rows
    for i, mr_type in enumerate(QAASKER_MR_TYPES):
        rates, cells = [], []
        for model in MODEL_ORDER:
            if model in qaasker_results:
                by_mr = qaasker_results[model].get("by_mr_type", {})
                stats = by_mr.get(mr_type, {"violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats["violation_rate"]
                    cells.append(f"{rate*100:.2f}")
                    rates.append(rate)
                else:
                    cells.append("--")
            else:
                cells.append("--")
        cells.append(f"{sum(rates)/len(rates)*100:.2f}" if rates else "--")
        display_name = QAASKER_MR_DISPLAY.get(mr_type, mr_type)
        if i == 0:
            lines.append(f"\\multirow{{3}}{{*}}{{QAASKER}} & {display_name} & " + " & ".join(cells) + r" \\")
        else:
            lines.append(f" & {display_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # KONTEST rows
    for mutation_type in KONTEST_MUTATION_TYPES:
        rates, cells = [], []
        for model in MODEL_ORDER:
            if model in kontest_results:
                by_mut = kontest_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"error_rate": 0, "violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats.get("error_rate", stats.get("violation_rate", 0))
                    cells.append(f"{rate*100:.2f}")
                    rates.append(rate)
                else:
                    cells.append("--")
            else:
                cells.append("--")
        cells.append(f"{sum(rates)/len(rates)*100:.2f}" if rates else "--")
        display_name = KONTEST_MUTATION_DISPLAY.get(mutation_type, mutation_type)
        lines.append(f"KONTEST & {display_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # HONEST rows
    for i, mutation_type in enumerate(HONEST_MUTATION_TYPES):
        rates, cells = [], []
        for model in MODEL_ORDER:
            if model in honest_results:
                by_mut = honest_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"error_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats["error_rate"]
                    cells.append(f"{rate*100:.2f}")
                    rates.append(rate)
                else:
                    cells.append("--")
            else:
                cells.append("--")
        cells.append(f"{sum(rates)/len(rates)*100:.2f}" if rates else "--")
        display_name = HONEST_MUTATION_DISPLAY.get(mutation_type, mutation_type)
        if i == 0:
            lines.append(f"\\multirow{{3}}{{*}}{{HONEST}} & {display_name} & " + " & ".join(cells) + r" \\")
        else:
            lines.append(f" & {display_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # DrHall rows
    drhall_results = drhall_results or {}
    for i, mr_type in enumerate(DRHALL_MR_TYPES):
        rates, cells = [], []
        for model in MODEL_ORDER:
            if model in drhall_results:
                by_mr = drhall_results[model].get("by_mutation_type", {})
                stats = by_mr.get(mr_type, {"violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats.get("violation_rate", 0)
                    cells.append(f"{rate*100:.2f}")
                    rates.append(rate)
                else:
                    cells.append("--")
            else:
                cells.append("--")
        cells.append(f"{sum(rates)/len(rates)*100:.2f}" if rates else "--")
        display_name = DRHALL_MR_DISPLAY.get(mr_type, mr_type)
        if i == 0:
            lines.append(f"\\multirow{{{len(DRHALL_MR_TYPES)}}}{{*}}{{DrHall}} & {display_name} & " + " & ".join(cells) + r" \\")
        else:
            lines.append(f" & {display_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # MetaQA rows
    metaqa_results = metaqa_results or {}
    for i, mutation_type in enumerate(METAQA_MUTATION_TYPES):
        rates, cells = [], []
        for model in MODEL_ORDER:
            if model in metaqa_results:
                by_mut = metaqa_results[model].get("by_mutation_type", {})
                stats = by_mut.get(mutation_type, {"violation_rate": 0, "total": 0})
                if stats["total"] > 0:
                    rate = stats.get("violation_rate", 0)
                    cells.append(f"{rate*100:.2f}")
                    rates.append(rate)
                else:
                    cells.append("--")
            else:
                cells.append("--")
        cells.append(f"{sum(rates)/len(rates)*100:.2f}" if rates else "--")
        display_name = METAQA_MUTATION_DISPLAY.get(mutation_type, mutation_type)
        if i == 0:
            lines.append(f"\\multirow{{{len(METAQA_MUTATION_TYPES)}}}{{*}}{{MetaQA}} & {display_name} & " + " & ".join(cells) + r" \\")
        else:
            lines.append(f" & {display_name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_latex_overall_table(
    qaqa_results: Dict[str, Dict],
    qaasker_results: Dict[str, Dict],
    kontest_results: Dict[str, Dict],
    honest_results: Dict[str, Dict],
    drhall_results: Dict[str, Dict] = None,
    metaqa_results: Dict[str, Dict] = None,
) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall Inconsistency Rate Comparison (\%)}")
    lines.append(r"\label{tab:rq4_overall}")
    lines.append(r"\begin{tabular}{l" + "r" * len(MODEL_ORDER) + "r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & " + " & ".join([r"\textbf{" + m + r"}" for m in MODEL_ORDER]) +
                r" & \textbf{Avg.} \\")
    lines.append(r"\midrule")

    for method_name, method_results in [
        ("QAQA", qaqa_results),
        ("QAASKER", qaasker_results),
        ("KONTEST", kontest_results),
        ("HONEST", honest_results),
        ("DrHall", drhall_results or {}),
        ("MetaQA", metaqa_results or {}),
    ]:
        rates, cells = [], [method_name]
        for model in MODEL_ORDER:
            if model in method_results:
                r = method_results[model]
                if "error_detection_rate" in r:
                    rate = r["error_detection_rate"]
                elif "violation_rate" in r:
                    rate = r["violation_rate"]
                else:
                    rate = 0
                cells.append(f"{rate*100:.2f}")
                rates.append(rate)
            else:
                cells.append("--")
        cells.append(f"{sum(rates)/len(rates)*100:.2f}" if rates else "--")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================================
# Venn Diagram Functions
# ============================================================================

def load_ours_violations(honest_dir: Path, original_df: pd.DataFrame) -> Tuple[Set[int], Set[str], pd.DataFrame]:
    """Load violations detected by HONEST method."""
    summary_file = honest_dir / "all_consistency_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    df = pd.read_csv(summary_file)

    question_to_idx = {}
    for idx, row in original_df.iterrows():
        q = row.get('original_question', '')
        if pd.notna(q):
            q_normalized = ' '.join(str(q).split())
            question_to_idx[q_normalized] = idx

    violations_df = df[df['answers_consistent'] == False].copy()

    base_indices = set()
    matched_questions = set()

    for _, row in violations_df.iterrows():
        q = row.get('original_question', '')
        if pd.notna(q):
            q_normalized = ' '.join(str(q).split())
            if q_normalized in question_to_idx:
                base_indices.add(question_to_idx[q_normalized])
                matched_questions.add(q_normalized)

    return base_indices, matched_questions, violations_df


def load_qaqa_nli_violations(data_file: Path) -> Tuple[Set[int], pd.DataFrame]:
    """Load violations detected by QAQA method using NLI evaluation."""
    if not data_file.exists():
        raise FileNotFoundError(f"QAQA file not found: {data_file}")

    df = pd.read_csv(data_file)

    if 'is_violation_nli' in df.columns:
        violations_df = df[df['is_violation_nli'] == True]
    else:
        violations_df = df[df['is_violation'] == True]

    base_indices = set(violations_df['index'].dropna().astype(int).unique())

    return base_indices, violations_df


def load_qaasker_nli_violations(data_file: Path) -> Tuple[Set[int], pd.DataFrame]:
    """Load violations detected by QAAsker method using NLI evaluation."""
    if not data_file.exists():
        raise FileNotFoundError(f"QAAsker file not found: {data_file}")

    df = pd.read_csv(data_file)

    if 'is_violation_nli' in df.columns:
        violations_df = df[df['is_violation_nli'] == True]
    else:
        violations_df = df[df['is_violation'] == True]

    base_indices = set(violations_df['row_index'].dropna().astype(int).unique())

    return base_indices, violations_df


def load_drhall_nli_violations(data_file: Path) -> Tuple[Set[int], pd.DataFrame]:
    """Load violations detected by DrHall method (NLI protocol).

    A DrHall detail CSV expands one golden base instance into multiple rows
    (multiple MRs / follow-ups), so we aggregate by ``row_index``: any instance
    whose any row is flagged as a violation counts toward the violation set.
    Prefer ``is_violation_nli`` (the raw ``is_violation`` is inflated by false
    positives); fall back when missing.
    """
    if not data_file.exists():
        raise FileNotFoundError(f"DrHall file not found: {data_file}")

    df = pd.read_csv(data_file)

    if 'is_violation_nli' in df.columns:
        violations_df = df[df['is_violation_nli'] == True]
    else:
        violations_df = df[df['is_violation'] == True]

    base_indices = set(violations_df['row_index'].dropna().astype(int).unique())

    return base_indices, violations_df


def load_metaqa_nli_violations(data_file: Path) -> Tuple[Set[int], pd.DataFrame]:
    """Load violations detected by MetaQA method.

    MetaQA's NLI-version CSV did not produce an ``is_violation_nli`` column, so
    it falls back to ``is_violation``; the rest of the aggregation logic matches
    DrHall (aggregate by ``row_index``).
    """
    if not data_file.exists():
        raise FileNotFoundError(f"MetaQA file not found: {data_file}")

    df = pd.read_csv(data_file)

    if 'is_violation_nli' in df.columns:
        violations_df = df[df['is_violation_nli'] == True]
    else:
        violations_df = df[df['is_violation'] == True]

    base_indices = set(violations_df['row_index'].dropna().astype(int).unique())

    return base_indices, violations_df


def get_venn_labels(set_a: Set, set_b: Set, set_c: Set) -> Dict[str, str]:
    """Calculate Venn diagram labels for 3 sets."""
    only_a = len(set_a - set_b - set_c)
    only_b = len(set_b - set_a - set_c)
    only_c = len(set_c - set_a - set_b)
    ab = len((set_a & set_b) - set_c)
    ac = len((set_a & set_c) - set_b)
    bc = len((set_b & set_c) - set_a)
    abc = len(set_a & set_b & set_c)

    return {
        '100': f'{only_a}', '010': f'{only_b}', '001': f'{only_c}',
        '110': f'{ab}', '101': f'{ac}', '011': f'{bc}', '111': f'{abc}',
    }


def draw_venn3_circles(ax, labels, names, colors=None, fontsize=10):
    """Draw a 3-set Venn diagram using circles."""
    if colors is None:
        colors = [
            [92/255, 192/255, 98/255, 0.5],
            [90/255, 155/255, 212/255, 0.5],
            [246/255, 236/255, 86/255, 0.6],
        ]

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    circle_positions = [
        (0.35, 0.65),   # Top-left (HONEST)
        (0.85, 0.65),   # Top-right (QAQA)
        (0.6, 0.3),     # Bottom (QAAsker)
    ]
    circle_radius = 0.4

    for (x, y), color in zip(circle_positions, colors):
        circle = mpatches.Circle((x, y), circle_radius, color=color, ec='black', linewidth=1.0)
        ax.add_patch(circle)

    # Labels (numbers inside circles)
    ax.text(0.2, 0.75, labels.get('100', ''), ha='center', va='center', fontsize=fontsize, fontweight='bold')
    ax.text(1.0, 0.75, labels.get('010', ''), ha='center', va='center', fontsize=fontsize, fontweight='bold')
    ax.text(0.6, 0.15, labels.get('001', ''), ha='center', va='center', fontsize=fontsize, fontweight='bold')
    ax.text(0.6, 0.78, labels.get('110', ''), ha='center', va='center', fontsize=fontsize-1)
    ax.text(0.4, 0.42, labels.get('101', ''), ha='center', va='center', fontsize=fontsize-1)
    ax.text(0.8, 0.42, labels.get('011', ''), ha='center', va='center', fontsize=fontsize-1)
    ax.text(0.6, 0.52, labels.get('111', ''), ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # Names (method labels)
    ax.text(0.0, 1.05, names[0], ha='center', va='bottom', fontsize=fontsize, fontweight='bold', color='#5CC062')
    ax.text(1.2, 1.05, names[1], ha='center', va='bottom', fontsize=fontsize, fontweight='bold', color='#5A9BD4')
    ax.text(0.6, -0.15, names[2], ha='center', va='top', fontsize=fontsize, fontweight='bold', color='#F6EC56')


def create_venn_diagram(
    honest_set: Set[int],
    qaqa_set: Set[int],
    qaasker_set: Set[int],
    output_path: Path,
    model_name: str
):
    """Create a 3-set Venn diagram showing the overlap between methods."""
    labels = get_venn_labels(honest_set, qaqa_set, qaasker_set)

    # Single column figure size (approximately 3.5 inches wide for IEEE/ACM format)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    draw_venn3_circles(ax, labels, names=['HONEST', 'QAQA', 'QAAsker'], fontsize=8)

    # No title - will use figure caption in paper

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return labels


def print_overlap_statistics(honest_set: Set[int], qaqa_set: Set[int], qaasker_set: Set[int], model_name: str):
    """Print detailed statistics about the overlaps."""
    print(f"\n{'='*70}")
    print(f"Overlap Statistics for {model_name}")
    print(f"{'='*70}")

    print(f"\nIndividual Method Counts:")
    print(f"  HONEST:  {len(honest_set)} samples")
    print(f"  QAQA:    {len(qaqa_set)} samples")
    print(f"  QAAsker: {len(qaasker_set)} samples")

    print(f"\nPairwise Overlaps:")
    print(f"  HONEST ∩ QAQA:    {len(honest_set & qaqa_set)} samples")
    print(f"  HONEST ∩ QAAsker: {len(honest_set & qaasker_set)} samples")
    print(f"  QAQA ∩ QAAsker:   {len(qaqa_set & qaasker_set)} samples")

    print(f"\nTriple Overlap:")
    print(f"  HONEST ∩ QAQA ∩ QAAsker: {len(honest_set & qaqa_set & qaasker_set)} samples")

    print(f"\nUnique to Each Method:")
    print(f"  Only HONEST:  {len(honest_set - qaqa_set - qaasker_set)} samples")
    print(f"  Only QAQA:    {len(qaqa_set - honest_set - qaasker_set)} samples")
    print(f"  Only QAAsker: {len(qaasker_set - honest_set - qaqa_set)} samples")

    def jaccard(set1, set2):
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0

    print(f"\nJaccard Similarity:")
    print(f"  HONEST vs QAQA:    {jaccard(honest_set, qaqa_set):.4f}")
    print(f"  HONEST vs QAAsker: {jaccard(honest_set, qaasker_set):.4f}")
    print(f"  QAQA vs QAAsker:   {jaccard(qaqa_set, qaasker_set):.4f}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ4 Final Analysis")
    parser.add_argument("--skip-tables", action="store_true", help="Skip generating tables")
    parser.add_argument("--skip-venn", action="store_true", help="Skip generating Venn diagrams")
    parser.add_argument("--model", type=str, help="Generate Venn diagram for specific model only")
    args = parser.parse_args()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RQ4 Final Analysis: Using NLI-based Evaluation Results")
    print("=" * 80)

    # =========================================================================
    # Generate Tables
    # =========================================================================
    if not args.skip_tables:
        print("\n" + "=" * 80)
        print("[1] Loading Results...")
        print("=" * 80)

        print("\nLoading QAQA NLI results...")
        qaqa_results = load_qaqa_nli_results(NLI_RESULTS_DIR)

        print("\nLoading QAASKER NLI results...")
        qaasker_results = load_qaasker_nli_results(NLI_RESULTS_DIR)

        print("\nLoading KONTEST exact-match results...")
        kontest_results = load_kontest_results(KONTEST_RESULTS_DIR)
        if not kontest_results:
            kontest_results = load_kontest_results(NLI_RESULTS_DIR)

        print("\nLoading HONEST results...")
        honest_results = load_honest_results(HONEST_RESULTS_BASE)

        print("\nLoading DrHall results...")
        drhall_results = load_drhall_results(KONTEST_RESULTS_DIR)

        print("\nLoading MetaQA results...")
        metaqa_results = load_metaqa_results(KONTEST_RESULTS_DIR)

        print("\n" + "=" * 80)
        print("Data Summary")
        print("=" * 80)
        print(f"QAQA:    {len(qaqa_results)} models - {list(qaqa_results.keys())}")
        print(f"QAASKER: {len(qaasker_results)} models - {list(qaasker_results.keys())}")
        print(f"KONTEST: {len(kontest_results)} models - {list(kontest_results.keys())}")
        print(f"HONEST:  {len(honest_results)} models - {list(honest_results.keys())}")
        print(f"DrHall:  {len(drhall_results)} models - {list(drhall_results.keys())}")
        print(f"MetaQA:  {len(metaqa_results)} models - {list(metaqa_results.keys())}")

        # Generate main comparison table
        print("\n" + "=" * 80)
        print("[2] Generating Main Comparison Table")
        print("=" * 80)

        main_table = generate_main_comparison_table(qaqa_results, qaasker_results, kontest_results, honest_results, drhall_results, metaqa_results)

        display_table = main_table.copy()
        for col in MODEL_ORDER + ["Avg."]:
            display_table[col] = display_table[col].apply(format_rate)
        print(display_table.to_string(index=False))

        main_csv_path = OUTPUT_DIR / "rq4_main_comparison_nli.csv"
        main_table.to_csv(main_csv_path, index=False)
        print(f"\nSaved to: {main_csv_path}")

        # Generate overall table
        print("\n" + "=" * 80)
        print("[3] Generating Overall Comparison Table")
        print("=" * 80)

        overall_table = generate_overall_table(qaqa_results, qaasker_results, kontest_results, honest_results, drhall_results, metaqa_results)
        display_overall = overall_table.copy()
        for col in MODEL_ORDER + ["Avg."]:
            display_overall[col] = display_overall[col].apply(format_rate)
        print(display_overall.to_string(index=False))

        overall_csv_path = OUTPUT_DIR / "rq4_overall_comparison_nli.csv"
        overall_table.to_csv(overall_csv_path, index=False)
        print(f"\nSaved to: {overall_csv_path}")

        # Generate LaTeX tables
        print("\n" + "=" * 80)
        print("[4] Generating LaTeX Tables...")
        print("=" * 80)

        latex_main = generate_latex_main_table(qaqa_results, qaasker_results, kontest_results, honest_results, drhall_results, metaqa_results)
        latex_main_path = OUTPUT_DIR / "rq4_main_table_nli.tex"
        with open(latex_main_path, "w") as f:
            f.write(latex_main)
        print(f"Main LaTeX table saved to: {latex_main_path}")

        latex_overall = generate_latex_overall_table(qaqa_results, qaasker_results, kontest_results, honest_results, drhall_results, metaqa_results)
        latex_overall_path = OUTPUT_DIR / "rq4_overall_table_nli.tex"
        with open(latex_overall_path, "w") as f:
            f.write(latex_overall)
        print(f"Overall LaTeX table saved to: {latex_overall_path}")

    # =========================================================================
    # Generate Venn Diagrams
    # =========================================================================
    if not args.skip_venn:
        print("\n" + "=" * 80)
        print("[5] Generating Venn Diagrams...")
        print("=" * 80)

        # Load original dataset for mapping
        original_data_file = DATA_BASE_PATH / "golden_dataset/golden_dataset_full.csv"
        print(f"\nLoading original dataset from: {original_data_file}")
        original_df = pd.read_csv(original_data_file)
        original_df['_idx'] = original_df.index
        print(f"  Loaded {len(original_df)} base samples")

        all_overlap_stats = {}
        all_honest_sets = {}
        all_qaqa_sets = {}
        all_qaqaasker_sets = {}

        models_to_process = [args.model] if args.model else MODEL_ORDER

        for model_name in models_to_process:
            print(f"\n{'='*70}")
            print(f"Processing {model_name}...")
            print(f"{'='*70}")

            internal_name = INTERNAL_MODEL_NAMES.get(model_name, model_name.lower())
            honest_dir_name = HONEST_DIR_MAPPING.get(model_name)

            if not honest_dir_name:
                print(f"  Warning: No HONEST directory mapping for {model_name}")
                continue

            honest_dir = HONEST_RESULTS_BASE / honest_dir_name

            # Find QAQA file (use regular file, not *_nli_* which has buggy evaluation)
            qaqa_file = NLI_RESULTS_DIR / f"qaqa_{internal_name}_evaluation_results.csv"
            if not qaqa_file.exists():
                # Try alternative patterns
                alt_files = list(NLI_RESULTS_DIR.glob(f"qaqa_{internal_name}*_evaluation_results.csv"))
                # Prefer non-nli files
                non_nli_files = [f for f in alt_files if "_nli_" not in f.name]
                if non_nli_files:
                    qaqa_file = non_nli_files[0]
                elif alt_files:
                    qaqa_file = alt_files[0]
                else:
                    print(f"  Warning: QAQA file not found for {model_name}")
                    continue

            # Find QAASKER file
            qaasker_file = NLI_RESULTS_DIR / f"qaasker_{internal_name}_evaluation_results.csv"
            if not qaasker_file.exists():
                alt_files = list(NLI_RESULTS_DIR.glob(f"qaasker_{internal_name}*_evaluation_results.csv"))
                if alt_files:
                    qaasker_file = alt_files[0]
                else:
                    print(f"  Warning: QAASKER file not found for {model_name}")
                    continue

            print(f"  HONEST dir: {honest_dir}")
            print(f"  QAQA file:  {qaqa_file}")
            print(f"  QAASKER file: {qaasker_file}")

            # Load violations
            try:
                honest_set, _, _ = load_ours_violations(honest_dir, original_df)
                qaqa_set, _ = load_qaqa_nli_violations(qaqa_file)
                qaasker_set, _ = load_qaasker_nli_violations(qaasker_file)

                print(f"  Loaded {len(honest_set)} HONEST violations")
                print(f"  Loaded {len(qaqa_set)} QAQA violations")
                print(f"  Loaded {len(qaasker_set)} QAASKER violations")

                # Print statistics
                print_overlap_statistics(honest_set, qaqa_set, qaasker_set, model_name)

                # Create Venn diagram
                output_path = FIGURES_DIR / f"rq4_overlap_{model_name.replace('-', '_')}.png"
                create_venn_diagram(honest_set, qaqa_set, qaasker_set, output_path, model_name)
                print(f"\n  Venn diagram saved to: {output_path}")
                print(f"  Venn diagram saved to: {output_path.with_suffix('.pdf')}")

                # Store sets for overall diagram
                all_honest_sets[model_name] = honest_set
                all_qaqa_sets[model_name] = qaqa_set
                all_qaqaasker_sets[model_name] = qaasker_set

                all_overlap_stats[model_name] = {
                    'honest': len(honest_set),
                    'qaqa': len(qaqa_set),
                    'qaasker': len(qaasker_set),
                    'honest_qaqa': len(honest_set & qaqa_set),
                    'honest_qaasker': len(honest_set & qaasker_set),
                    'qaqa_qaasker': len(qaqa_set & qaasker_set),
                    'all_three': len(honest_set & qaqa_set & qaasker_set),
                }

            except Exception as e:
                print(f"  Error processing {model_name}: {e}")
                import traceback
                traceback.print_exc()

        # Save overlap statistics
        if all_overlap_stats:
            stats_file = FIGURES_DIR / "rq4_overlap_statistics.json"
            with open(stats_file, "w") as f:
                json.dump(all_overlap_stats, f, indent=2)
            print(f"\nOverlap statistics saved to: {stats_file}")

        # Generate overall Venn diagram (combining all models)
        if all_honest_sets and all_qaqa_sets and all_qaqaasker_sets:
            print("\n" + "=" * 70)
            print("Generating Overall Venn Diagram (All Models Combined)...")
            print("=" * 70)

            # Merge all sets
            all_honest = set()
            all_qaqa = set()
            all_qaqaasker = set()
            for model_name in all_honest_sets:
                # Use (model_name, sample_index) tuple to distinguish samples from different models
                for idx in all_honest_sets[model_name]:
                    all_honest.add((model_name, idx))
                for idx in all_qaqa_sets[model_name]:
                    all_qaqa.add((model_name, idx))
                for idx in all_qaqaasker_sets[model_name]:
                    all_qaqaasker.add((model_name, idx))

            print(f"  Combined HONEST violations: {len(all_honest)}")
            print(f"  Combined QAQA violations: {len(all_qaqa)}")
            print(f"  Combined QAASKER violations: {len(all_qaqaasker)}")

            # Print overall statistics
            print_overlap_statistics(all_honest, all_qaqa, all_qaqaasker, "All Models Combined")

            # Create overall Venn diagram
            output_path = FIGURES_DIR / "rq4_overlap_all_models.png"
            create_venn_diagram(all_honest, all_qaqa, all_qaqaasker, output_path, "All Models")
            print(f"\n  Overall Venn diagram saved to: {output_path}")
            print(f"  Overall Venn diagram saved to: {output_path.with_suffix('.pdf')}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
