#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 Fair Comparison Table (all three methods on the same golden set)

Generates an inconsistency-rate comparison table for QAQA / QAASKER / HONEST all
evaluated on the SAME golden dataset, replacing the "unfair" comparison in
rq4_final_analysis.py that arose from HONEST mistakenly using the full dataset.

Alignment of evaluation protocol:
- QAQA    : reuses rq4_final_analysis.load_qaqa_nli_results logic
            -> qaqa_<model>_statistics.json (skips the buggy _nli_ files)
            -> by_mutation_type[EQ/EC/EQC]['error_rate']
- QAASKER : reuses load_qaasker_nli_results logic
            -> qaasker_<model>_statistics.json
            -> by_mr_type[MR1/MR2/MR3]['violation_rate']
- HONEST  : new protocol -- reads from this repo's accompanying golden NLI
            results directory
            -> consistency_results_nli_golden_<model>/kg_rule_*/<op>_llm_answers.csv
            -> (~answers_consistent).mean()   (identical to load_honest_results)

Note: the "inconsistency" verdict for all three is itself NLI-based, and
QAQA/QAASKER were already run on golden, so this table is a genuinely fair
comparison on the same golden set under the same NLI protocol.

Output: data/examples/rq4_summary/fair_comparison_*.{csv,tex}
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_BASE = Path("$PROJECT_ROOT/data/examples")
RQ4_NLI_DIR = DATA_BASE / "rq4_results_nli"
RQ4_RESULTS_DIR = DATA_BASE / "rq4_results"   # drhall / metaqa statistics live here
OUTPUT_DIR = DATA_BASE / "rq4_summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (display_name, stats_suffix for qaqa/qaasker json, golden_dir_suffix for the HONEST golden dir)
MODELS = [
    ("DeepSeek-V3",      "deepseek_v3",         "deepseek-v3"),
    ("DeepSeek-V4-Flash","deepseek_v4_flash",   "deepseek-v4-flash"),
    ("GLM-5-Turbo",      "GLM_5_Turbo",         "GLM-5-Turbo"),
    ("GPT-5.5",          "gpt_5.5",             "gpt-5.5"),
    ("Qwen2.5-7B",       "Qwen2.5_7B_Instruct", "Qwen2.5-7B-Instruct"),
]
MODEL_ORDER = [m[0] for m in MODELS]

# Operators per method: (code, display_en, display_zh)
QAQA_OPS = [
    ("EQ",  "Equiv. Question", "Equiv. Question"),
    ("EC",  "Equiv. Context",  "Equiv. Context"),
    ("EQC", "Equiv. Context+", "Equiv. Context+"),
]
QAASKER_OPS = [
    ("MR1", "Special->Special", "Special->Special"),
    ("MR2", "Special->General", "Special->General"),
    ("MR3", "General->Special", "General->Special"),
]
HONEST_OPS = [
    ("body_permutation", "Body Permutation", "Body Permutation"),
    ("entity_rename",    "Entity Rename",    "Entity Rename"),
    ("body_augmentation", "Body Augmentation", "Body Augmentation"),
]
DRHALL_OPS = [
    ("QMR1", "CoT (QMR1)",        "CoT (QMR1)"),
    ("QMR2", "Multilingual (QMR2)", "Multilingual (QMR2)"),
    ("AMR1", "Negation (AMR1)",   "Negation (AMR1)"),
]
METAQA_OPS = [
    ("Synonym", "Synonym Sub.",   "Synonym Sub."),
    ("Antonym", "Antonym Sub.",   "Antonym Sub."),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_qaqa() -> Dict[str, dict]:
    """Read QAQA's by_mutation_type (reuse statistics.json, skip the buggy _nli_ files)."""
    res = {}
    for disp, suf, _ in MODELS:
        f = RQ4_NLI_DIR / f"qaqa_{suf}_statistics.json"
        if not f.exists():
            print(f"  [QAQA] missing: {f.name}")
            res[disp] = None
            continue
        d = json.load(open(f))
        res[disp] = d.get("by_mutation_type", {})
    return res


def load_qaasker() -> Dict[str, dict]:
    """Read QAASKER's by_mr_type."""
    res = {}
    for disp, suf, _ in MODELS:
        f = RQ4_NLI_DIR / f"qaasker_{suf}_statistics.json"
        if not f.exists():
            print(f"  [QAASKER] missing: {f.name}")
            res[disp] = None
            continue
        d = json.load(open(f))
        res[disp] = d.get("by_mr_type", {})
    return res


def load_drhall() -> Dict[str, dict]:
    """Read DrHall's by_mutation_type (violation_rate corrected by the strong judge)."""
    res = {}
    for disp, suf, _ in MODELS:
        f = RQ4_RESULTS_DIR / f"drhall_{suf}_statistics.json"
        if not f.exists():
            print(f"  [DrHall] missing: {f.name}")
            res[disp] = None
            continue
        d = json.load(open(f))
        res[disp] = d.get("by_mutation_type", {})
    return res


def load_metaqa() -> Dict[str, dict]:
    """Read MetaQA's by_mutation_type (Synonym / Antonym)."""
    res = {}
    for disp, suf, _ in MODELS:
        f = RQ4_RESULTS_DIR / f"metaqa_{suf}_statistics.json"
        if not f.exists():
            print(f"  [MetaQA] missing: {f.name}")
            res[disp] = None
            continue
        d = json.load(open(f))
        res[disp] = d.get("by_mutation_type", {})
    return res


def load_honest_golden() -> Dict[str, dict]:
    """Read each HONEST operator's (total, violations) from the golden NLI results dir."""
    res = {}
    for disp, _, gsuf in MODELS:
        ddir = DATA_BASE / f"consistency_results_nli_golden_{gsuf}"
        if not ddir.exists():
            print(f"  [HONEST] missing golden dir: {ddir.name}")
            res[disp] = None
            continue
        by_op = defaultdict(lambda: {"total": 0, "violations": 0})
        n_files = 0
        for rule_dir in ddir.glob("kg_rule_*"):
            for code, _, _ in HONEST_OPS:
                cf = rule_dir / f"{code}_llm_answers.csv"
                if cf.exists():
                    try:
                        df = pd.read_csv(cf)
                        if "answers_consistent" not in df.columns:
                            continue
                        by_op[code]["total"] += len(df)
                        by_op[code]["violations"] += int((~df["answers_consistent"]).sum())
                        n_files += 1
                    except Exception as e:
                        print(f"  [HONEST] read error {cf}: {e}")
        if n_files == 0:
            print(f"  [HONEST] no processed files in {ddir.name} (NLI run may be incomplete)")
        res[disp] = dict(by_op)
    return res


# ---------------------------------------------------------------------------
# Value extraction: returns (rate, n); returns (None, 0) when data is missing
# ---------------------------------------------------------------------------
def get_qaqa(bymt: Optional[dict], code: str) -> Tuple[Optional[float], int]:
    if not bymt:
        return None, 0
    s = bymt.get(code)
    if not s or s.get("total", 0) == 0:
        return None, 0
    return s.get("error_rate", 0.0), s.get("total", 0)


def get_qaasker(bymr: Optional[dict], code: str) -> Tuple[Optional[float], int]:
    if not bymr:
        return None, 0
    s = bymr.get(code)
    if not s or s.get("total", 0) == 0:
        return None, 0
    return s.get("violation_rate", 0.0), s.get("total", 0)


def get_drhall(bymt: Optional[dict], code: str) -> Tuple[Optional[float], int]:
    if not bymt:
        return None, 0
    s = bymt.get(code)
    if not s or s.get("total", 0) == 0:
        return None, 0
    return s.get("violation_rate", 0.0), s.get("total", 0)


def get_metaqa(bymt: Optional[dict], code: str) -> Tuple[Optional[float], int]:
    if not bymt:
        return None, 0
    s = bymt.get(code)
    if not s or s.get("total", 0) == 0:
        return None, 0
    return s.get("violation_rate", 0.0), s.get("total", 0)


def get_honest(by_op: Optional[dict], code: str) -> Tuple[Optional[float], int]:
    if not by_op:
        return None, 0
    s = by_op.get(code)
    if not s or s.get("total", 0) == 0:
        return None, 0
    return s["violations"] / s["total"], s["total"]


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------
def build_table(qaqa, qaasker, honest, drhall, metaqa) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rates_df wide table in %, counts_df wide table of sample sizes)."""
    rate_rows, count_rows = [], []
    sources = [
        ("HONEST",  HONEST_OPS,  honest,  get_honest),
        ("DRHALL",  DRHALL_OPS,  drhall,  get_drhall),
        ("METAQA",  METAQA_OPS,  metaqa,  get_metaqa),
        ("QAQA",    QAQA_OPS,    qaqa,    get_qaqa),
        ("QAASKER", QAASKER_OPS, qaasker, get_qaasker),
    ]
    for method, ops, data, getter in sources:
        for code, en, zh in ops:
            rate_row = {"Method": method, "Operator (en)": en, "Operator (zh)": zh}
            count_row = {"Method": method, "Operator (en)": en, "Operator (zh)": zh}
            rates = []
            for disp in MODEL_ORDER:
                rate, n = getter(data.get(disp), code)
                rate_row[disp] = rate
                count_row[disp] = n
                if rate is not None:
                    rates.append(rate)
            rate_row["Avg."] = (sum(rates) / len(rates)) if rates else None
            count_row["Avg."] = ""  # sample sizes are not averaged; left blank
            rate_rows.append(rate_row)
            count_rows.append(count_row)
    cols = ["Method", "Operator (en)", "Operator (zh)"] + MODEL_ORDER + ["Avg."]
    return pd.DataFrame(rate_rows, columns=cols), pd.DataFrame(count_rows, columns=cols)


def to_latex(rates: pd.DataFrame, counts: pd.DataFrame) -> str:
    """Generate a LaTeX table isomorphic to the paper's table:honest-baselines (English labels, per CLAUDE.md)."""
    def fmt(v):
        return f"{v*100:.2f}" if v is not None and pd.notna(v) else "--"

    # Operator count per method (MetaQA=2, others=3), used for \multirow
    op_counts = rates["Method"].value_counts().to_dict()

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inconsistency rate (\%) of each method and metamorphic "
                 r"operator across models --- all methods evaluated on the "
                 r"\emph{same} golden subset (fair comparison).}")
    lines.append(r"\label{table:honest-baselines-fair}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    hdr = (r"\textbf{Method} & \textbf{Operator} & " +
           " & ".join(rf"\textbf{{{m}}}" for m in MODEL_ORDER) +
           r" & \textbf{Avg.} \\")
    lines.append(hdr)
    lines.append(r"\midrule")
    cur_method = None
    for _, r in rates.iterrows():
        method = r["Method"]
        op = r["Operator (en)"]
        # Insert a separator between method groups (not before the first group)
        if cur_method is not None and method != cur_method:
            lines.append(r"\midrule")
        n_ops = op_counts.get(method, 1)
        lead = f"\\multirow{{{n_ops}}}{{*}}{{{method}}}" if method != cur_method else ""
        cur_method = method
        cells = [fmt(r[m]) for m in MODEL_ORDER] + [fmt(r["Avg."])]
        lines.append(f"{lead} & {op} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("RQ4 Fair Comparison Table (QAQA / QAASKER / HONEST on SAME golden)")
    print("=" * 70)

    qaqa = load_qaqa()
    qaasker = load_qaasker()
    honest = load_honest_golden()
    drhall = load_drhall()
    metaqa = load_metaqa()

    rates, counts = build_table(qaqa, qaasker, honest, drhall, metaqa)

    # Save
    rates_path = OUTPUT_DIR / "fair_comparison_rates.csv"
    counts_path = OUTPUT_DIR / "fair_comparison_counts.csv"
    tex_path = OUTPUT_DIR / "fair_comparison_table.tex"
    rates.to_csv(rates_path, index=False)
    counts.to_csv(counts_path, index=False)
    tex = to_latex(rates, counts)
    with open(tex_path, "w") as f:
        f.write(tex)

    # Print the percentage table (Chinese operators, for easy pasting into a Chinese paper)
    print("\n===== Inconsistency rate (%) — all three on the same golden =====")
    print(rates.to_string(index=False, float_format=lambda v: f"{v*100:.2f}"
          if pd.notna(v) else "--"))

    print("\n===== Per-cell sample size n — for checking instance alignment =====")
    print(counts.to_string(index=False))

    print(f"\nSaved:\n  {rates_path}\n  {counts_path}\n  {tex_path}")


if __name__ == "__main__":
    main()
