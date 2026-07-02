#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 Fair UpSet Plot (all five methods on the same golden set)

For each model, draws an UpSet plot of the violation-set overlap across the five
methods HONEST / QAQA / QAAskER / DrHall / MetaQA. UpSet is the standard
replacement for Venn when there are more than 3 sets: it uses a matrix + bar
chart to clearly show every intersection size, avoiding the unreadability of a
5-set Venn diagram.

Fairness protocol is identical to rq4_fair_venn.py:
  - HONEST uses the golden NLI rerun directory
    (consistency_results_nli_golden_<model>), sharing the same batch of golden
    base instances and the same NLI protocol as QAQA/QAAskER/DrHall/MetaQA.
  - DrHall prefers is_violation_nli (the raw is_violation is inflated by false
    positives).
  - MetaQA's NLI-version CSV did not produce is_violation_nli, so it falls back
    to is_violation.
  - All five violation sets live in the same idx space [0, 1451] (the row index
    of golden_dataset_full).

Output (non-destructive, parallel to the existing venn output): figures/rq4_fair/
  - rq4_upset_<Model>.png / .pdf     per-model 5-set UpSet plot
  - rq4_upset_all_models.png / .pdf  combined plot across models
  - rq4_upset_statistics.json        per-method violation counts / per-intersection
                                     counts / Jaccard matrix
"""

import sys
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_contents, UpSet

# Reuse the loading functions and constants from the existing module (importing
# does not trigger main())
sys.path.insert(0, str(Path(__file__).parent))
from rq4_final_analysis import (  # noqa: E402
    MODEL_ORDER,
    INTERNAL_MODEL_NAMES,
    DATA_BASE_PATH,
    NLI_RESULTS_DIR,
    load_ours_violations,
    load_qaqa_nli_violations,
    load_qaasker_nli_violations,
    load_drhall_nli_violations,
    load_metaqa_nli_violations,
)
from rq4_fair_venn import HONEST_GOLDEN_DIR, find_baseline_file  # noqa: E402

OUT_DIR = Path("$PROJECT_ROOT/figures/rq4_fair")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_DATA_FILE = DATA_BASE_PATH / "golden_dataset" / "golden_dataset_full.csv"

# Method order and display names for the paper
METHOD_ORDER = ["HONEST", "QAQA", "QAAsker", "DrHall", "MetaQA"]


# DrHall/MetaQA detail CSVs are in rq4_results/, while QAQA/QAAskER are in
# rq4_results_nli/, so lookups cover both directories.
RESULTS_DIRS = [NLI_RESULTS_DIR, DATA_BASE_PATH / "rq4_results"]


def find_nli_baseline_file(prefix: str, internal_name: str) -> Path:
    """Prefer the ``_nli_`` detail CSV (DrHall/MetaQA need the NLI protocol); search across both result dirs."""
    candidates = []
    for d in RESULTS_DIRS:
        candidates.append(d / f"{prefix}_{internal_name}_nli_evaluation_results.csv")
    for d in RESULTS_DIRS:
        candidates.append(d / f"{prefix}_{internal_name}_evaluation_results.csv")
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob all candidates, preferring _nli_
    alts = []
    for d in RESULTS_DIRS:
        alts.extend(sorted(d.glob(f"{prefix}_{internal_name}*_evaluation_results.csv")))
    nli_like = [f for f in alts if "_nli_" in f.name]
    return (nli_like or alts or [candidates[-1]])[0]


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def collect_violation_sets(model_name: str, original_df: pd.DataFrame):
    """Return (dict[method]->set[int], dict[method]->Path); None for missing entries."""
    internal = INTERNAL_MODEL_NAMES.get(model_name, model_name)
    honest_dir = DATA_BASE_PATH / HONEST_GOLDEN_DIR[model_name]
    # QAQA/QAAskER reuse the lookup protocol from rq4_fair_venn (non-nli first) so
    # the three HONEST/QAQA/QAAskER sets stay fully comparable with the existing
    # 3-set Venn; DrHall/MetaQA use the nli version.
    qaqa_file = find_baseline_file("qaqa", internal)
    qaasker_file = find_baseline_file("qaasker", internal)
    drhall_file = find_nli_baseline_file("drhall", internal)
    metaqa_file = find_nli_baseline_file("metaqa", internal)

    sources = {
        "HONEST": honest_dir,
        "QAQA": qaqa_file,
        "QAAsker": qaasker_file,
        "DrHall": drhall_file,
        "MetaQA": metaqa_file,
    }

    sets = {}
    sets["HONEST"], _, _ = load_ours_violations(honest_dir, original_df)
    sets["QAQA"], _ = load_qaqa_nli_violations(qaqa_file)
    sets["QAAsker"], _ = load_qaasker_nli_violations(qaasker_file)
    sets["DrHall"], _ = load_drhall_nli_violations(drhall_file)
    sets["MetaQA"], _ = load_metaqa_nli_violations(metaqa_file)
    return sets, sources


def render_upset(contents: dict, out_png: Path, title: str):
    """Draw with upsetplot and save png + pdf. contents: {method: set[idx]}."""
    data = from_contents(contents)

    upset = UpSet(
        data,
        orientation="horizontal",
        sort_by="degree",
        sort_categories_by="input",
        show_counts=True,
        show_percentages=False,
        min_degree=1,            # exclude "detected by none of the five" instances, focus on real overlaps
        facecolor="#5A9BD4",
    )

    fig = plt.figure(figsize=(7.0, 3.6))
    upset.plot(fig=fig)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Report the count of "violated by none of the five" instances filtered out
    # by min_degree=1 (for transparency)
    total_elements = len(set().union(*contents.values()))
    n_shown = int(data.shape[0]) if hasattr(data, "shape") else len(data)
    n_empty = total_elements - n_shown
    print(f"    [upset] {len(contents)} sets, {total_elements} unique idx, "
          f"{n_shown} in ≥1 set, {n_empty} in none (excluded by min_degree=1)")
    return n_shown, n_empty


def intersection_table(sets: dict) -> dict:
    """Compute the size of every non-empty intersection (2^5-1 of them); return {sorted-methods-tuple: count}."""
    methods = METHOD_ORDER
    # Which methods each idx belongs to
    membership = {}
    for m in methods:
        for idx in sets[m]:
            membership.setdefault(idx, set()).add(m)

    counts = {}
    for members in membership.values():
        key = tuple(sorted(members))
        counts[key] = counts.get(key, 0) + 1
    # Serialize: turn tuple keys into "HONEST+QAQA" strings
    return {"+".join(k): v for k, v in sorted(counts.items(), key=lambda x: -len(x[0]))}


def main():
    print("=" * 80)
    print("RQ4 Fair UpSet Plot (HONEST/QAQA/QAAskER/DrHall/MetaQA on golden)")
    print("=" * 80)

    original_df = pd.read_csv(ORIGINAL_DATA_FILE)
    original_df["_idx"] = original_df.index
    print(f"Loaded {len(original_df)} golden base samples from {ORIGINAL_DATA_FILE.name}")

    all_stats = {}
    combined = {m: set() for m in METHOD_ORDER}  # (model, idx) composite key

    for model_name in MODEL_ORDER:
        if model_name not in HONEST_GOLDEN_DIR:
            print(f"\n[skip] {model_name}: no golden HONEST dir mapping")
            continue
        honest_dir = DATA_BASE_PATH / HONEST_GOLDEN_DIR[model_name]
        if not honest_dir.exists():
            print(f"\n[skip] {model_name}: HONEST golden dir missing: {honest_dir.name}")
            continue

        print(f"\n{'=' * 70}\nProcessing {model_name}\n{'=' * 70}")
        try:
            sets, sources = collect_violation_sets(model_name, original_df)
        except Exception as e:
            print(f"  ERROR collecting violations: {e}")
            import traceback; traceback.print_exc()
            continue

        for m in METHOD_ORDER:
            print(f"  {m:<9}: {len(sets[m]):>5} violations  ({sources[m].name})")

        # Jaccard matrix
        jac = {}
        for a, b in combinations(METHOD_ORDER, 2):
            jac[f"{a}|{b}"] = round(jaccard(sets[a], sets[b]), 4)
        print("  Pairwise Jaccard:")
        for k, v in jac.items():
            print(f"    {k:<22}: {v}")

        out_png = OUT_DIR / f"rq4_upset_{model_name.replace('-', '_')}.png"
        render_upset(sets, out_png, f"{model_name} (golden, NLI)")
        print(f"  saved: {out_png} (+.pdf)")

        all_stats[model_name] = {
            "sizes": {m: len(sets[m]) for m in METHOD_ORDER},
            "jaccard": jac,
            "intersections": intersection_table(sets),
        }

        # Accumulate into the cross-model combined set (use (model, idx) to
        # distinguish samples from different models)
        for m in METHOD_ORDER:
            for idx in sets[m]:
                combined[m].add((model_name, idx))

    if all_stats:
        with open(OUT_DIR / "rq4_upset_statistics.json", "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nUpSet stats saved: {OUT_DIR / 'rq4_upset_statistics.json'}")

    # Combined UpSet plot across models
    if any(combined.values()):
        print(f"\n{'=' * 70}\nCombined (all models, (model,idx) keys)\n{'=' * 70}")
        for m in METHOD_ORDER:
            print(f"  {m:<9}: {len(combined[m]):>5}")
        all_png = OUT_DIR / "rq4_upset_all_models.png"
        render_upset(combined, all_png, "All Models (golden, NLI)")
        print(f"saved: {all_png} (+.pdf)")

    print("\nDone. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
