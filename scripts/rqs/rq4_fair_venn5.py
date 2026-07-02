#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 Fair 5-set Venn Diagram (Edwards five ellipses)

For each model, draws an Edwards five-set Venn diagram (5 congruent ellipses,
2^5=32 regions) of the violation sets of the five methods HONEST / QAQA /
QAAskER / DrHall / MetaQA. Uses the preset Edwards template from the third-party
``venn`` package (tctianchi/pyvenn, PyPI: venn).

Protocol is fully consistent with rq4_fair_upset.py / rq4_fair_venn.py:
  - HONEST uses the golden NLI rerun directory
    (consistency_results_nli_golden_<model>)
  - QAQA/QAAskER reuse rq4_fair_venn.find_baseline_file (non-nli first, fully
    comparable with the 3-set Venn)
  - DrHall/MetaQA use the _nli_ detail CSVs (load_drhall_nli_violations /
    load_metaqa_nli_violations, aggregated from row_index to the golden instance
    level)

Note (instance-level vs operator-level protocol):
  The "violation set" in this diagram is **instance-level** -- a golden question
  flagged as a violation by any operator/metamorphic relation of a method counts
  toward that method's violation set. This is a different level from the
  **operator-level violation_rate** in the fair comparison table (violating rows
  per operator / total rows). Because DrHall's AMR1 (negation) operator already
  has a high per-operator rate, instance-level "any operator violates" amplifies
  it to the question level, so the DrHall ellipse looks large in the diagram --
  this reflects a method characteristic, not false-positive inflation.

Output (non-destructive): figures/rq4_fair/
  - rq4_venn5_<Model>.png / .pdf     per-model 5-set Edwards Venn
  - rq4_venn5_all_models.png / .pdf  combined plot across models
"""

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from venn import generate_petal_labels, venn5

sys.path.insert(0, str(Path(__file__).parent))
from rq4_final_analysis import MODEL_ORDER  # noqa: E402
from rq4_fair_upset import (  # noqa: E402
    collect_violation_sets,
    METHOD_ORDER,
    OUT_DIR,
)

# Display names of the 5 methods for the Edwards Venn (order = METHOD_ORDER;
# first entry HONEST = our HONEST method)
VENN_NAMES = METHOD_ORDER  # ['HONEST','QAQA','QAAsker','DrHall','MetaQA']

# Paper-friendly 5 colors (consistent with the existing 3-set Venn palette)
VENN_COLORS = [
    [92/255, 192/255, 98/255, 0.45],   # HONEST green
    [90/255, 155/255, 212/255, 0.45],  # QAQA blue
    [246/255, 236/255, 86/255, 0.55],  # QAAsker yellow
    [244/255, 154/255, 98/255, 0.45],  # DrHall orange
    [180/255, 128/255, 200/255, 0.45], # MetaQA purple
]


def render_venn5(sets: dict, out_png: Path):
    """Draw one 5-set Edwards Venn (no title; the paper supplies a caption). sets: {method: set[idx]}."""
    # generate_petal_labels needs a list-of-sets (passing a dict triggers a
    # set.union('str') bug)
    set_list = [sets[m] for m in METHOD_ORDER]
    labels = generate_petal_labels(set_list, fmt="{size}")

    venn5(
        labels,
        names=VENN_NAMES,
        colors=VENN_COLORS,
        figsize=(6.5, 6.5),
        fontsize=11,
    )
    fig = plt.gcf()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sizes = {m: len(sets[m]) for m in METHOD_ORDER}
    allfive = int(labels["11111"]) if labels["11111"] not in ("", "0") else 0
    print(f"    [venn5] sizes={sizes} | all-five overlap={allfive}")
    return labels


def main():
    print("=" * 80)
    print("RQ4 Fair 5-set Edwards Venn (HONEST/QAQA/QAAskER/DrHall/MetaQA)")
    print("=" * 80)

    import pandas as pd
    from rq4_final_analysis import DATA_BASE_PATH
    from rq4_fair_venn import HONEST_GOLDEN_DIR

    original_df = pd.read_csv(DATA_BASE_PATH / "golden_dataset" / "golden_dataset_full.csv")
    original_df["_idx"] = original_df.index
    print(f"Loaded {len(original_df)} golden base samples")

    all_labels = {}
    combined = {m: set() for m in METHOD_ORDER}

    for model_name in MODEL_ORDER:
        if model_name not in HONEST_GOLDEN_DIR:
            continue
        if not (DATA_BASE_PATH / HONEST_GOLDEN_DIR[model_name]).exists():
            print(f"\n[skip] {model_name}: HONEST golden dir missing")
            continue

        print(f"\n{'=' * 70}\nProcessing {model_name}\n{'=' * 70}")
        try:
            sets, _ = collect_violation_sets(model_name, original_df)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        for m in METHOD_ORDER:
            print(f"  {m:<9}: {len(sets[m]):>5}")

        out_png = OUT_DIR / f"rq4_venn5_{model_name.replace('-', '_')}.png"
        labels = render_venn5(sets, out_png)
        print(f"  saved: {out_png} (+.pdf)")

        all_labels[model_name] = labels
        for m in METHOD_ORDER:
            combined[m].update((model_name, i) for i in sets[m])

    if all_labels:
        with open(OUT_DIR / "rq4_venn5_labels.json", "w") as f:
            json.dump(all_labels, f, indent=2)
        print(f"\nVenn5 labels saved: {OUT_DIR / 'rq4_venn5_labels.json'}")

    if any(combined.values()):
        print(f"\n{'=' * 70}\nCombined (all models, (model,idx) keys)\n{'=' * 70}")
        for m in METHOD_ORDER:
            print(f"  {m:<9}: {len(combined[m]):>5}")
        all_png = OUT_DIR / "rq4_venn5_all_models.png"
        render_venn5(combined, all_png)
        print(f"saved: {all_png} (+.pdf)")

    print("\nDone. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
