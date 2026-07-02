#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 Fair Venn Diagrams (all three methods on the same golden set)

Reuses the Venn plotting / overlap statistics functions from rq4_final_analysis,
but changes HONEST's violation set from the FULL consistency_results_nli_<model>
to the GOLDEN consistency_results_nli_golden_<model>.

This way the violation sets of all three (HONEST / QAQA / QAASKER) come from the
same batch of golden base instances under the same NLI protocol, so the overlap
(Venn) diagram is fair.

QAQA / QAASKER violation files were already evaluated on golden and are kept
unchanged; original_df (the source of base-instance indices) is still
golden_dataset/golden_dataset_full.csv, unchanged.

Output (non-destructive, new directory): figures/rq4_fair/
  - rq4_overlap_<Model>.png / .pdf   per-model 3-set Venn diagram
  - rq4_overlap_all_models.png/.pdf  combined plot across models
  - rq4_overlap_statistics.json      overlap counts / Jaccard
"""

import sys
import json
from pathlib import Path

import pandas as pd

# Reuse the plotting and statistics functions from the existing module (importing
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
    create_venn_diagram,
    print_overlap_statistics,
)

# Fair HONEST directory (the golden NLI results generated alongside this repo)
HONEST_GOLDEN_DIR = {
    "DeepSeek-V3":       "consistency_results_nli_golden_deepseek-v3",
    "DeepSeek-V4-Flash": "consistency_results_nli_golden_deepseek-v4-flash",
    "GLM-5-Turbo":       "consistency_results_nli_golden_GLM-5-Turbo",
    "GPT-5.5":           "consistency_results_nli_golden_gpt-5.5",
    "Qwen2.5-7B":        "consistency_results_nli_golden_Qwen2.5-7B-Instruct",
}

OUT_DIR = Path("$PROJECT_ROOT/figures/rq4_fair")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_DATA_FILE = DATA_BASE_PATH / "golden_dataset" / "golden_dataset_full.csv"


def find_baseline_file(prefix: str, internal_name: str) -> Path:
    """Same lookup logic as rq4_final_analysis.main (prefer the non-_nli_ file for QAQA)."""
    direct = NLI_RESULTS_DIR / f"{prefix}_{internal_name}_evaluation_results.csv"
    if direct.exists():
        return direct
    alts = sorted(NLI_RESULTS_DIR.glob(f"{prefix}_{internal_name}*_evaluation_results.csv"))
    non_nli = [f for f in alts if "_nli_" not in f.name]
    return (non_nli or alts)[0] if (non_nli or alts) else direct


def main():
    print("=" * 80)
    print("RQ4 Fair Venn Diagrams (HONEST on golden — same as QAQA/QAASKER)")
    print("=" * 80)

    original_df = pd.read_csv(ORIGINAL_DATA_FILE)
    original_df["_idx"] = original_df.index
    print(f"Loaded {len(original_df)} golden base samples from {ORIGINAL_DATA_FILE.name}")

    all_overlap_stats = {}
    all_honest, all_qaqa, all_qaasker = {}, {}, {}

    for model_name in MODEL_ORDER:
        if model_name not in HONEST_GOLDEN_DIR:
            print(f"\n[skip] {model_name}: no golden HONEST dir")
            continue
        internal = INTERNAL_MODEL_NAMES.get(model_name, model_name)
        honest_dir = DATA_BASE_PATH / HONEST_GOLDEN_DIR[model_name]
        qaqa_file = find_baseline_file("qaqa", internal)
        qaasker_file = find_baseline_file("qaasker", internal)

        if not honest_dir.exists():
            print(f"\n[skip] {model_name}: HONEST golden dir missing: {honest_dir.name}")
            continue

        print(f"\n{'='*70}\nProcessing {model_name}\n{'='*70}")
        print(f"  HONEST(golden): {honest_dir.name}")
        print(f"  QAQA:  {qaqa_file.name}")
        print(f"  QAASKER: {qaasker_file.name}")

        try:
            honest_set, _, _ = load_ours_violations(honest_dir, original_df)
            qaqa_set, _ = load_qaqa_nli_violations(qaqa_file)
            qaasker_set, _ = load_qaasker_nli_violations(qaasker_file)
        except Exception as e:
            print(f"  ERROR loading violations: {e}")
            import traceback; traceback.print_exc()
            continue

        print(f"  HONEST violations: {len(honest_set)} (golden)")
        print(f"  QAQA violations:   {len(qaqa_set)}")
        print(f"  QAASKER violations:{len(qaasker_set)}")
        print_overlap_statistics(honest_set, qaqa_set, qaasker_set, model_name)

        out_png = OUT_DIR / f"rq4_overlap_{model_name.replace('-', '_')}.png"
        create_venn_diagram(honest_set, qaqa_set, qaasker_set, out_png, model_name)
        print(f"  saved: {out_png} (+.pdf)")

        all_honest[model_name] = honest_set
        all_qaqa[model_name] = qaqa_set
        all_qaasker[model_name] = qaasker_set
        all_overlap_stats[model_name] = {
            "honest": len(honest_set), "qaqa": len(qaqa_set), "qaasker": len(qaasker_set),
            "honest_qaqa": len(honest_set & qaqa_set),
            "honest_qaasker": len(honest_set & qaasker_set),
            "qaqa_qaasker": len(qaqa_set & qaasker_set),
            "all_three": len(honest_set & qaqa_set & qaasker_set),
        }

    if all_overlap_stats:
        with open(OUT_DIR / "rq4_overlap_statistics.json", "w") as f:
            json.dump(all_overlap_stats, f, indent=2)
        print(f"\nOverlap stats saved: {OUT_DIR / 'rq4_overlap_statistics.json'}")

    # Combined Venn diagram across models (use (model, idx) to distinguish
    # samples from different models)
    if all_honest:
        mk = {(m, i) for m, s in all_honest.items() for i in s}
        mq = {(m, i) for m, s in all_qaqa.items() for i in s}
        ma = {(m, i) for m, s in all_qaasker.items() for i in s}
        print(f"\nCombined: HONEST={len(mk)} QAQA={len(mq)} QAASKER={len(ma)}")
        all_png = OUT_DIR / "rq4_overlap_all_models.png"
        create_venn_diagram(mk, mq, ma, all_png, "All Models (golden)")
        print(f"saved: {all_png} (+.pdf)")

    print("\nDone. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
