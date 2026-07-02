#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 MetaQA Evaluation Script

Evaluates MetaQA hallucination detection results by:
  1. Reading the hallucination score for each question
  2. Comparing with ground truth (is the original answer correct?)
  3. Calculating violation rates by MR type, question type, kg_rule, etc.

Output format is compatible with rq4_final_analysis.py.

Usage:
    conda activate karma
    python scripts/rqs/rq4_metaqa_evaluation.py \\
        --input-dir data/examples/golden_dataset_metaqa_answer_Qwen2.5_7B_Instruct

    # Specify model name explicitly
    python scripts/rqs/rq4_metaqa_evaluation.py \\
        --input-dir data/examples/golden_dataset_metaqa_answer_Qwen2.5_7B_Instruct \\
        --model-name Qwen2.5_7B_Instruct

    # Custom threshold
    python scripts/rqs/rq4_metaqa_evaluation.py \\
        --threshold 0.6
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.metaqa import MetaQA, MetaQAMutationType
from baselines.qaqa.wrapper import is_same_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DEFAULT_OUTPUT_DIR = Path("data/examples/rq4_results")


# ---------------------------------------------------------------------------
# Evaluation result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetaQAEvalResult:
    """Result of evaluating a single MetaQA mutation verification."""
    file_path: str
    row_index: int
    mr_type: str                    # "Synonym" or "Antonym"
    original_question: str
    base_response: str
    mutation_text: str
    verification_result: str        # "Yes" / "No" / "Not Sure"
    hallucination_contribution: float
    is_violation: bool              # whether this mutation indicates hallucination
    hallucination_score: float      # overall score for this row
    question_type: str
    kg_rule: str
    honest_mutation_type: str


# ---------------------------------------------------------------------------
# Violation detection
# ---------------------------------------------------------------------------

def is_synonym_violation(verification: str) -> bool:
    """
    Synonym mutation: if the original answer was correct, its synonym
    should be verified as factual (Yes). If verification says No → violation.
    """
    return verification == "No"


def is_antonym_violation(verification: str) -> bool:
    """
    Antonym mutation: if the original answer was correct, its antonym
    should NOT be factual (No). If verification says Yes → violation.
    """
    return verification == "Yes"


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def parse_json_list(value: Any) -> List[Dict]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def detect_question_type(row: pd.Series) -> str:
    qt = row.get("original_question_type", "")
    if pd.notna(qt) and qt:
        return str(qt)
    return "unknown"


def process_file(
    df: pd.DataFrame,
    model_name: str,
    file_path: str,
    threshold: float = 0.5,
) -> Tuple[List[MetaQAEvalResult], Dict]:
    """Process a single CSV file and evaluate all MetaQA mutations."""
    results: List[MetaQAEvalResult] = []
    stats = {
        "total": 0,
        "violations": 0,
        "by_mr": defaultdict(lambda: {"total": 0, "violations": 0}),
    }

    base_col = f"metaqa_base_response_{model_name}"
    syn_col = f"metaqa_synonym_mutations_{model_name}"
    ant_col = f"metaqa_antonym_mutations_{model_name}"
    score_col = f"metaqa_hallucination_score_{model_name}"

    for idx, row in df.iterrows():
        # Get base response
        base_response = ""
        if base_col in df.columns and pd.notna(row.get(base_col)):
            base_response = str(row[base_col]).strip()
        if not base_response or base_response.startswith("ERROR:"):
            continue

        question_type = detect_question_type(row)
        kg_rule = str(row.get("kg_rule", ""))
        honest_mutation_type = str(row.get("mutation_type", ""))
        original_question = str(row.get("original_question", ""))

        # Get hallucination score
        h_score = 0.0
        if score_col in df.columns and pd.notna(row.get(score_col)):
            try:
                h_score = float(row[score_col])
            except (ValueError, TypeError):
                h_score = 0.0

        # Process synonym mutations
        if syn_col in df.columns:
            syn_mutations = parse_json_list(row.get(syn_col, ""))
            for mut in syn_mutations:
                verification = mut.get("verification_result", "")
                contribution = mut.get("hallucination_contribution", 0.5)
                mutation_text = mut.get("mutation_text", "")

                if not verification:
                    continue

                is_viol = is_synonym_violation(verification)
                stats["total"] += 1
                stats["by_mr"]["Synonym"]["total"] += 1
                if is_viol:
                    stats["violations"] += 1
                    stats["by_mr"]["Synonym"]["violations"] += 1

                results.append(MetaQAEvalResult(
                    file_path=file_path,
                    row_index=int(idx) if idx is not None else -1,
                    mr_type="Synonym",
                    original_question=original_question,
                    base_response=base_response,
                    mutation_text=mutation_text,
                    verification_result=verification,
                    hallucination_contribution=contribution,
                    is_violation=is_viol,
                    hallucination_score=h_score,
                    question_type=question_type,
                    kg_rule=kg_rule,
                    honest_mutation_type=honest_mutation_type,
                ))

        # Process antonym mutations
        if ant_col in df.columns:
            ant_mutations = parse_json_list(row.get(ant_col, ""))
            for mut in ant_mutations:
                verification = mut.get("verification_result", "")
                contribution = mut.get("hallucination_contribution", 0.5)
                mutation_text = mut.get("mutation_text", "")

                if not verification:
                    continue

                is_viol = is_antonym_violation(verification)
                stats["total"] += 1
                stats["by_mr"]["Antonym"]["total"] += 1
                if is_viol:
                    stats["violations"] += 1
                    stats["by_mr"]["Antonym"]["violations"] += 1

                results.append(MetaQAEvalResult(
                    file_path=file_path,
                    row_index=int(idx) if idx is not None else -1,
                    mr_type="Antonym",
                    original_question=original_question,
                    base_response=base_response,
                    mutation_text=mutation_text,
                    verification_result=verification,
                    hallucination_contribution=contribution,
                    is_violation=is_viol,
                    hallucination_score=h_score,
                    question_type=question_type,
                    kg_rule=kg_rule,
                    honest_mutation_type=honest_mutation_type,
                ))

    return results, dict(stats)


# ---------------------------------------------------------------------------
# Statistics calculation
# ---------------------------------------------------------------------------

def calculate_statistics(results: List[MetaQAEvalResult]) -> Dict:
    """Calculate overall and per-group statistics."""
    total = len(results)
    violations = sum(1 for r in results if r.is_violation)
    violation_rate = violations / total if total > 0 else 0.0

    # By MR type
    by_mr: Dict[str, Dict] = {}
    mr_groups = defaultdict(lambda: {"total": 0, "violations": 0})
    for r in results:
        mr_groups[r.mr_type]["total"] += 1
        if r.is_violation:
            mr_groups[r.mr_type]["violations"] += 1
    for mr_name, mr_stats in mr_groups.items():
        t = mr_stats["total"]
        v = mr_stats["violations"]
        by_mr[mr_name] = {
            "total": t,
            "violations": v,
            "violation_rate": v / t if t > 0 else 0.0,
        }

    # By question type
    by_question_type: Dict[str, Dict] = {}
    qt_groups = defaultdict(lambda: {"total": 0, "violations": 0})
    for r in results:
        qt_groups[r.question_type]["total"] += 1
        if r.is_violation:
            qt_groups[r.question_type]["violations"] += 1
    for qt, qt_stats in qt_groups.items():
        t = qt_stats["total"]
        v = qt_stats["violations"]
        by_question_type[qt] = {
            "total": t,
            "violations": v,
            "violation_rate": v / t if t > 0 else 0.0,
        }

    # By honest_mutation_type
    by_honest_mutation: Dict[str, Dict] = {}
    kmt_groups = defaultdict(lambda: {"total": 0, "violations": 0})
    for r in results:
        kmt_groups[r.honest_mutation_type]["total"] += 1
        if r.is_violation:
            kmt_groups[r.honest_mutation_type]["violations"] += 1
    for kmt, kmt_stats in kmt_groups.items():
        t = kmt_stats["total"]
        v = kmt_stats["violations"]
        by_honest_mutation[kmt] = {
            "total": t,
            "violations": v,
            "violation_rate": v / t if t > 0 else 0.0,
        }

    # By kg_rule
    by_kg_rule: Dict[str, Dict] = {}
    kr_groups = defaultdict(lambda: {"total": 0, "violations": 0})
    for r in results:
        kr_groups[r.kg_rule]["total"] += 1
        if r.is_violation:
            kr_groups[r.kg_rule]["violations"] += 1
    for kr, kr_stats in kr_groups.items():
        t = kr_stats["total"]
        v = kr_stats["violations"]
        by_kg_rule[kr] = {
            "total": t,
            "violations": v,
            "violation_rate": v / t if t > 0 else 0.0,
        }

    return {
        "total": total,
        "violations": violations,
        "violation_rate": violation_rate,
        "by_mr": by_mr,
        "by_mutation_type": by_mr,  # alias for rq4_final_analysis.py compatibility
        "by_question_type": by_question_type,
        "by_honest_mutation_type": by_honest_mutation,
        "by_kg_rule": by_kg_rule,
    }


def convert_for_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Auto-detect model name
# ---------------------------------------------------------------------------

def auto_detect_model_name(input_path: Path) -> str:
    prefix = "golden_dataset_metaqa_answer_"
    if input_path.name.startswith(prefix):
        return input_path.name[len(prefix):]
    return "Qwen2.5_7B_Instruct"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MetaQA hallucination detection results for RQ4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    conda activate karma
    python scripts/rqs/rq4_metaqa_evaluation.py \\
        --input-dir data/examples/golden_dataset_metaqa_answer_Qwen2.5_7B_Instruct
        """,
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Hallucination score threshold (default: 0.5)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.input_file:
        input_path = Path(args.input_file)
    elif args.input_dir:
        input_path = Path(args.input_dir)
    else:
        # Auto-detect
        candidates = sorted(Path("data/examples").glob("golden_dataset_metaqa_answer_*"))
        if candidates:
            input_path = candidates[0]
            print(f"Auto-detected input: {input_path}")
        else:
            print("Error: No input specified and no MetaQA answer directory found")
            sys.exit(1)

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)

    model_name = args.model_name or auto_detect_model_name(input_path)
    # Normalize model name for column matching.
    # NOTE: the answering phase keeps dots when writing column names
    # (e.g. gpt_5.5 / Qwen2.5_7B_Instruct), so here we only normalize hyphens /
    # slashes and keep dots, to match the actual column names.
    column_model_name = model_name.replace("-", "_").replace("/", "_")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RQ4 MetaQA Evaluation")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Hallucination threshold: {args.threshold}")

    # Find CSV files
    if input_path.is_file():
        csv_files = [input_path]
    else:
        csv_files = [
            p for p in input_path.rglob("*.csv")
            if "summary" not in p.name and "evaluation" not in p.name
        ]

    if not csv_files:
        print("No CSV files found")
        sys.exit(1)

    all_results: List[MetaQAEvalResult] = []

    for csv_file in tqdm(csv_files, desc="Evaluating MetaQA files"):
        try:
            df = pd.read_csv(csv_file)
            rel_path = str(csv_file.relative_to(input_path)) if input_path.is_dir() else csv_file.name
            results, stats = process_file(df, column_model_name, rel_path, args.threshold)
            all_results.extend(results)

            if args.verbose and stats["total"] > 0:
                print(f"\n  {rel_path}: {stats['total']} mutations, "
                      f"{stats['violations']} violations "
                      f"({stats['violations']/stats['total']*100:.1f}%)")
        except Exception as exc:
            print(f"  Warning: Failed to process {csv_file}: {exc}")

    if not all_results:
        print("\nNo results to evaluate!")
        sys.exit(0)

    # Calculate statistics
    combined_stats = calculate_statistics(all_results)

    # Save detailed results
    results_df = pd.DataFrame([asdict(r) for r in all_results])
    results_file = output_dir / f"metaqa_{column_model_name}_evaluation_results.csv"
    results_df.to_csv(results_file, index=False)

    # NLI-style results for rq4_final_analysis.py compatibility
    nli_results_file = output_dir / f"metaqa_{column_model_name}_nli_evaluation_results.csv"
    results_df.to_csv(nli_results_file, index=False)

    # Build statistics JSON compatible with rq4_final_analysis.py
    overall_stats = {
        "model_name": column_model_name,
        "total_count": combined_stats["total"],
        "original_violations": combined_stats["violations"],
        "original_rate": combined_stats["violation_rate"],
        "nli_violations": combined_stats["violations"],
        "nli_rate": combined_stats["violation_rate"],
        "error_detection_rate": combined_stats["violation_rate"],
        "by_mutation_type": combined_stats["by_mr"],
        "by_question_type": combined_stats["by_question_type"],
        "by_honest_mutation_type": combined_stats["by_honest_mutation_type"],
        "by_kg_rule": combined_stats["by_kg_rule"],
        "threshold": args.threshold,
    }

    stats_file = output_dir / f"metaqa_{column_model_name}_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(convert_for_json(overall_stats), f, indent=2, ensure_ascii=False)

    # NLI-style statistics for rq4_final_analysis.py compatibility
    nli_stats_file = output_dir / f"metaqa_{column_model_name}_nli_statistics.json"
    with open(nli_stats_file, "w", encoding="utf-8") as f:
        json.dump(convert_for_json(overall_stats), f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"MetaQA Evaluation Results for {model_name}")
    print(f"{'=' * 70}")
    print(f"Total mutations evaluated: {combined_stats['total']}")
    print(f"Violations (hallucination indicators): {combined_stats['violations']}")
    print(f"Violation rate: {combined_stats['violation_rate']:.4f}")
    print(f"\nBy MR type:")
    for mr_name, mr_stats in combined_stats["by_mr"].items():
        print(f"  {mr_name}: {mr_stats['violations']}/{mr_stats['total']} "
              f"({mr_stats['violation_rate']:.4f})")
    print(f"\nBy question type:")
    for qt, qt_stats in combined_stats["by_question_type"].items():
        print(f"  {qt}: {qt_stats['violations']}/{qt_stats['total']} "
              f"({qt_stats['violation_rate']:.4f})")
    print(f"\nBy honest mutation type:")
    for kmt, kmt_stats in combined_stats["by_honest_mutation_type"].items():
        print(f"  {kmt}: {kmt_stats['violations']}/{kmt_stats['total']} "
              f"({kmt_stats['violation_rate']:.4f})")
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {stats_file}")
    print(f"  - {nli_stats_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
