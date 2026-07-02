#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 KONTEST-inspired Evaluation Script

This script evaluates KONTEST-style atomic Yes/No paraphrase pairs using the
strict startswith yes/no exact-match oracle from KONTEST.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.kontest import KONTEST

DEFAULT_INPUT_DIR = Path("data/examples/golden_dataset_kontest_answer_Qwen2.5_7B_Instruct")
DEFAULT_OUTPUT_DIR = Path("data/examples/rq4_results")
MUTATION_COLUMN = "kontest_atomic_mutations"


@dataclass
class EvaluationResult:
    file_path: str
    row_index: int
    mutation_index: int
    subject_id: str
    predicate: str
    object_id: str
    subject_label: str
    object_label: str
    q1: str
    q2: str
    q1_answer: str
    q2_answer: str
    q1_label: str
    q2_label: str
    both_valid: bool
    is_violation: bool
    source_fact: str
    source_part: str
    kg_rule: str
    honest_mutation_type: str
    metadata: Dict[str, Any]


class KONTESTEvaluator:
    def __init__(self, model_name: str = "Unknown"):
        self.model_name = model_name
        self.kontest = KONTEST()
        self.column_model_name = model_name.replace(".", "_").replace("-", "_").replace("/", "_")

    @staticmethod
    def parse_json_list(value: Any) -> List[Dict]:
        if value is None or pd.isna(value) or not str(value).strip():
            return []
        try:
            parsed = json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    def find_answer_column(self, df: pd.DataFrame) -> str:
        exact = f"{MUTATION_COLUMN}_{self.column_model_name}_answers"
        if exact in df.columns:
            return exact

        candidates = [col for col in df.columns if col.startswith(f"{MUTATION_COLUMN}_") and col.endswith("_answers")]
        if len(candidates) == 1:
            return candidates[0]

        for col in candidates:
            if self.column_model_name in col or self.model_name in col:
                return col

        return ""

    def evaluate_row(self, row: pd.Series, file_path: str, answer_column: str) -> List[EvaluationResult]:
        mutations = self.parse_json_list(row.get(MUTATION_COLUMN, ""))
        answers = self.parse_json_list(row.get(answer_column, ""))
        results: List[EvaluationResult] = []

        for idx, mutation in enumerate(mutations):
            answer_obj = answers[idx] if idx < len(answers) else {}
            q1_answer = str(answer_obj.get("q1_answer", "")) if isinstance(answer_obj, dict) else ""
            q2_answer = str(answer_obj.get("q2_answer", "")) if isinstance(answer_obj, dict) else ""
            eval_result = self.kontest.evaluate_pair(q1_answer, q2_answer)
            metadata = mutation.get("metadata", {}) if isinstance(mutation.get("metadata", {}), dict) else {}

            results.append(EvaluationResult(
                file_path=file_path,
                row_index=int(row.name) if row.name is not None else -1,
                mutation_index=idx,
                subject_id=mutation.get("subject_id", ""),
                predicate=mutation.get("predicate", ""),
                object_id=mutation.get("object_id", ""),
                subject_label=mutation.get("subject_label", ""),
                object_label=mutation.get("object_label", ""),
                q1=mutation.get("original_question", answer_obj.get("q1", "") if isinstance(answer_obj, dict) else ""),
                q2=mutation.get("mutated_question", answer_obj.get("q2", "") if isinstance(answer_obj, dict) else ""),
                q1_answer=q1_answer,
                q2_answer=q2_answer,
                q1_label=eval_result["q1_label"] or "",
                q2_label=eval_result["q2_label"] or "",
                both_valid=eval_result["both_valid"],
                is_violation=eval_result["is_violation"],
                source_fact=mutation.get("source_fact", ""),
                source_part=mutation.get("source_part", ""),
                kg_rule=str(row.get("kg_rule", metadata.get("kg_rule", ""))),
                honest_mutation_type=str(row.get("mutation_type", metadata.get("honest_mutation_type", ""))),
                metadata=metadata,
            ))

        return results

    def evaluate_file(self, file_path: str, verbose: bool = False) -> Tuple[List[EvaluationResult], Dict]:
        df = pd.read_csv(file_path)
        answer_column = self.find_answer_column(df)
        if not answer_column:
            if verbose:
                print(f"  Warning: Could not find KONTEST answer column in {file_path}")
            return [], self.calculate_statistics([])

        all_results: List[EvaluationResult] = []
        for _, row in df.iterrows():
            row_results = self.evaluate_row(row, file_path, answer_column)
            all_results.extend(row_results)
            if verbose:
                for result in row_results:
                    if result.is_violation:
                        print(f"    VIOLATION row={result.row_index} pair={result.mutation_index}: {result.q1_label} vs {result.q2_label}")

        return all_results, self.calculate_statistics(all_results)

    def calculate_statistics(self, results: List[EvaluationResult]) -> Dict:
        total_pairs = len(results)
        valid_pairs = sum(1 for result in results if result.both_valid)
        invalid_pairs = total_pairs - valid_pairs
        violations = sum(1 for result in results if result.both_valid and result.is_violation)
        consistent = valid_pairs - violations

        by_source_part = self._group_stats(results, "source_part")
        by_predicate = self._group_stats(results, "predicate")
        by_honest_mutation_type = self._group_stats(results, "honest_mutation_type")

        return {
            "model_name": self.model_name,
            "total_pairs": total_pairs,
            "valid_pairs": valid_pairs,
            "invalid_pairs": invalid_pairs,
            "invalid_rate": invalid_pairs / total_pairs if total_pairs else 0.0,
            "violations": violations,
            "consistent": consistent,
            "violation_rate": violations / valid_pairs if valid_pairs else 0.0,
            "consistency_rate": consistent / valid_pairs if valid_pairs else 0.0,
            "by_source_part": by_source_part,
            "by_predicate": by_predicate,
            "by_honest_mutation_type": by_honest_mutation_type,
            "by_mutation_type": {
                "atomic": {
                    "total": valid_pairs,
                    "violations": violations,
                    "error_rate": violations / valid_pairs if valid_pairs else 0.0,
                    "violation_rate": violations / valid_pairs if valid_pairs else 0.0,
                }
            },
        }

    def evaluate_directory(self, input_dir: Path, output_dir: Path, verbose: bool = False) -> Dict:
        csv_files = [
            path for path in input_dir.rglob("*.csv")
            if "summary" not in path.name and "evaluation" not in path.name
        ]
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return {}

        all_results: List[EvaluationResult] = []
        by_file = {}
        for file_path in tqdm(csv_files, desc="Evaluating KONTEST files"):
            if verbose:
                print(f"\nEvaluating {file_path.relative_to(input_dir)}...")
            results, stats = self.evaluate_file(str(file_path), verbose=verbose)
            rel_path = str(file_path.relative_to(input_dir))
            by_file[rel_path] = {
                "total_pairs": stats["total_pairs"],
                "valid_pairs": stats["valid_pairs"],
                "violations": stats["violations"],
                "violation_rate": stats["violation_rate"],
            }
            all_results.extend(results)

        combined_stats = self.calculate_statistics(all_results)
        combined_stats["by_file"] = by_file
        self.save_results(all_results, combined_stats, output_dir)
        return combined_stats

    def save_results(self, results: List[EvaluationResult], stats: Dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame([asdict(result) for result in results])
        results_file = output_dir / f"kontest_{self.model_name}_evaluation_results.csv"
        stats_file = output_dir / f"kontest_{self.model_name}_statistics.json"

        results_df.to_csv(results_file, index=False)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 70}")
        print(f"KONTEST-inspired Evaluation Results for {self.model_name}")
        print(f"{'=' * 70}")
        print(f"Total pairs: {stats['total_pairs']}")
        print(f"Valid pairs: {stats['valid_pairs']}")
        print(f"Invalid pairs: {stats['invalid_pairs']} ({stats['invalid_rate']:.2%})")
        print(f"Violations: {stats['violations']} ({stats['violation_rate']:.2%} of valid pairs)")
        print(f"Results saved to:")
        print(f"  - {results_file}")
        print(f"  - {stats_file}")
        print(f"{'=' * 70}\n")

    @staticmethod
    def _group_stats(results: List[EvaluationResult], attr: str) -> Dict[str, Dict]:
        grouped = defaultdict(lambda: {"total_pairs": 0, "valid_pairs": 0, "invalid_pairs": 0, "violations": 0})
        for result in results:
            key = str(getattr(result, attr) or "unknown")
            grouped[key]["total_pairs"] += 1
            if result.both_valid:
                grouped[key]["valid_pairs"] += 1
                if result.is_violation:
                    grouped[key]["violations"] += 1
            else:
                grouped[key]["invalid_pairs"] += 1

        output = {}
        for key, stats in grouped.items():
            valid_pairs = stats["valid_pairs"]
            output[key] = {
                **stats,
                "violation_rate": stats["violations"] / valid_pairs if valid_pairs else 0.0,
                "invalid_rate": stats["invalid_pairs"] / stats["total_pairs"] if stats["total_pairs"] else 0.0,
            }
        return output


def auto_detect_model_name(input_path: Path) -> str:
    prefix = "golden_dataset_kontest_answer_"
    if input_path.name.startswith(prefix):
        return input_path.name[len(prefix):]
    return "Qwen2.5_7B_Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate KONTEST-inspired atomic mutations for RQ4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    conda activate karma
    python scripts/rqs/rq4_kontest_evaluation.py \
        --input-dir data/examples/golden_dataset_kontest_answer_Qwen2.5_7B_Instruct \
        --model-name Qwen2.5_7B_Instruct
        """,
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.input_file:
        input_path = Path(args.input_file)
    elif args.input_dir:
        input_path = Path(args.input_dir)
    else:
        input_path = DEFAULT_INPUT_DIR

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)

    model_name = args.model_name or auto_detect_model_name(input_path)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("RQ4 KONTEST-inspired Evaluation")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")

    evaluator = KONTESTEvaluator(model_name=model_name)
    if input_path.is_file():
        results, stats = evaluator.evaluate_file(str(input_path), verbose=args.verbose)
        evaluator.save_results(results, stats, output_dir)
    else:
        evaluator.evaluate_directory(input_path, output_dir, verbose=args.verbose)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
