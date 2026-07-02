#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3 Feature Extractor Module

This module provides a unified interface for extracting features from
knowledge graph rules and questions for inconsistency analysis.

Design Principles:
1. Single Responsibility: Each feature extractor handles one type of feature
2. Composability: Features can be combined and computed independently
3. Caching: Expensive computations (e.g., corpus-level statistics) are cached
4. Type Safety: Clear input/output types for each function

Usage:
    from rq3_feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    features_df = extractor.extract_all_features(df)
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from functools import lru_cache

# Import configuration
from rq3_config import (
    MODELS, MUTATION_FILES, QUESTION_TYPE_DISPLAY, YESNO_ALIASES,
    Config, get_all_data_files, FEATURE_DEFINITIONS
)


# =============================================================================
# Feature Extractor Classes
# =============================================================================

@dataclass
class RuleFeatures:
    """Data class for rule-level features."""
    body_atom_count: int = 0
    head_atom_count: int = 0
    total_atom_count: int = 0
    unique_variable_count: int = 0
    unique_predicate_count: int = 0
    max_variable_reuse: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body_atom_count": self.body_atom_count,
            "head_atom_count": self.head_atom_count,
            "total_atom_count": self.total_atom_count,
            "unique_variable_count": self.unique_variable_count,
            "unique_predicate_count": self.unique_predicate_count,
            "max_variable_reuse": self.max_variable_reuse,
        }


@dataclass
class EntityFeatures:
    """Data class for entity-level features."""
    entity_count: int = 0
    entity_ids: Set[str] = field(default_factory=set)
    avg_label_length: float = 0.0
    avg_word_count: float = 0.0
    has_special_chars: bool = False
    has_numbers: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "avg_entity_label_length": self.avg_label_length,
            "avg_entity_word_count": self.avg_word_count,
            "has_special_chars": int(self.has_special_chars),
            "has_numbers": int(self.has_numbers),
        }


class RuleFeatureExtractor:
    """
    Extract structural features from logical rules.

    Rule format: "?h P1002 ?b ?a P527 ?h => ?a P1002 ?b"
    """

    @staticmethod
    def parse_rule(rule_str: str) -> Tuple[str, str]:
        """Split rule into body and head."""
        if pd.isna(rule_str) or not isinstance(rule_str, str):
            return "", ""

        parts = rule_str.split("=>")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return rule_str.strip(), ""

    @staticmethod
    def extract_variables(rule_str: str) -> List[str]:
        """Extract all variable references (?X pattern)."""
        if pd.isna(rule_str) or not isinstance(rule_str, str):
            return []
        return re.findall(r"\?[\w\d]+", rule_str)

    @staticmethod
    def extract_predicates(rule_str: str) -> List[str]:
        """Extract all predicate IDs (P followed by digits)."""
        if pd.isna(rule_str) or not isinstance(rule_str, str):
            return []
        return re.findall(r"\bP\d+\b", rule_str)

    @staticmethod
    def count_atoms(text: str) -> int:
        """Count number of atoms (triples) in text."""
        if not text:
            return 0
        tokens = text.split()
        return len(tokens) // 3

    def extract(self, rule_str: str) -> RuleFeatures:
        """Extract all rule features."""
        if pd.isna(rule_str) or not isinstance(rule_str, str):
            return RuleFeatures()

        body, head = self.parse_rule(rule_str)

        # Atom counts
        body_atoms = self.count_atoms(body)
        head_atoms = self.count_atoms(head)

        # Variables
        variables = self.extract_variables(rule_str)
        unique_vars = set(variables)
        var_counts = Counter(variables)
        max_reuse = max(var_counts.values()) if var_counts else 0

        # Predicates
        predicates = self.extract_predicates(rule_str)
        unique_preds = set(predicates)

        return RuleFeatures(
            body_atom_count=body_atoms,
            head_atom_count=head_atoms,
            total_atom_count=body_atoms + head_atoms,
            unique_variable_count=len(unique_vars),
            unique_predicate_count=len(unique_preds),
            max_variable_reuse=max_reuse,
        )


class EntityFeatureExtractor:
    """
    Extract features related to entities in the knowledge graph.

    Handles extraction from:
    - instantiation_mapping (JSON format)
    - original_instance (rule with entity IDs)
    - entity_labels (JSON array format)
    """

    @staticmethod
    def extract_entity_ids_from_mapping(mapping_str: str) -> Set[str]:
        """Extract entity IDs from instantiation mapping JSON."""
        if pd.isna(mapping_str) or not isinstance(mapping_str, str):
            return set()

        try:
            mapping = json.loads(mapping_str)
            if isinstance(mapping, dict):
                return set(mapping.values())
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: regex extraction
        return set(re.findall(r"\bQ\d+\b", mapping_str))

    @staticmethod
    def extract_entity_ids_from_instance(instance_str: str) -> Set[str]:
        """Extract entity IDs from instance string."""
        if pd.isna(instance_str) or not isinstance(instance_str, str):
            return set()
        return set(re.findall(r"\bQ\d+\b", instance_str))

    @staticmethod
    def parse_entity_labels(labels_str: str) -> List[str]:
        """
        Parse entity labels from JSON array.

        Format: ["Q109704630: Label Name | additional info", ...]
        Returns list of label texts.
        """
        if pd.isna(labels_str) or not isinstance(labels_str, str):
            return []

        try:
            labels = json.loads(labels_str)
            result = []
            for label_str in labels:
                if ":" in label_str:
                    label = label_str.split(":", 1)[1].strip()
                    if "|" in label:
                        label = label.split("|")[0].strip()
                    result.append(label)
            return result
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def calculate_label_complexity(label: str) -> Dict[str, Any]:
        """Calculate complexity metrics for a single label."""
        if pd.isna(label) or not isinstance(label, str):
            return {
                "length": 0,
                "word_count": 0,
                "has_special_chars": False,
                "has_numbers": False,
            }

        return {
            "length": len(label),
            "word_count": len(label.split()),
            "has_special_chars": any(c in label for c in ["-", "/", "_", "&"]),
            "has_numbers": any(c.isdigit() for c in label),
        }

    def extract(
        self,
        instantiation_mapping: str = "",
        original_instance: str = "",
        entity_labels: str = "",
    ) -> EntityFeatures:
        """Extract all entity features."""

        # Collect all entity IDs
        ids_from_mapping = self.extract_entity_ids_from_mapping(instantiation_mapping)
        ids_from_instance = self.extract_entity_ids_from_instance(original_instance)
        all_entity_ids = ids_from_mapping | ids_from_instance

        # Parse labels and calculate complexity
        labels = self.parse_entity_labels(entity_labels)

        if labels:
            complexities = [self.calculate_label_complexity(l) for l in labels]
            avg_length = np.mean([c["length"] for c in complexities])
            avg_word_count = np.mean([c["word_count"] for c in complexities])
            has_special = any(c["has_special_chars"] for c in complexities)
            has_nums = any(c["has_numbers"] for c in complexities)
        else:
            avg_length = 0.0
            avg_word_count = 0.0
            has_special = False
            has_nums = False

        return EntityFeatures(
            entity_count=len(all_entity_ids),
            entity_ids=all_entity_ids,
            avg_label_length=avg_length,
            avg_word_count=avg_word_count,
            has_special_chars=has_special,
            has_numbers=has_nums,
        )


class CorpusStatistics:
    """
    Compute corpus-level statistics that require aggregation across samples.

    These statistics are computed once and cached for efficiency.
    """

    def __init__(self):
        self._predicate_frequency: Dict[str, int] = {}
        self._pattern_instance_count: Dict[str, int] = {}
        self._computed = False

    def compute(self, df: pd.DataFrame) -> None:
        """Compute corpus-level statistics from the full dataset."""

        # Predicate frequency
        rule_extractor = RuleFeatureExtractor()
        all_predicates = []

        for rule_str in df["original_rule"].dropna():
            predicates = rule_extractor.extract_predicates(rule_str)
            all_predicates.extend(predicates)

        self._predicate_frequency = dict(Counter(all_predicates))

        # Pattern instance count (normalized rule patterns)
        patterns = df["original_rule"].apply(self._normalize_pattern)
        self._pattern_instance_count = dict(patterns.value_counts())

        self._computed = True

    @staticmethod
    def _normalize_pattern(rule_str: str) -> str:
        """Normalize rule by replacing variables with generic placeholders."""
        if pd.isna(rule_str) or not isinstance(rule_str, str):
            return ""

        var_to_normalized = {}
        counter = 0

        def replace_var(match):
            nonlocal counter
            var = match.group()
            if var not in var_to_normalized:
                counter += 1
                var_to_normalized[var] = f"?V{counter}"
            return var_to_normalized[var]

        return re.sub(r"\?[\w\d]+", replace_var, rule_str)

    def get_predicate_frequency(self, predicate: str) -> int:
        """Get frequency of a predicate in the corpus."""
        return self._predicate_frequency.get(predicate, 0)

    def get_predicate_scarcity_score(self, predicate: str) -> float:
        """Get scarcity score (1 - percentile rank) for a predicate."""
        if not self._predicate_frequency:
            return 0.0

        freq = self.get_predicate_frequency(predicate)
        all_freqs = list(self._predicate_frequency.values())

        # Calculate percentile rank
        rank = sum(1 for f in all_freqs if f <= freq) / len(all_freqs)
        return 1 - rank

    def get_pattern_instance_count(self, rule_str: str) -> int:
        """Get instance count for a rule's normalized pattern."""
        pattern = self._normalize_pattern(rule_str)
        return self._pattern_instance_count.get(pattern, 0)

    def get_pattern_support_score(self, rule_str: str) -> float:
        """Get support score (percentile rank) for a pattern."""
        if not self._pattern_instance_count:
            return 0.0

        count = self.get_pattern_instance_count(rule_str)
        all_counts = list(self._pattern_instance_count.values())

        rank = sum(1 for c in all_counts if c <= count) / len(all_counts)
        return rank


class QuestionFeatureExtractor:
    """Extract features from questions."""

    @staticmethod
    def get_question_type(question_type: str) -> str:
        """Normalize question type."""
        if pd.isna(question_type):
            return "unknown"

        qtype = str(question_type).lower()

        if qtype in YESNO_ALIASES:
            return "yes_no"
        elif qtype in ["multiple_choice", "wh_question"]:
            return qtype
        else:
            return "unknown"

    @staticmethod
    def get_question_length(question: str) -> int:
        """Get length of question text."""
        if pd.isna(question) or not isinstance(question, str):
            return 0
        return len(question.split())


# =============================================================================
# Main Feature Extractor
# =============================================================================

class FeatureExtractor:
    """
    Main feature extraction orchestrator.

    This class coordinates all feature extractors and provides
    a unified interface for extracting all features.
    """

    def __init__(self):
        self.rule_extractor = RuleFeatureExtractor()
        self.entity_extractor = EntityFeatureExtractor()
        self.corpus_stats: Optional[CorpusStatistics] = None
        self.question_extractor = QuestionFeatureExtractor()

    def extract_sample_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract all features for a single sample."""

        features = {}

        # Rule features
        rule_str = row.get("original_rule", "")
        rule_features = self.rule_extractor.extract(rule_str)
        features.update(rule_features.to_dict())

        # Primary predicate (first predicate in body)
        predicates = self.rule_extractor.extract_predicates(rule_str)
        primary_predicate = predicates[0] if predicates else ""

        # Corpus-level features
        if self.corpus_stats:
            features["primary_predicate_frequency"] = (
                self.corpus_stats.get_predicate_frequency(primary_predicate)
            )
            features["predicate_scarcity_score"] = (
                self.corpus_stats.get_predicate_scarcity_score(primary_predicate)
            )
            features["pattern_instance_count"] = (
                self.corpus_stats.get_pattern_instance_count(rule_str)
            )
            features["pattern_support_score"] = (
                self.corpus_stats.get_pattern_support_score(rule_str)
            )
        else:
            features["primary_predicate_frequency"] = 0
            features["predicate_scarcity_score"] = 0.0
            features["pattern_instance_count"] = 0
            features["pattern_support_score"] = 0.0

        # Entity features
        entity_features = self.entity_extractor.extract(
            instantiation_mapping=row.get("instantiation_mapping", ""),
            original_instance=row.get("original_instance", ""),
            entity_labels=row.get("entity_labels", ""),
        )
        features.update(entity_features.to_dict())

        # Question features
        features["question_type"] = self.question_extractor.get_question_type(
            row.get("original_question_type", "")
        )
        features["question_length"] = self.question_extractor.get_question_length(
            row.get("original_question", "")
        )

        # Target variable
        features["inconsistent"] = int(not row.get("answers_consistent", True))

        # Preserve metadata columns
        if "model" in row.index:
            features["model"] = row["model"]
        if "mutation_type" in row.index:
            features["mutation_type"] = row["mutation_type"]

        return features

    def extract_all_features(
        self,
        df: pd.DataFrame,
        compute_corpus_stats: bool = True,
    ) -> pd.DataFrame:
        """
        Extract all features from the dataset.

        Args:
            df: DataFrame with rule data
            compute_corpus_stats: Whether to compute corpus-level statistics

        Returns:
            DataFrame with extracted features
        """
        print(f"Extracting features from {len(df)} samples...")

        # Compute corpus statistics if requested
        if compute_corpus_stats:
            print("  Computing corpus-level statistics...")
            self.corpus_stats = CorpusStatistics()
            self.corpus_stats.compute(df)

        # Extract features for each sample
        print("  Extracting sample-level features...")
        features_list = []

        for idx, row in df.iterrows():
            features = self.extract_sample_features(row)
            features_list.append(features)

            if (idx + 1) % 1000 == 0:
                print(f"    Processed {idx + 1}/{len(df)} samples")

        features_df = pd.DataFrame(features_list)

        print(f"  Extracted {len(features_df.columns)} features")

        return features_df


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_all_data() -> pd.DataFrame:
    """
    Load all consistency data from all models.

    Returns:
        Combined DataFrame with all samples
    """
    all_data = []
    files = get_all_data_files()

    print(f"Loading data from {len(files)} files...")

    for file_info in files:
        file_path = file_info["path"]
        model = file_info["model"]
        mutation_type = file_info["mutation_type"]

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                continue

            # Add metadata
            df["model"] = model
            df["mutation_type"] = mutation_type

            all_data.append(df)

        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    if not all_data:
        raise ValueError("No data found!")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total samples loaded: {len(combined_df)}")

    return combined_df


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Test feature extraction."""

    print("=" * 70)
    print("RQ3 Feature Extractor Test")
    print("=" * 70)

    # Load data
    df = load_all_data()

    # Initialize extractor
    extractor = FeatureExtractor()

    # Extract features
    features_df = extractor.extract_all_features(df)

    # Summary
    print("\n" + "=" * 70)
    print("Feature Summary")
    print("=" * 70)

    print(f"\nTotal samples: {len(features_df)}")
    print(f"Total features: {len(features_df.columns)}")

    print("\nFeature Statistics:")
    for col in features_df.columns:
        if features_df[col].dtype in [np.int64, np.float64]:
            print(f"  {col}:")
            print(f"    Mean: {features_df[col].mean():.2f}")
            print(f"    Std: {features_df[col].std():.2f}")
            print(f"    Min: {features_df[col].min()}")
            print(f"    Max: {features_df[col].max()}")

    # Save features
    output_path = Config.ensure_output_dir() / "features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
