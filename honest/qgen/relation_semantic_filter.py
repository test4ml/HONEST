#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation Semantic Filter for MCQ Distractor Generation

Uses semantic similarity to avoid selecting semantically similar relations
when generating distractors, preventing multiple valid answers.
"""

from typing import List, Dict, Set, Optional
import logging
from ..qgen.semantic_validator import semantic_validator

logger = logging.getLogger(__name__)


class RelationSemanticFilter:
    """Filter relations based on semantic similarity to avoid misleading distractors"""

    def __init__(self,
                 similarity_threshold: float = 0.7,
                 transitive_relation_keywords: Set[str] = None):
        """
        Initialize relation semantic filter

        Args:
            similarity_threshold: Threshold above which relations are considered
                                 semantically similar (0.0-1.0)
            transitive_relation_keywords: Keywords that indicate transitive relations
        """
        self.similarity_threshold = similarity_threshold

        # Default transitive relation keywords
        self.transitive_relation_keywords = transitive_relation_keywords or {
            'part of', 'contains', 'has part', 'member of', 'subclass of',
            'appears in', 'listed in', 'included in', 'belongs to',
            'component of', 'constituent of', 'element of'
        }

        # Cache for relation similarity computations
        self.similarity_cache: Dict[str, Dict[str, float]] = {}

    def filter_similar_relations(self,
                                current_relation_label: str,
                                candidate_relation_labels: List[str]) -> List[str]:
        """
        Filter out relations that are semantically similar to the current relation

        Args:
            current_relation_label: The relation being replaced
            candidate_relation_labels: List of candidate relation labels

        Returns:
            Filtered list of semantically distinct relation labels
        """
        if not candidate_relation_labels:
            return []

        distinct_relations = []

        for candidate_label in candidate_relation_labels:
            # Skip if identical
            if candidate_label.lower() == current_relation_label.lower():
                continue

            # Check semantic similarity
            similarity = self._compute_relation_similarity(current_relation_label, candidate_label)

            if similarity < self.similarity_threshold:
                distinct_relations.append(candidate_label)
            else:
                logger.debug(
                    f"Filtered out similar relation: '{candidate_label}' "
                    f"(similarity={similarity:.3f} to '{current_relation_label}')"
                )

        return distinct_relations

    def _compute_relation_similarity(self, rel1: str, rel2: str) -> float:
        """Compute semantic similarity between two relation labels"""
        # Check cache first
        cache_key1 = f"{rel1}_{rel2}"
        cache_key2 = f"{rel2}_{rel1}"

        if cache_key1 in self.similarity_cache:
            return self.similarity_cache[cache_key1][rel2]
        if cache_key2 in self.similarity_cache:
            return self.similarity_cache[cache_key2][rel1]

        # Compute similarity using semantic validator
        similarity = semantic_validator.compute_similarity(rel1, rel2)

        # Update cache
        if cache_key1 not in self.similarity_cache:
            self.similarity_cache[cache_key1] = {}
        self.similarity_cache[cache_key1][rel2] = similarity

        return similarity

    def is_transitive_relation(self, relation_label: str) -> bool:
        """Check if a relation is likely to be transitive based on keywords"""
        relation_lower = relation_label.lower()
        return any(keyword in relation_lower for keyword in self.transitive_relation_keywords)

    def filter_transitive_relations(self,
                                  current_relation_label: str,
                                  candidate_relation_labels: List[str]) -> List[str]:
        """
        Filter out transitive relations when replacing another transitive relation
        to avoid multiple valid answers

        Args:
            current_relation_label: The relation being replaced
            candidate_relation_labels: List of candidate relation labels

        Returns:
            Filtered list of non-transitive relation labels
        """
        current_is_transitive = self.is_transitive_relation(current_relation_label)

        if not current_is_transitive:
            # If current relation is not transitive, any replacement is safe
            return candidate_relation_labels

        # If current relation is transitive, prefer non-transitive replacements
        non_transitive_candidates = [
            rel for rel in candidate_relation_labels
            if not self.is_transitive_relation(rel)
        ]

        # If no non-transitive candidates found, return all candidates
        # (but this might still cause issues - semantic similarity will catch this)
        return non_transitive_candidates if non_transitive_candidates else candidate_relation_labels

    def get_safe_relation_replacements(self,
                                      current_relation_label: str,
                                      candidate_relation_labels: List[str]) -> List[str]:
        """
        Get safe relation replacements that avoid semantic similarity and transitive conflicts

        Args:
            current_relation_label: The relation being replaced
            candidate_relation_labels: List of candidate relation labels

        Returns:
            List of safe relation replacement labels
        """
        if not candidate_relation_labels:
            return []

        # Step 1: Filter out transitive relations if current is transitive
        filtered_candidates = self.filter_transitive_relations(
            current_relation_label, candidate_relation_labels
        )

        # Step 2: Filter out semantically similar relations
        safe_candidates = self.filter_similar_relations(
            current_relation_label, filtered_candidates
        )

        logger.debug(
            f"Relation replacement safety check: "
            f"'{current_relation_label}' -> {len(safe_candidates)}/{len(candidate_relation_labels)} safe candidates"
        )

        return safe_candidates


# Global instance with default settings
relation_semantic_filter = RelationSemanticFilter(
    similarity_threshold=0.7,  # Conservative threshold for relation similarity
    transitive_relation_keywords={
        'part of', 'contains', 'has part', 'member of', 'subclass of',
        'appears in', 'listed in', 'included in', 'belongs to',
        'component of', 'constituent of', 'element of', 'subset of',
        'parent of', 'child of', 'ancestor of', 'descendant of'
    }
)