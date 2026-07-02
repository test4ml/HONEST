#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distractor Validator for Multiple Choice Question Generation

Ensures that distractors (wrong options) are genuinely false by validating
against the knowledge graph using the Closed World Assumption (CWA).

Core principle: If a triple does not exist in the knowledge graph, it is considered false.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional, Set, Any, Protocol
from functools import lru_cache

logger = logging.getLogger(__name__)


# Type aliases
Triple = Tuple[str, str, str]  # (subject, predicate, object)


class KGWithContains(Protocol):
    """Protocol for knowledge graph with contains method"""
    def contains(self, subject: str, predicate: str, obj: str) -> bool:
        """Check if a triple exists in the knowledge graph"""
        ...


@dataclass
class ValidationResult:
    """Result of distractor validation"""
    is_valid: bool      # True if distractor is confirmed to be false (valid distractor)
    reason: str         # Explanation of validation result
    confidence: float   # Confidence level 0.0-1.0


class DistractorValidator:
    """
    Validates that distractor triples are genuinely false.

    Uses the Closed World Assumption (CWA):
    - If a triple exists in KG -> it's TRUE -> invalid distractor
    - If a triple does NOT exist in KG -> it's FALSE -> valid distractor
    """

    def __init__(self, kg: Optional[Any] = None, cache_size: int = 10000):
        """
        Initialize the validator.

        Args:
            kg: Knowledge graph instance with 'contains' method.
                If None, validation will be skipped (all distractors considered valid).
            cache_size: Size of LRU cache for contains queries
        """
        self.kg = kg
        self._has_contains = kg is not None and hasattr(kg, 'contains')
        self._cache_size = cache_size

        # Create cached contains method if KG supports it
        if self._has_contains:
            self._cached_contains = lru_cache(maxsize=cache_size)(self._kg_contains)

        if not self._has_contains:
            logger.warning(
                "DistractorValidator initialized without KG contains support. "
                "Validation will be skipped - distractors may be logically correct!"
            )

    def _kg_contains(self, subject: str, predicate: str, obj: str) -> bool:
        """Internal method for cached KG contains query"""
        try:
            return self.kg.contains(subject, predicate, obj)
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"KG contains query failed for ({subject}, {predicate}, {obj}): {e}")
            # On error, assume triple doesn't exist (conservative approach)
            return False

    def triple_exists_in_kg(self, triple: Triple) -> bool:
        """
        Check if a triple exists in the knowledge graph.

        Args:
            triple: (subject, predicate, object) tuple

        Returns:
            True if triple exists in KG, False otherwise
        """
        if not self._has_contains:
            return False  # Cannot verify, assume doesn't exist

        subject, predicate, obj = triple
        return self._cached_contains(subject, predicate, obj)

    def validate(self,
                 distractor_triple: Triple,
                 correct_triple: Triple,
                 body_triples: List[Triple],
                 entity_labels: Optional[dict] = None) -> ValidationResult:
        """
        Validate that a distractor triple is genuinely false.

        Args:
            distractor_triple: The candidate distractor (s, p, o)
            correct_triple: The correct answer triple (s, p, o)
            body_triples: List of premise triples (for checking duplication)
            entity_labels: Optional dict mapping entity IDs to labels (for self-reference check)

        Returns:
            ValidationResult with is_valid=True if distractor is confirmed false
        """
        # Check 1: Distractor must not equal correct answer
        if distractor_triple == correct_triple:
            return ValidationResult(
                is_valid=False,
                reason="Identical to correct answer",
                confidence=1.0
            )

        # Check 2: Distractor must not be a premise
        if distractor_triple in body_triples:
            return ValidationResult(
                is_valid=False,
                reason="Duplicates a premise",
                confidence=1.0
            )

        # Check 3: Self-referential check (subject and object have same label)
        if entity_labels:
            s, p, o = distractor_triple
            s_label = entity_labels.get(s, s)
            o_label = entity_labels.get(o, o)

            # Extract base labels (remove descriptions in parentheses for comparison)
            import re
            s_base = re.sub(r'\s*\([^)]*\)\s*', '', s_label).strip()
            o_base = re.sub(r'\s*\([^)]*\)\s*', '', o_label).strip()

            if s_base and o_base and s_base.lower() == o_base.lower():
                return ValidationResult(
                    is_valid=False,
                    reason=f"Self-referential: subject and object have same label '{s_base}'",
                    confidence=1.0
                )

        # Check 4: Distractor must NOT exist in knowledge graph (CWA)
        if self._has_contains:
            if self.triple_exists_in_kg(distractor_triple):
                return ValidationResult(
                    is_valid=False,
                    reason="Triple exists in KG (may be true)",
                    confidence=1.0
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    reason="Triple does not exist in KG (confirmed false by CWA)",
                    confidence=1.0
                )
        else:
            # Without KG verification, we cannot guarantee falseness
            return ValidationResult(
                is_valid=True,
                reason="Cannot verify (KG contains not available)",
                confidence=0.5
            )

    def validate_batch(self,
                       candidate_triples: List[Triple],
                       correct_triple: Triple,
                       body_triples: List[Triple]) -> List[Tuple[Triple, ValidationResult]]:
        """
        Validate multiple distractor candidates.

        Args:
            candidate_triples: List of candidate distractor triples
            correct_triple: The correct answer triple
            body_triples: List of premise triples

        Returns:
            List of (triple, ValidationResult) tuples
        """
        results = []
        for triple in candidate_triples:
            result = self.validate(triple, correct_triple, body_triples)
            results.append((triple, result))
        return results

    def filter_valid_distractors(self,
                                  candidate_triples: List[Triple],
                                  correct_triple: Triple,
                                  body_triples: List[Triple],
                                  entity_labels: Optional[dict] = None) -> List[Triple]:
        """
        Filter candidate triples to keep only valid distractors.

        Args:
            candidate_triples: List of candidate distractor triples
            correct_triple: The correct answer triple
            body_triples: List of premise triples
            entity_labels: Optional dict mapping entity IDs to labels (for self-reference check)

        Returns:
            List of validated distractor triples (confirmed to be false)
        """
        valid_triples = []
        seen_triples: Set[Triple] = set()

        for triple in candidate_triples:
            # Skip duplicates
            if triple in seen_triples:
                continue
            seen_triples.add(triple)

            result = self.validate(triple, correct_triple, body_triples, entity_labels)
            if result.is_valid:
                valid_triples.append(triple)
                logger.debug(f"Valid distractor: {triple} - {result.reason}")
            else:
                logger.debug(f"Rejected distractor: {triple} - {result.reason}")

        return valid_triples

    def clear_cache(self):
        """Clear the contains query cache"""
        if self._has_contains:
            self._cached_contains.cache_clear()


# Convenience function for creating validator with optional KG
def create_validator(kg: Optional[Any] = None) -> DistractorValidator:
    """
    Create a DistractorValidator instance.

    Args:
        kg: Knowledge graph with optional 'contains' method

    Returns:
        DistractorValidator instance
    """
    return DistractorValidator(kg)
