#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation negation configuration system.

Provides a more elegant relation negation scheme, replacing hardcoded special
handling.

IMPORTANT: For inverse relation pairs (e.g. upstream/downstream), negated
distractors should not be used, because negating one direction is logically
equivalent to affirming the other direction, which leads to multiple correct
answers.
"""

from typing import Dict, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NegationType(Enum):
    """Negation type enum."""
    DIRECT_NEGATION = "direct"  # Direct negation: is -> is not
    OPPOSITE_RELATION = "opposite"  # Use the opposite relation: upstream -> downstream
    SEMANTIC_NEGATION = "semantic"  # Semantic negation: connected -> not connected
    CUSTOM_FUNCTION = "custom"  # Custom function handling


@dataclass
class RelationNegationRule:
    """Relation negation rule."""
    property_id: str
    negation_type: NegationType
    target_relation: Optional[str] = None  # For OPPOSITE_RELATION type
    negation_phrase: Optional[str] = None  # For SEMANTIC_NEGATION type
    custom_function: Optional[Callable[[str, str, str], str]] = None  # For CUSTOM_FUNCTION type
    condition: Optional[Callable[[str], bool]] = None  # Optional condition function


@dataclass
class InverseRelationPair:
    """Inverse relation pair configuration.

    Used to identify pairs of relations that are logically inverse to each
    other. For these relations, negated distractors should not be generated,
    because:
    - If A relation1 B is true, then A NOT relation2 B is also true (since
      relation2 is the inverse of relation1).
    - This would cause multiple correct answers in multiple-choice questions.
    """
    relation1_id: str
    relation2_id: str
    relation1_label: str = ""
    relation2_label: str = ""
    description: str = ""


class InverseRelationRegistry:
    """Inverse relation registry.

    Maintains all known inverse relation pairs, used to avoid logical errors
    when generating distractors.
    """

    def __init__(self):
        self._pairs: Dict[str, str] = {}  # relation_id -> inverse_relation_id
        self._pair_info: Dict[str, InverseRelationPair] = {}  # relation_id -> full info
        self._initialize_known_pairs()

    def _initialize_known_pairs(self):
        """Initialize known inverse relation pairs."""
        known_pairs = [
            # Bridge/tunnel crossing relations
            InverseRelationPair(
                relation1_id="P2673",
                relation2_id="P2674",
                relation1_label="next crossing upstream",
                relation2_label="next crossing downstream",
                description="River crossing direction"
            ),
            # Rank relations
            InverseRelationPair(
                relation1_id="P3729",
                relation2_id="P3730",
                relation1_label="next lower rank",
                relation2_label="next higher rank",
                description="Rank hierarchy"
            ),
            # Parent-child relations
            InverseRelationPair(
                relation1_id="P22",
                relation2_id="P40",
                relation1_label="father",
                relation2_label="child",
                description="Parent-child relationship (father)"
            ),
            InverseRelationPair(
                relation1_id="P25",
                relation2_id="P40",
                relation1_label="mother",
                relation2_label="child",
                description="Parent-child relationship (mother)"
            ),
            # Administrative divisions
            InverseRelationPair(
                relation1_id="P131",
                relation2_id="P150",
                relation1_label="located in administrative territorial entity",
                relation2_label="contains administrative territorial entity",
                description="Administrative containment"
            ),
            # Predecessor/successor relations
            InverseRelationPair(
                relation1_id="P155",
                relation2_id="P156",
                relation1_label="follows",
                relation2_label="followed by",
                description="Sequence order"
            ),
            # Part-whole relations
            InverseRelationPair(
                relation1_id="P361",
                relation2_id="P527",
                relation1_label="part of",
                relation2_label="has part",
                description="Part-whole relationship"
            ),
        ]

        for pair in known_pairs:
            self.register_pair(pair)

    def register_pair(self, pair: InverseRelationPair):
        """Register an inverse relation pair."""
        self._pairs[pair.relation1_id] = pair.relation2_id
        self._pairs[pair.relation2_id] = pair.relation1_id
        self._pair_info[pair.relation1_id] = pair
        self._pair_info[pair.relation2_id] = pair

    def get_inverse(self, relation_id: str) -> Optional[str]:
        """Get the inverse relation ID of a relation."""
        return self._pairs.get(relation_id)

    def has_inverse(self, relation_id: str) -> bool:
        """Check whether a relation has a known inverse."""
        return relation_id in self._pairs

    def are_inverse_pair(self, relation1_id: str, relation2_id: str) -> bool:
        """Check whether two relations are inverses of each other."""
        return self._pairs.get(relation1_id) == relation2_id

    def get_pair_info(self, relation_id: str) -> Optional[InverseRelationPair]:
        """Get detailed information about an inverse relation pair."""
        return self._pair_info.get(relation_id)

    def is_negation_safe(self, conclusion_relation: str, premise_relations: Set[str]) -> bool:
        """
        Check whether using negation on the conclusion relation is safe.

        If the inverse of the conclusion relation appears among the premises,
        then negation is unsafe, because negating the inverse relation yields a
        logically correct statement.

        Args:
            conclusion_relation: The relation ID used in the conclusion
            premise_relations: Set of all relation IDs used in the premises

        Returns:
            True if negation is safe, False if negation may produce multiple
            correct answers
        """
        inverse = self.get_inverse(conclusion_relation)
        if inverse is None:
            # No known inverse relation; negation is safe
            return True

        if inverse in premise_relations:
            # The inverse relation appears in the premises; negation is unsafe
            logger.debug(
                f"Negation unsafe: conclusion relation {conclusion_relation} "
                f"has inverse {inverse} in premises"
            )
            return False

        return True


# Global inverse relation registry instance
inverse_relation_registry = InverseRelationRegistry()


class RelationNegationManager:
    """Relation negation manager."""

    def __init__(self):
        self.rules: Dict[str, RelationNegationRule] = {}
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default rules."""
        # Directional relation pairs (upstream/downstream)
        self.add_rule(RelationNegationRule(
            property_id="P2673",  # next crossing upstream
            negation_type=NegationType.OPPOSITE_RELATION,
            target_relation="not the next crossing downstream from"
        ))

        self.add_rule(RelationNegationRule(
            property_id="P2674",  # next crossing downstream
            negation_type=NegationType.OPPOSITE_RELATION,
            target_relation="not the next crossing upstream from"
        ))

        # Border/connection relations
        self.add_rule(RelationNegationRule(
            property_id="border",  # This is a special key, matched on the label
            negation_type=NegationType.SEMANTIC_NEGATION,
            negation_phrase="not connected to",
            condition=lambda label: label and 'border' in label.lower()  # Added None check
        ))

        # Other directional relation pairs
        self.add_rule(RelationNegationRule(
            property_id="P22",  # father
            negation_type=NegationType.SEMANTIC_NEGATION,
            negation_phrase="not the father of"
        ))

        self.add_rule(RelationNegationRule(
            property_id="P25",  # mother
            negation_type=NegationType.SEMANTIC_NEGATION,
            negation_phrase="not the mother of"
        ))

    def add_rule(self, rule: RelationNegationRule):
        """Add a negation rule."""
        self.rules[rule.property_id] = rule

    def get_negation(self, property_id: str, property_label: str,
                    subject_label: str, object_label: str,
                    format_function: Callable[[str, str, str, str], str]) -> Optional[str]:
        """
        Get the negation expression for a relation.

        Args:
            property_id: Property ID
            property_label: Property label
            subject_label: Subject label
            object_label: Object label
            format_function: Formatting function

        Returns:
            The negation expression string, or None if there is no special rule
        """
        # First, check for an exact property ID match
        if property_id in self.rules:
            rule = self.rules[property_id]
            return self._apply_rule(rule, property_id, property_label,
                                  subject_label, object_label, format_function)

        # Then, check condition-based rules
        for rule_key, rule in self.rules.items():
            # Added None check for property_label before calling condition
            if rule.condition and property_label and rule.condition(property_label):
                return self._apply_rule(rule, property_id, property_label,
                                      subject_label, object_label, format_function)

        return None

    def _apply_rule(self, rule: RelationNegationRule, property_id: str,
                   property_label: str, subject_label: str, object_label: str,
                   format_function: Callable[[str, str, str, str], str]) -> str:
        """Apply a negation rule."""
        if rule.negation_type == NegationType.OPPOSITE_RELATION:
            # Do not pass property_id, to avoid using the predefined pattern and
            # let format_function use the passed predicate
            return format_function(subject_label, rule.target_relation, object_label, None)

        elif rule.negation_type == NegationType.SEMANTIC_NEGATION:
            # Do not pass property_id, to avoid using the predefined pattern and
            # let format_function use the passed predicate
            return format_function(subject_label, rule.negation_phrase, object_label, None)

        elif rule.negation_type == NegationType.CUSTOM_FUNCTION:
            if rule.custom_function:
                return rule.custom_function(subject_label, property_label, object_label)

        elif rule.negation_type == NegationType.DIRECT_NEGATION:
            # Direct negation: prepend "not" to the property
            negated_label = f"is not {property_label.lower()}"
            return format_function(subject_label, negated_label, object_label, None)

        return format_function(subject_label, f"is not {property_label.lower()}", object_label, None)


# Global instance
relation_negation_manager = RelationNegationManager()
