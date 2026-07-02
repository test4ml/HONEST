#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base mutation class definitions
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from dataclasses import dataclass

from ..kg_interfaces import IndexedKnowledgeGraph
from ..rule_parser import Rule, Fact, LogicExpression
from ..rule_parser import RuleParser


@dataclass
class FactInstance:
    """Inference rule instance"""
    original_rule_str: str  # Original rule string format, e.g.: ?b P2673 ?a => ?a P2674 ?b
    matched_instance_str: str  # Matched instance string
    instantiation_mapping: Dict[str, str]  # Mapping from variables to entities
    entity_labels: List[str]  # Entity label descriptions
    natural_language: str  # Natural language description

    def __post_init__(self):
        """Parse the rule into a Rule object after initialization"""
        self._original_rule: Rule = RuleParser.parse_rule(self.original_rule_str)
        self._matched_instance: Rule = RuleParser.parse_rule(self.matched_instance_str)

    @property
    def original_rule(self) -> Rule:
        """Get the Rule object of the original rule"""
        return self._original_rule

    @property
    def matched_instance(self) -> Rule:
        """Get the Rule object of the matched instance"""
        return self._matched_instance

    # ============================================================================
    # Instance-level properties (contain actual entity IDs, not variables)
    # ============================================================================

    @property
    def instance_body_atoms(self) -> List[Fact]:
        """Get body atoms from the matched instance (contains entity IDs, not variables)

        This returns the instantiated body facts where variables have been replaced
        with actual entity IDs from the knowledge graph.

        Example:
            original_rule: "?h P1080 ?b ?a P31 ?h => ?a P1080 ?b"
            matched_instance: "Q1187421 P1080 Q18043309 Q107273026 P31 Q1187421 => ..."
            instance_body_atoms: [Fact("Q1187421", "P1080", "Q18043309"),
                                  Fact("Q107273026", "P31", "Q1187421")]

        Note: This property is crucial for mutation operators like EntityRename
        that need to work with actual entity IDs.

        Returns:
            List[Fact]: List of instantiated body facts with entity IDs

        Raises:
            ValueError: If body contains OR/NOT logic that cannot be converted to simple list
        """
        if isinstance(self.matched_instance.body, list):
            return self.matched_instance.body
        elif isinstance(self.matched_instance.body, LogicExpression):
            # Try to convert to fact list (only works for AND-only expressions)
            return self.matched_instance.body.to_fact_list()
        else:
            raise ValueError(f"Unexpected body type: {type(self.matched_instance.body)}")

    @property
    def instance_head_atom(self) -> Fact:
        """Get head atom from the matched instance (contains entity IDs, not variables)

        This returns the instantiated head fact where variables have been replaced
        with actual entity IDs from the knowledge graph.

        Example:
            original_rule: "?h P1080 ?b ?a P31 ?h => ?a P1080 ?b"
            matched_instance: "... => Q107273026 P1080 Q18043309"
            instance_head_atom: Fact("Q107273026", "P1080", "Q18043309")

        Returns:
            Fact: Instantiated head fact with entity IDs
        """
        return self.matched_instance.head

    # ============================================================================
    # Rule-level properties (contain variables like ?a, ?b, ?h)
    # ============================================================================

    @property
    def rule_body_atoms(self) -> List[Fact]:
        """Get body atoms from the original rule (contains variables, not entity IDs)

        This returns the rule template body facts with variable placeholders.

        Example:
            original_rule: "?h P1080 ?b ?a P31 ?h => ?a P1080 ?b"
            rule_body_atoms: [Fact("?h", "P1080", "?b"), Fact("?a", "P31", "?h")]

        Returns:
            List[Fact]: List of rule template body facts with variables

        Raises:
            ValueError: If body contains OR/NOT logic that cannot be converted to simple list
        """
        if isinstance(self.original_rule.body, list):
            return self.original_rule.body
        elif isinstance(self.original_rule.body, LogicExpression):
            return self.original_rule.body.to_fact_list()
        else:
            raise ValueError(f"Unexpected body type: {type(self.original_rule.body)}")

    @property
    def rule_head_atom(self) -> Fact:
        """Get head atom from the original rule (contains variables, not entity IDs)

        This returns the rule template head fact with variable placeholders.

        Example:
            original_rule: "?h P1080 ?b ?a P31 ?h => ?a P1080 ?b"
            rule_head_atom: Fact("?a", "P1080", "?b")

        Returns:
            Fact: Rule template head fact with variables
        """
        return self.original_rule.head

    # ============================================================================
    # Other properties
    # ============================================================================

    @property
    def variables(self) -> List[str]:
        """Get all variables in the rule template

        Returns:
            List[str]: List of variable names (e.g., ['?a', '?b', '?h'])
        """
        return self.original_rule.variables


class MutationOperator(ABC):
    """Base class for mutation operators"""

    @abstractmethod
    def mutate(self, rule: FactInstance, kg: IndexedKnowledgeGraph) -> List[Tuple[str, str]]:
        """
        Apply a mutation operation

        Args:
            rule: Rule instance to be mutated
            kg: Knowledge graph interface

        Returns:
            List[(mutated_body, expected_head)]: List of mutated rule bodies and expected rule heads
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the mutation operator"""
        pass


class MutationEngine:
    """Mutation engine"""

    def __init__(self, operators: List[MutationOperator]) -> None:
        self.operators = operators

    def mutate_rule(self, rule: FactInstance, kg: IndexedKnowledgeGraph) -> Dict[str, List[Tuple[str, str]]]:
        """
        Apply all mutation operators to a rule

        Returns:
            Dict[operator_name, List[(mutated_body, expected_head)]]
        """
        results = {}
        for operator in self.operators:
            try:
                mutations = operator.mutate(rule, kg)
                if mutations:
                    results[operator.name] = mutations
            except ValueError as e:
                # Skip operators that cannot handle this rule type (e.g., OR/NOT logic)
                # This is expected behavior - not all operators support all rule types
                continue
            except (RuntimeError, AttributeError, TypeError, KeyError) as e:
                # Unexpected errors should still be logged
                print(f"Error applying {operator.name}: {e}")
                continue
        return results
