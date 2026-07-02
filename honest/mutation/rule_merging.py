#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule Merging - rule merging mutation operator
A AND B -> C, D AND E -> F, yields (A AND B) OR (D AND E) -> (C OR F)
"""

from typing import List, Tuple, Optional, Any
from .base import MutationOperator, FactInstance
from ..rule_parser import Fact


class RuleMerging(MutationOperator):
    """Rule merging mutation operator - merges two rules to form a disjunction"""

    def __init__(self, additional_rules: Optional[List[FactInstance]] = None):
        """
        Initialize the rule merging operator

        Args:
            additional_rules: Extra rule pool used for merging operations
        """
        self.additional_rules: List[FactInstance] = additional_rules or []

    @property
    def name(self) -> str:
        return "Rule Merging"

    def mutate(self, rule: FactInstance, kg: Any) -> List[Tuple[str, str]]:
        """
        Merge the current rule with other rules
        Logic: A AND B -> C, D AND E -> F, yields (A AND B) OR (D AND E) -> (C OR F)

        Args:
            rule: Rule instance to be mutated
            kg: Knowledge graph interface (required parameter)

        Returns:
            List[(mutated_body, expected_head)]: Mutated rule body and expected rule head
        """
        mutations: List[Tuple[str, str]] = []

        # If there are no extra rules, create a self-merge
        if not self.additional_rules:
            return self._self_merge(rule)

        # Merge with extra rules
        for other_rule in self.additional_rules[:3]:
            merged_rule: Optional[Tuple[str, str]] = self._merge_two_rules(rule, other_rule)
            if merged_rule:
                mutations.append(merged_rule)

        return mutations

    def _self_merge(self, rule: FactInstance) -> List[Tuple[str, str]]:
        """Rule self-merge - creates a simplified version"""
        # Use the matched_instance Rule object so the output is concrete instances rather than variables
        matched_rule = rule.matched_instance
        body_atoms = matched_rule.body
        head_atom = matched_rule.head

        if len(body_atoms) < 2:
            return []

        # Split the rule body to create a disjunction
        mid_point: int = len(body_atoms) // 2
        part1: List[Fact] = body_atoms[:mid_point]
        part2: List[Fact] = body_atoms[mid_point:]

        # Create the disjunctive rule body
        part1_str: str = " AND ".join(str(fact) for fact in part1)
        part2_str: str = " AND ".join(str(fact) for fact in part2)
        merged_body: str = f"({part1_str}) OR ({part2_str})"

        return [(merged_body, str(head_atom))]

    def _merge_two_rules(self, rule1: FactInstance, rule2: FactInstance) -> Tuple[str, str]:
        """
        Merge two rules: A AND B -> C, D AND E -> F, yields (A AND B) OR (D AND E) -> (C OR F)

        Args:
            rule1: First rule (A AND B -> C)
            rule2: Second rule (D AND E -> F)

        Returns:
            (merged_body, merged_head): The merged rule
        """
        # Build the disjunction of the rule bodies: (A AND B) OR (D AND E)
        # Use instance_body_atoms to get instantiated facts with entity IDs
        body1_str: str = " AND ".join(str(fact) for fact in rule1.instance_body_atoms)
        body2_str: str = " AND ".join(str(fact) for fact in rule2.instance_body_atoms)
        merged_body: str = f"({body1_str}) OR ({body2_str})"

        # Build the disjunction of the rule heads: C OR F
        # Use instance_head_atom to get instantiated fact with entity IDs
        merged_head: str = f"({rule1.instance_head_atom}) OR ({rule2.instance_head_atom})"

        return (merged_body, merged_head)

    def add_rule(self, rule: FactInstance) -> None:
        """Add an extra rule to the merging pool"""
        self.additional_rules.append(rule)
