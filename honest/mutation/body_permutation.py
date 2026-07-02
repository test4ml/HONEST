#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Body Permutation - body permutation mutation operator
"""

from typing import List, Tuple, Any
import itertools
from .base import MutationOperator, FactInstance
from ..rule_parser import Fact


class BodyPermutation(MutationOperator):
    """Body permutation mutation operator - reorders the atomic formulas in the rule body"""

    @property
    def name(self) -> str:
        return "Body Permutation"

    def mutate(self, rule: FactInstance, kg: Any) -> List[Tuple[str, str]]:
        """
        Reorder the atomic formulas in the rule body

        Args:
            rule: Rule instance to be mutated
            kg: Knowledge graph interface (required parameter)

        Returns:
            List[(mutated_body, expected_head)]: Mutated rule body and expected rule head

        Raises:
            ValueError: If rule body contains OR/NOT logic that cannot be permuted
        """
        # Use instance_body_atoms property to get instantiated facts with entity IDs
        # (not rule_body_atoms which would contain variables like ?h, ?a, ?b)
        # This will raise ValueError for OR/NOT expressions
        body_atoms: List[Fact] = rule.instance_body_atoms
        head_atom: Fact = rule.instance_head_atom

        if len(body_atoms) <= 1:
            # Only one atomic formula, cannot permute
            return []

        mutations: List[Tuple[str, str]] = []
        # Generate all possible permutations (except the original order)
        for perm in itertools.permutations(body_atoms):
            if list(perm) != body_atoms:  # Skip the original order
                mutated_body: str = "   ".join(str(atom) for atom in perm)  # Join with three spaces to keep the format consistent
                mutations.append((mutated_body, str(head_atom)))

        return mutations[:10]  # Limit the output count to avoid too many permutations
