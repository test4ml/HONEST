#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Body Reduction - body reduction mutation operator
"""

from typing import List, Tuple, Any
import itertools
from .base import MutationOperator, FactInstance
from ..rule_parser import Fact


class BodyReduction(MutationOperator):
    """Body reduction mutation operator - removes some atomic formulas from the rule body"""

    @property
    def name(self) -> str:
        return "Body Reduction"

    def mutate(self, rule: FactInstance, kg: Any) -> List[Tuple[str, str]]:
        """
        Remove one or more atomic formulas from the rule body

        Args:
            rule: Rule instance to be mutated
            kg: Knowledge graph interface (required parameter)

        Returns:
            List[(mutated_body, expected_head)]: Mutated rule body and expected rule head

        Raises:
            ValueError: If rule body contains OR/NOT logic that cannot be reduced
        """
        # Use instance_body_atoms property to get instantiated facts with entity IDs
        # (not rule_body_atoms which would contain variables like ?h, ?a, ?b)
        # This will raise ValueError for OR/NOT expressions
        body_atoms: List[Fact] = rule.instance_body_atoms
        head_atom: Fact = rule.instance_head_atom

        if len(body_atoms) <= 1:
            # Only one atomic formula, cannot reduce further
            return []

        mutations: List[Tuple[str, str]] = []

        # All combinations of removing one atomic formula
        for i in range(len(body_atoms)):
            reduced_atoms: List[Fact] = body_atoms[:i] + body_atoms[i+1:]
            if reduced_atoms:  # Ensure at least one atom is kept
                mutated_body: str = "   ".join(str(atom) for atom in reduced_atoms)
                mutations.append((mutated_body, str(head_atom)))

        # If there are many atomic formulas, also try removing two atomic formulas
        if len(body_atoms) > 3:
            for combination in itertools.combinations(range(len(body_atoms)), 2):
                indices_to_remove: set[int] = set(combination)
                reduced_atoms: List[Fact] = [atom for i, atom in enumerate(body_atoms)
                               if i not in indices_to_remove]
                if len(reduced_atoms) >= 1:  # Ensure at least one atom is kept
                    mutated_body: str = "   ".join(str(atom) for atom in reduced_atoms)
                    mutations.append((mutated_body, str(head_atom)))

        return mutations[:10]  # Limit the output count
