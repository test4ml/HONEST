#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horn Rule Parser Module

Horn rules (Horn clauses) are a subset of first-order logic where:
- The rule body contains only conjunctions (AND) of predicates
- No disjunctions (OR) or negations (NOT) are allowed
- Format: "?a P1 ?b ?b P2 ?c => ?a P3 ?c"

This parser can handle any Horn rule format, including AMIE-style rules output.
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class HornFact:
    """Represents a fact triple (subject, predicate, object) for Horn rules"""
    subject: str
    predicate: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to tuple"""
        return (self.subject, self.predicate, self.object)


@dataclass
class HornRule:
    """Represents a Horn rule with body and head

    Horn rules have:
    - body: List[HornFact] (only AND conjunctions allowed)
    - head: HornFact (single fact)
    """
    body: List[HornFact]  # Rule body (only AND conjunctions)
    head: HornFact        # Rule head, single fact

    def __str__(self) -> str:
        body_str = " ".join(str(fact) for fact in self.body)
        return f"{body_str} => {self.head}"

    @property
    def variables(self) -> List[str]:
        """Get all variables in the rule"""
        variables = set()

        # Extract variables from body
        for fact in self.body:
            for term in [fact.subject, fact.predicate, fact.object]:
                if term.startswith('?'):
                    variables.add(term)

        # Extract variables from head
        for term in [self.head.subject, self.head.predicate, self.head.object]:
            if term.startswith('?'):
                variables.add(term)

        return list(variables)


class HornRuleParser:
    """Horn rule parser (only supports AND conjunctions)

    Parses Horn rules where the body consists of space-separated triples
    that are implicitly connected by AND operations.

    Format: "?a P1 ?b ?b P2 ?c => ?a P3 ?c"

    This parser is suitable for:
    - AMIE rules output
    - Any simple rule format with implicit AND conjunctions
    - Knowledge graph rules without OR/NOT logic
    """

    @staticmethod
    def parse_rule(rule_str: str) -> HornRule:
        """Parse Horn rule string

        Args:
            rule_str: Horn rule string in format "body => head"

        Returns:
            HornRule object with body as List[HornFact]

        Raises:
            ValueError: If rule format is invalid
        """
        if '=>' not in rule_str:
            raise ValueError(f"Invalid rule format: {rule_str}")

        body_str, head_str = rule_str.split('=>', 1)
        body_str = body_str.strip()
        head_str = head_str.strip()

        # Parse body (supports multiple conditions)
        body_patterns = HornRuleParser._parse_body_patterns(body_str)

        # Parse head
        head_pattern = HornRuleParser._parse_pattern(head_str)

        # Convert to HornFact objects
        body_facts = [HornFact(subject=s, predicate=p, object=o) for s, p, o in body_patterns]
        head_fact = HornFact(subject=head_pattern[0], predicate=head_pattern[1], object=head_pattern[2])

        return HornRule(body=body_facts, head=head_fact)

    @staticmethod
    def _parse_body_patterns(body_str: str) -> List[Tuple[str, str, str]]:
        """Parse multiple patterns in the body

        Example: "?a  P1423  ?g  ?g  P910  ?b"
        Returns: [('?a', 'P1423', '?g'), ('?g', 'P910', '?b')]
        """
        patterns = []
        parts = body_str.split()

        # Group by threes (subject predicate object)
        i = 0
        while i + 2 < len(parts):
            subject = parts[i]
            predicate = parts[i + 1]
            obj = parts[i + 2]
            patterns.append((subject, predicate, obj))
            i += 3

        if not patterns:
            raise ValueError(f"Invalid body format, cannot parse: {body_str}")

        return patterns

    @staticmethod
    def _parse_pattern(pattern_str: str) -> Tuple[str, str, str]:
        """Parse a single pattern

        Example: "?b  P2673  ?a" returns ('?b', 'P2673', '?a')
        """
        parts = pattern_str.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid pattern format: {pattern_str}")
        return tuple(parts)


# Backward compatibility: AMIERuleParser is an alias for HornRuleParser
AMIERuleParser = HornRuleParser