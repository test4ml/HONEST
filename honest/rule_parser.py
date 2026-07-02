#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference rule data structure definition module

Provides shared data structures needed for rule parsing:
- LogicOperator: logical operator enum
- Fact: triple fact
- LogicExpression: logic expression (supports AND, OR, NOT)
- Rule: inference rule
- Token: lexical analysis token

For concrete parser implementations, see:
- horn_rule_parser.py: pure Horn rule parser (AND only, fully independent)
"""

import re
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum


class LogicOperator(Enum):
    """Logical operators"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class Fact:
    """Represents a fact triple (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def to_tuple(self) -> Tuple[str, str, str]:
        """Convert to tuple"""
        return (self.subject, self.predicate, self.object)


@dataclass
class LogicExpression:
    """Represents a logic expression (supports AND, OR, NOT)"""
    operator: Optional[LogicOperator]  # None for single fact
    operands: List[Union['LogicExpression', Fact]]

    def __str__(self) -> str:
        if self.operator is None:
            # Single fact
            return str(self.operands[0]) if self.operands else ""
        elif self.operator == LogicOperator.NOT:
            # NOT expression
            return f"NOT ({self.operands[0]})"
        else:
            # AND/OR expression
            op_str = f" {self.operator.value} "
            return f"({op_str.join(str(op) for op in self.operands)})"

    def to_fact_list(self) -> List[Fact]:
        """Convert to list of facts (only works for AND-only expressions)

        Returns:
            List of facts if this is an AND-only expression

        Raises:
            ValueError: If expression contains OR or NOT
        """
        facts = []
        if self.operator is None:
            # Single fact
            if self.operands and isinstance(self.operands[0], Fact):
                facts.append(self.operands[0])
        elif self.operator == LogicOperator.AND:
            # AND: recursively collect all facts
            for operand in self.operands:
                if isinstance(operand, Fact):
                    facts.append(operand)
                elif isinstance(operand, LogicExpression):
                    facts.extend(operand.to_fact_list())
        elif self.operator in [LogicOperator.OR, LogicOperator.NOT]:
            # OR and NOT cannot be converted to simple fact list
            raise ValueError(f"Cannot convert {self.operator.value} expression to simple fact list")
        return facts


@dataclass
class Rule:
    """Represents an inference rule with body and head

    body can be:
    - List[Fact]: Traditional format (only AND, for backward compatibility)
    - LogicExpression: New format (supports OR/AND/NOT)
    """
    body: Union[List[Fact], LogicExpression]  # Rule body
    head: Fact                                 # Rule head, single fact

    def __str__(self) -> str:
        if isinstance(self.body, list):
            body_str = " ".join(str(fact) for fact in self.body)
        else:
            body_str = str(self.body)
        return f"{body_str} => {self.head}"

    @property
    def variables(self) -> List[str]:
        """Get all variables in the rule"""
        variables = set()

        # Extract variables from body
        if isinstance(self.body, list):
            for fact in self.body:
                for term in [fact.subject, fact.predicate, fact.object]:
                    if term.startswith('?'):
                        variables.add(term)
        else:
            self._extract_variables_from_expr(self.body, variables)

        # Extract variables from head
        for term in [self.head.subject, self.head.predicate, self.head.object]:
            if term.startswith('?'):
                variables.add(term)

        return list(variables)

    def _extract_variables_from_expr(self, expr: LogicExpression, variables: set):
        """Recursively extract variables from logic expression"""
        for operand in expr.operands:
            if isinstance(operand, Fact):
                for term in [operand.subject, operand.predicate, operand.object]:
                    if term.startswith('?'):
                        variables.add(term)
            elif isinstance(operand, LogicExpression):
                self._extract_variables_from_expr(operand, variables)


class Token:
    """Token for lexical analysis"""
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


class RuleParser:
    """Rule parser supporting OR/AND/NOT/parentheses

    This parser can handle both traditional simple rules and complex logical expressions:
    - Simple: "?a P1 ?b ?b P2 ?c => ?a P3 ?c" (implicit AND)
    - Complex: "(?a P1 ?b) OR (?b P2 ?c) => ?a P3 ?c" (explicit OR/AND/NOT)
    """

    @staticmethod
    def parse_rule(rule_str: str) -> Rule:
        """Parse rule string to Rule object

        Args:
            rule_str: Rule string in format "body => head"

        Returns:
            Rule object with body as List[Fact] or LogicExpression
            - If rule contains OR/AND/NOT keywords or parentheses, returns LogicExpression
            - Otherwise returns List[Fact] (traditional format)

        Raises:
            ValueError: If rule format is invalid
        """
        if '=>' not in rule_str:
            raise ValueError(f"Invalid rule format (missing '=>'): {rule_str}")

        # Check if rule contains advanced logic operators or parentheses
        has_advanced_logic = any(keyword in rule_str for keyword in [' OR ', ' AND ', ' NOT ', '(', ')'])

        if has_advanced_logic:
            # Use advanced parser for OR/AND/NOT logic
            tokens = RuleParser._tokenize(rule_str)

            # Find the => token
            arrow_idx = next((i for i, t in enumerate(tokens) if t.type == 'ARROW'), None)
            if arrow_idx is None:
                raise ValueError(f"Missing '=>' in rule: {rule_str}")

            body_tokens = tokens[:arrow_idx]
            head_tokens = tokens[arrow_idx + 1:]

            # Parse body expression
            body_expr, _ = RuleParser._parse_or_expression(body_tokens, 0)

            # Parse head (should be a simple triple)
            head_fact = RuleParser._parse_simple_triple(head_tokens)

            return Rule(body=body_expr, head=head_fact)
        else:
            # Use traditional parser for simple space-separated format
            body_str, head_str = rule_str.split('=>', 1)
            body_str = body_str.strip()
            head_str = head_str.strip()

            # Parse body patterns (space-separated triples)
            body_patterns = RuleParser._parse_traditional_body(body_str)

            # Parse head
            head_pattern = RuleParser._parse_traditional_triple(head_str)

            # Convert to Fact objects
            body_facts = [Fact(subject=s, predicate=p, object=o) for s, p, o in body_patterns]
            head_fact = Fact(subject=head_pattern[0], predicate=head_pattern[1], object=head_pattern[2])

            return Rule(body=body_facts, head=head_fact)

    @staticmethod
    def _tokenize(rule_str: str) -> List[Token]:
        """Lexical analysis: tokenize rule string"""
        tokens = []

        # Regular expression matches:
        # - Arrow: =>
        # - Operators: AND, OR, NOT
        # - Parentheses: ( )
        # - Terms: Q..., P..., ?... (entities, properties, variables)
        pattern = r'=>|\b(AND|OR|NOT)\b|[()]|[QP?\w]+'

        for match in re.finditer(pattern, rule_str):
            value = match.group()

            if value == '=>':
                tokens.append(Token('ARROW', '=>'))
            elif value in ['AND', 'OR', 'NOT']:
                tokens.append(Token('OPERATOR', value))
            elif value == '(':
                tokens.append(Token('LPAREN', '('))
            elif value == ')':
                tokens.append(Token('RPAREN', ')'))
            else:
                # Entity, property, or variable
                tokens.append(Token('TERM', value))

        return tokens

    @staticmethod
    def _parse_or_expression(tokens: List[Token], start: int) -> Tuple[LogicExpression, int]:
        """Parse OR expression (lowest precedence)"""
        left, pos = RuleParser._parse_and_expression(tokens, start)

        or_operands = [left]

        while pos < len(tokens) and tokens[pos].type == 'OPERATOR' and tokens[pos].value == 'OR':
            pos += 1  # Skip OR
            right, pos = RuleParser._parse_and_expression(tokens, pos)
            or_operands.append(right)

        if len(or_operands) == 1:
            return or_operands[0], pos
        else:
            return LogicExpression(operator=LogicOperator.OR, operands=or_operands), pos

    @staticmethod
    def _parse_and_expression(tokens: List[Token], start: int) -> Tuple[LogicExpression, int]:
        """Parse AND expression (medium precedence)"""
        left, pos = RuleParser._parse_not_expression(tokens, start)

        and_operands = [left]

        while pos < len(tokens):
            if tokens[pos].type == 'OPERATOR' and tokens[pos].value == 'AND':
                pos += 1  # Skip AND
                right, pos = RuleParser._parse_not_expression(tokens, pos)
                and_operands.append(right)
            else:
                break

        if len(and_operands) == 1:
            return and_operands[0], pos
        else:
            return LogicExpression(operator=LogicOperator.AND, operands=and_operands), pos

    @staticmethod
    def _parse_not_expression(tokens: List[Token], start: int) -> Tuple[LogicExpression, int]:
        """Parse NOT expression (highest precedence)"""
        if start >= len(tokens):
            raise ValueError("Unexpected end of expression")

        if tokens[start].type == 'OPERATOR' and tokens[start].value == 'NOT':
            # NOT expression
            operand, pos = RuleParser._parse_atom(tokens, start + 1)
            return LogicExpression(operator=LogicOperator.NOT, operands=[operand]), pos
        else:
            return RuleParser._parse_atom(tokens, start)

    @staticmethod
    def _parse_atom(tokens: List[Token], start: int) -> Tuple[LogicExpression, int]:
        """Parse atomic expression (triple or parenthesized expression)"""
        if start >= len(tokens):
            raise ValueError("Unexpected end of expression")

        if tokens[start].type == 'LPAREN':
            # Parenthesized expression
            pos = start + 1
            expr, pos = RuleParser._parse_or_expression(tokens, pos)

            if pos >= len(tokens) or tokens[pos].type != 'RPAREN':
                raise ValueError("Missing closing parenthesis")

            return expr, pos + 1

        elif tokens[start].type == 'TERM':
            # Triple (subject predicate object)
            if start + 2 >= len(tokens):
                raise ValueError(f"Incomplete triple at position {start}")

            subject = tokens[start].value
            predicate = tokens[start + 1].value
            obj = tokens[start + 2].value

            fact = Fact(subject=subject, predicate=predicate, object=obj)

            # Wrap in LogicExpression for consistency
            return LogicExpression(operator=None, operands=[fact]), start + 3

        else:
            raise ValueError(f"Unexpected token at position {start}: {tokens[start]}")

    @staticmethod
    def _parse_simple_triple(tokens: List[Token]) -> Fact:
        """Parse simple triple (for head)"""
        if len(tokens) != 3 or any(t.type != 'TERM' for t in tokens):
            raise ValueError(f"Invalid head format, expected 3 terms, got: {tokens}")

        return Fact(
            subject=tokens[0].value,
            predicate=tokens[1].value,
            object=tokens[2].value
        )

    @staticmethod
    def _parse_traditional_body(body_str: str) -> List[Tuple[str, str, str]]:
        """Parse traditional format body (space-separated triples)

        Example: "?a P1 ?b ?b P2 ?c" → [('?a', 'P1', '?b'), ('?b', 'P2', '?c')]
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
    def _parse_traditional_triple(triple_str: str) -> Tuple[str, str, str]:
        """Parse traditional format single triple

        Example: "?a P1 ?b" → ('?a', 'P1', '?b')
        """
        parts = triple_str.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid triple format: {triple_str}")
        return tuple(parts)
