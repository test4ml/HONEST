#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relation negation configuration examples.

Shows how to extend and customize relation negation rules.
"""

from honest.qgen.relation_negation_config import (
    RelationNegationRule,
    NegationType,
    relation_negation_manager
)


def setup_custom_negation_rules():
    """Example of setting up custom negation rules."""

    # Add family relation rules
    relation_negation_manager.add_rule(RelationNegationRule(
        property_id="P40",  # child
        negation_type=NegationType.SEMANTIC_NEGATION,
        negation_phrase="is not the child of"
    ))

    # Add geographic relation rules
    relation_negation_manager.add_rule(RelationNegationRule(
        property_id="P17",  # country
        negation_type=NegationType.SEMANTIC_NEGATION,
        negation_phrase="is not located in"
    ))

    # Add temporal relation rules
    relation_negation_manager.add_rule(RelationNegationRule(
        property_id="P585",  # point in time
        negation_type=NegationType.SEMANTIC_NEGATION,
        negation_phrase="did not occur in"
    ))

    # Add occupation relation rules
    relation_negation_manager.add_rule(RelationNegationRule(
        property_id="P106",  # occupation
        negation_type=NegationType.SEMANTIC_NEGATION,
        negation_phrase="is not a"
    ))

    # Example of a complex rule using a custom function
    def custom_marriage_negation(subject: str, relation: str, object: str) -> str:
        # For marriage relations, use a more natural negation
        return f"{subject} is not married to {object}"

    relation_negation_manager.add_rule(RelationNegationRule(
        property_id="P26",  # spouse
        negation_type=NegationType.CUSTOM_FUNCTION,
        custom_function=custom_marriage_negation
    ))


if __name__ == "__main__":
    # Set up custom rules
    setup_custom_negation_rules()

    # Test custom rules
    def test_format(s, rel, o, p):
        return f"{s} {rel} {o}"

    test_cases = [
        ("P40", "child", "Alice", "Bob"),
        ("P17", "country", "Paris", "France"),
        ("P26", "spouse", "John", "Mary"),
        ("P106", "occupation", "Einstein", "physicist")
    ]

    print("Testing custom negation rules:")
    for prop_id, rel, subj, obj in test_cases:
        result = relation_negation_manager.get_negation(
            prop_id, rel, subj, obj, test_format
        )
        print(f"{prop_id}: {result}")
