#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikidata property formatting configuration.

This module defines how to correctly format Wikidata properties into natural
language sentences. It provides a mapping from property IDs to grammatical
patterns to avoid common grammatical errors.

Supports type-aware formatting: some properties require different patterns based
on the semantic type of the subject.
For example, P1313 (office held by head of government):
- When the subject is a place: "Fournels has Mayor of Fournels as its head of government office"
- When the subject is a person: "Pierre holds the office of Mayor" (although this case is unlikely)
"""

import json
import os
import re
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from ..langtool.english.article_analyzer import EnglishArticleAnalyzer


# Semantic type enumeration
class SemanticType:
    """Semantic type constants."""
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    POSITION = "position"
    EVENT = "event"
    WORK = "work"  # Artworks, books, etc.
    UNKNOWN = "unknown"


@dataclass
class PropertyFormatRule:
    """Property formatting rule.

    Supports type-aware formatting:
    - pattern: default pattern
    - type_patterns: subject-type-specific patterns
    """
    pattern: str  # Formatting pattern, using {subject}, {object} placeholders
    needs_article_subject: bool = True  # Whether the subject needs an article
    needs_article_object: bool = True   # Whether the object needs an article
    description: str = ""  # Rule description
    # Type-aware patterns: {semantic_type: pattern}
    type_patterns: Dict[str, str] = field(default_factory=dict)


class EntityTypeDetector:
    """Entity semantic type detector.

    Detects the semantic type from entity descriptions to select the
    appropriate formatting pattern.
    """

    # Type detection patterns
    PERSON_PATTERNS = [
        'politician', 'actor', 'actress', 'writer', 'author', 'artist',
        'musician', 'singer', 'composer', 'director', 'scientist',
        'philosopher', 'historian', 'journalist', 'engineer', 'lawyer',
        'doctor', 'professor', 'teacher', 'athlete', 'footballer',
        'basketball player', 'tennis player', 'person', 'human',
        # Birth/death year patterns
        r'\(\d{4}-\)', r'\(\d{4}-\d{4}\)', r'\(born \d{4}\)',
    ]

    PLACE_PATTERNS = [
        'commune', 'city', 'town', 'village', 'municipality', 'county',
        'district', 'region', 'province', 'state', 'country', 'nation',
        'capital', 'prefecture', 'department', 'borough', 'township',
        'located in', 'administrative', 'territorial', 'settlement'
    ]

    POSITION_PATTERNS = [
        'mayor', 'governor', 'president', 'minister', 'head of',
        'office', 'position', 'role', 'title', 'post', 'seat'
    ]

    ORGANIZATION_PATTERNS = [
        'company', 'organization', 'corporation', 'institution',
        'university', 'college', 'school', 'hospital', 'government',
        'agency', 'department', 'ministry', 'party', 'association',
        'foundation', 'institute', 'council', 'committee'
    ]

    @classmethod
    def detect_type(cls, label: str, description: str = "") -> str:
        """Detect the semantic type of an entity.

        Args:
            label: Entity label
            description: Entity description

        Returns:
            SemanticType: The detected semantic type
        """
        if not description or description == "<<NO_DESCRIPTION>>":
            # When there is no description, try to infer from the label
            return cls._infer_from_label(label)

        desc_lower = description.lower()

        # Check person patterns
        for pattern in cls.PERSON_PATTERNS:
            if pattern.startswith(r'\('):
                # Regex pattern
                if re.search(pattern, description):
                    return SemanticType.PERSON
            elif pattern in desc_lower:
                return SemanticType.PERSON

        # Check place patterns
        for pattern in cls.PLACE_PATTERNS:
            if pattern in desc_lower:
                return SemanticType.PLACE

        # Check position patterns
        for pattern in cls.POSITION_PATTERNS:
            if pattern in desc_lower:
                return SemanticType.POSITION

        # Check organization patterns
        for pattern in cls.ORGANIZATION_PATTERNS:
            if pattern in desc_lower:
                return SemanticType.ORGANIZATION

        return SemanticType.UNKNOWN

    @classmethod
    def _infer_from_label(cls, label: str) -> str:
        """Infer the type from the label."""
        label_lower = label.lower()

        # Position labels usually contain these words
        if any(p in label_lower for p in ['mayor', 'governor', 'president', 'minister']):
            return SemanticType.POSITION

        return SemanticType.UNKNOWN


class WikidataPropertyFormatter:
    """Wikidata property formatter.

    Supports type-aware formatting: some properties select different patterns
    based on the semantic type of the subject.
    """

    # Properties that require type-aware formatting and their type-specific patterns.
    # The default patterns for these properties live in the JSONL file; only the
    # type-specific overrides are defined here.
    TYPE_AWARE_PATTERNS = {
        # P1313: office held by head of government
        # When the subject is a place, use "has ... as its head of government office"
        # rather than "holds the office of", because a place cannot "hold" a position.
        "P1313": {
            SemanticType.PLACE: "{subject} has {object} as its head of government office",
            SemanticType.ORGANIZATION: "{subject} has {object} as its head of government office",
        },
        # More type-aware properties can be added here
    }

    def __init__(self):
        """Initialize the property formatter."""
        self.article_analyzer = EnglishArticleAnalyzer()
        self.property_format_rules: Dict[str, PropertyFormatRule] = {}
        self.type_detector = EntityTypeDetector()

        # Load property formatting rules from the jsonl file
        self._load_property_rules()

        # Default formatting rule (used when there is no specific rule)
        self.default_rule = PropertyFormatRule(
            pattern="{subject} {predicate} {object}",
            needs_article_subject=True,
            needs_article_object=True,
            description="Default formatting rule"
        )

    def _load_property_rules(self):
        """Load property formatting rules from the jsonl file."""
        rules_file = os.path.join(os.path.dirname(__file__), "property_format_rules.jsonl")

        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rule_data = json.loads(line)
                        property_id = rule_data['property_id']
                        self.property_format_rules[property_id] = PropertyFormatRule(
                            pattern=rule_data['pattern'],
                            needs_article_subject=rule_data['needs_article_subject'],
                            needs_article_object=rule_data['needs_article_object'],
                            description=rule_data['description']
                        )
        except FileNotFoundError:
            print(f"Warning: Property rules file not found at {rules_file}")
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing property rules file: {e}")
        except (OSError, KeyError, ValueError) as e:
            print(f"Warning: Error loading property rules: {e}")

    def get_format_rule(self, property_id: str) -> PropertyFormatRule:
        """
        Get the formatting rule for a property.

        Args:
            property_id: Wikidata property ID (e.g. P1435)

        Returns:
            PropertyFormatRule: The formatting rule
        """
        return self.property_format_rules.get(property_id, self.default_rule)

    def _get_type_aware_pattern(self, property_id: str, subject_type: str,
                                 default_pattern: str) -> str:
        """Get a type-aware formatting pattern.

        Args:
            property_id: Property ID
            subject_type: Semantic type of the subject
            default_pattern: Default pattern

        Returns:
            str: The formatting pattern appropriate for the subject type
        """
        if property_id in self.TYPE_AWARE_PATTERNS:
            type_patterns = self.TYPE_AWARE_PATTERNS[property_id]
            if subject_type in type_patterns:
                return type_patterns[subject_type]
        return default_pattern

    def format_statement(self, subject: str, predicate: str, object_: str,
                        property_id: str, subject_description: str = "") -> str:
        """
        Format a statement by property ID (note: only affirmative sentences are
        formatted; the predicate is ignored).

        Supports type-aware formatting: when subject_description is provided, the
        most appropriate formatting pattern is selected based on the semantic
        type of the subject.

        Args:
            subject: Subject
            predicate: Predicate (original property label)
            object_: Object
            property_id: Property ID
            subject_description: Wikidata description of the subject (optional,
                                used for type-aware formatting)

        Returns:
            str: The formatted statement
        """
        rule = self.get_format_rule(property_id)

        # Decide whether to add an article based on the rule
        formatted_subject = self.article_analyzer.add_article(subject) if rule.needs_article_subject else subject

        # Determine whether the subject is plural, to adjust the object form
        is_subject_plural = self.article_analyzer.is_subject_plural(subject)

        # For some relations, if the subject is plural, the object should be adjusted accordingly
        adjusted_object = self._adjust_object_for_plural_subject(object_, property_id, is_subject_plural)
        formatted_object = self.article_analyzer.add_article(adjusted_object) if rule.needs_article_object else adjusted_object

        # Get the formatting pattern
        pattern = rule.pattern

        # Type-aware pattern selection
        if property_id in self.TYPE_AWARE_PATTERNS and subject_description:
            subject_type = EntityTypeDetector.detect_type(subject, subject_description)
            pattern = self._get_type_aware_pattern(property_id, subject_type, pattern)

        # Adjust the verb form according to the subject's number.
        # For certain inverted patterns (e.g. P674), the verb must be adjusted based
        # on the actual grammatical subject of the sentence.
        grammatical_subject = self._determine_grammatical_subject(pattern, subject, object_)
        adjusted_pattern = self._adjust_verb_for_subject_number(pattern, grammatical_subject)

        # Format the statement using the adjusted pattern
        if '{predicate}' in adjusted_pattern:
            return adjusted_pattern.format(
                subject=formatted_subject,
                predicate=predicate,
                object=formatted_object
            )
        else:
            return adjusted_pattern.format(
                subject=formatted_subject,
                object=formatted_object
            )

    def add_property_rule(self, property_id: str, rule: PropertyFormatRule):
        """Add a new property formatting rule."""
        self.property_format_rules[property_id] = rule

    def _determine_grammatical_subject(self, pattern: str, subject: str, object_: str) -> str:
        """
        Determine the grammatical subject of the sentence.

        For inverted patterns (e.g. P674: "{object} is one of the characters in {subject}"),
        the grammatical subject is actually the object, not the subject.

        Args:
            pattern: Formatting pattern
            subject: The subject of the triple
            object_: The object of the triple

        Returns:
            str: The actual grammatical subject of the sentence
        """
        # Check whether the pattern starts with {object}, indicating an inverted pattern
        if pattern.strip().startswith('{object}'):
            return object_
        else:
            return subject

    def _adjust_verb_for_subject_number(self, pattern: str, subject: str) -> str:
        """Adjust the verb form according to the subject's number (singular/plural)."""
        if not pattern or not subject:
            return pattern

        # Determine whether the subject is plural
        is_plural = self.article_analyzer.is_subject_plural(subject)

        # Verb form mapping: singular -> plural
        verb_mappings = {
            'is': 'are',
            'was': 'were',
            'has': 'have',
            'does': 'do'
        }

        # If the subject is plural, replace the verb form
        if is_plural:
            adjusted_pattern = pattern
            for singular, plural in verb_mappings.items():
                # Use word boundaries to ensure exact matching
                import re
                adjusted_pattern = re.sub(r'\b' + singular + r'\b', plural, adjusted_pattern)
            return adjusted_pattern

        return pattern

    def _adjust_object_for_plural_subject(self, object_: str, property_id: str, is_subject_plural: bool) -> str:
        """Adjust the object according to the subject's plural form."""
        if not is_subject_plural:
            return object_

        # For the "instance of" relation, when the subject is plural, the object is usually plural as well
        if property_id in ['P31', 'P279']:  # instance of, subclass of
            return self._pluralize_noun(object_)

        # Occupation relations may also need the plural form
        if property_id == 'P106':  # occupation
            return self._pluralize_noun(object_)

        return object_

    def _pluralize_noun(self, noun: str) -> str:
        """Simple noun pluralization."""
        if not noun:
            return noun

        noun = noun.strip()
        noun_lower = noun.lower()

        # Cases that should not be pluralized
        if noun_lower in ['water', 'music', 'information', 'advice', 'research', 'equipment']:  # Uncountable nouns
            return noun

        # Irregular plurals
        irregular_plurals = {
            'child': 'children',
            'person': 'people',
            'man': 'men',
            'woman': 'women',
            'foot': 'feet',
            'tooth': 'teeth',
            'mouse': 'mice',
            'goose': 'geese'
        }

        if noun_lower in irregular_plurals:
            # Preserve the original capitalization
            if noun[0].isupper():
                return irregular_plurals[noun_lower].capitalize()
            return irregular_plurals[noun_lower]

        # Already plural
        if noun.endswith(('s', 'es', 'ies', 'ves')):
            return noun

        # Simple plural rules
        if noun.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return noun + 'es'
        elif noun.endswith('y') and len(noun) > 1 and noun[-2] not in 'aeiou':
            return noun[:-1] + 'ies'
        elif noun.endswith(('f', 'fe')):
            if noun.endswith('fe'):
                return noun[:-2] + 'ves'
            else:
                return noun[:-1] + 'ves'
        else:
            return noun + 's'

    def get_all_supported_properties(self) -> Dict[str, str]:
        """Get all supported properties and their descriptions."""
        return {prop_id: rule.description
                for prop_id, rule in self.property_format_rules.items()}


# Create a global instance
wikidata_formatter = WikidataPropertyFormatter()
