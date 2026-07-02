#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized metadata store implementation

Provides memory-optimized metadata storage, supporting management of labels, descriptions, and other info for entities and properties.
"""

from typing import List, Tuple, Optional, Dict
from honest.utils.profiling import profile


class OptimizedMetadataStore:
    """Memory-optimized metadata store

    Uses string interning and __slots__ to optimize memory usage
    """
    __slots__ = ['_entity_labels', '_entity_descriptions', '_property_labels', '_enabled']

    @profile
    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        if enabled:
            # Use more efficient dict initialization, pre-allocate space
            self._entity_labels: Dict[str, str] = dict()
            self._entity_descriptions: Dict[str, str] = dict()
            self._property_labels: Dict[str, str] = dict()
        else:
            self._entity_labels = None
            self._entity_descriptions = None
            self._property_labels = None

    @profile
    def add_entity_label(self, entity_id: str, label: str) -> None:
        """Add an entity label with automatic standardization"""
        if self._enabled and label:
            # Standardize the label
            standardized_label = self._standardize_label(label)
            self._entity_labels[entity_id] = standardized_label

    def _standardize_label(self, label: str) -> str:
        """Standardize an entity label to ensure consistency"""
        if not label:
            return label

        label = label.strip()

        # 1. Handle Wikidata special namespaces
        if ':' in label:
            parts = label.split(':', 1)
            if len(parts) == 2:
                prefix, content = parts
                prefix_lower = prefix.lower()

                # Standardize the Wikidata namespace prefix
                if prefix_lower in ['category', 'template', 'file', 'user']:
                    prefix = prefix_lower.capitalize()
                    # Ensure the content part keeps correct casing
                    content = self._normalize_entity_content(content.strip())
                    return f"{prefix}:{content}"

        # 2. Handle common entity names
        return self._normalize_entity_content(label)

    def _normalize_entity_content(self, content: str) -> str:
        """Standardize the content part of an entity"""
        if not content:
            return content

        # Keep special formats unchanged
        if content == 'COVID-19':
            return content

        # Handle multi-word entity names: Title Case
        words = content.split()
        if len(words) > 1:
            # Title Case, but keep conjunctions and prepositions lowercase (except at the beginning)
            articles_prepositions = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'and', 'or', 'but'}
            normalized_words = []

            for i, word in enumerate(words):
                # Handle special cases
                if word.lower() == 'covid-19':
                    normalized_words.append('COVID-19')
                elif i == 0 or word.lower() not in articles_prepositions:
                    # First word or non-preposition/article: capitalize the first letter
                    normalized_words.append(word.capitalize())
                else:
                    # Preposition/article: keep lowercase
                    normalized_words.append(word.lower())

            return ' '.join(normalized_words)

        # Single-word case: capitalize the first letter
        return content.capitalize()

    @profile
    def add_entity_description(self, entity_id: str, description: str) -> None:
        """Add an entity description"""
        if self._enabled and description:
            self._entity_descriptions[entity_id] = description

    @profile
    def add_property_label(self, property_id: str, label: str) -> None:
        """Add a property label"""
        if self._enabled and label:
            self._property_labels[property_id] = label

    @profile
    def add_entity_labels_batch(self, labels_batch: List[Tuple[str, str]]) -> None:
        """Add entity labels in batch, efficient version"""
        if not self._enabled:
            return

        # Use the dict update method, faster than assigning one by one
        labels_dict = {entity_id: label for entity_id, label in labels_batch if label}
        self._entity_labels.update(labels_dict)

    def get_entity_label(self, entity_id: str) -> Optional[str]:
        """Get an entity label"""
        if not self._enabled:
            return None
        return self._entity_labels.get(entity_id)

    @profile
    def get_entity_description(self, entity_id: str) -> Optional[str]:
        """Get an entity description"""
        if not self._enabled:
            return None
        return self._entity_descriptions.get(entity_id)

    @profile
    def get_property_label(self, property_id: str) -> Optional[str]:
        """Get a property label"""
        if not self._enabled:
            return None
        return self._property_labels.get(property_id)

    def get_stats(self) -> Dict[str, int]:
        """Get metadata statistics"""
        if not self._enabled:
            return {
                'total_entity_labels': 0,
                'total_entity_descriptions': 0,
                'total_property_labels': 0,
            }

        return {
            'total_entity_labels': len(self._entity_labels),
            'total_entity_descriptions': len(self._entity_descriptions),
            'total_property_labels': len(self._property_labels),
        }

    def clear(self) -> None:
        """Clear all metadata"""
        if self._enabled:
            self._entity_labels.clear()
            self._entity_descriptions.clear()
            self._property_labels.clear()
