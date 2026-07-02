#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity Rename - Entity Renaming Mutation Operator
Focuses ONLY on entity virtualization (replacing concrete entities with abstract placeholders)
Does NOT modify predicates or swap entities - those are handled by other mutation operators
"""

import random
from typing import List, Tuple, Optional, Any, Dict
from .base import MutationOperator, FactInstance
from ..rule_parser import Rule, Fact


class EntityRename(MutationOperator):
    """Entity renaming mutation operator - replaces concrete entities with semantic-appropriate virtual names"""

    @property
    def name(self) -> str:
        return "Entity Rename"

    def mutate(self, rule: FactInstance, kg: Any) -> List[Tuple[str, str]]:
        """
        Entity virtualization mutation: replaces concrete entities with appropriate virtual names
        based on the entity's semantic type (person, place, organization, etc.)

        Args:
            rule: Rule instance to be mutated
            kg: Knowledge graph interface (required parameter)

        Returns:
            List[(mutated_body, expected_head)]: Mutated rule body and expected rule head

        Raises:
            ValueError: If rule body contains OR/NOT logic that cannot be renamed
        """
        mutations: List[Tuple[str, str]] = []

        # Use instance_body_atoms property to get instantiated facts with entity IDs
        # (not rule_body_atoms which would contain variables like ?h, ?a, ?b)
        # This will raise ValueError for OR/NOT expressions
        matched_body_facts: List[Fact] = rule.instance_body_atoms
        matched_head_fact: Fact = rule.instance_head_atom

        # ONLY perform entity virtualization - no predicate substitution or entity swapping
        entity_mutations = self._mutate_entity_virtualization(rule, matched_body_facts, matched_head_fact, kg)
        mutations.extend(entity_mutations)

        return mutations[:5]  # Limit return count

    def _mutate_entity_virtualization(self, rule: FactInstance, body_facts: List[Fact], head_fact: Fact, kg: Any) -> List[Tuple[str, str]]:
        """Entity virtualization: convert instantiated variables back to virtual descriptions

        Uses semantic-aware entity type detection to choose appropriate virtual names.
        """
        mutations: List[Tuple[str, str]] = []

        # Virtual description mapping (semantic categories)
        # Using number suffixes (e.g., Place1, Person2) to avoid spaCy plural misclassification
        # Number suffixes are reliably recognized as singular by spaCy's POS tagger
        virtual_names: Dict[str, List[str]] = {
            # People (for PERSON entities)
            'person': ['Person1', 'Person2', 'Person3'],
            # Places (for GPE, LOC, FAC entities - countries, cities, locations)
            'place': ['Place1', 'Place2', 'Place3', 'City1', 'City2', 'Town1'],
            # Organizations (for ORG entities - companies, institutions)
            'organization': ['Organization1', 'Organization2', 'Company1', 'Company2', 'Institute1'],
            # Objects/Things (for PRODUCT, WORK_OF_ART, or unknown entities)
            'object': ['Object1', 'Object2', 'Thing1', 'Thing2', 'Entity1', 'Entity2'],
        }

        # Get instantiation mapping
        instantiation_mapping: Dict[str, str] = rule.instantiation_mapping
        if not instantiation_mapping:
            return mutations

        # Generate virtual name replacements for each instantiated variable
        for variable, entity in list(instantiation_mapping.items())[:2]:  # Limit mutation count
            # Get entity label for semantic type detection
            entity_label = kg.get_entity_label(entity) if kg else entity

            # Detect semantic entity type based on label and Wikidata properties
            entity_type = self._detect_entity_type(entity, entity_label, kg)

            # Select appropriate virtual name based on entity type
            if entity_type in virtual_names:
                virtual_name = random.choice(virtual_names[entity_type])
            else:
                # Fallback to generic object names
                virtual_name = random.choice(virtual_names['object'])

            # Replace entities in body (only replace entity IDs, NOT predicates)
            mutated_body_facts = []
            for fact in body_facts:
                # Only replace subject and object, never predicate
                new_subject = fact.subject.replace(entity, virtual_name) if entity in fact.subject else fact.subject
                new_object = fact.object.replace(entity, virtual_name) if entity in fact.object else fact.object
                mutated_body_facts.append(Fact(subject=new_subject, predicate=fact.predicate, object=new_object))

            # Replace entities in head (only replace entity IDs, NOT predicates)
            new_head_subject = head_fact.subject.replace(entity, virtual_name) if entity in head_fact.subject else head_fact.subject
            new_head_object = head_fact.object.replace(entity, virtual_name) if entity in head_fact.object else head_fact.object
            mutated_head_fact = Fact(subject=new_head_subject, predicate=head_fact.predicate, object=new_head_object)

            # Convert to string format
            mutated_body_str = " ".join(str(fact) for fact in mutated_body_facts)
            mutated_head_str = str(mutated_head_fact)

            mutations.append((mutated_body_str, mutated_head_str))

        return mutations

    def _detect_entity_type(self, entity_id: str, entity_label: str, kg: Any) -> str:
        """Detect entity semantic type using label analysis and Wikidata instance-of property

        Args:
            entity_id: Wikidata entity ID (e.g., Q8646)
            entity_label: Entity label (e.g., "Hong Kong")
            kg: Knowledge graph interface

        Returns:
            Entity type: 'person', 'place', 'organization', or 'object'
        """
        # Try to use spaCy NER for label-based detection
        try:
            import spacy
            from honest.constants import SPACY_MODEL_NAME
            nlp = spacy.load(SPACY_MODEL_NAME)
            doc = nlp(entity_label)

            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    return 'person'
                elif ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geopolitical, Location, Facility
                    return 'place'
                elif ent.label_ == 'ORG':
                    return 'organization'
        except (OSError, ImportError, AttributeError, RuntimeError):
            pass

        # Fallback: keyword-based detection
        label_lower = entity_label.lower()

        # Place keywords (the non-ASCII entry is a Chinese character appearing in
        # Hong Kong related labels; 'kong' covers the English form)
        place_keywords = ['city', 'country', 'state', 'province', 'region', 'town', 'village',
                         'island', 'mountain', 'river', 'sea', 'ocean', 'continent', '港', 'kong']
        if any(keyword in label_lower for keyword in place_keywords):
            return 'place'

        # Organization keywords
        org_keywords = ['company', 'corporation', 'inc', 'ltd', 'university', 'institute',
                       'organization', 'association', 'foundation', 'group']
        if any(keyword in label_lower for keyword in org_keywords):
            return 'organization'

        # Person name patterns (capitalized words without common place/org indicators)
        words = entity_label.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            # Could be a person name if not clearly a place or organization
            return 'person'

        # Default to generic object
        return 'object'

