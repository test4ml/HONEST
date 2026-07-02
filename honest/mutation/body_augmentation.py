#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Body Augmentation - body augmentation mutation operator
Randomly adds a true SPO to a horn clause
"""

import random
from typing import List, Tuple, Any
from .base import MutationOperator, FactInstance
from ..rule_parser import Fact


class BodyAugmentation(MutationOperator):
    """Body augmentation mutation operator - randomly adds a true SPO to a horn clause"""

    @property
    def name(self) -> str:
        return "Body Augmentation"

    def mutate(self, rule: FactInstance, kg: Any) -> List[Tuple[str, str]]:
        """
        Randomly add a true triple (real facts queried from the knowledge graph) to the rule body

        Args:
            rule: Rule instance to be mutated
            kg: Knowledge graph interface (required parameter)

        Returns:
            List[(mutated_body, expected_head)]: Mutated rule body and expected rule head

        Raises:
            ValueError: If rule body contains OR/NOT logic that cannot be augmented
        """
        mutations: List[Tuple[str, str]] = []

        # Use instance_body_atoms property to get instantiated facts with entity IDs
        # (not rule_body_atoms which would contain variables like ?h, ?a, ?b)
        # This will raise ValueError for OR/NOT expressions
        matched_body_facts: List[Fact] = rule.instance_body_atoms
        matched_head_fact: Fact = rule.instance_head_atom

        # Get the entities involved in the rule
        entities: List[str] = list(rule.instantiation_mapping.values())
        if not entities:
            return mutations

        # Query the real properties of each entity from the knowledge graph
        for entity in entities:
            try:
                # Query the real properties of the entity
                real_facts: List[Tuple[str, str]] = self._query_entity_facts(entity, kg)

                # Generate a mutation for each real property, avoiding duplicate SPOs
                for predicate, obj in random.sample(real_facts, min(3, len(real_facts))):
                    new_fact = Fact(subject=entity, predicate=predicate, object=obj)

                    # Check whether the new Fact already exists in the rule body
                    if new_fact in matched_body_facts:
                        continue

                    # Create the mutated body
                    mutated_body_facts = matched_body_facts + [new_fact]
                    mutated_body_str = " ".join(str(fact) for fact in mutated_body_facts)
                    matched_head_str = str(matched_head_fact)

                    mutations.append((mutated_body_str, matched_head_str))

            except (AttributeError, TypeError, KeyError, RuntimeError):
                # If the query fails, skip this entity
                continue

        return mutations[:5]  # Limit the return count

    def _query_entity_facts(self, entity: str, kg: Any) -> List[Tuple[str, str]]:
        """
        Query the real properties of an entity, fully leveraging Neo4j's query capability
        Priority order: execute_cypher > get_triples_by_pattern

        Args:
            entity: Entity ID to query
            kg: Knowledge graph interface

        Returns:
            List[Tuple[str, str]]: List of the entity's properties [(predicate, object), ...]
        """
        facts: List[Tuple[str, str]] = []

        try:
            # Method 1 (preferred): if a direct Neo4j query interface is available, use a more powerful Cypher query
            if hasattr(kg, 'execute_cypher') or hasattr(kg, 'query'):
                # Build a Cypher query to get the most relevant properties of the entity
                cypher_query = f"""
                MATCH (n:Entity)-[r:HAS_PROPERTY]->(m:Entity)
                WHERE n.id = '{entity}'
                RETURN r.property_id as predicate, m.id as object, m.label as object_label
                ORDER BY rand()
                LIMIT 20
                """

                query_method = getattr(kg, 'execute_cypher', None) or getattr(kg, 'query', None)
                if query_method:
                    results = query_method(cypher_query)

                    for result in results:
                        if isinstance(result, dict):
                            predicate = result.get('predicate', '')
                            obj = result.get('object', '')
                            if predicate and obj and self._is_suitable_predicate(predicate):
                                facts.append((predicate, obj))
                        elif isinstance(result, (list, tuple)) and len(result) >= 2:
                            predicate, obj = result[0], result[1]
                            if predicate and obj and self._is_suitable_predicate(predicate):
                                facts.append((predicate, obj))

                    # If the Cypher query succeeded and returned results, return directly without trying other methods
                    if facts:
                        print(f"DEBUG: Successfully queried {len(facts)} facts using Cypher for entity {entity}")
                        random.shuffle(facts)
                        return facts[:15]

            # Method 2 (fallback): use the knowledge graph's pattern query method to get all out-edges of the entity
            if hasattr(kg, 'get_triples_by_pattern'):
                print(f"DEBUG: Fallback to get_triples_by_pattern for entity {entity}")
                triples = kg.get_triples_by_pattern(subject=entity, predicate=None, obj=None)

                for triple in triples:
                    if len(triple) >= 3:
                        predicate, obj = triple[1], triple[2]
                        # Basic filter: exclude some properties that are not very suitable as augmentation conditions
                        if self._is_suitable_predicate(predicate):
                            facts.append((predicate, obj))

                # If the pattern query succeeded and returned results, return directly without trying other methods
                if facts:
                    print(f"DEBUG: Successfully queried {len(facts)} facts using pattern query for entity {entity}")
                    random.shuffle(facts)
                    return facts[:15]
        except (AttributeError, TypeError, KeyError, ValueError, RuntimeError) as e:
            # Log the error but do not interrupt execution
            print(f"Warning: Failed to query facts for entity {entity}: {str(e)}")

        # Shuffle randomly and limit the count to increase mutation randomness
        if facts:
            random.shuffle(facts)
            facts = facts[:15]  # Keep more options to improve mutation quality
        else:
            print(f"WARNING: No facts found for entity {entity}, all query methods failed")

        return facts

    def _is_suitable_predicate(self, predicate: str) -> bool:
        """
        Determine whether a property is suitable for use as a body augmentation condition

        Args:
            predicate: Property/predicate

        Returns:
            bool: Whether it is suitable
        """
        # Exclude unsuitable property types
        unsuitable_patterns = [
            'label',           # Label property
            'description',     # Description property
            'alias',           # Alias property
            'sameAs',          # Equivalent link
            'wikidata',        # External link
            'wikipedia',       # Wikipedia link
            'image',           # Image link
            'url',             # URL link
            'identifier',      # Various identifiers
        ]

        predicate_lower = predicate.lower()

        # If it contains an unsuitable pattern, return False
        for pattern in unsuitable_patterns:
            if pattern in predicate_lower:
                return False

        # For Wikidata property IDs (starting with P), all are considered suitable
        if predicate.startswith('P') and predicate[1:].isdigit():
            return True

        # For other types of properties, also considered suitable (let the KG decide)
        return True
