#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distractor Triple Generator

Generates candidate distractor triples using various strategies.
Works at the triple level, enabling validation before text conversion.
"""

import logging
from typing import Tuple, List, Optional, Set, Any

# Import Triple type from validator module
from .validator import Triple

logger = logging.getLogger(__name__)


# OPTIMIZATION: Pre-computed inverse relation pairs to avoid expensive KG queries
# These are the most common inverse pairs in Wikidata
PRECOMPUTED_INVERSE_PAIRS = {
    'P22': 'P40',    # father <-> child
    'P40': 'P22',    # child <-> father
    'P25': 'P40',    # mother <-> child
    'P7': 'P26',     # brother <-> sibling
    'P26': 'P7',     # sibling <-> brother
    'P3373': 'P3373', # sibling <-> sibling (symmetric)
    'P2673': 'P2674', # next crossing upstream <-> next crossing downstream
    'P2674': 'P2673', # next crossing downstream <-> next crossing upstream
    'P3729': 'P3730', # next lower rank <-> next higher rank
    'P3730': 'P3729', # next higher rank <-> next lower rank
    'P155': 'P156',  # follows <-> followed by
    'P156': 'P155',  # followed by <-> follows
    'P1365': 'P1366', # replaces <-> replaced by
    'P1366': 'P1365', # replaced by <-> replaces
}

# OPTIMIZATION: Pre-computed symmetric relations (R(a,b) <=> R(b,a))
PRECOMPUTED_SYMMETRIC_RELATIONS = {
    'P3373',  # sibling
    'P451',   # unmarried partner
    'P1327',  # partner in business or sport
    'P2350',  # member of sports team
}


class DistractorTripleGenerator:
    """
    Generates candidate distractor triples using various strategies.

    Unlike the text-based generator, this works at the triple level,
    enabling validation before text conversion.
    """

    def __init__(self, kg: Optional[Any] = None):
        """
        Initialize the generator.

        Args:
            kg: Knowledge graph instance for querying entities/relations
        """
        self.kg = kg

        # Caches for KG metadata queries (for performance)
        self._symmetric_cache = {}      # property_id -> bool
        self._inverse_cache = {}        # property_id -> inverse_property_id
        self._type_cache = {}           # entity_id -> type_id
        self._type_hierarchy_cache = {} # type_id -> set of ancestor types

    def generate_candidates(self,
                            correct_triple: Triple,
                            body_triples: List[Triple],
                            entity_ids: List[str],
                            relation_ids: List[str],
                            num_candidates: int = 10,
                            use_only_local_entities: bool = True) -> List[Triple]:
        """
        Generate candidate distractor triples using multiple strategies.

        Args:
            correct_triple: The correct answer (s, p, o)
            body_triples: Premise triples
            entity_ids: Available entity IDs (from question context)
            relation_ids: Available relation IDs (from question context)
            num_candidates: Maximum number of candidates to generate
            use_only_local_entities: If True, only use entities from question context.
                                      If False, may use entities from KG (NOT RECOMMENDED)

        Returns:
            List of candidate distractor triples
        """
        candidates = []
        s, p, o = correct_triple

        # IMPORTANT: We now ONLY use entities and relations from the question context
        # to ensure distractors are semantically relevant to the topic

        # Strategy 1: Swap subject and object
        swapped = self._strategy_swap_entities(correct_triple)
        if swapped:
            candidates.append(swapped)

        # Strategy 2: Change relation (only use relations from question context)
        relation_variants = self._strategy_change_relation(
            correct_triple, relation_ids
        )
        candidates.extend(relation_variants)

        # Strategy 3: Change object entity (only use entities from question context)
        object_variants = self._strategy_change_object(
            correct_triple, entity_ids
        )
        candidates.extend(object_variants)

        # Strategy 4: Change subject entity (only use entities from question context)
        subject_variants = self._strategy_change_subject(
            correct_triple, entity_ids
        )
        candidates.extend(subject_variants)

        # Strategy 5: Use entities from body with different relation
        # This is the most semantically relevant strategy
        body_variants = self._strategy_body_entity_combinations(
            correct_triple, body_triples, relation_ids
        )
        candidates.extend(body_variants)

        # NEW STRATEGIES (6-10):

        # Strategy 6: Functional property violation
        # Generates distractors by violating functional constraints
        functional_variants = self._strategy_functional_property_violation(
            correct_triple, entity_ids
        )
        candidates.extend(functional_variants)

        # Strategy 7: Inverse relation misapplication
        # Uses inverse relations incorrectly
        inverse_variants = self._strategy_inverse_relation_error(
            correct_triple, relation_ids
        )
        candidates.extend(inverse_variants)

        # Strategy 8: Break transitivity
        # Creates wrong intermediate or reversed conclusions
        transitivity_variants = self._strategy_break_transitivity(
            correct_triple, body_triples, relation_ids
        )
        candidates.extend(transitivity_variants)

        # Strategy 9: Negate premise
        # Uses modified premises as wrong conclusions
        premise_variants = self._strategy_negate_premise(
            body_triples, correct_triple, relation_ids
        )
        candidates.extend(premise_variants)

        # Strategy 10: Type mismatch - DISABLED FOR PERFORMANCE
        # REASON: Requires expensive type hierarchy queries (19s per query)
        # Creates semantically incompatible triples
        # type_variants = self._strategy_type_mismatch(
        #     correct_triple, entity_ids, relation_ids
        # )
        # candidates.extend(type_variants)
        logger.debug("Strategy 10 (type mismatch) disabled for performance optimization")

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for triple in candidates:
            if triple not in seen and triple != correct_triple:
                seen.add(triple)
                unique_candidates.append(triple)

        return unique_candidates[:num_candidates]

    def _strategy_swap_entities(self, correct_triple: Triple) -> Optional[Triple]:
        """Swap subject and object"""
        s, p, o = correct_triple
        if s != o:  # Only swap if different
            return (o, p, s)
        return None

    def _strategy_change_relation(self,
                                   correct_triple: Triple,
                                   relation_ids: List[str],
                                   max_variants: int = 3) -> List[Triple]:
        """Generate variants with different relations"""
        s, p, o = correct_triple
        variants = []

        # Deterministically select relations based on hash
        other_relations = [r for r in relation_ids if r != p]
        if not other_relations:
            return variants

        # Sort for determinism
        sorted_relations = sorted(other_relations)
        seed = hash(f"rel_{s}_{p}_{o}")

        for i in range(min(max_variants, len(sorted_relations))):
            idx = (seed + i) % len(sorted_relations)
            new_p = sorted_relations[idx]
            variants.append((s, new_p, o))

        return variants

    def _strategy_change_object(self,
                                 correct_triple: Triple,
                                 entity_ids: List[str],
                                 max_variants: int = 3) -> List[Triple]:
        """Generate variants with different object entities"""
        s, p, o = correct_triple
        variants = []

        other_entities = [e for e in entity_ids if e != o and e != s]
        if not other_entities:
            return variants

        sorted_entities = sorted(other_entities)
        seed = hash(f"obj_{s}_{p}_{o}")

        for i in range(min(max_variants, len(sorted_entities))):
            idx = (seed + i) % len(sorted_entities)
            new_o = sorted_entities[idx]
            variants.append((s, p, new_o))

        return variants

    def _strategy_change_subject(self,
                                  correct_triple: Triple,
                                  entity_ids: List[str],
                                  max_variants: int = 2) -> List[Triple]:
        """Generate variants with different subject entities"""
        s, p, o = correct_triple
        variants = []

        other_entities = [e for e in entity_ids if e != s and e != o]
        if not other_entities:
            return variants

        sorted_entities = sorted(other_entities)
        seed = hash(f"subj_{s}_{p}_{o}")

        for i in range(min(max_variants, len(sorted_entities))):
            idx = (seed + i) % len(sorted_entities)
            new_s = sorted_entities[idx]
            variants.append((new_s, p, o))

        return variants

    def _strategy_body_entity_combinations(self,
                                           correct_triple: Triple,
                                           body_triples: List[Triple],
                                           relation_ids: List[str],
                                           max_variants: int = 2) -> List[Triple]:
        """Generate variants using entities from body with different relations"""
        variants = []
        s, p, o = correct_triple

        # Collect all entities from body
        body_entities = set()
        for bs, bp, bo in body_triples:
            body_entities.add(bs)
            body_entities.add(bo)

        # Use body entities with a different relation
        other_relations = [r for r in relation_ids if r != p]
        if not other_relations or len(body_entities) < 2:
            return variants

        body_entities_list = sorted(list(body_entities))
        sorted_relations = sorted(other_relations)
        seed = hash(f"body_{s}_{p}_{o}")

        count = 0
        for i, e1 in enumerate(body_entities_list):
            for j, e2 in enumerate(body_entities_list):
                if e1 != e2 and count < max_variants:
                    rel_idx = (seed + count) % len(sorted_relations)
                    new_rel = sorted_relations[rel_idx]
                    variant = (e1, new_rel, e2)
                    if variant != correct_triple:
                        variants.append(variant)
                        count += 1

        return variants

    def _strategy_functional_property_violation(self,
                                                correct_triple: Triple,
                                                entity_ids: List[str],
                                                max_variants: int = 2) -> List[Triple]:
        """
        Strategy 6: Functional Property Violation

        Generate distractors by violating functional properties.
        If a property is functional (1-to-1 or many-to-1), then for the same subject
        and property, only one object can be true. Any other object makes it false.

        Example:
        - Correct: (Person, birthPlace, CityA)
        - Distractor: (Person, birthPlace, CityB)  # Wrong! Can only have one birthplace

        Args:
            correct_triple: The correct answer (s, p, o)
            entity_ids: Available entity IDs
            max_variants: Maximum number of variants to generate

        Returns:
            List of distractor triples that violate functional property constraint
        """
        s, p, o = correct_triple
        variants = []

        # Check if this property is functional
        if not self._is_functional_property(p):
            return variants

        # Generate variants with same subject and predicate, different object
        # This violates the functional constraint (one subject can only have one object)
        other_entities = [e for e in entity_ids if e != o and e != s]
        if not other_entities:
            return variants

        sorted_entities = sorted(other_entities)
        seed = hash(f"func_{s}_{p}_{o}")

        for i in range(min(max_variants, len(sorted_entities))):
            idx = (seed + i) % len(sorted_entities)
            new_o = sorted_entities[idx]
            variants.append((s, p, new_o))

        return variants

    def _is_functional_property(self, property_id: str) -> bool:
        """
        Check if a property is functional using KG metadata.

        A functional property means each subject can have at most one object.
        Examples: birthPlace, birthDate, deathDate, etc.

        Args:
            property_id: Property ID to check

        Returns:
            True if property is functional (functionality > 0.95)
        """
        if self.kg and hasattr(self.kg, 'functionality'):
            try:
                func_value = self.kg.functionality(property_id)
                return func_value > 0.95  # High functionality threshold
            except (AttributeError, TypeError, ValueError):
                return False
        return False

    def _strategy_inverse_relation_error(self,
                                        correct_triple: Triple,
                                        relation_ids: List[str]) -> List[Triple]:
        """
        Strategy 7: Inverse Relation Misapplication

        Generate distractors using inverse relations incorrectly.

        Example:
        - Correct: (John, father, Mary)  # John is Mary's father
        - Distractor: (Mary, father, John)  # Wrong! Mary is not John's father

        Or with inverse property:
        - Correct: (John, father, Mary)
        - Distractor: (John, child, Mary)  # Wrong direction! Should be (Mary, child, John)

        Args:
            correct_triple: The correct answer (s, p, o)
            relation_ids: Available relation IDs

        Returns:
            List of distractor triples with inverse relation errors
        """
        s, p, o = correct_triple
        variants = []

        # Strategy 7a: Direct swap (keep same relation)
        # This creates (Object, Relation, Subject) which is often wrong
        # unless the relation is symmetric
        if not self._is_symmetric_relation(p):
            variants.append((o, p, s))

        # Strategy 7b: Use inverse relation but keep subject/object order
        inverse_p = self._get_inverse_relation(p, relation_ids)
        if inverse_p and inverse_p != p:
            # Using inverse relation without swapping entities = wrong!
            variants.append((s, inverse_p, o))

        return variants

    def _is_symmetric_relation(self, property_id: str) -> bool:
        """
        Check if a relation is symmetric (R(a,b) <=> R(b,a)).

        OPTIMIZED: Uses pre-computed list first, falls back to KG query with caching.
        """
        # OPTIMIZATION: Check pre-computed list first (instant lookup)
        if property_id in PRECOMPUTED_SYMMETRIC_RELATIONS:
            logger.debug(f"Property {property_id} is symmetric (pre-computed)")
            return True

        # Check cache
        if property_id in self._symmetric_cache:
            return self._symmetric_cache[property_id]

        is_symmetric = False

        # Fall back to KG query (only if not in pre-computed list)
        if self.kg and hasattr(self.kg, 'query'):
            try:
                query = """
                    MATCH (p:Property {id: $property_id})-[:HAS_PROPERTY {property_id: 'P1696'}]->(inv:Property)
                    RETURN inv.id as inverse_id
                """
                result = self.kg.query(query, property_id=property_id)

                if result and len(result) > 0:
                    inverse_id = result[0].get('inverse_id')
                    is_symmetric = (inverse_id == property_id)
                    logger.debug(f"Property {property_id} inverse: {inverse_id}, symmetric: {is_symmetric} (KG query)")

            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"Failed to query symmetric relation for {property_id}: {e}")
                is_symmetric = False

        # Cache the result
        self._symmetric_cache[property_id] = is_symmetric
        return is_symmetric

    def _get_inverse_relation(self, property_id: str, available_relations: List[str]) -> Optional[str]:
        """
        Get the inverse relation.

        OPTIMIZED: Uses pre-computed pairs first, falls back to KG query with caching.
        """
        # OPTIMIZATION: Check pre-computed pairs first (instant lookup)
        if property_id in PRECOMPUTED_INVERSE_PAIRS:
            inverse_id = PRECOMPUTED_INVERSE_PAIRS[property_id]
            logger.debug(f"Found inverse property for {property_id}: {inverse_id} (pre-computed)")
            return inverse_id if inverse_id in available_relations else None

        # Check cache
        if property_id in self._inverse_cache:
            inverse = self._inverse_cache[property_id]
            return inverse if inverse in available_relations else None

        inverse_id = None

        # Fall back to KG query (only if not in pre-computed pairs)
        if self.kg and hasattr(self.kg, 'query'):
            try:
                query = """
                    MATCH (p:Property {id: $property_id})-[:HAS_PROPERTY {property_id: 'P1696'}]->(inv:Property)
                    RETURN inv.id as inverse_id
                """
                result = self.kg.query(query, property_id=property_id)

                if result and len(result) > 0:
                    inverse_id = result[0].get('inverse_id')
                    logger.debug(f"Found inverse property for {property_id}: {inverse_id} (KG query)")

            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"Failed to query inverse property for {property_id}: {e}")
                inverse_id = None

        # Cache the result (even if None)
        self._inverse_cache[property_id] = inverse_id

        # Only return if inverse is in available relations
        return inverse_id if inverse_id and inverse_id in available_relations else None

    def _strategy_break_transitivity(self,
                                    correct_triple: Triple,
                                    body_triples: List[Triple],
                                    relation_ids: List[str]) -> List[Triple]:
        """
        Strategy 8: Transitive Relation Path Breaking

        Generate distractors by breaking transitive reasoning chains.

        If the rule is: (A, R1, B), (B, R2, C) => (A, R3, C)
        Generate wrong paths like:
        - (B, R3, A)  # Reversed conclusion
        - (A, R3, B)  # Intermediate instead of end
        - (B, R3, C)  # Start from middle

        Example:
        - Premises: (Alice, parentOf, Bob), (Bob, parentOf, Charlie)
        - Correct: (Alice, grandparentOf, Charlie)
        - Wrong: (Alice, grandparentOf, Bob)  # Bob is child, not grandchild

        Args:
            correct_triple: The correct answer (s, p, o)
            body_triples: Premise triples
            relation_ids: Available relation IDs

        Returns:
            List of distractor triples that break transitivity
        """
        variants = []
        s, p, o = correct_triple

        # Check if this is a transitive inference (need at least 2 premises)
        if len(body_triples) < 2:
            return variants

        # Find the chain: look for connected triples in body
        chain_entities = set()
        for bs, bp, bo in body_triples:
            chain_entities.add(bs)
            chain_entities.add(bo)

        # Strategy 8a: Use intermediate entities as endpoints
        # Intermediate entities are those in the chain but not at the final endpoints
        intermediate_entities = chain_entities - {s, o}

        for entity in intermediate_entities:
            # Create wrong conclusions to/from intermediate points
            if entity != s:
                variants.append((s, p, entity))  # Conclusion to intermediate
            if entity != o:
                variants.append((entity, p, o))  # Intermediate to end

        # Strategy 8b: Reverse the conclusion
        if s != o:  # Avoid self-loops
            variants.append((o, p, s))

        return variants[:3]  # Limit to avoid too many variants

    def _strategy_negate_premise(self,
                                body_triples: List[Triple],
                                correct_triple: Triple,
                                relation_ids: List[str]) -> List[Triple]:
        """
        Strategy 9: Negated Premise as Conclusion

        Generate distractors by modifying premises.
        If premise is (A, R, B), create distractor that changes it:
        - (A, different_R, B)  # Different relation for same entities
        - (A, R, different_entity)  # Different object

        This is logically wrong because we're concluding something that
        contradicts or differs from the premises.

        Example:
        - Premise: (Paris, locatedIn, France)
        - Correct conclusion: (Eiffel Tower, country, France)
        - Wrong: (Paris, capitalOf, France)  # Changed premise relation

        Args:
            body_triples: Premise triples
            correct_triple: The correct answer
            relation_ids: Available relation IDs

        Returns:
            List of distractor triples based on negated/modified premises
        """
        variants = []

        for premise_triple in body_triples:
            ps, pp, po = premise_triple

            # Don't duplicate the correct answer
            if premise_triple == correct_triple:
                continue

            # Strategy 9a: Use different relation for same entities from premise
            other_relations = [r for r in relation_ids if r != pp]
            for new_p in other_relations[:2]:  # Limit to 2 per premise
                variant = (ps, new_p, po)
                if variant != correct_triple:
                    variants.append(variant)

        return variants[:3]  # Limit to avoid too many

    def _strategy_type_mismatch(self,
                               correct_triple: Triple,
                               entity_ids: List[str],
                               relation_ids: List[str]) -> List[Triple]:
        """
        Strategy 10: Type Mismatch

        Generate distractors with type mismatches.
        Uses entity type information to create semantically incompatible triples.

        Example:
        - Correct: (Person, worksAt, University)
        - Wrong: (Person, worksAt, Date)  # Date is wrong type for worksAt
        - Wrong: (Building, authorOf, Book)  # Building can't be author

        Args:
            correct_triple: The correct answer (s, p, o)
            entity_ids: Available entity IDs
            relation_ids: Available relation IDs

        Returns:
            List of distractor triples with type mismatches
        """
        variants = []
        s, p, o = correct_triple

        # Get entity types from KG
        s_type = self._get_entity_type(s)
        o_type = self._get_entity_type(o)

        if not o_type:  # Can't do type checking without type info
            return variants

        # Find entities with incompatible types for the object position
        incompatible_objects = []
        for entity_id in entity_ids:
            if entity_id == s or entity_id == o:
                continue

            entity_type = self._get_entity_type(entity_id)
            # If types are very different, it's likely incompatible
            if entity_type and not self._are_compatible_types(entity_type, o_type):
                incompatible_objects.append(entity_id)

        # Generate variants with type-mismatched objects
        sorted_incompatible = sorted(incompatible_objects)
        seed = hash(f"type_{s}_{p}_{o}")

        for i in range(min(2, len(sorted_incompatible))):
            idx = (seed + i) % len(sorted_incompatible)
            new_o = sorted_incompatible[idx]
            variants.append((s, p, new_o))

        return variants

    def _get_entity_type(self, entity_id: str) -> Optional[str]:
        """
        Get the primary type of an entity by querying KG.

        In Wikidata: Query P31 (instance of) for the entity.

        Args:
            entity_id: Entity ID to get type for

        Returns:
            Entity type string if available, None otherwise
        """
        # Check cache first
        if entity_id in self._type_cache:
            return self._type_cache[entity_id]

        entity_type = None

        if self.kg and hasattr(self.kg, 'query'):
            try:
                # Query for instance of (P31 in Wikidata)
                # Get the most specific type (first result)
                query = """
                    MATCH (e:Entity {id: $entity_id})-[:HAS_PROPERTY {property_id: 'P31'}]->(type:Entity)
                    RETURN type.id as type_id, type.label as type_label
                    LIMIT 1
                """
                result = self.kg.query(query, entity_id=entity_id)

                if result and len(result) > 0:
                    entity_type = result[0].get('type_id')
                    logger.debug(f"Entity {entity_id} type: {entity_type}")

            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"Failed to query entity type for {entity_id}: {e}")
                entity_type = None

        # Cache the result
        self._type_cache[entity_id] = entity_type
        return entity_type

    def _get_type_ancestors(self, type_id: str, max_depth: int = 3) -> set:
        """
        Get ancestor types by following subclass hierarchy.

        In Wikidata: Follow P279 (subclass of) relationships.

        Args:
            type_id: Type ID to get ancestors for
            max_depth: Maximum depth to traverse

        Returns:
            Set of ancestor type IDs
        """
        # Check cache first
        if type_id in self._type_hierarchy_cache:
            return self._type_hierarchy_cache[type_id]

        ancestors = {type_id}  # Include self

        if self.kg and hasattr(self.kg, 'query') and max_depth > 0:
            try:
                # Query for subclass of (P279 in Wikidata)
                # Use iterative approach to limit depth
                current_level = {type_id}
                for depth in range(max_depth):
                    if not current_level:
                        break

                    # Query parents of current level
                    query = """
                        MATCH (t:Entity)-[:HAS_PROPERTY {property_id: 'P279'}]->(parent:Entity)
                        WHERE t.id IN $type_ids
                        RETURN DISTINCT parent.id as parent_id
                    """
                    result = self.kg.query(query, type_ids=list(current_level))

                    # Add parents to ancestors and prepare for next level
                    next_level = set()
                    for row in result:
                        parent_id = row.get('parent_id')
                        if parent_id and parent_id not in ancestors:
                            ancestors.add(parent_id)
                            next_level.add(parent_id)

                    current_level = next_level

                logger.debug(f"Type {type_id} ancestors: {ancestors}")

            except (AttributeError, TypeError, KeyError) as e:
                logger.debug(f"Failed to query type hierarchy for {type_id}: {e}")

        # Cache the result
        self._type_hierarchy_cache[type_id] = ancestors
        return ancestors

    def _are_compatible_types(self, type1: str, type2: str) -> bool:
        """
        Check if two types are compatible by querying type hierarchy.

        Two types are compatible if:
        1. They are the same
        2. One is an ancestor of the other
        3. They share a common close ancestor

        Args:
            type1: First type
            type2: Second type

        Returns:
            True if types are compatible (same or closely related)
        """
        if type1 == type2:
            return True

        # Get type hierarchies
        ancestors1 = self._get_type_ancestors(type1)
        ancestors2 = self._get_type_ancestors(type2)

        # Check if one is ancestor of the other
        if type1 in ancestors2 or type2 in ancestors1:
            return True

        # Check if they share common ancestors (besides very general ones)
        # Exclude very general types like Q35120 (entity), Q488383 (object)
        very_general_types = {'Q35120', 'Q488383', 'Q16889133'}
        common = (ancestors1 & ancestors2) - very_general_types

        # If they share specific common ancestors, consider compatible
        if common:
            logger.debug(f"Types {type1} and {type2} share ancestors: {common}")
            return True

        return False
