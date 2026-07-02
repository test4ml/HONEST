#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distractor generation engine for multiple choice questions"""
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

from ..types import QuestionType
from ..relation_negation_config import relation_negation_manager, inverse_relation_registry
from ...rule_parser import LogicExpression, Fact
from ..semantic_validator import semantic_validator

# Type aliases
TriplePattern = Tuple[str, str, str]
BodyPatterns = List[TriplePattern]
EntityLabels = Dict[str, str]
RelationLabels = Dict[str, str]


class DistractorEngine:
    """Engine for generating distractor options for multiple choice questions"""

    def __init__(self, question_generator):
        """Initialize distractor engine with reference to parent QuestionGenerator

        Args:
            question_generator: Reference to the parent QuestionGenerator instance
                              to access shared resources like kg, validators, etc.
        """
        self.qg = question_generator

        # Cache for relation-specific entity pools
        # Key: relation_id (e.g., 'P6')
        # Value: {'subjects': set of entity IDs, 'objects': set of entity IDs}
        # This helps generate semantically reasonable distractors
        self._relation_entity_cache: Dict[str, Dict[str, set]] = {}

    def generate_distractors_deterministic(self,
                                          body_patterns: BodyPatterns,
                                          head_pattern: TriplePattern,
                                          entity_labels: EntityLabels,
                                          relation_labels: RelationLabels,
                                          num_distractors: int) -> List[str]:
        """Generate deterministic distractor options using KG validation

        Pipeline:
        1. Generate negation of head as first distractor (guaranteed false)
        1.5. Generate negation of premises as additional distractors (guaranteed false)
        2. Prepare entity/relation pool
        3. Generate candidate triples
        4. Validate candidates against KG (Closed World Assumption)
        5. Convert to text and deduplicate
        6. Generate generic distractors if needed

        Args:
            body_patterns: Premise triples
            head_pattern: Conclusion triple
            entity_labels: Entity ID to label mapping
            relation_labels: Relation ID to label mapping
            num_distractors: Number of distractors to generate

        Returns:
            List of distractor statements
        """
        correct_conclusion = self.qg._format_conclusion(head_pattern, entity_labels, relation_labels)
        premises_statements = self.get_all_premise_statements(body_patterns, entity_labels, relation_labels)
        augmented_features = self.extract_augmented_features(body_patterns, head_pattern)

        # Step 1: Generate negation of head as first distractor (guaranteed to be false)
        valid_distractors = []
        seen_distractors = set()

        negation_distractor = self.negate_conclusion(
            head_pattern, entity_labels, relation_labels, body_patterns
        )
        if negation_distractor and negation_distractor != correct_conclusion:
            if not self.is_duplicate_of_premise(negation_distractor, premises_statements):
                valid_distractors.append(negation_distractor)
                seen_distractors.add(negation_distractor)
                logger.debug(f"Head negation distractor added: {negation_distractor[:60]}...")

        # Step 1.5: Generate negation of premises as additional distractors (also guaranteed to be false)
        # Handle LogicExpression format
        if isinstance(body_patterns, LogicExpression):
            facts = self.qg._extract_all_facts(body_patterns)
            premise_patterns = [fact.to_tuple() for fact in facts]
        else:
            premise_patterns = body_patterns

        # Generate negation for each premise
        for premise_pattern in premise_patterns:
            premise_negation = self.negate_conclusion(
                premise_pattern, entity_labels, relation_labels, body_patterns
            )
            if premise_negation and premise_negation != correct_conclusion:
                if premise_negation not in seen_distractors:
                    valid_distractors.append(premise_negation)
                    seen_distractors.add(premise_negation)
                    logger.debug(f"Premise negation distractor added: {premise_negation[:60]}...")

        # Step 2: Prepare entity and relation pools
        # Extract entity/relation IDs from patterns (not from label keys)
        # This ensures consistency across mutations: only entities actually used in
        # the patterns are included, not entities from instantiation mappings
        entity_ids = []
        relation_ids = []

        # Extract from body patterns and cache relation-entity pairs
        if isinstance(body_patterns, list):
            for s, p, o in body_patterns:
                # Collect entities
                if s.startswith('Q') and s not in entity_ids:
                    entity_ids.append(s)
                if o.startswith('Q') and o not in entity_ids:
                    entity_ids.append(o)
                if p.startswith('P') and p not in relation_ids:
                    relation_ids.append(p)

                # Cache relation-entity pairs
                if p.startswith('P'):
                    if p not in self._relation_entity_cache:
                        self._relation_entity_cache[p] = {'subjects': set(), 'objects': set()}
                    if s.startswith('Q'):
                        self._relation_entity_cache[p]['subjects'].add(s)
                    if o.startswith('Q'):
                        self._relation_entity_cache[p]['objects'].add(o)

        # Extract from head pattern and cache
        s, p, o = head_pattern
        if s.startswith('Q') and s not in entity_ids:
            entity_ids.append(s)
        if o.startswith('Q') and o not in entity_ids:
            entity_ids.append(o)
        if p.startswith('P') and p not in relation_ids:
            relation_ids.append(p)

        # Cache head relation-entity pairs
        if p.startswith('P'):
            if p not in self._relation_entity_cache:
                self._relation_entity_cache[p] = {'subjects': set(), 'objects': set()}
            if s.startswith('Q'):
                self._relation_entity_cache[p]['subjects'].add(s)
            if o.startswith('Q'):
                self._relation_entity_cache[p]['objects'].add(o)

        # Add augmented entities/relations from body patterns
        if augmented_features['relations']:
            for relation_id in augmented_features['relations']:
                if relation_id not in relation_labels:
                    relation_labels[relation_id] = self.qg._get_relation_label_from_kg(relation_id)
            relation_ids = list(set(relation_ids + list(augmented_features['relations'])))
        if augmented_features['entities']:
            for entity_id in augmented_features['entities']:
                if entity_id not in entity_labels:
                    entity_labels[entity_id] = self.qg._get_entity_label_from_kg(entity_id)
            entity_ids = list(set(entity_ids + list(augmented_features['entities'])))

        # Step 3: Generate candidate triples
        # OPTIMIZATION: Reduced from 30x to 10x (most candidates get filtered anyway)
        candidate_triples = self.qg.distractor_triple_generator.generate_candidates(
            correct_triple=head_pattern,
            body_triples=body_patterns,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            num_candidates=num_distractors * 10  # Reduced from 30 to 10
        )
        logger.debug(f"Generated {len(candidate_triples)} candidate triples")

        # Step 4: Validate candidates using KG (Closed World Assumption + self-reference check)
        validated_triples = self.qg.distractor_validator.filter_valid_distractors(
            candidate_triples=candidate_triples,
            correct_triple=head_pattern,
            body_triples=body_patterns,
            entity_labels=entity_labels
        )
        logger.debug(f"Validated {len(validated_triples)} triples after KG validation")

        # Step 5: Convert to text and deduplicate
        candidate_distractors = []
        for triple in validated_triples:
            text = self.triple_to_text(triple, entity_labels, relation_labels)
            if text and text != correct_conclusion and text not in seen_distractors:
                if not self.is_duplicate_of_premise(text, premises_statements):
                    candidate_distractors.append(text)
                    seen_distractors.add(text)

        # Add candidate distractors to valid list (after negation distractor)
        valid_distractors.extend(candidate_distractors)
        logger.debug(f"After text conversion: {len(valid_distractors)} valid distractors")

        # Step 6: Generate generic distractors if needed
        generic_attempt = 0
        max_generic_attempts = 50

        while len(valid_distractors) < num_distractors and generic_attempt < max_generic_attempts:
            generic_distractor = self.generate_validated_generic_distractor(
                entity_labels, relation_labels, body_patterns, head_pattern, generic_attempt
            )
            generic_attempt += 1

            # Filter out invalid generic distractors
            if not generic_distractor or generic_distractor == correct_conclusion:
                continue
            if self.is_duplicate_of_premise(generic_distractor, premises_statements):
                continue
            if generic_distractor in seen_distractors:
                continue

            seen_distractors.add(generic_distractor)
            valid_distractors.append(generic_distractor)
            logger.debug(f"Generic distractor added: {generic_distractor[:60]}... ({len(valid_distractors)}/{num_distractors})")

        return valid_distractors[:num_distractors]

    def triple_to_text(self,
                       triple: TriplePattern,
                       entity_labels: EntityLabels,
                       relation_labels: RelationLabels) -> Optional[str]:
        """Convert a triple to natural language text."""
        s, p, o = triple
        s_label = entity_labels.get(s, self.qg._get_entity_label_from_kg(s))
        o_label = entity_labels.get(o, self.qg._get_entity_label_from_kg(o))
        p_label = relation_labels.get(p, self.qg._get_relation_label_from_kg(p))

        if not s_label or not o_label or not p_label:
            return None

        return self.qg._format_statement(s_label, p_label, o_label, p)

    def generate_validated_generic_distractor(self,
                                              entity_labels: EntityLabels,
                                              relation_labels: RelationLabels,
                                              body_patterns: BodyPatterns,
                                              head_pattern: TriplePattern,
                                              index: int) -> Optional[str]:
        """Generate generic distractor by replacing entities from cache

        Strategy:
        - For conclusion A P B, replace A or B with cached entities
        - Only use entities from cache that differ from current entities
        - Ensures distractors use semantically plausible entity types

        Example:
        - Conclusion: United States P6 Joe Biden
        - Cache: P6 subjects={United States, France, UK}, objects={Joe Biden, Macron}
        - Replace subject: France P6 Joe Biden (France ≠ United States)
        - Replace object: United States P6 Macron (Macron ≠ Joe Biden)
        - Replace both: France P6 Macron

        Args:
            entity_labels: Entity ID to label mapping
            relation_labels: Relation ID to label mapping
            body_patterns: Premise triples
            head_pattern: Conclusion triple
            index: Deterministic index for entity selection

        Returns:
            Generated distractor statement, or None if no cached entities available
        """
        head_s, head_p, head_o = head_pattern
        relation_label = relation_labels.get(head_p, head_p)

        # Must use cached entities only
        if head_p not in self._relation_entity_cache:
            return None

        cache = self._relation_entity_cache[head_p]

        # Filter out current entities - only use cached entities that differ
        cached_subjects = sorted([s for s in cache['subjects'] if s != head_s and s != head_o])
        cached_objects = sorted([o for o in cache['objects'] if o != head_s and o != head_o])

        if len(cached_subjects) == 0 and len(cached_objects) == 0:
            return None

        # Try 3 strategies deterministically based on index
        strategy = index % 3

        # Strategy 1: Replace subject only (if possible)
        if strategy == 0 and len(cached_subjects) > 0:
            subj_idx = (index // 3) % len(cached_subjects)
            new_subject = cached_subjects[subj_idx]

            # Keep original object
            subject_label = entity_labels.get(new_subject, new_subject)
            object_label = entity_labels.get(head_o, head_o)

            return self.qg._format_statement(subject_label, relation_label, object_label, head_p)

        # Strategy 2: Replace object only (if possible)
        if strategy == 1 and len(cached_objects) > 0:
            obj_idx = (index // 3) % len(cached_objects)
            new_object = cached_objects[obj_idx]

            # Keep original subject
            subject_label = entity_labels.get(head_s, head_s)
            object_label = entity_labels.get(new_object, new_object)

            return self.qg._format_statement(subject_label, relation_label, object_label, head_p)

        # Strategy 3: Replace both (if possible)
        if strategy == 2 and len(cached_subjects) > 0 and len(cached_objects) > 0:
            subj_idx = (index // 3) % len(cached_subjects)
            obj_idx = ((index // 3) + 1) % len(cached_objects)

            new_subject = cached_subjects[subj_idx]
            new_object = cached_objects[obj_idx]

            # Skip self-reference
            if new_subject == new_object:
                obj_idx = (obj_idx + 1) % len(cached_objects)
                new_object = cached_objects[obj_idx]

                if new_subject == new_object:
                    return None

            subject_label = entity_labels.get(new_subject, new_subject)
            object_label = entity_labels.get(new_object, new_object)

            return self.qg._format_statement(subject_label, relation_label, object_label, head_p)

        # Fallback: try other strategies if current one failed
        for fallback_strategy in range(3):
            if fallback_strategy == strategy:
                continue

            if fallback_strategy == 0 and len(cached_subjects) > 0:
                subj_idx = (index // 3) % len(cached_subjects)
                new_subject = cached_subjects[subj_idx]
                subject_label = entity_labels.get(new_subject, new_subject)
                object_label = entity_labels.get(head_o, head_o)
                return self.qg._format_statement(subject_label, relation_label, object_label, head_p)

            if fallback_strategy == 1 and len(cached_objects) > 0:
                obj_idx = (index // 3) % len(cached_objects)
                new_object = cached_objects[obj_idx]
                subject_label = entity_labels.get(head_s, head_s)
                object_label = entity_labels.get(new_object, new_object)
                return self.qg._format_statement(subject_label, relation_label, object_label, head_p)

            if fallback_strategy == 2 and len(cached_subjects) > 0 and len(cached_objects) > 0:
                subj_idx = (index // 3) % len(cached_subjects)
                obj_idx = ((index // 3) + 1) % len(cached_objects)
                new_subject = cached_subjects[subj_idx]
                new_object = cached_objects[obj_idx]

                if new_subject != new_object:
                    subject_label = entity_labels.get(new_subject, new_subject)
                    object_label = entity_labels.get(new_object, new_object)
                    return self.qg._format_statement(subject_label, relation_label, object_label, head_p)

        return None

    def get_all_premise_statements(self,
                                   body_patterns: BodyPatterns,
                                   entity_labels: EntityLabels,
                                   relation_labels: RelationLabels) -> List[str]:
        """Get all premise statements from the premises."""
        premises_statements = []

        # Handle LogicExpression format
        if isinstance(body_patterns, LogicExpression):
            facts = self.qg._extract_all_facts(body_patterns)
            patterns = [fact.to_tuple() for fact in facts]
        else:
            patterns = body_patterns

        for s, p, o in patterns:
            s_label = entity_labels.get(s, s)
            o_label = entity_labels.get(o, o)
            p_label = relation_labels.get(p, p)
            statement = self.qg._format_statement(s_label, p_label, o_label, p)
            premises_statements.append(statement)
        return premises_statements

    def is_duplicate_of_premise(self, statement: str, premises_statements: List[str]) -> bool:
        """Check if statement duplicates any premise

        OPTIMIZATION: Semantic similarity filtering temporarily disabled for performance.
        Only checks for exact text match.
        """
        # TEMPORARY: Disable semantic similarity filtering for performance
        # TODO: Re-enable with optimized implementation (batched inference, caching)

        # Only check for exact text match
        for premise_statement in premises_statements:
            if statement.strip().lower() == premise_statement.strip().lower():
                logger.debug(f"Statement exactly duplicates premise (exact match)")
                logger.debug(f"  Statement: {statement[:80]}...")
                logger.debug(f"  Premise: {premise_statement[:80]}...")
                return True
        return False

        # ORIGINAL CODE (disabled for performance):
        # for premise_statement in premises_statements:
        #     similarity = semantic_validator.compute_similarity(statement, premise_statement)
        #     has_inverse_relations = self._contains_inverse_relations(statement, premise_statement)
        #     threshold = 0.995 if has_inverse_relations else semantic_validator.similarity_threshold
        #     if similarity >= threshold:
        #         logger.debug(f"Statement duplicate of premise (similarity={similarity:.4f} >= {threshold})")
        #         return True
        # return False

    def _contains_inverse_relations(self, statement1: str, statement2: str) -> bool:
        """Check if two statements contain words from inverse relation pairs

        Returns True if the statements likely use inverse relations
        (e.g., one uses 'upstream' and the other 'downstream')
        """
        # Get all known inverse relation pairs
        keywords_by_relation = {
            # Extract keywords from relation labels
            "P2673": ["upstream"],
            "P2674": ["downstream"],
            "P3729": ["lower"],
            "P3730": ["higher"],
            "P40": ["child", "children"],
            "P22": ["father"],
            "P25": ["mother"],
        }

        # Check if statements contain keywords from different relations in an inverse pair
        for rel1, rel2 in [("P2673", "P2674"), ("P3729", "P3730"), ("P40", "P22"), ("P40", "P25")]:
            keywords1 = keywords_by_relation.get(rel1, [])
            keywords2 = keywords_by_relation.get(rel2, [])

            statement1_lower = statement1.lower()
            statement2_lower = statement2.lower()

            # Check if statement1 has rel1 keywords and statement2 has rel2 keywords (or vice versa)
            has_rel1_in_s1 = any(kw in statement1_lower for kw in keywords1)
            has_rel2_in_s2 = any(kw in statement2_lower for kw in keywords2)
            has_rel2_in_s1 = any(kw in statement1_lower for kw in keywords2)
            has_rel1_in_s2 = any(kw in statement2_lower for kw in keywords1)

            if (has_rel1_in_s1 and has_rel2_in_s2) or (has_rel2_in_s1 and has_rel1_in_s2):
                return True

        return False

    def negate_conclusion(self,
                         head_pattern: TriplePattern,
                         entity_labels: EntityLabels,
                         relation_labels: RelationLabels,
                         body_patterns: Optional[BodyPatterns] = None) -> Optional[str]:
        """Negate the conclusion to generate a distractor (uses the configuration-driven negation system).

        IMPORTANT: For inverse relation pairs, return None to avoid generating
        a logically correct statement.
        """
        s, p, o = head_pattern
        s_label = entity_labels.get(s, s)
        o_label = entity_labels.get(o, o)
        p_label = relation_labels.get(p, p)

        # Check whether this is an inverse relation pair
        premise_relations = set()
        if body_patterns:
            for _, bp, _ in body_patterns:
                premise_relations.add(bp)

        if not inverse_relation_registry.is_negation_safe(p, premise_relations):
            logger.debug(
                f"Skipping negation for conclusion relation {p} - "
                f"inverse relation found in premises"
            )
            # Return a safe alternative: a positive statement with swapped
            # subject and object (rather than a negation)
            return self.qg._format_statement(o_label, p_label, s_label, p)

        # Use the configuration system to handle negation of special relations
        special_negation = relation_negation_manager.get_negation(
            property_id=p,
            property_label=p_label,
            subject_label=s_label,
            object_label=o_label,
            format_function=self.qg._format_statement
        )

        if special_negation:
            return special_negation

        # Default handling: build a positive statement, then use the intelligent negation transformer
        positive_statement = self.qg._format_statement(s_label, p_label, o_label, p)
        negative_statement = self.qg.negation_transformer.transform(positive_statement)

        return negative_statement

    def generate_negation_variants(self,
                                   head_pattern: TriplePattern,
                                   entity_labels: EntityLabels,
                                   relation_labels: RelationLabels,
                                   body_patterns: Optional[BodyPatterns] = None) -> List[str]:
        """Generate multiple distractor variants to supplement distractors when resources are scarce.

        When there are only two entities, triple-level strategies cannot generate
        enough distractors, so this method generates multiple text-level variants
        as a supplement.

        IMPORTANT: For inverse relation pairs (e.g. upstream/downstream), avoid
        using negation, because negating one direction is logically equivalent to
        affirming the other direction, which leads to multiple correct answers.
        """
        variants = []
        s, p, o = head_pattern
        s_label = entity_labels.get(s, s)
        o_label = entity_labels.get(o, o)
        p_label = relation_labels.get(p, p)

        # Extract relation IDs from the premises for the inverse relation check
        premise_relations = set()
        if body_patterns:
            for _, bp, _ in body_patterns:
                premise_relations.add(bp)

        # Check whether negation is safe to use
        negation_safe = inverse_relation_registry.is_negation_safe(p, premise_relations)

        if not negation_safe:
            # For inverse relation pairs, do not use negation; use safe alternatives instead
            logger.debug(
                f"Skipping negation variants for relation {p} - "
                f"inverse relation found in premises, using safe alternatives"
            )

            # Safe alternative strategy 1: use a relation from the premises as a false conclusion.
            # Example: premise is "B upstream A", conclusion is "A downstream B"
            # Distractor: "A upstream B" (using the premise relation as the conclusion is invalid reasoning)
            for bp in premise_relations:
                if bp != p:  # Use a relation different from the conclusion
                    bp_label = relation_labels.get(bp, bp)
                    variant = self.qg._format_statement(s_label, bp_label, o_label, bp)
                    if variant and variant not in variants:
                        variants.append(variant)

            # Safe alternative strategy 2: swap subject and object but keep the same relation (not a negation).
            # Example: correct is "A downstream B"
            # Distractor: "B downstream A" (wrong, because B is upstream)
            swapped = self.qg._format_statement(o_label, p_label, s_label, p)
            if swapped and swapped not in variants:
                variants.append(swapped)

            # Safe alternative strategy 3: use other available relations
            for rel_id, rel_label in relation_labels.items():
                if rel_id != p and rel_id not in premise_relations:
                    variant = self.qg._format_statement(s_label, rel_label, o_label, rel_id)
                    if variant and variant not in variants:
                        variants.append(variant)
                    if len(variants) >= 5:  # Limit the count
                        break

            # Safe alternative strategy 4: use mixed subject/object and relation combinations.
            # Example: swap subject/object + use a premise relation
            for bp in premise_relations:
                if bp != p:
                    bp_label = relation_labels.get(bp, bp)
                    # Swap subject/object + premise relation
                    variant = self.qg._format_statement(o_label, bp_label, s_label, bp)
                    if variant and variant not in variants:
                        variants.append(variant)

            # Safe alternative strategy 5: use a descriptive false statement.
            # "The relationship between A and B is not established through P"
            variant5 = f"The relationship between {s_label} and {o_label} is not established through {p_label}"
            if variant5 not in variants:
                variants.append(variant5)

            # Safe alternative strategy 6: "B has a different relationship with A"
            variant6 = f"{o_label} has a different relationship with {s_label}"
            if variant6 not in variants:
                variants.append(variant6)

            # Safe alternative strategy 7: "Neither A nor B has the P relationship"
            variant7 = f"Neither {s_label} nor {o_label} has the {p_label} relationship"
            if variant7 not in variants:
                variants.append(variant7)

            return variants

        # For cases without inverse-relation issues, negation can be used.
        # Variant 1: standard negation "A is not P of B"
        positive = self.qg._format_statement(s_label, p_label, o_label, p)
        neg1 = self.qg.negation_transformer.transform(positive)
        if neg1:
            variants.append(neg1)

        # Variant 2: reverse negation "B is not P of A" (swap subject/object)
        swapped_positive = self.qg._format_statement(o_label, p_label, s_label, p)
        neg2 = self.qg.negation_transformer.transform(swapped_positive)
        if neg2 and neg2 not in variants:
            variants.append(neg2)

        # Variant 3: "It is false that A is P of B"
        neg3 = f"It is false that {positive}"
        if neg3 not in variants:
            variants.append(neg3)

        # Variant 4: "A has no P relationship with B"
        neg4 = f"{s_label} has no {p_label} relationship with {o_label}"
        if neg4 not in variants:
            variants.append(neg4)

        # Variant 5: "B is not related to A via P"
        neg5 = f"{o_label} is not related to {s_label} via {p_label}"
        if neg5 not in variants:
            variants.append(neg5)

        return variants

    def extract_augmented_features(self,
                                   body_patterns: BodyPatterns,
                                   head_pattern: TriplePattern) -> Dict[str, set]:
        """Extract features from augmented body triples to enable body-aware distractor generation

        This method identifies which triples in the body are "augmented" (i.e., not strictly
        necessary for the head inference) and extracts their relations and entities for use
        in generating more diverse, context-aware distractors.

        Args:
            body_patterns: Body triple patterns (may be List[TriplePattern] or LogicExpression)
            head_pattern: Head triple pattern

        Returns:
            Dict with keys:
            - 'relations': set of relation IDs from all body triples
            - 'entities': set of entity IDs from all body triples
        """
        features = {
            'relations': set(),
            'entities': set()
        }

        # Extract all triples from body_patterns
        if isinstance(body_patterns, LogicExpression):
            facts = self.qg._extract_all_facts(body_patterns)
            triples = [fact.to_tuple() for fact in facts]
        else:
            triples = body_patterns

        # Extract all relations and entities from body triples
        for s, p, o in triples:
            features['relations'].add(p)
            features['entities'].add(s)
            features['entities'].add(o)

        # Remove head pattern entities and relation from augmented features
        # (they're already in the primary entity/relation pool)
        head_s, head_p, head_o = head_pattern
        features['relations'].discard(head_p)

        return features

