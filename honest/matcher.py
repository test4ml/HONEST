#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Positive example matcher module
Used to match positive example instances of inference rules
"""

from typing import List, Dict, Any, Tuple, Optional
import gc
from tqdm import tqdm
from .kg_interfaces import IndexedKnowledgeGraph
from .rule_parser import RuleParser
from honest.utils.profiling import profile


class PositiveExampleMatcher:
    """Positive example matcher, supports traditional triples and Wikidata format"""

    @profile
    def __init__(self, kg: IndexedKnowledgeGraph):
        self.kg = kg
        # Check whether it is a Wikidata graph
        self.is_wikidata = hasattr(kg, 'load_wikidata_files')

        # Cache pattern selectivity info, keyed by the full pattern
        self._pattern_selectivity_cache = {}

        # Enable query optimization
        self.enable_query_optimization = True

    @profile
    def match_rule(self, rule_str: str) -> List[Dict[str, Any]]:
        """Match positive examples for the given rule"""
        try:
            rule = RuleParser.parse_rule(rule_str)
            body_patterns = [fact.to_tuple() for fact in rule.body]
            head_pattern = rule.head.to_tuple()
        except ValueError as e:
            print(f"Failed to parse rule: {e}")
            return []

        positive_examples = []

        # Get variable bindings
        variables = self._extract_variables(body_patterns + [head_pattern])

        # Use an optimized depth-first search to find variable bindings that satisfy all body conditions
        max_bindings = 1000  # Maximum binding count limit

        if self.enable_query_optimization and len(body_patterns) > 1:
            all_bindings = self._find_all_valid_bindings_optimized(body_patterns, max_bindings)
        else:
            all_bindings = self._find_all_valid_bindings(body_patterns, max_bindings)

        # For each valid binding, check whether the head is also satisfied
        for binding in tqdm(all_bindings, desc="Verifying rule matches", unit="bindings", leave=False):
            head_instantiated = self._instantiate_pattern(head_pattern, binding)
            if self._pattern_exists(head_instantiated):
                # Instantiate all body conditions
                body_triples = []
                for pattern in body_patterns:
                    body_triples.append(self._instantiate_pattern(pattern, binding))

                example = {
                    'variables': binding,
                    'body_triples': body_triples,
                    'head_triple': head_instantiated
                }

                # If it is Wikidata, add extra context info
                if self.is_wikidata:
                    example.update(self._get_wikidata_context(head_instantiated, body_triples))

                positive_examples.append(example)

        return positive_examples

    @profile
    def _find_all_valid_bindings(self, body_patterns: List[Tuple[str, str, str]], max_results: int = 1000) -> List[Dict[str, str]]:
        """Find variable bindings that satisfy all body conditions"""
        if not body_patterns:
            return []

        # Recursively build bindings that satisfy all conditions
        return self._build_bindings(body_patterns, 0, {}, max_results)

    def _build_bindings(self, patterns: List[Tuple[str, str, str]], pattern_index: int, current_binding: Dict[str, str], max_results: int = 1000) -> List[Dict[str, str]]:
        """Recursively build variable bindings - supports a result count limit"""
        if pattern_index >= len(patterns):
            # All patterns processed, return current binding
            return [current_binding.copy()]

        current_pattern = patterns[pattern_index]
        valid_bindings = []

        # Partially instantiate the current pattern (using existing bindings)
        partially_instantiated = self._partially_instantiate_pattern(current_pattern, current_binding)

        # Get all triples matching this (possibly partially instantiated) pattern
        # Limit the result count directly at the query level to avoid memory waste
        max_candidates = min(10000, max_results * 10)
        candidates = self._find_candidates_for_pattern(partially_instantiated, limit=max_candidates)

        # For each candidate triple, try to extend the binding
        for candidate in candidates:
            # If enough results have been found, exit early
            if len(valid_bindings) >= max_results:
                break

            # Try to match the candidate triple with the pattern to produce a new variable binding
            new_binding = self._extend_binding(current_binding, current_pattern, candidate)
            if new_binding is not None:
                # Recursively process the next pattern
                remaining_results = max_results - len(valid_bindings)
                sub_bindings = self._build_bindings(patterns, pattern_index + 1, new_binding, remaining_results)
                valid_bindings.extend(sub_bindings)

        return valid_bindings

    def _estimate_pattern_selectivity(self, pattern: Tuple[str, str, str]) -> int:
        """Estimate the selectivity of a pattern (returns an estimate of the result count)"""
        subject, predicate, obj = pattern

        # Use the full pattern as the cache key
        cache_key = (subject, predicate, obj)

        if cache_key in self._pattern_selectivity_cache:
            return self._pattern_selectivity_cache[cache_key]

        # Query the materialized pattern directly via the count method
        try:
            # Prepare count query parameters
            count_subject = None if subject.startswith('?') else subject
            count_predicate = None if predicate.startswith('?') else predicate
            count_obj = None if obj.startswith('?') else obj

            # Query the count of the materialized pattern directly
            estimated_count = self.kg.count(count_subject, count_predicate, count_obj)

            # Cache the result
            self._pattern_selectivity_cache[cache_key] = estimated_count

            return estimated_count

        except (AttributeError, ValueError, RuntimeError, TypeError) as e:
            print(f"    Warning: count query failed for pattern {pattern}: {e}")
            # Return a conservative estimate on failure
            fallback_count = 1000 if not predicate.startswith('?') else 10000
            self._pattern_selectivity_cache[cache_key] = fallback_count
            return fallback_count

    def _find_all_valid_bindings_optimized(self, body_patterns: List[Tuple[str, str, str]], max_results: int = 1000) -> List[Dict[str, str]]:
        """Find variable bindings that satisfy all body conditions using an optimized query order"""
        if not body_patterns:
            return []

        # Estimate the selectivity of each pattern
        pattern_selectivity = []
        for i, pattern in enumerate(body_patterns):
            selectivity = self._estimate_pattern_selectivity(pattern)
            pattern_selectivity.append((selectivity, i, pattern))
            print(f"    Pattern {i+1}: {pattern} -> estimated result count: {selectivity:,}")

        # Sort by selectivity (high selectivity first, i.e. few results first)
        pattern_selectivity.sort(key=lambda x: x[0])
        optimized_order = [x[1] for x in pattern_selectivity]
        optimized_patterns = [x[2] for x in pattern_selectivity]

        print(f"    Optimized query order: {[i+1 for i in optimized_order]}")

        # Execute the query using the optimized order
        return self._build_bindings_optimized(optimized_patterns, 0, {}, optimized_order, max_results)

    def _build_bindings_optimized(self, patterns: List[Tuple[str, str, str]], pattern_index: int,
                                current_binding: Dict[str, str], original_order: List[int], max_results: int = 1000) -> List[Dict[str, str]]:
        """Recursively build variable bindings using the optimized order - supports a result count limit"""
        if pattern_index >= len(patterns):
            return [current_binding.copy()]

        current_pattern = patterns[pattern_index]
        valid_bindings = []

        # Partially instantiate the current pattern
        partially_instantiated = self._partially_instantiate_pattern(current_pattern, current_binding)

        # Get matching candidate triples
        # Limit the result count directly at the query level
        max_candidates = min(10000, max_results * 10)
        candidates = self._find_candidates_for_pattern(partially_instantiated, limit=max_candidates)

        # Early pruning: if there are no candidates, return an empty list directly
        if not candidates:
            return []

        # If the query results were already capped, show a message
        if len(candidates) == max_candidates:
            print(f"      Pattern {pattern_index+1} produced too many candidates, already capped at {max_candidates:,} at the query level")

        # For each candidate triple, try to extend the binding
        for candidate in candidates:
            # If enough results have been found, exit early
            if len(valid_bindings) >= max_results:
                break

            new_binding = self._extend_binding(current_binding, current_pattern, candidate)
            if new_binding is not None:
                # Recursively process the next pattern
                remaining_results = max_results - len(valid_bindings)
                sub_bindings = self._build_bindings_optimized(patterns, pattern_index + 1, new_binding, original_order, remaining_results)
                valid_bindings.extend(sub_bindings)

        return valid_bindings

    def _partially_instantiate_pattern(self, pattern: Tuple[str, str, str], binding: Dict[str, str]) -> Tuple[str, str, str]:
        """Partially instantiate a pattern (only replace already-bound variables)"""
        result = []
        for element in pattern:
            if element.startswith('?') and element in binding:
                result.append(binding[element])
            else:
                result.append(element)
        return tuple(result)

    def _find_candidates_for_pattern(self, pattern: Tuple[str, str, str], limit: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """Find candidate triples matching the (partially instantiated) pattern"""
        subject, predicate, obj = pattern

        # Build query parameters
        query_subject = None if subject.startswith('?') else subject
        query_predicate = None if predicate.startswith('?') else predicate
        query_obj = None if obj.startswith('?') else obj

        return self.kg.get_triples_by_pattern(
            subject=query_subject,
            predicate=query_predicate,
            obj=query_obj,
            limit=limit  # Pass the limit parameter directly to the underlying query
        )

    def _extend_binding(self, current_binding: Dict[str, str], pattern: Tuple[str, str, str], triple: Tuple[str, str, str]) -> Optional[Dict[str, str]]:
        """Try to extend the current binding with a triple"""
        new_binding = current_binding.copy()
        subject_var, predicate_var, obj_var = pattern
        triple_subject, triple_predicate, triple_obj = triple

        # Check and bind the subject
        if subject_var.startswith('?'):
            if subject_var in new_binding:
                # Variable already bound, check for consistency
                if new_binding[subject_var] != triple_subject:
                    return None
            else:
                new_binding[subject_var] = triple_subject
        elif subject_var != triple_subject:
            return None

        # Check and bind the predicate
        if predicate_var.startswith('?'):
            if predicate_var in new_binding:
                if new_binding[predicate_var] != triple_predicate:
                    return None
            else:
                new_binding[predicate_var] = triple_predicate
        elif predicate_var != triple_predicate:
            return None

        # Check and bind the object
        if obj_var.startswith('?'):
            if obj_var in new_binding:
                if new_binding[obj_var] != triple_obj:
                    return None
            else:
                new_binding[obj_var] = triple_obj
        elif obj_var != triple_obj:
            return None

        return new_binding

    def _extract_variables(self, patterns: List[Tuple[str, str, str]]) -> List[str]:
        """Extract all variables"""
        variables = set()
        for pattern in patterns:
            for element in pattern:
                if element.startswith('?'):
                    variables.add(element)
        return list(variables)

    def _instantiate_pattern(self, pattern: Tuple[str, str, str], binding: Dict[str, str]) -> Tuple[str, str, str]:
        """Instantiate a pattern according to variable bindings"""
        result = []
        for element in pattern:
            if element.startswith('?') and element in binding:
                result.append(binding[element])
            else:
                result.append(element)
        return tuple(result)

    def _pattern_exists(self, triple: Tuple[str, str, str]) -> bool:
        """Check whether a triple exists in the knowledge graph"""
        s, p, o = triple
        result = self.kg.get_triples_by_pattern(subject=s, predicate=p, obj=o)
        return len(result) > 0

    def _get_wikidata_context(self, head_triple: Tuple[str, str, str], body_triples: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Get Wikidata-specific context info"""
        context = {}

        try:
            # Collect all involved entity QIDs
            entities = set()
            properties = set()

            # Extract entities and properties from the head and body triples
            for triple in [head_triple] + body_triples:
                s, p, o = triple
                if s.startswith('Q'):
                    entities.add(s)
                if o.startswith('Q'):
                    entities.add(o)
                if p.startswith('P'):
                    properties.add(p)

            # Get entity labels (if any)
            entity_labels = {}
            if hasattr(self.kg, 'conn'):  # Ensure it is a Wikidata graph
                for entity in list(entities)[:10]:  # Limit the query count
                    try:
                        cursor = self.kg.conn.execute(
                            "SELECT label FROM entity_info WHERE qid = ? LIMIT 1",
                            (entity,)
                        )
                        result = cursor.fetchone()
                        if result:
                            entity_labels[entity] = result[0]
                    except (AttributeError, TypeError, ValueError):
                        pass

            # Get property labels
            property_labels = {}
            if hasattr(self.kg, 'conn'):
                for prop in list(properties)[:10]:  # Limit the query count
                    try:
                        cursor = self.kg.conn.execute(
                            "SELECT label FROM properties WHERE property_id = ? LIMIT 1",
                            (prop,)
                        )
                        result = cursor.fetchone()
                        if result:
                            property_labels[prop] = result[0]
                    except (AttributeError, TypeError, ValueError):
                        pass

            if entity_labels:
                context['entity_labels'] = entity_labels
            if property_labels:
                context['property_labels'] = property_labels

        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # If getting context fails, do not affect the main functionality
            pass

        return context
