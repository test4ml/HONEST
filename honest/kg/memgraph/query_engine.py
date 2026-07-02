#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memgraph Knowledge Graph Query Engine Module

Query execution, statistics computation, and pattern matching functionality
for Memgraph-based knowledge graph.
"""

import time
import logging
from typing import List, Tuple, Optional, Dict, Any

try:
    from neo4j.exceptions import Neo4jError
except ImportError:
    # Fallback if neo4j is not installed
    Neo4jError = Exception

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Statistics and functionality measures
# -------------------------------------------------------------------------

def compute_functionality(kg, predicate: str) -> float:
    """Compute relation functionality (prefer precomputed cache, query when necessary)"""
    # Check cache (including precomputed data)
    if predicate in kg.functionality_cache:
        return kg.functionality_cache[predicate]

    # Not in cache, perform real-time computation
    driver = kg._get_driver()
    try:
        with driver.session() as session:
            # Use standard functionality query, avoid complex aggregation operations
            query = """
            MATCH (s:Entity)-[r:HAS_PROPERTY {property_id: $predicate}]->(o:Entity)
            WITH s, count(DISTINCT o) as obj_count
            RETURN avg(1.0/obj_count) as functionality
            """
            result = session.run(query, predicate=predicate)
            record = result.single()
            functionality_value = record["functionality"] if record and record["functionality"] else 0.0

            # Cache result
            kg.functionality_cache[predicate] = functionality_value
            logger.debug(f"Real-time compute functionality {predicate}: {functionality_value:.4f}")
            return functionality_value

    except Neo4jError as e:
        logger.error(f"Failed to compute functionality: {e}")
        # Cache failure result to avoid repeated queries
        kg.functionality_cache[predicate] = 0.0
        return 0.0


def compute_inverse_functionality(kg, predicate: str) -> float:
    """Compute relation inverse functionality (prefer precomputed cache, query when necessary)"""
    # Check cache (including precomputed data)
    if predicate in kg.inv_functionality_cache:
        return kg.inv_functionality_cache[predicate]

    # Not in cache, perform real-time computation
    driver = kg._get_driver()
    try:
        with driver.session() as session:
            # Use standard inverse functionality query, avoid complex aggregation operations
            query = """
            MATCH (s:Entity)-[r:HAS_PROPERTY {property_id: $predicate}]->(o:Entity)
            WITH o, count(DISTINCT s) as subj_count
            RETURN avg(1.0/subj_count) as inv_functionality
            """
            result = session.run(query, predicate=predicate)
            record = result.single()
            inv_functionality_value = record["inv_functionality"] if record and record["inv_functionality"] else 0.0

            # Cache result
            kg.inv_functionality_cache[predicate] = inv_functionality_value
            logger.debug(f"Real-time compute inverse functionality {predicate}: {inv_functionality_value:.4f}")
            return inv_functionality_value

    except Neo4jError as e:
        logger.error(f"Failed to compute inverse functionality: {e}")
        # Cache failure result to avoid repeated queries
        kg.inv_functionality_cache[predicate] = 0.0
        return 0.0


def batch_compute_functionality(kg, predicates: List[str]) -> Dict[str, Tuple[float, float]]:
    """Batch compute functionality and inverse functionality for multiple predicates (standard query + cache optimization)"""
    results = {}
    uncached_predicates = []

    # Check which predicates need computation
    for predicate in predicates:
        if predicate in kg.functionality_cache and predicate in kg.inv_functionality_cache:
            results[predicate] = (kg.functionality_cache[predicate], kg.inv_functionality_cache[predicate])
        else:
            uncached_predicates.append(predicate)

    if not uncached_predicates:
        return results

    # For uncached predicates, use separate computation to avoid complex batch queries
    for predicate in uncached_predicates:
        try:
            func_value = compute_functionality(kg, predicate)
            inv_func_value = compute_inverse_functionality(kg, predicate)
            results[predicate] = (func_value, inv_func_value)
            logger.debug(f"Compute predicate {predicate}: func={func_value:.3f}, inv_func={inv_func_value:.3f}")
        except Neo4jError as e:
            logger.error(f"Failed to compute predicate {predicate}: {e}")
            # Set default values
            kg.functionality_cache[predicate] = 0.0
            kg.inv_functionality_cache[predicate] = 0.0
            results[predicate] = (0.0, 0.0)

    return results


def precompute_common_functionality(kg, top_n: int = 100):
    """Precompute functionality for most common predicates (startup optimization)"""
    driver = kg._get_driver()
    try:
        with driver.session() as session:
            # Get most common predicates
            query = """
            MATCH (s:Entity)-[r:HAS_PROPERTY]->(o:Entity)
            WITH r.property_id as pred, count(*) as usage_count
            ORDER BY usage_count DESC
            LIMIT $top_n
            RETURN pred
            """
            result = session.run(query, top_n=top_n)
            common_predicates = [record["pred"] for record in result]

            if common_predicates:
                logger.info(f"Precomputing functionality for top {len(common_predicates)} common predicates...")
                batch_compute_functionality(kg, common_predicates)
                logger.info(f"Precomputation complete, cached functionality for {len(common_predicates)} predicates")

    except Neo4jError as e:
        logger.error(f"Failed to precompute functionality: {e}")


def clear_functionality_cache(kg):
    """Clear functionality cache"""
    func_count = len(kg.functionality_cache)
    inv_func_count = len(kg.inv_functionality_cache)

    kg.functionality_cache.clear()
    kg.inv_functionality_cache.clear()

    total_cleared = func_count + inv_func_count
    logger.info(f"Cleared {total_cleared} functionality cache entries (functionality: {func_count}, inverse functionality: {inv_func_count})")


def clear_relations_entities_cache(kg):
    """Clear relations and entities cache"""
    relations_count = len(kg.relations_cache) if kg.relations_cache else 0
    entities_count = len(kg.entities_cache) if kg.entities_cache else 0

    kg.relations_cache = None
    kg.entities_cache = None

    logger.info(f"Cleared relations and entities cache (relations: {relations_count}, entities: {entities_count})")


def get_functionality_cache_stats(kg) -> Dict[str, int]:
    """Get functionality cache statistics"""
    return {
        'functionality_cached': len(kg.functionality_cache),
        'inverse_functionality_cached': len(kg.inv_functionality_cache),
        'total_cache_size': len(kg.functionality_cache) + len(kg.inv_functionality_cache),
        'relations_cached': len(kg.relations_cache) if kg.relations_cache else 0,
        'entities_cached': len(kg.entities_cache) if kg.entities_cache else 0
    }


def is_functional(kg, predicate: str) -> bool:
    """Check if relation is functional"""
    return compute_functionality(kg, predicate) > 0.95


def relation_size(kg, predicate: str) -> int:
    """Get number of triples in relation"""
    return kg.count_triples(predicate=predicate)


def relation_column_size(kg, predicate: str, column: int) -> int:
    """Get number of distinct entities in specified column of relation"""
    driver = kg._get_driver()
    try:
        with driver.session() as session:
            if column == 0:  # Subject
                query = """
                MATCH (s:Entity)-[r:HAS_PROPERTY {property_id: $predicate}]->(o:Entity)
                RETURN count(DISTINCT s) as count
                """
            elif column == 2:  # Object
                query = """
                MATCH (s:Entity)-[r:HAS_PROPERTY {property_id: $predicate}]->(o:Entity)
                RETURN count(DISTINCT o) as count
                """
            else:
                return 0

            result = session.run(query, predicate=predicate)
            record = result.single()
            return record["count"] if record else 0
    except Neo4jError as e:
        logger.error(f"Failed to compute relation column size: {e}")
        return 0


def overlap(kg, predicate1: str, predicate2: str, overlap_type: int) -> int:
    """Compute overlap between two relations"""
    # Simplified implementation
    if overlap_type == 0:  # Subject overlap
        entities1 = kg.select_distinct(0, [(None, predicate1, None)])
        entities2 = kg.select_distinct(0, [(None, predicate2, None)])
    elif overlap_type == 2:  # Object overlap
        entities1 = kg.select_distinct(2, [(None, predicate1, None)])
        entities2 = kg.select_distinct(2, [(None, predicate2, None)])
    else:
        return 0

    return len(entities1.intersection(entities2))


# -------------------------------------------------------------------------
# Statistics information
# -------------------------------------------------------------------------

def get_stats(kg) -> Dict[str, Any]:
    """Get knowledge graph statistics (ultra-fast version)"""
    from honest.utils.profiling import profile

    @profile
    def _get_stats_internal():
        if kg.stats_cache:
            return kg.stats_cache

        driver = kg._get_driver()
        try:
            with driver.session() as session:
                # Use minimal statistics, avoid time-consuming queries
                stats = {
                    'engine_type': 'MemgraphKnowledgeGraph',
                    'database_type': 'Memgraph',
                    'query_language': 'Cypher',
                    'status': 'connected',
                    **kg.metadata.get_stats()
                }

                # Optional: only do sampling statistics when needed
                try:
                    # Ultra-fast sampling: only query minimal data
                    sample_result = session.run("""
                        MATCH (n:Entity)
                        RETURN count(n) as entity_count
                        LIMIT 1
                    """)
                    entity_sample = sample_result.single()
                    if entity_sample:
                        stats['entity_sample_available'] = True
                        stats['connection_verified'] = True
                    else:
                        stats['entity_sample_available'] = False

                except Neo4jError as sample_error:
                    stats['entity_sample_available'] = False
                    stats['sample_error'] = str(sample_error)

                # Add quick availability check
                stats['database_accessible'] = True
                stats['query_ready'] = True

                kg.stats_cache = stats
                return stats

        except Neo4jError as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'error': str(e),
                'engine_type': 'MemgraphKnowledgeGraph',
                'status': 'connection_failed',
                'database_accessible': False
            }

    return _get_stats_internal()


# -------------------------------------------------------------------------
# Memgraph-specific advanced query methods
# -------------------------------------------------------------------------

def execute_cypher(kg, query: str, **params) -> List[Dict[str, Any]]:
    """Execute custom Cypher query"""
    driver = kg._get_driver()
    try:
        with driver.session() as session:
            result = session.run(query, **params)
            records = []
            for record in result:
                records.append(dict(record))
            return records
    except Neo4jError as e:
        logger.error(f"Cypher query failed: {e}")
        return []


def query(kg, query_str: str, **params) -> List[Dict[str, Any]]:
    """Execute query (alias of execute_cypher)"""
    return execute_cypher(kg, query_str, **params)


def match_rule_patterns(kg, body_patterns: List[Tuple], head_pattern: Tuple,
                       max_examples: int = 100) -> List[Dict[str, Any]]:
    """Match rule patterns, return positive instances - fixed version"""
    driver = kg._get_driver()
    try:
        with driver.session() as session:
            # Build Cypher query
            query_parts = []
            where_conditions = []
            variable_mapping = {}  # Variable to Cypher node mapping
            params = {}

            def get_cypher_node(entity_or_var: str) -> str:
                """Map entity or variable to Cypher node variable"""
                if entity_or_var.startswith('?'):
                    # Variable - ensure same variable uses same node
                    if entity_or_var not in variable_mapping:
                        variable_mapping[entity_or_var] = f"n{len(variable_mapping)}"
                    return variable_mapping[entity_or_var]
                else:
                    # Concrete entity, create unique node
                    node_var = f"e_{abs(hash(entity_or_var)) % 10000}"
                    where_conditions.append(f"{node_var}.id = $entity_{node_var}")
                    params[f"entity_{node_var}"] = entity_or_var
                    return node_var

            # Process body patterns
            for i, (s, p, o) in enumerate(body_patterns):
                s_node = get_cypher_node(s)
                o_node = get_cypher_node(o)

                rel_var = f"r_body_{i}"
                if p.startswith('?'):
                    # Variable property
                    query_parts.append(f"MATCH ({s_node}:Entity)-[{rel_var}:HAS_PROPERTY]->({o_node}:Entity)")
                    if p not in variable_mapping:
                        variable_mapping[p] = f"{rel_var}.property_id"
                else:
                    # Concrete property
                    query_parts.append(f"MATCH ({s_node}:Entity)-[{rel_var}:HAS_PROPERTY]->({o_node}:Entity)")
                    where_conditions.append(f"{rel_var}.property_id = $prop_body_{i}")
                    params[f"prop_body_{i}"] = p

            # Process head pattern
            if head_pattern:
                s, p, o = head_pattern
                s_node = get_cypher_node(s)
                o_node = get_cypher_node(o)

                rel_var = f"r_head"
                if p.startswith('?'):
                    query_parts.append(f"MATCH ({s_node}:Entity)-[{rel_var}:HAS_PROPERTY]->({o_node}:Entity)")
                    if p not in variable_mapping:
                        variable_mapping[p] = f"{rel_var}.property_id"
                else:
                    query_parts.append(f"MATCH ({s_node}:Entity)-[{rel_var}:HAS_PROPERTY]->({o_node}:Entity)")
                    where_conditions.append(f"{rel_var}.property_id = $prop_head")
                    params["prop_head"] = p

            # Build return clause - return all variable values
            return_items = []
            for var, cypher_ref in variable_mapping.items():
                if var.startswith('?'):
                    var_name = var[1:]  # Remove ? prefix
                    if '.' in cypher_ref:
                        # Property variable
                        return_items.append(f"{cypher_ref} as {var_name}")
                    else:
                        # Entity variable
                        return_items.append(f"{cypher_ref}.id as {var_name}")

            # Assemble full query
            full_query = "\n".join(query_parts)
            if where_conditions:
                full_query += f"\nWHERE {' AND '.join(where_conditions)}"

            if return_items:
                full_query += f"\nRETURN {', '.join(return_items)}"
            else:
                # If no variables, only return match indicator
                full_query += f"\nRETURN 1 as match_found"

            # Add LIMIT to improve performance
            full_query += f"\nLIMIT {max_examples}"

            logger.debug(f"Execute Cypher query: {full_query}")
            logger.debug(f"Parameters: {params}")

            # Execute query
            start_time = time.time()
            result = session.run(full_query, **params)
            end_time = time.time()

            query_time = end_time - start_time
            logger.debug(f"Query execution time: {query_time*1000:.2f} ms")

            examples = []
            for record in result:
                if 'match_found' in record:
                    # No variables case
                    examples.append({
                        'variables': {},
                        'body_triples': body_patterns,
                        'head_triple': head_pattern
                    })
                else:
                    # Has variables case - create variable bindings
                    variables = {}

                    # Extract variable values from record
                    for var_name in record.keys():
                        # Find corresponding original variable name
                        original_var = None
                        for var in variable_mapping:
                            if var.startswith('?') and var[1:] == var_name:
                                original_var = var
                                break

                        if original_var:
                            variables[original_var] = record[var_name]

                    # Instantiate triples
                    instantiated_body = []
                    for s, p, o in body_patterns:
                        inst_s = variables.get(s, s)
                        inst_p = variables.get(p, p)
                        inst_o = variables.get(o, o)
                        instantiated_body.append((inst_s, inst_p, inst_o))

                    inst_head = None
                    if head_pattern:
                        s, p, o = head_pattern
                        inst_head = (
                            variables.get(s, s),
                            variables.get(p, p),
                            variables.get(o, o)
                        )

                    examples.append({
                        'variables': variables,
                        'body_triples': instantiated_body,
                        'head_triple': inst_head or (instantiated_body[0] if instantiated_body else None)
                    })

            logger.debug(f"Found {len(examples)} matching results")
            return examples

    except Neo4jError as e:
        logger.error(f"Pattern matching failed: {e}")
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return []
