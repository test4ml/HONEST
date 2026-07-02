#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memgraph Knowledge Graph Core Module

Core functionality for Memgraph-based knowledge graph implementation,
including initialization, persistence, and metadata management.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Set
from ...kg_interfaces import IndexedKnowledgeGraph
from honest.utils.profiling import profile
from ..metadata_store import OptimizedMetadataStore

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
except ImportError:
    raise ImportError("Need to install neo4j driver: conda install -c conda-forge neo4j-python-driver")

logger = logging.getLogger(__name__)


class MemgraphKnowledgeGraph(IndexedKnowledgeGraph):
    """Memgraph-based knowledge graph implementation

    Features:
    - Cypher query language for graph pattern matching
    - High-performance in-memory analytical mode
    - SPO triple index optimization
    - Complete entity labels and descriptions metadata
    - Native graph database performance advantages
    """

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "", password: str = "",
                 enable_metadata: bool = True,
                 functionality_cache: Optional[Dict[str, float]] = None,
                 inv_functionality_cache: Optional[Dict[str, float]] = None):
        """Initialize Memgraph Knowledge Graph

        Args:
            uri: Memgraph connection URI
            user: Username
            password: Password
            enable_metadata: Whether to enable metadata management
            functionality_cache: Precomputed functionality cache dictionary
            inv_functionality_cache: Precomputed inverse functionality cache dictionary
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._metadata_enabled = enable_metadata
        self.metadata = OptimizedMetadataStore(enable_metadata)

        # Initialize caches, use provided precomputed cache or empty dict
        self.functionality_cache = functionality_cache.copy() if functionality_cache else {}
        self.inv_functionality_cache = inv_functionality_cache.copy() if inv_functionality_cache else {}
        self.stats_cache = {}

        # Add relation and entity caches
        self.relations_cache: Optional[Set[str]] = None
        self.entities_cache: Optional[Set[str]] = None

    def _get_driver(self):
        """Get database driver"""
        if self.driver is None:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=30,
                encrypted=False
            )
        return self.driver

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                return record and record["test"] == 1
        except Neo4jError as e:
            logger.error(f"Memgraph connection failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Basic data operations (Memgraph doesn't support direct addition, these are interface compatibility methods)
    # -------------------------------------------------------------------------

    def add_triple(self, subject: str, predicate: str, obj: str) -> bool:
        """Add a single triple (Memgraph version)"""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                # Check if node exists, create if not
                create_query = """
                MERGE (s:Entity {id: $subject})
                MERGE (o:Entity {id: $object})
                MERGE (s)-[r:HAS_PROPERTY {property_id: $predicate}]->(o)
                """
                session.run(create_query, subject=subject, predicate=predicate, object=obj)
                return True
        except Neo4jError as e:
            logger.error(f"Failed to add triple: {e}")
            return False

    @profile
    def add_triples_batch(self, triples: List[Tuple[str, str, str]]) -> int:
        """Batch add triples"""
        driver = self._get_driver()
        added_count = 0
        try:
            with driver.session() as session:
                for s, p, o in triples:
                    try:
                        create_query = """
                        MERGE (s:Entity {id: $subject})
                        MERGE (o:Entity {id: $object})
                        MERGE (s)-[r:HAS_PROPERTY {property_id: $predicate}]->(o)
                        """
                        session.run(create_query, subject=s, predicate=p, object=o)
                        added_count += 1
                    except Neo4jError as e:
                        logger.warning(f"Failed to add triple {s} {p} {o}: {e}")
        except Neo4jError as e:
            logger.error(f"Batch add triples failed: {e}")
        return added_count

    def load_from_file(self, file_path: str, delimiter: str = '\t') -> None:
        """Load triples from file (not implemented)"""
        raise NotImplementedError("Memgraph version recommends using CSV import tool")

    # -------------------------------------------------------------------------
    # Basic query operations
    # -------------------------------------------------------------------------

    def contains(self, subject: str, predicate: str, obj: str) -> bool:
        """Check if the specified triple exists"""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                query = """
                MATCH (s:Entity {id: $subject})-[r:HAS_PROPERTY {property_id: $predicate}]->(o:Entity {id: $object})
                RETURN count(r) as count
                """
                result = session.run(query, subject=subject, predicate=predicate, object=obj)
                record = result.single()
                return record and record["count"] > 0
        except Neo4jError as e:
            logger.error(f"Query triple failed: {e}")
            return False

    @profile
    def get_triples_by_pattern(self,
                              subject: Optional[str] = None,
                              predicate: Optional[str] = None,
                              obj: Optional[str] = None,
                              limit: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """Query triples by pattern"""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                # Build query conditions
                conditions = []
                params = {}

                if subject:
                    conditions.append("s.id = $subject")
                    params["subject"] = subject
                if predicate:
                    conditions.append("r.property_id = $predicate")
                    params["predicate"] = predicate
                if obj:
                    conditions.append("o.id = $object")
                    params["object"] = obj

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                limit_clause = f"LIMIT {limit}" if limit else ""

                query = f"""
                MATCH (s:Entity)-[r:HAS_PROPERTY]->(o:Entity)
                WHERE {where_clause}
                RETURN s.id as subject, r.property_id as predicate, o.id as object
                {limit_clause}
                """

                result = session.run(query, **params)
                triples = []
                for record in result:
                    triples.append((record["subject"], record["predicate"], record["object"]))
                return triples
        except Neo4jError as e:
            logger.error(f"Pattern query failed: {e}")
            return []

    def count_triples(self,
                     subject: Optional[str] = None,
                     predicate: Optional[str] = None,
                     obj: Optional[str] = None) -> int:
        """Count triples matching the specified pattern"""
        driver = self._get_driver()
        try:
            with driver.session() as session:
                # Build query conditions
                conditions = []
                params = {}

                if subject:
                    conditions.append("s.id = $subject")
                    params["subject"] = subject
                if predicate:
                    conditions.append("r.property_id = $predicate")
                    params["predicate"] = predicate
                if obj:
                    conditions.append("o.id = $object")
                    params["object"] = obj

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                query = f"""
                MATCH (s:Entity)-[r:HAS_PROPERTY]->(o:Entity)
                WHERE {where_clause}
                RETURN count(r) as count
                """

                result = session.run(query, **params)
                record = result.single()
                return record["count"] if record else 0
        except Neo4jError as e:
            logger.error(f"Count query failed: {e}")
            return 0

    def count(self,
             subject: Optional[str] = None,
             predicate: Optional[str] = None,
             obj: Optional[str] = None) -> int:
        """Count triples matching the specified pattern (alias)"""
        return self.count_triples(subject, predicate, obj)

    # -------------------------------------------------------------------------
    # Advanced query operations (simplified implementation)
    # -------------------------------------------------------------------------

    @profile
    def select_distinct(self, variable_position: int,
                       query_patterns: List[Tuple[Optional[str], Optional[str], Optional[str]]]) -> Set[str]:
        """Select distinct entities satisfying query conditions"""
        # Simplified implementation, only supports single pattern
        if len(query_patterns) != 1:
            logger.warning("Memgraph version currently only supports single query pattern")
            return set()

        pattern = query_patterns[0]
        triples = self.get_triples_by_pattern(pattern[0], pattern[1], pattern[2])

        entities = set()
        for triple in triples:
            if variable_position < len(triple):
                entities.add(triple[variable_position])

        return entities

    def count_distinct(self, variable_position: int,
                      query_patterns: List[Tuple[Optional[str], Optional[str], Optional[str]]]) -> int:
        """Count distinct entities satisfying query conditions"""
        return len(self.select_distinct(variable_position, query_patterns))

    def count_distinct_pairs(self, var1_pos: int, var2_pos: int,
                            query_patterns: List[Tuple[Optional[str], Optional[str], Optional[str]]]) -> int:
        """Count distinct entity pairs satisfying query conditions"""
        # Simplified implementation
        if len(query_patterns) != 1:
            return 0

        pattern = query_patterns[0]
        triples = self.get_triples_by_pattern(pattern[0], pattern[1], pattern[2])

        pairs = set()
        for triple in triples:
            if var1_pos < len(triple) and var2_pos < len(triple):
                pairs.add((triple[var1_pos], triple[var2_pos]))

        return len(pairs)

    def size(self) -> int:
        """Get total number of triples in the knowledge graph"""
        return self.count_triples()

    def get_relations(self) -> Set[str]:
        """Get all relations set (with cache)"""
        # If cache exists, return directly
        if self.relations_cache is not None:
            return self.relations_cache

        driver = self._get_driver()
        try:
            with driver.session() as session:
                query = """
                MATCH ()-[r:HAS_PROPERTY]->()
                RETURN DISTINCT r.property_id as predicate
                LIMIT 10000
                """
                result = session.run(query)
                relations = set()
                for record in result:
                    relations.add(record["predicate"])

                # Cache result
                self.relations_cache = relations
                logger.debug(f"Cached {len(relations)} relations")
                return relations
        except Neo4jError as e:
            logger.error(f"Failed to get relations set: {e}")
            return set()

    def get_entities(self) -> Set[str]:
        """Get all entities set (with cache)"""
        # If cache exists, return directly
        if self.entities_cache is not None:
            return self.entities_cache

        driver = self._get_driver()
        try:
            with driver.session() as session:
                query = """
                MATCH (n:Entity)
                RETURN DISTINCT n.id as entity
                LIMIT 100000
                """
                result = session.run(query)
                entities = set()
                for record in result:
                    entities.add(record["entity"])

                # Cache result
                self.entities_cache = entities
                logger.debug(f"Cached {len(entities)} entities")
                return entities
        except Neo4jError as e:
            logger.error(f"Failed to get entities set: {e}")
            return set()

    # -------------------------------------------------------------------------
    # Metadata management
    # -------------------------------------------------------------------------

    def get_entity_label(self, entity_id: str) -> Optional[str]:
        """Get entity label"""
        # First check cache
        cached_label = self.metadata.get_entity_label(entity_id)
        if cached_label:
            return cached_label

        # Query from Memgraph
        driver = self._get_driver()
        try:
            with driver.session() as session:
                query = "MATCH (e:Entity {id: $entity_id}) RETURN e.label as label"
                result = session.run(query, entity_id=entity_id)
                record = result.single()
                if record and record["label"]:
                    label = record["label"]
                    # Cache result
                    self.metadata.add_entity_label(entity_id, label)
                    return label
        except Neo4jError as e:
            logger.warning(f"Failed to get entity label {entity_id}: {e}")

        return None

    def get_entity_description(self, entity_id: str) -> Optional[str]:
        """Get entity description"""
        # First check cache
        cached_desc = self.metadata.get_entity_description(entity_id)
        if cached_desc:
            return cached_desc

        # Query from Memgraph
        driver = self._get_driver()
        try:
            with driver.session() as session:
                query = "MATCH (e:Entity {id: $entity_id}) RETURN e.description as description"
                result = session.run(query, entity_id=entity_id)
                record = result.single()
                if record and record["description"]:
                    desc = record["description"]
                    # Cache result
                    self.metadata.add_entity_description(entity_id, desc)
                    return desc
        except Neo4jError as e:
            logger.warning(f"Failed to get entity description {entity_id}: {e}")

        return None

    def get_property_label(self, property_id: str) -> Optional[str]:
        """Get property label"""
        return self.metadata.get_property_label(property_id)

    def add_entity_metadata(self, entity_id: str, label: Optional[str] = None,
                           description: Optional[str] = None) -> None:
        """Add entity metadata"""
        if label is not None:
            self.metadata.add_entity_label(entity_id, label)
        if description is not None:
            self.metadata.add_entity_description(entity_id, description)

    def add_property_metadata(self, property_id: str, label: str) -> None:
        """Add property metadata"""
        self.metadata.add_property_label(property_id, label)

    def add_entity_label(self, entity_id: str, label: str) -> None:
        """Add entity label"""
        if self._metadata_enabled:
            self.metadata.add_entity_label(entity_id, label)

    def add_entity_description(self, entity_id: str, description: str) -> None:
        """Add entity description"""
        if self._metadata_enabled:
            self.metadata.add_entity_description(entity_id, description)

    def add_property_label(self, property_id: str, label: str) -> None:
        """Add property label"""
        if self._metadata_enabled:
            self.metadata.add_property_label(property_id, label)

    # -------------------------------------------------------------------------
    # Lifecycle management
    # -------------------------------------------------------------------------

    def finalize(self) -> None:
        """Finalize construction, optimize data structures"""
        print(f"\nMemgraph knowledge graph connection completed:")
        stats = self.get_stats()
        print(f"  Number of entities: {stats.get('total_entities', 0):,}")
        print(f"  Number of triples: {stats.get('total_triples', 0):,}")
        print(f"  Number of relation types: {stats.get('total_relations', 0):,}")
        if self._metadata_enabled:
            print(f"  Cached entity labels: {stats.get('total_entity_labels', 0):,}")
            print(f"  Cached entity descriptions: {stats.get('total_entity_descriptions', 0):,}")

    @profile
    def close(self) -> None:
        """Close connection, release resources"""
        if self.driver:
            self.driver.close()
            self.driver = None
        if self._metadata_enabled:
            self.metadata.clear()
        # Clean up caches
        self.functionality_cache.clear()
        self.inv_functionality_cache.clear()
        self.stats_cache.clear()
        self.relations_cache = None
        self.entities_cache = None

    # -------------------------------------------------------------------------
    # Serialization and deserialization interfaces (not supported yet)
    # -------------------------------------------------------------------------

    def serialize(self, file_path: str) -> None:
        """Serialize (Memgraph version not supported yet)"""
        raise NotImplementedError("Memgraph version does not support serialization yet")

    def deserialize(self, file_path: str) -> None:
        """Deserialize (Memgraph version not supported yet)"""
        raise NotImplementedError("Memgraph version does not support deserialization yet")

    def supports_serialization(self) -> bool:
        """Check if serialization is supported"""
        return False

    # -------------------------------------------------------------------------
    # Query engine methods (will be imported from query_engine module)
    # -------------------------------------------------------------------------

    def functionality(self, predicate: str) -> float:
        """Compute relation functionality (imported from query_engine)"""
        from .query_engine import compute_functionality
        return compute_functionality(self, predicate)

    def inverse_functionality(self, predicate: str) -> float:
        """Compute relation inverse functionality (imported from query_engine)"""
        from .query_engine import compute_inverse_functionality
        return compute_inverse_functionality(self, predicate)

    def batch_compute_functionality(self, predicates: List[str]) -> Dict[str, Tuple[float, float]]:
        """Batch compute functionality and inverse functionality for multiple predicates (imported from query_engine)"""
        from .query_engine import batch_compute_functionality
        return batch_compute_functionality(self, predicates)

    def precompute_common_functionality(self, top_n: int = 100):
        """Precompute functionality for most common predicates (imported from query_engine)"""
        from .query_engine import precompute_common_functionality
        return precompute_common_functionality(self, top_n)

    def clear_functionality_cache(self):
        """Clear functionality cache (imported from query_engine)"""
        from .query_engine import clear_functionality_cache
        return clear_functionality_cache(self)

    def clear_relations_entities_cache(self):
        """Clear relations and entities cache (imported from query_engine)"""
        from .query_engine import clear_relations_entities_cache
        return clear_relations_entities_cache(self)

    def get_functionality_cache_stats(self) -> Dict[str, int]:
        """Get functionality cache statistics (imported from query_engine)"""
        from .query_engine import get_functionality_cache_stats
        return get_functionality_cache_stats(self)

    def is_functional(self, predicate: str) -> bool:
        """Check if relation is functional (imported from query_engine)"""
        from .query_engine import is_functional
        return is_functional(self, predicate)

    def relation_size(self, predicate: str) -> int:
        """Get number of triples in relation (imported from query_engine)"""
        from .query_engine import relation_size
        return relation_size(self, predicate)

    def relation_column_size(self, predicate: str, column: int) -> int:
        """Get number of distinct entities in specified column of relation (imported from query_engine)"""
        from .query_engine import relation_column_size
        return relation_column_size(self, predicate, column)

    def overlap(self, predicate1: str, predicate2: str, overlap_type: int) -> int:
        """Compute overlap between two relations (imported from query_engine)"""
        from .query_engine import overlap
        return overlap(self, predicate1, predicate2, overlap_type)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics (imported from query_engine)"""
        from .query_engine import get_stats
        return get_stats(self)

    def execute_cypher(self, query: str, **params) -> List[Dict[str, Any]]:
        """Execute custom Cypher query (imported from query_engine)"""
        from .query_engine import execute_cypher
        return execute_cypher(self, query, **params)

    def query(self, query_str: str, **params) -> List[Dict[str, Any]]:
        """Execute query (alias of execute_cypher) (imported from query_engine)"""
        from .query_engine import query
        return query(self, query_str, **params)

    def match_rule_patterns(self, body_patterns: List[Tuple], head_pattern: Tuple,
                           max_examples: int = 100) -> List[Dict[str, Any]]:
        """Match rule patterns, return positive instances (imported from query_engine)"""
        from .query_engine import match_rule_patterns
        return match_rule_patterns(self, body_patterns, head_pattern, max_examples)
