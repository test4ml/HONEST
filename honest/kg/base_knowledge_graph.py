#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base unified knowledge graph interface implementation

Integrates the AbstractKB engine and metadata management, providing a unified knowledge graph interface.
Memory-optimized version, supports optional metadata storage.
"""

import sys
from typing import List, Tuple, Optional, Dict, Any, Set, TYPE_CHECKING
from ..kg_interfaces import IndexedKnowledgeGraph
from honest.utils.profiling import profile
from .metadata_store import OptimizedMetadataStore

if TYPE_CHECKING:
    from ..engines.base_engine import KGQueryEngine


class KnowledgeGraph(IndexedKnowledgeGraph):
    """Unified knowledge graph interface implementation

    Built on top of AbstractKB, providing:
    - A unified query interface
    - Optional metadata management (memory optimization)
    - Full AMIE-style statistical functionality
    """
    __slots__ = ['engine', 'metadata', '_metadata_enabled']

    @profile
    def __init__(self, query_engine: 'KGQueryEngine', enable_metadata: bool = True):
        """Initialize the knowledge graph

        Args:
            query_engine: Query engine instance
            enable_metadata: Whether to enable metadata storage (default True)
                           Setting this to False can significantly reduce memory usage, but loses label and description functionality
        """
        self.engine = query_engine
        self._metadata_enabled = enable_metadata
        self.metadata = OptimizedMetadataStore(enable_metadata)

    # -------------------------------------------------------------------------
    # Basic data operations
    # -------------------------------------------------------------------------

    def add_triple(self, subject: str, predicate: str, obj: str) -> bool:
        """Add a single triple"""
        return self.engine.add_triple(subject, predicate, obj)

    @profile
    def add_triples_batch(self, triples: List[Tuple[str, str, str]]) -> int:
        """Add triples in batch"""
        return self.engine.add_triples_batch(triples)

    def load_from_file(self, file_path: str, delimiter: str = '\t') -> None:
        """Load triples from a file"""
        self.engine.load_from_file(file_path, delimiter)

    # -------------------------------------------------------------------------
    # Basic query operations
    # -------------------------------------------------------------------------

    def contains(self, subject: str, predicate: str, obj: str) -> bool:
        """Check whether the specified triple is contained"""
        return self.engine.contains(subject, predicate, obj)

    @profile
    def get_triples_by_pattern(self,
                              subject: Optional[str] = None,
                              predicate: Optional[str] = None,
                              obj: Optional[str] = None,
                              limit: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """Query triples by pattern"""
        return self.engine.query_triples(subject, predicate, obj, limit or -1)

    def count_triples(self,
                     subject: Optional[str] = None,
                     predicate: Optional[str] = None,
                     obj: Optional[str] = None) -> int:
        """Count the number of triples matching the specified pattern"""
        return self.engine.count(subject, predicate, obj)

    def count(self,
             subject: Optional[str] = None,
             predicate: Optional[str] = None,
             obj: Optional[str] = None) -> int:
        """Count the number of triples matching the specified pattern (alias)"""
        return self.engine.count(subject, predicate, obj)

    # -------------------------------------------------------------------------
    # Advanced query operations (AMIE style)
    # -------------------------------------------------------------------------

    @profile
    def select_distinct(self, variable_position: int,
                       query_patterns: List[Tuple[Optional[str], Optional[str], Optional[str]]]) -> Set[str]:
        """Select distinct entities that satisfy the query conditions"""
        return self.engine.select_distinct(variable_position, query_patterns)

    def count_distinct(self, variable_position: int,
                      query_patterns: List[Tuple[Optional[str], Optional[str], Optional[str]]]) -> int:
        """Count the number of distinct entities that satisfy the query conditions"""
        return self.engine.count_distinct(variable_position, query_patterns)

    def count_distinct_pairs(self, var1_pos: int, var2_pos: int,
                            query_patterns: List[Tuple[Optional[str], Optional[str], Optional[str]]]) -> int:
        """Count the number of distinct entity pairs that satisfy the query conditions"""
        return self.engine.count_distinct_pairs(var1_pos, var2_pos, query_patterns)

    # -------------------------------------------------------------------------
    # Statistics and functionality metrics
    # -------------------------------------------------------------------------

    def functionality(self, predicate: str) -> float:
        """Compute the functionality of a relation"""
        return self.engine.functionality(predicate)

    def inverse_functionality(self, predicate: str) -> float:
        """Compute the inverse functionality of a relation"""
        return self.engine.inverse_functionality(predicate)

    def is_functional(self, predicate: str) -> bool:
        """Determine whether a relation is a functional relation"""
        return self.engine.is_functional(predicate)

    def relation_size(self, predicate: str) -> int:
        """Get the number of triples of a relation"""
        return self.engine.relation_size(predicate)

    def relation_column_size(self, predicate: str, column: int) -> int:
        """Get the number of distinct entities of a relation in the specified column"""
        return self.engine.relation_column_size(predicate, column)

    def overlap(self, predicate1: str, predicate2: str, overlap_type: int) -> int:
        """Compute the overlap between two relations"""
        return self.engine.overlap(predicate1, predicate2, overlap_type)

    def size(self) -> int:
        """Get the total number of triples in the knowledge graph"""
        return self.engine.size()

    def get_relations(self) -> Set[str]:
        """Get the set of all relations"""
        return self.engine.get_relations()

    def get_entities(self) -> Set[str]:
        """Get the set of all entities (subjects + objects)"""
        return self.engine.get_entities()

    # -------------------------------------------------------------------------
    # Metadata management (optional functionality)
    # -------------------------------------------------------------------------

    def get_entity_label(self, entity_id: str) -> Optional[str]:
        """Get the entity label"""
        return self.metadata.get_entity_label(entity_id)

    def get_entity_description(self, entity_id: str) -> Optional[str]:
        """Get the entity description"""
        return self.metadata.get_entity_description(entity_id)

    def get_property_label(self, property_id: str) -> Optional[str]:
        """Get the property label"""
        return self.metadata.get_property_label(property_id)

    @profile
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
        """Add an entity label (convenience method)"""
        if self._metadata_enabled:
            self.metadata.add_entity_label(entity_id, label)

    def add_entity_description(self, entity_id: str, description: str) -> None:
        """Add an entity description (convenience method)"""
        if self._metadata_enabled:
            self.metadata.add_entity_description(entity_id, description)

    def add_property_label(self, property_id: str, label: str) -> None:
        """Add a property label (convenience method)"""
        if self._metadata_enabled:
            self.metadata.add_property_label(property_id, label)

    @profile
    def add_entity_metadata_batch(self, metadata_list: List[Tuple[str, str]]) -> None:
        """Add entity labels in batch, reducing function call overhead"""
        if not self._metadata_enabled:
            return

        # Use a more efficient batch method
        self.metadata.add_entity_labels_batch(metadata_list)

    def get_entity_info(self, entity_id: str) -> Dict[str, Optional[str]]:
        """Get complete entity info"""
        return {
            'id': entity_id,
            'label': self.get_entity_label(entity_id),
            'description': self.get_entity_description(entity_id)
        }

    def get_entities_info(self, entity_ids: List[str]) -> List[Dict[str, Optional[str]]]:
        """Get entity info in batch"""
        return [self.get_entity_info(eid) for eid in entity_ids]

    def format_entity(self, entity_id: str, show_description: bool = True) -> str:
        """Format entity display"""
        if not self._metadata_enabled:
            return entity_id

        label = self.get_entity_label(entity_id)
        description = self.get_entity_description(entity_id)

        if label:
            result = f"{entity_id}: {label}"
            if show_description and description:
                result += f" | {description}"
        else:
            result = f"{entity_id}: <<no_label>>"
            if show_description:
                result += " | <<no_description>>"

        return result

    def format_entities(self, entity_ids: List[str], show_description: bool = True) -> List[str]:
        """Format entity display in batch"""
        return [self.format_entity(eid, show_description) for eid in entity_ids]

    # -------------------------------------------------------------------------
    # Statistics info
    # -------------------------------------------------------------------------

    @profile
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        # Get basic stats directly from the engine
        engine_stats = self.engine.get_stats()

        # Add metadata stats
        metadata_stats = self.metadata.get_stats()
        metadata_stats['metadata_enabled'] = self._metadata_enabled

        # Merge stats info
        return {**engine_stats, **metadata_stats}


    @profile
    def finalize(self) -> None:
        """Finish building and optimize data structures"""
        self.engine.finalize()

        print(f"\nKnowledge graph build complete:")
        stats = self.get_stats()
        print(f"  Number of triples: {stats.get('total_triples', 0):,}")
        if self._metadata_enabled:
            print(f"  Number of entity labels: {stats.get('total_entity_labels', 0):,}")
            print(f"  Number of entity descriptions: {stats.get('total_entity_descriptions', 0):,}")
            print(f"  Number of property labels: {stats.get('total_property_labels', 0):,}")
        else:
            print("  Metadata storage: disabled (saves memory)")

    @profile
    def close(self) -> None:
        """Close the connection and release resources"""
        self.engine.close()
        if self._metadata_enabled:
            self.metadata.clear()

    # -------------------------------------------------------------------------
    # Serialization and deserialization interface
    # -------------------------------------------------------------------------

    def serialize(self, file_path: str) -> None:
        """Serialize the knowledge graph to a file

        Serialization includes:
        1. Engine data (triples, indexes, etc.)
        2. Metadata (entity labels, descriptions, property labels)
        3. Configuration info (whether metadata is enabled, etc.)

        Args:
            file_path: Serialization file path (without extension)
        """
        import json
        import os

        print(f"Start serializing knowledge graph to: {file_path}")

        # Create the directory
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)

        # Serialize engine data
        engine_file = f"{file_path}.engine"
        self.engine.serialize(engine_file)
        print(f"  Engine data serialized: {engine_file}")

        # Serialize metadata
        if self._metadata_enabled:
            metadata_file = f"{file_path}.metadata.json"
            metadata_data = {
                'entity_labels': dict(self.metadata._entity_labels) if self.metadata._entity_labels else {},
                'entity_descriptions': dict(self.metadata._entity_descriptions) if self.metadata._entity_descriptions else {},
                'property_labels': dict(self.metadata._property_labels) if self.metadata._property_labels else {}
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_data, f, ensure_ascii=False, indent=2)
            print(f"  Metadata serialized: {metadata_file}")

        # Serialize configuration info
        config_file = f"{file_path}.config.json"
        config_data = {
            'metadata_enabled': self._metadata_enabled,
            'engine_type': self.engine.__class__.__name__
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        print(f"  Configuration info serialized: {config_file}")

        print("Knowledge graph serialization complete")

    def deserialize(self, file_path: str) -> None:
        """Deserialize the knowledge graph from a file

        Note: this method will clear all current data!

        Args:
            file_path: Serialization file path (without extension)
        """
        import json
        import os

        print(f"Start deserializing knowledge graph from file: {file_path}")

        # Check that the required files exist
        engine_file = f"{file_path}.engine"
        config_file = f"{file_path}.config.json"

        if not os.path.exists(engine_file):
            raise FileNotFoundError(f"Engine data file does not exist: {engine_file}")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file does not exist: {config_file}")

        # Read the configuration info
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Deserialize engine data
        self.engine.deserialize(engine_file)
        print(f"  Engine data deserialized: {engine_file}")

        # Deserialize metadata
        metadata_file = f"{file_path}.metadata.json"
        if config_data.get('metadata_enabled', True) and os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)

            # Rebuild the metadata store
            if self.metadata._entity_labels is not None:
                self.metadata._entity_labels.clear()
                self.metadata._entity_labels.update(metadata_data.get('entity_labels', {}))

            if self.metadata._entity_descriptions is not None:
                self.metadata._entity_descriptions.clear()
                self.metadata._entity_descriptions.update(metadata_data.get('entity_descriptions', {}))

            if self.metadata._property_labels is not None:
                self.metadata._property_labels.clear()
                self.metadata._property_labels.update(metadata_data.get('property_labels', {}))

            print(f"  Metadata deserialized: {metadata_file}")

        print("Knowledge graph deserialization complete")

    def supports_serialization(self) -> bool:
        """Check whether serialization is supported

        Returns:
            bool: True if the underlying engine supports serialization
        """
        return self.engine.supports_serialization()


# Keep the original class name for backward compatibility
ComprehensiveKnowledgeGraph = KnowledgeGraph
