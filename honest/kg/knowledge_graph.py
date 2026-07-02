#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified knowledge graph interface implementation

Integrates the AbstractKB engine and metadata management, providing a unified knowledge graph interface.
Memory-optimized version, supports optional metadata storage.

For backward compatibility, this file re-exports the split classes.
"""

# Import classes from the split files
from .metadata_store import OptimizedMetadataStore
from .base_knowledge_graph import KnowledgeGraph, ComprehensiveKnowledgeGraph
from .memgraph_knowledge_graph import MemgraphKnowledgeGraph

# Re-export in this module for backward compatibility
__all__ = [
    'OptimizedMetadataStore',
    'KnowledgeGraph',
    'ComprehensiveKnowledgeGraph',
    'MemgraphKnowledgeGraph'
]
