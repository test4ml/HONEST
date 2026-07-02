from .knowledge_graph import KnowledgeGraph, ComprehensiveKnowledgeGraph, OptimizedMetadataStore, MemgraphKnowledgeGraph
from .base_knowledge_graph import KnowledgeGraph as BaseKnowledgeGraph
from .metadata_store import OptimizedMetadataStore as MetadataStore
from .memgraph_knowledge_graph import MemgraphKnowledgeGraph as MemgraphKG

__all__ = [
    'KnowledgeGraph',
    'ComprehensiveKnowledgeGraph',  # Backward-compatible alias
    'BaseKnowledgeGraph',
    'OptimizedMetadataStore',
    'MetadataStore',
    'MemgraphKnowledgeGraph',
    'MemgraphKG'
]