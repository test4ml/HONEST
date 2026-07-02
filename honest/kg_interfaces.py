#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal knowledge graph interface definition
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


class IndexedKnowledgeGraph(ABC):
    """Base class for the knowledge graph interface"""

    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection"""
        pass

    @abstractmethod
    def close(self):
        """Close the connection"""
        pass

    def query_entities_by_predicate(self, predicate: str) -> List[str]:
        """Query entities that use a specific predicate"""
        return []

    def query_predicates_by_domain_range(self, domain_type: str, range_type: str) -> List[str]:
        """Query predicates that connect entities of specific types"""
        return []

    def query_entities_by_type(self, entity_type: str) -> List[str]:
        """Query entities of a specific type"""
        return []


class KGIndexBuilder(ABC):
    """Base class for the knowledge graph index builder"""
    pass
