#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memgraph Knowledge Graph - Backward Compatibility Module

This module provides backward compatibility by re-exporting MemgraphKnowledgeGraph
from the refactored memgraph submodule.
"""

from .memgraph import MemgraphKnowledgeGraph

__all__ = ['MemgraphKnowledgeGraph']
