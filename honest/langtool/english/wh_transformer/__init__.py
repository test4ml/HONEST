#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WH-Question Transformer Module

This module provides WH-question transformation functionality for English.
Re-exports the main EnglishWhTransformer class for backward compatibility.
"""

from .core import EnglishWhTransformer, WHTransformer
from .entity_matcher import EntityMatcher

__all__ = ['EnglishWhTransformer', 'WHTransformer', 'EntityMatcher']
