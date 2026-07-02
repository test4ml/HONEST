#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distractor Validator for Multiple Choice Question Generation

Ensures that distractors (wrong options) are genuinely false by validating
against the knowledge graph using the Closed World Assumption (CWA).
"""

from .validator import (
    DistractorValidator,
    ValidationResult,
    KGWithContains,
    Triple,
    create_validator
)
from .generator import DistractorTripleGenerator

__all__ = [
    # Validator components
    'DistractorValidator',
    'ValidationResult',
    'KGWithContains',

    # Generator
    'DistractorTripleGenerator',

    # Type alias
    'Triple',

    # Convenience function
    'create_validator',
]
