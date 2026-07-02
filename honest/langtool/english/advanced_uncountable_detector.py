#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for AdvancedUncountableDetector

This module is deprecated. The functionality has been merged into
EnglishNumberAnalyzer for better code organization and performance.

Please use EnglishNumberAnalyzer.detect_uncountability() instead.
"""

import warnings
from typing import List, Dict, Tuple
from .number_analyzer import EnglishNumberAnalyzer


class AdvancedUncountableDetector:
    """
    Deprecated: Use EnglishNumberAnalyzer instead.

    This class is kept for backward compatibility only.
    All uncountable detection functionality has been merged into
    EnglishNumberAnalyzer for better integration and performance.
    """

    def __init__(self):
        """Initialize with a deprecation warning"""
        warnings.warn(
            "AdvancedUncountableDetector is deprecated. "
            "Use EnglishNumberAnalyzer.detect_uncountability() instead. "
            "This class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        self._analyzer = EnglishNumberAnalyzer()

    def detect_uncountability(self, word: str, context: str = "") -> Tuple[bool, float, str]:
        """
        Detect if a word is uncountable

        Deprecated: Use EnglishNumberAnalyzer.detect_uncountability() instead.
        """
        return self._analyzer.detect_uncountability(word, context)

    def batch_analyze(self, words: List[str]) -> Dict[str, Tuple[bool, float, str]]:
        """
        Batch analyze multiple words

        Deprecated: Use EnglishNumberAnalyzer.batch_detect_uncountability() instead.
        """
        return self._analyzer.batch_detect_uncountability(words)

    def analyze_text_nouns(self, text: str) -> Dict[str, Tuple[bool, float, str]]:
        """
        Analyze all nouns in text

        Deprecated: Use EnglishNumberAnalyzer.analyze_text_uncountability() instead.
        """
        return self._analyzer.analyze_text_uncountability(text)
