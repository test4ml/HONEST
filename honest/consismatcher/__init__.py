# -*- coding: utf-8 -*-
"""
ConsisMatcher: Deep Learning Based Answer Consistency Checker

Answer consistency checker based on NLI pre-trained models.
"""

from .consismatcher import ConsisMatcher
from .mbart_conclusion_extractor import MBartConclusionExtractor
from .llm_conclusion_extractor import LLMConclusionExtractor
from .nli_checker import NLIConsistencyChecker
from .llm_nli_checker import LLMNLIConsistencyChecker

__all__ = [
    'ConsisMatcher',
    'MBartConclusionExtractor',
    'LLMConclusionExtractor',
    'NLIConsistencyChecker',
    'LLMNLIConsistencyChecker',
]

__version__ = '0.1.0'
