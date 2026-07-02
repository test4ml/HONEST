# -*- coding: utf-8 -*-
"""
ConsisMatcher: A complete consistency-checking pipeline

Integrates conclusion extraction and NLI consistency checking.
"""

import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

from .mbart_conclusion_extractor import MBartConclusionExtractor, MBartExtractionResult
from .nli_checker import NLIConsistencyChecker, ConsistencyResult, ConsistencyLevel
from .llm_conclusion_extractor import LLMConclusionExtractor
from .llm_nli_checker import LLMNLIConsistencyChecker

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Complete match result"""
    # Consistency decision
    is_consistent: bool
    confidence: float
    explanation: str

    # NLI scores
    entailment_score: float
    contradiction_score: float
    neutral_score: float

    # Extracted conclusions
    conclusion1: str
    conclusion2: str
    extraction_confidence1: float
    extraction_confidence2: float

    # Metadata
    extraction_method1: str
    extraction_method2: str
    consistency_method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return asdict(self)


class ConsisMatcher:
    """
    ConsisMatcher: NLI-based answer consistency checker

    Pipeline:
    1. Extract conclusions from two LLM answers
    2. Use an NLI model to judge conclusion consistency
    """

    # Preset configurations (unified to use the mBART extractor)
    PRESETS = {
        'base': {
            'nli_model': 'cross-encoder/nli-deberta-v3-base',
        },
        'light': {
            'nli_model': 'cross-encoder/nli-MiniLM2-L6-H768',
        },
        'fast': {
            'nli_model': 'cross-encoder/nli-MiniLM2-L6-H768',
        },
        'accurate': {
            'nli_model': 'cross-encoder/nli-deberta-v3-base',
        },
    }

    def __init__(self,
                 preset: str = 'base',
                 nli_model: str = 'cross-encoder/nli-deberta-v3-base',
                 bidirectional: bool = True,
                 device: str = None,
                 extractor: str = 'mbart',
                 extractor_kwargs: Dict[str, Any] = None,
                 nli_method: str = 'cross-encoder',
                 nli_llm_kwargs: Dict[str, Any] = None):
        """
        Initialize ConsisMatcher

        Args:
            preset: Preset configuration ('base', 'light', 'fast', 'accurate')
            nli_model: NLI model name (only effective when nli_method='cross-encoder')
            bidirectional: Whether to perform bidirectional NLI inference (cross-encoder only)
            device: Device to run on ('cuda', 'cpu', None=auto)
            extractor: Conclusion extractor type
                - 'mbart': Local mBART extractive model (default, offline)
                - 'llm' : Calls a local small model (e.g., Qwen3-8B) via URL for more robust extraction
            extractor_kwargs: Initialization parameters for the LLM extractor (only effective when extractor='llm');
                commonly used: base_url, api_key, model_name, protocol, max_concurrent, rate_limit
            nli_method: NLI judging backend
                - 'cross-encoder': Local Cross-Encoder NLI small model (default, e.g., deberta-v3-base)
                - 'llm'          : Calls an LLM via URL for judging (e.g., DeepSeek-V4-Flash), outputs JSON
            nli_llm_kwargs: Initialization parameters for the LLM NLI backend (only effective when nli_method='llm');
                commonly used: base_url, api_key, model_name, protocol, max_concurrent, rate_limit
        """
        # Apply preset configuration
        if preset and preset in self.PRESETS:
            config = self.PRESETS[preset]
            nli_model = config.get('nli_model', nli_model)
            logger.info(f"Using preset: {preset}")

        self.device = device
        self.extractor_type = extractor
        self.nli_method = nli_method

        # Initialize the conclusion extractor
        if extractor == 'llm':
            kw = dict(extractor_kwargs or {})
            logger.info(f"Initializing ConsisMatcher with LLM extractor "
                        f"({kw.get('model_name', 'Qwen3-8B')} @ {kw.get('base_url', 'http://localhost:8001/v1')})...")
            self.extractor = LLMConclusionExtractor(**kw)
        elif extractor == 'mbart':
            logger.info("Initializing ConsisMatcher with mBART extractor...")
            self.extractor = MBartConclusionExtractor(device=device)
        else:
            raise ValueError(f"Unknown extractor type: {extractor!r} (expected 'mbart' or 'llm')")

        # Initialize the NLI consistency checker (cross-encoder or llm)
        if nli_method == 'cross-encoder':
            self.nli_checker = NLIConsistencyChecker(
                model_name=nli_model,
                device=device,
                bidirectional=bidirectional
            )
        elif nli_method == 'llm':
            kw = dict(nli_llm_kwargs or {})
            # method_suffix tags the extractor source for downstream distinction (e.g., nli_llm_entailment_llm)
            kw.setdefault('method_suffix', extractor)
            logger.info(f"Initializing ConsisMatcher with LLM NLI "
                        f"({kw.get('model_name', 'deepseek-v4-flash')} @ {kw.get('base_url', 'https://api.deepseek.com/anthropic')})...")
            self.nli_checker = LLMNLIConsistencyChecker(**kw)
        else:
            raise ValueError(f"Unknown nli_method: {nli_method!r} (expected 'cross-encoder' or 'llm')")

        logger.info("ConsisMatcher initialized successfully")

    def check_consistency(self,
                          answer1: str,
                          answer2: str,
                          question: str = None,
                          question_type: str = None) -> MatchResult:
        """
        Check the consistency of two LLM answers

        Args:
            answer1: First LLM answer (answer to the original question)
            answer2: Second LLM answer (answer to the mutated question)
            question: Original question (optional)
            question_type: Question type (optional)

        Returns:
            MatchResult: Complete match result
        """
        # Step 1: Extract conclusions
        extraction1 = self.extractor.extract(
            answer=answer1,
            question=question,
            question_type=question_type
        )

        extraction2 = self.extractor.extract(
            answer=answer2,
            question=question,
            question_type=question_type
        )

        # Step 2: Check consistency
        consistency = self.nli_checker.check_consistency(
            text1=extraction1.conclusion,
            text2=extraction2.conclusion
        )

        # Build the complete result
        return MatchResult(
            is_consistent=consistency.is_consistent,
            confidence=consistency.confidence,
            explanation=consistency.explanation,
            entailment_score=consistency.entailment_score,
            contradiction_score=consistency.contradiction_score,
            neutral_score=consistency.neutral_score,
            conclusion1=extraction1.conclusion,
            conclusion2=extraction2.conclusion,
            extraction_confidence1=extraction1.confidence,
            extraction_confidence2=extraction2.confidence,
            extraction_method1=extraction1.method,
            extraction_method2=extraction2.method,
            consistency_method=consistency.method
        )

    def check_consistency_simple(self,
                                 answer1: str,
                                 answer2: str,
                                 question: str = None,
                                 question_type: str = None) -> Tuple[bool, str]:
        """
        Simplified consistency-checking interface

        Args:
            answer1: First LLM answer
            answer2: Second LLM answer
            question: Original question (optional)
            question_type: Question type (optional)

        Returns:
            Tuple[bool, str]: (is_consistent, explanation)
        """
        result = self.check_consistency(
            answer1=answer1,
            answer2=answer2,
            question=question,
            question_type=question_type
        )
        return result.is_consistent, result.explanation

    def extract_conclusion(self,
                           answer: str,
                           question: str = None,
                           question_type: str = None) -> Tuple[str, float]:
        """
        Extract the conclusion only

        Args:
            answer: LLM answer
            question: Original question (optional)
            question_type: Question type (optional)

        Returns:
            Tuple[str, float]: (conclusion text, confidence)
        """
        result = self.extractor.extract(
            answer=answer,
            question=question,
            question_type=question_type
        )
        return result.conclusion, result.confidence

    def check_conclusions_consistency(self,
                                      conclusion1: str,
                                      conclusion2: str) -> Tuple[bool, str, float]:
        """
        Directly check the consistency of two conclusions (skips the extraction step)

        Args:
            conclusion1: First conclusion
            conclusion2: Second conclusion

        Returns:
            Tuple[bool, str, float]: (is_consistent, explanation, confidence)
        """
        result = self.nli_checker.check_consistency(
            text1=conclusion1,
            text2=conclusion2
        )
        return result.is_consistent, result.explanation, result.confidence


def create_consismatcher(preset: str = 'base', device: str = None) -> ConsisMatcher:
    """
    Factory function: create a ConsisMatcher instance

    Args:
        preset: Preset configuration
            - 'base': DeBERTa NLI (recommended)
            - 'light': MiniLM NLI (faster)
            - 'fast': MiniLM NLI (same as light)
            - 'accurate': DeBERTa NLI (same as base, most precise)
        device: Device to run on

    Returns:
        ConsisMatcher instance
    """
    return ConsisMatcher(preset=preset, device=device)
