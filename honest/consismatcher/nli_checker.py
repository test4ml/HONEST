# -*- coding: utf-8 -*-
"""
NLI Consistency Checker: A consistency checker based on natural language inference

Uses a pre-trained NLI model to determine whether two conclusions are consistent
- Entailment -> consistent
- Contradiction -> inconsistent
- Neutral -> needs further judgment
"""

import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Consistency level"""
    CONSISTENT = "consistent"           # consistent
    INCONSISTENT = "inconsistent"       # inconsistent
    NEUTRAL = "neutral"                 # neutral / uncertain
    ERROR = "error"                     # processing error


@dataclass
class ConsistencyResult:
    """Consistency check result"""
    is_consistent: bool                 # Whether consistent
    level: ConsistencyLevel             # Consistency level
    confidence: float                   # Confidence (0-1)
    entailment_score: float             # Entailment score
    contradiction_score: float          # Contradiction score
    neutral_score: float                # Neutral score
    explanation: str                    # Explanation
    method: str                         # Method used


class NLIConsistencyChecker:
    """
    NLI-based consistency checker

    Uses a Cross-Encoder model for natural language inference to determine the
    relationship between two texts.
    """

    # Supported models list
    SUPPORTED_MODELS = {
        'deberta-base': 'cross-encoder/nli-deberta-v3-base',
        'deberta-small': 'cross-encoder/nli-deberta-v3-small',
        'roberta-base': 'cross-encoder/nli-roberta-base',
        'minilm': 'cross-encoder/nli-MiniLM2-L6-H768',
    }

    # NLI label mapping (different models may have different orderings)
    NLI_LABELS = ['contradiction', 'entailment', 'neutral']

    def __init__(self,
                 model_name: str = 'cross-encoder/nli-deberta-v3-base',
                 device: str = None,
                 bidirectional: bool = True):
        """
        Initialize the NLI consistency checker

        Args:
            model_name: Model name or HuggingFace ID
            device: Device to run on ('cuda', 'cpu', None=auto)
            bidirectional: Whether to perform bidirectional inference (A->B and B->A)
        """
        # Resolve the model name
        if model_name in self.SUPPORTED_MODELS:
            model_name = self.SUPPORTED_MODELS[model_name]

        self.model_name = model_name
        self.device = device
        self.bidirectional = bidirectional
        self.model = None

        # Dynamic label mapping (initialized in _load_model)
        self.label2id = None
        self.id2label = None
        self.contradiction_idx = 0
        self.entailment_idx = 1
        self.neutral_idx = 2

        self._load_model()

    def _load_model(self):
        """Load the NLI model and dynamically obtain the label mapping"""
        try:
            from sentence_transformers import CrossEncoder
            import torch

            # Determine device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            logger.info(f"Loading NLI model: {self.model_name} on {self.device}")

            self.model = CrossEncoder(
                self.model_name,
                device=self.device
            )

            # Dynamically obtain the label mapping
            self._init_label_mapping()

            logger.info("NLI model loaded successfully")
            logger.info(f"Label mapping: {self.id2label}")

        except (OSError, RuntimeError, ImportError, ValueError) as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise RuntimeError(f"Failed to load NLI model: {e}")

    def _init_label_mapping(self):
        """Initialize the label mapping, dynamically obtaining the model's label order"""
        try:
            # Try to obtain the label mapping from the model config
            if hasattr(self.model.model, 'config') and hasattr(self.model.model.config, 'label2id'):
                self.label2id = self.model.model.config.label2id
                self.id2label = {v: k for k, v in self.label2id.items()}

                # Update the indices
                self.contradiction_idx = self.label2id.get('contradiction', self.label2id.get('CONTRADICTION', 0))
                self.entailment_idx = self.label2id.get('entailment', self.label2id.get('ENTAILMENT', 1))
                self.neutral_idx = self.label2id.get('neutral', self.label2id.get('NEUTRAL', 2))

                logger.info(f"Dynamically loaded label mapping from model config")
            else:
                # Fall back to the default mapping (works for most cross-encoder/nli-* models)
                logger.warning("Model config does not have label2id, using default mapping")
                self.label2id = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
                self.id2label = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
                self.contradiction_idx = 0
                self.entailment_idx = 1
                self.neutral_idx = 2

        except Exception as e:
            logger.warning(f"Failed to get label mapping from model, using defaults: {e}")
            # Use the default mapping
            self.label2id = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
            self.id2label = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
            self.contradiction_idx = 0
            self.entailment_idx = 1
            self.neutral_idx = 2

    def check_consistency(self,
                          text1: str,
                          text2: str) -> ConsistencyResult:
        """
        Check the consistency of two texts

        Args:
            text1: First text (conclusion 1)
            text2: Second text (conclusion 2)

        Returns:
            ConsistencyResult: Consistency check result
        """
        if not text1 or not text2:
            return ConsistencyResult(
                is_consistent=False,
                level=ConsistencyLevel.ERROR,
                confidence=0.0,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=0.0,
                explanation="Empty text input",
                method='error'
            )

        try:
            # Perform NLI inference
            if self.bidirectional:
                scores = self._bidirectional_inference(text1, text2)
            else:
                scores = self._single_inference(text1, text2)

            # Parse the scores using the dynamic indices
            contradiction_score = scores[self.contradiction_idx]
            entailment_score = scores[self.entailment_idx]
            neutral_score = scores[self.neutral_idx]

            # Determine consistency
            result = self._determine_consistency(
                entailment_score,
                contradiction_score,
                neutral_score
            )

            return result

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error during consistency check: {e}")
            return ConsistencyResult(
                is_consistent=False,
                level=ConsistencyLevel.ERROR,
                confidence=0.0,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=0.0,
                explanation=f"Error: {str(e)}",
                method='error'
            )

    def _single_inference(self, text1: str, text2: str) -> np.ndarray:
        """Unidirectional NLI inference"""
        scores = self.model.predict([(text1, text2)], show_progress_bar=False)
        return np.array(scores[0])

    def _bidirectional_inference(self, text1: str, text2: str) -> np.ndarray:
        """Bidirectional NLI inference, taking the averaged scores"""
        # Forward: text1 -> text2
        scores_forward = self.model.predict([(text1, text2)], show_progress_bar=False)[0]

        # Backward: text2 -> text1
        scores_backward = self.model.predict([(text2, text1)], show_progress_bar=False)[0]

        # Average the scores
        avg_scores = (np.array(scores_forward) + np.array(scores_backward)) / 2

        return avg_scores

    def _determine_consistency(self,
                               entailment_score: float,
                               contradiction_score: float,
                               neutral_score: float) -> ConsistencyResult:
        """
        Determine consistency from the scores

        Simplified logic: directly compare the three scores and take the label of the maximum
        - max entailment -> consistent
        - max contradiction -> inconsistent
        - max neutral -> neutral (treated as consistent, conservative strategy)
        """
        # Find the maximum score and its corresponding label
        scores = {
            'entailment': entailment_score,
            'contradiction': contradiction_score,
            'neutral': neutral_score
        }

        max_label = max(scores, key=scores.get)
        max_score = scores[max_label]

        # Determine consistency based on the label of the maximum score
        if max_label == 'entailment':
            return ConsistencyResult(
                is_consistent=True,
                level=ConsistencyLevel.CONSISTENT,
                confidence=max_score,
                entailment_score=entailment_score,
                contradiction_score=contradiction_score,
                neutral_score=neutral_score,
                explanation=f"Entailment has highest score ({entailment_score:.3f})",
                method='nli_entailment'
            )
        elif max_label == 'contradiction':
            return ConsistencyResult(
                is_consistent=False,
                level=ConsistencyLevel.INCONSISTENT,
                confidence=max_score,
                entailment_score=entailment_score,
                contradiction_score=contradiction_score,
                neutral_score=neutral_score,
                explanation=f"Contradiction has highest score ({contradiction_score:.3f})",
                method='nli_contradiction'
            )
        else:  # neutral
            # Neutral state, conservative strategy: treat as consistent
            return ConsistencyResult(
                is_consistent=True,
                level=ConsistencyLevel.NEUTRAL,
                confidence=max_score,
                entailment_score=entailment_score,
                contradiction_score=contradiction_score,
                neutral_score=neutral_score,
                explanation=f"Neutral has highest score ({neutral_score:.3f}), treating as consistent",
                method='nli_neutral'
            )

    def batch_check_consistency(self,
                                pairs: List[Tuple[str, str]]) -> List[ConsistencyResult]:
        """
        Batch-check consistency

        Args:
            pairs: List of text pairs [(text1, text2), ...]

        Returns:
            List[ConsistencyResult]: List of consistency check results
        """
        if not pairs:
            return []

        try:
            # Batch inference
            if self.bidirectional:
                # Build forward and backward pairs
                forward_pairs = pairs
                backward_pairs = [(t2, t1) for t1, t2 in pairs]

                all_pairs = forward_pairs + backward_pairs
                all_scores = self.model.predict(all_pairs)

                # Split and average
                n = len(pairs)
                forward_scores = np.array(all_scores[:n])
                backward_scores = np.array(all_scores[n:])
                avg_scores = (forward_scores + backward_scores) / 2
            else:
                avg_scores = np.array(self.model.predict(pairs))

            # Parse the results
            results = []
            for i, scores in enumerate(avg_scores):
                result = self._determine_consistency(
                    entailment_score=scores[self.entailment_idx],
                    contradiction_score=scores[self.contradiction_idx],
                    neutral_score=scores[self.neutral_idx]
                )
                results.append(result)

            return results

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error during batch consistency check: {e}")
            # Return an error result
            return [
                ConsistencyResult(
                    is_consistent=False,
                    level=ConsistencyLevel.ERROR,
                    confidence=0.0,
                    entailment_score=0.0,
                    contradiction_score=0.0,
                    neutral_score=0.0,
                    explanation=f"Batch error: {str(e)}",
                    method='error'
                )
                for _ in pairs
            ]
