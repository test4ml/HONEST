#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Similarity Validator for MCQ Distractor Generation

Uses sentence embeddings to detect semantically equivalent statements,
avoiding generation of multiple correct answers in multiple choice questions.
"""

from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SemanticValidator:
    """Validate distractors are semantically different from correct answer"""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.96,
                 device: str = None):
        """
        Initialize semantic validator

        Args:
            model_name: Sentence-BERT model name
                - 'all-MiniLM-L6-v2': Fast, English (80MB)
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual (420MB)
            similarity_threshold: Threshold above which statements are considered
                                  semantically equivalent (0.0-1.0)
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.model_name = model_name
        self.device = device

        # Lazy loading - only load when first used
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Lazy load the sentence transformer model"""
        if not self._model_loaded:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                logger.info(f"Loading semantic similarity model: {self.model_name}")

                # Determine device
                if self.device is None:
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

                logger.info(f"Using device: {self.device}")

                # Load model and immediately move to target device
                self.model = SentenceTransformer(self.model_name, device=self.device)

                # Ensure model is in eval mode (no gradient computation needed)
                self.model.eval()

                self._model_loaded = True
                logger.info(f"Model loaded successfully on {self.device}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Semantic validation will use simple string matching. "
                    "Install with: pip install sentence-transformers"
                )
                self.model = None
            except (RuntimeError, OSError, ValueError) as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Semantic validation will use simple string matching")
                self.model = None

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        self._ensure_model_loaded()

        if self.model is None:
            # Fallback: if model not available, use simple string matching
            # Normalize texts for comparison
            norm1 = text1.lower().strip()
            norm2 = text2.lower().strip()
            return 1.0 if norm1 == norm2 else 0.0

        try:
            import torch

            # Encode sentences to embeddings
            # Set show_progress_bar=False to disable progress bar display
            # Use no_grad context to disable gradient computation for inference
            with torch.no_grad():
                embeddings = self.model.encode([text1, text2],
                                              convert_to_numpy=True,
                                              show_progress_bar=False,
                                              device=self.device)

            # Compute cosine similarity
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            return float(similarity)
        except (RuntimeError, ValueError, ImportError) as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def is_semantically_distinct(self,
                                correct_answer: str,
                                distractor: str) -> bool:
        """
        Check if distractor is semantically distinct from correct answer

        Args:
            correct_answer: The correct answer statement
            distractor: A candidate distractor statement

        Returns:
            True if distractor is sufficiently different (valid distractor)
            False if distractor is too similar (should be rejected)
        """
        similarity = self.compute_similarity(correct_answer, distractor)

        is_distinct = similarity < self.similarity_threshold

        if not is_distinct:
            logger.debug(
                f"Rejected distractor (similarity={similarity:.3f}): "
                f"\n  Correct: {correct_answer}"
                f"\n  Distractor: {distractor}"
            )

        return is_distinct

    def filter_distractors(self,
                          correct_answer: str,
                          candidate_distractors: List[str]) -> List[str]:
        """
        Filter out semantically similar distractors

        Args:
            correct_answer: The correct answer statement
            candidate_distractors: List of candidate distractor statements

        Returns:
            Filtered list of valid distractors
        """
        valid_distractors = []

        for distractor in candidate_distractors:
            if self.is_semantically_distinct(correct_answer, distractor):
                valid_distractors.append(distractor)

        return valid_distractors

    def filter_distractors_with_scores(self,
                                      correct_answer: str,
                                      candidate_distractors: List[str]) -> List[Tuple[str, float]]:
        """
        Filter distractors and return with similarity scores

        Args:
            correct_answer: The correct answer statement
            candidate_distractors: List of candidate distractor statements

        Returns:
            List of (distractor, similarity_score) tuples for valid distractors
        """
        results = []

        for distractor in candidate_distractors:
            similarity = self.compute_similarity(correct_answer, distractor)
            if similarity < self.similarity_threshold:
                results.append((distractor, similarity))

        # Sort by similarity (ascending) - prefer more distinct distractors
        results.sort(key=lambda x: x[1])

        return results


# Global instance with default settings
semantic_validator = SemanticValidator(
    model_name='all-MiniLM-L6-v2',
    similarity_threshold=0.99,  # Raised from 0.95 to 0.99 to reduce over-filtering
    device=None  # Auto-detect: use CUDA if available, otherwise CPU
)
