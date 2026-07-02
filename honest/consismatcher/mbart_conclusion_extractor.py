# -*- coding: utf-8 -*-
"""
MBart Conclusion Extractor: Use mbart-large-50-extractive-conclusion model for conclusion extraction

This module provides a conclusion extractor based on the mBART model fine-tuned for
extractive conclusion generation from long-form text.
"""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MBartExtractionResult:
    """Extraction result from MBart model"""
    conclusion: str           # Extracted conclusion text
    confidence: float         # Confidence score (0-1)
    start_position: int       # Start position in original text (approximated)
    end_position: int         # End position in original text (approximated)
    method: str               # Extraction method: 'mbart_model'


class MBartConclusionExtractor:
    """
    Conclusion Extractor using mbart-large-50-extractive-conclusion model

    This model is specifically fine-tuned for extracting conclusions from long-form text.
    It performs extractive summarization to identify the key conclusion sentences.
    """

    def __init__(self,
                 model_name: str = 'XiaHan19/mbart-large-50-extractive-conclusion',
                 device: str = None,
                 max_input_length: int = 1024,
                 max_output_length: int = 256,
                 num_beams: int = 4,
                 tail_chars: int = 1536):
        """
        Initialize MBart Conclusion Extractor

        Args:
            model_name: Model name from Hugging Face Hub
            device: Device to run on ('cuda', 'cpu', None=auto)
            max_input_length: Maximum input length in tokens
            max_output_length: Maximum output length in tokens
            num_beams: Number of beams for beam search
            tail_chars: Number of characters to extract from the end (conclusions are typically at the end)
        """
        self.model_name = model_name
        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.tail_chars = tail_chars

        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load mBART model and tokenizer"""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            # Determine device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            logger.info(f"Loading mBART model: {self.model_name} on {self.device}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            logger.info("mBART model loaded successfully")

        except (OSError, RuntimeError, ImportError, ValueError) as e:
            logger.error(f"Failed to load mBART model: {e}")
            raise

    def _prepare_input_text(self, answer: str):
        """
        Preprocess the raw answer into model input text (tail-truncation strategy)

        Conclusions typically appear at the end of the answer, so for long text we take only
        the tail to avoid extracting intermediate analysis steps.

        Args:
            answer: The LLM's full answer text

        Returns:
            (input_text, method): The preprocessed input text and the extraction method tag;
            returns None if the input is empty.
        """
        if not answer or not answer.strip():
            return None

        # Strategy: Extract the tail of the text since conclusions are typically at the end
        if len(answer) > self.tail_chars:
            # Find a good breaking point (sentence/paragraph boundary) to avoid cutting mid-sentence
            tail_start = len(answer) - self.tail_chars

            # Look for paragraph break first (cleaner break)
            para_break = answer.find('\n\n', tail_start, min(tail_start + 200, len(answer)))
            if para_break != -1:
                tail_start = para_break + 2
            else:
                # Look for sentence break
                sentence_break_pos = answer.find('. ', tail_start, min(tail_start + 150, len(answer)))
                if sentence_break_pos != -1:
                    tail_start = sentence_break_pos + 2

            input_text = answer[tail_start:].strip()
            method = 'mbart_tail'
            logger.debug(f"Using tail extraction: {len(input_text)} chars from position {tail_start}")
        else:
            # Text is short enough, use it all
            input_text = answer
            method = 'mbart_full'
            logger.debug(f"Using full-text extraction: {len(input_text)} chars")

        return input_text, method

    def _generate_batch(self, input_texts: list, gen_subbatch_size: int = 8) -> list:
        """
        Run mBART generation over a batch of preprocessed input texts (true batching)

        Both single-item extract and batch batch_extract share this method, uniformly using
        the generate path with output_scores=True, and directly taking HF's native
        sequences_scores as the confidence. This guarantees exact consistency with the original
        per-item implementation (including sequences truncated at max_length, for which HF
        applies a different length normalization than a natural EOS — only the native path can
        reproduce this precisely).

        VRAM control: output_scores=True accumulates the score tensor for every decoding step,
        across batch x num_beams and the full vocabulary (~250k), so memory grows linearly with
        batch x num_beams x steps x vocab. For long answers that generate up to max_length
        (~255 steps), running generate on the whole batch at once easily OOMs. Therefore we
        internally re-batch by gen_subbatch_size (default 8) and call empty_cache after each
        sub-batch to release the accumulated score tensors, keeping peak memory in a safe range
        (8x4x255xvocab ~= 8GB).

        Args:
            input_texts: List of preprocessed input texts (all non-empty)
            gen_subbatch_size: Maximum number of samples per generate() call (controls memory)

        Returns:
            List[Tuple[str, float]]: The (conclusion text, confidence) for each input
        """
        import torch

        results = [None] * len(input_texts)

        for start in range(0, len(input_texts), gen_subbatch_size):
            sub_texts = input_texts[start:start + gen_subbatch_size]

            # Tokenize the sub-batch — padding is aligned to the longest within the sub-batch, attention_mask masks pads
            inputs = self.tokenizer(
                sub_texts,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=True
            ).to(self.device)

            # Generate — output_scores=True to obtain precise sequences_scores
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=self.num_beams,
                    early_stopping=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            seq_scores = outputs.sequences_scores
            for j in range(len(sub_texts)):
                conclusion = self.tokenizer.decode(
                    outputs.sequences[j],
                    skip_special_tokens=True
                ).strip()
                if seq_scores is not None:
                    confidence = max(0.0, min(1.0, torch.exp(seq_scores[j]).item()))
                else:
                    confidence = 0.85
                results[start + j] = (conclusion, confidence)

            # Immediately release the full-vocabulary score tensors accumulated by this sub-batch to avoid stacking across sub-batches
            del outputs
            torch.cuda.empty_cache()

        return results

    @staticmethod
    def _locate_conclusion(answer: str, conclusion: str):
        """
        Locate the conclusion's position in the original text (used for position tracking)

        Consistent with the original implementation: fall back to an approximate tail
        position if exact matching fails.
        """
        start_pos = answer.lower().find(conclusion.lower())
        if start_pos != -1:
            end_pos = start_pos + len(conclusion)
        else:
            # If exact match not found, approximate position
            # (mBART may paraphrase slightly)
            start_pos = max(0, len(answer) - len(conclusion) * 2)
            end_pos = len(answer)
        return start_pos, end_pos

    def extract(self,
                answer: str,
                question: str = None,
                question_type: str = None) -> MBartExtractionResult:
        """
        Extract conclusion from LLM answer using mBART model

        Single-item extraction; internally delegates to _prepare_input_text + _generate_batch
        (batch=1). Behavior is identical to before the refactor.

        Args:
            answer: LLM's full answer text
            question: Original question (optional, not used by this model)
            question_type: Question type (optional, not used by this model)

        Returns:
            MBartExtractionResult: Extraction result
        """
        prepared = self._prepare_input_text(answer)
        if prepared is None:
            return MBartExtractionResult(
                conclusion="",
                confidence=0.0,
                start_position=0,
                end_position=0,
                method='mbart_empty'
            )

        input_text, method = prepared

        try:
            conclusion, confidence = self._generate_batch([input_text])[0]

            # Position tracking
            start_pos, end_pos = self._locate_conclusion(answer, conclusion)

            return MBartExtractionResult(
                conclusion=conclusion,
                confidence=confidence,
                start_position=start_pos,
                end_position=end_pos,
                method=method
            )

        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"mBART extraction failed: {e}")
            raise RuntimeError(f"mBART extraction failed: {e}") from e

    def batch_extract(self, answers: list, batch_size: int = 8) -> list:
        """
        True batched conclusion extraction

        Groups non-empty answers by batch_size, performing a single padding + generate pass,
        which can significantly improve throughput over per-item extract calls (typically
        5~15x on GPU). Uses exactly the same _generate_batch numerical path as single-item
        extract, so results are consistent.

        Empty answers behave the same as single-item extract and return an mbart_empty result
        directly.

        Args:
            answers: List of answer texts
            batch_size: Batch size

        Returns:
            List[MBartExtractionResult], in the same order and one-to-one correspondence with the input answers
        """
        import torch  # noqa: F401  (keep import style consistent with single-item)

        # 1) Preprocessing: tail-truncate each answer and record the input text and method for non-empty items
        prepared_items = []  # same length as answers; each element is None or (input_text, method, answer)
        for answer in answers:
            prepared = self._prepare_input_text(answer)
            if prepared is None:
                prepared_items.append(None)
            else:
                input_text, method = prepared
                prepared_items.append((input_text, method, answer))

        # Placeholder result (empty answer)
        empty_result = MBartExtractionResult(
            conclusion="", confidence=0.0,
            start_position=0, end_position=0, method='mbart_empty'
        )

        results = [None] * len(answers)

        # 2) Collect the global indices of all non-empty items
        nonempty_global_idxs = [i for i, item in enumerate(prepared_items) if item is not None]

        # 3) Generate in batches of batch_size
        for chunk_start in range(0, len(nonempty_global_idxs), batch_size):
            chunk_global_idxs = nonempty_global_idxs[chunk_start:chunk_start + batch_size]
            chunk_input_texts = [prepared_items[i][0] for i in chunk_global_idxs]
            chunk_answers = [prepared_items[i][2] for i in chunk_global_idxs]

            try:
                gen_results = self._generate_batch(chunk_input_texts)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                # On batch generation failure, fall back to per-answer extract for robustness and single-item parity
                logger.warning(
                    f"Batch generation failed ({e}); falling back to per-answer extraction"
                )
                for gi, ans in zip(chunk_global_idxs, chunk_answers):
                    results[gi] = self.extract(ans)
                continue

            for gi, ans, (conclusion, confidence) in zip(
                    chunk_global_idxs, chunk_answers, gen_results):
                method = prepared_items[gi][1]
                start_pos, end_pos = self._locate_conclusion(ans, conclusion)
                results[gi] = MBartExtractionResult(
                    conclusion=conclusion,
                    confidence=confidence,
                    start_position=start_pos,
                    end_position=end_pos,
                    method=method
                )

        # 4) Fill in the empty-answer placeholder results
        for i, item in enumerate(prepared_items):
            if item is None:
                results[i] = empty_result

        return results
