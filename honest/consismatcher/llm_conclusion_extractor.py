# -*- coding: utf-8 -*-
"""
LLM Conclusion Extractor: extracts conclusions via a local small model (e.g., Qwen3-8B) over an OpenAI-compatible URL

Its interface (extract / batch_extract) is fully compatible with MBartConclusionExtractor and can serve as a drop-in
replacement. The mBART extractive model often grabs the wrong sentence on long answers (a title, a premise fragment,
an intermediate step, or even a flipped stance); here we switch to an instruction-following LLM and explicitly require
it to output only the "final conclusion sentence" to avoid these false positives.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class LLMExtractionResult:
    """LLM extraction result (fields aligned with MBartExtractionResult)"""
    conclusion: str           # Extracted conclusion text
    confidence: float         # Confidence (0-1); the LLM does not natively produce a score, so we use a heuristic
    start_position: int       # Approximate start offset in the original text
    end_position: int         # Approximate end offset in the original text
    method: str               # Extraction method tag: 'llm'


# Extraction prompt: force the model to output a complete conclusion sentence; forbid bare tokens and forbid the model from answering on its own
_EXTRACT_SYSTEM = (
    "You are a strict conclusion extractor. A model's ANSWER to a question is given to you. "
    "Your ONLY job is to restate the FINAL CONCLUSION that the given answer reaches, as "
    "ONE complete declarative sentence.\n\n"
    "CRITICAL: You are NOT answering the question yourself. Do NOT use your own world knowledge "
    "to judge whether the answer is true or false. Just faithfully restate what the given answer "
    "concludes, even if you believe it is factually wrong.\n\n"
    "Requirements:\n"
    "1. Output exactly ONE complete sentence stating the final verdict/claim of the given answer.\n"
    "2. The sentence MUST contain the subject AND the claim, e.g. "
    "'Yes, captain has the next higher rank of major.' or "
    "'The next crossing downstream from X is Y.'\n"
    "3. NEVER output a bare token or a lone entity name. Forbidden outputs include: 'Yes', 'No', "
    "'True', 'False', a lone option letter ('B'), a lone entity ('Mayport Ferry', 'Marshal of the branch'), "
    "or a vague stance without the claim ('Yes, it is true'). Always restate the SPECIFIC claim (the entity + relationship + value).\n"
    "4. Prefer the explicit conclusion sentence at the END of the given answer; copy it verbatim.\n"
    "5. Do NOT copy reasoning steps, premises, definitions, or titles.\n"
    "6. No quotation marks; no prefix like 'Conclusion:' or 'Answer:'.\n\n"
    "Good: 'Yes, METI is a child organization of the Cabinet of Japan.'\n"
    "Bad (forbidden, too short): 'Yes.' | 'Mayport Ferry.' | 'B.'"
)

_EXTRACT_USER_TMPL = """Question:
{question}

Model's full answer:
{answer}

Final conclusion:"""


def _truncate_tail(text: str, tail_chars: int = 4000) -> str:
    """For long answers take the tail (conclusions are usually at the end); consistent with mBART's tail strategy and saves tokens"""
    text = (text or "").strip()
    if len(text) > tail_chars:
        return text[-tail_chars:]
    return text


def _clean_conclusion(raw: str, answer: str) -> str:
    """Clean the LLM output: strip markdown bold/quotes/prefixes/extra whitespace"""
    if not raw:
        return ""
    c = raw.strip()
    # Strip markdown bold **...** (keep the inner text)
    c = re.sub(r'\*\*(.+?)\*\*', r'\1', c)
    c = re.sub(r'\*(.+?)\*', r'\1', c)
    c = re.sub(r'__(.+?)__', r'\1', c)
    # Strip common prefixes such as "Conclusion:", "最终结论：", "The conclusion is"
    c = re.sub(r'^(conclusion|final conclusion|最终结论|结论|answer)\s*[:：]?\s*', '', c, flags=re.IGNORECASE)
    # Strip surrounding quotes
    if (c.startswith('"') and c.endswith('"')) or (c.startswith("'") and c.endswith("'")):
        c = c[1:-1].strip()
    if (c.startswith('“') and c.endswith('”')) or (c.startswith('‘') and c.endswith('’')):
        c = c[1:-1].strip()
    # The model may split a "question + answer" across multiple lines; convert newlines to spaces to avoid dropping the real conclusion on the second line
    c = c.replace('\n', ' ').replace('\r', ' ').strip()
    c = re.sub(r'\s+', ' ', c)
    return c


# Decide whether a conclusion is a "bare token / no substantive claim" — these cause an asymmetric comparison on both NLI sides
_BARE_WORDS = {"yes", "no", "true", "false", "maybe", "unknown", "uncertain",
               "correct", "incorrect", "none", "n/a", "na", "ok"}

# A qualified conclusion sentence must contain at least one verb / relation / stance word; otherwise it is treated as a bare entity name / phrase
_CLAIM_WORDS = {
    # copula / auxiliary
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    # common relation / attribute words (high frequency in KG conclusions)
    "opposite", "located", "encoded", "adjacent", "child", "parent", "subordinate",
    "higher", "lower", "rank", "synonym", "basionym", "pronounced", "below", "above",
    "contributes", "married", "member", "part", "belongs", "derived", "studied",
    "coextensive", "coordinate", "instance", "described", "contains", "includes",
    "crossing", "level", "tier", "valid", "invalid", "follows", "means", "called",
    # stance words
    "yes", "no", "true", "false", "not", "cannot", "option",
}


def _is_bare_conclusion(conclusion: str) -> bool:
    """Whether the conclusion is too short / a bare token / verb-less (needs fallback to a full sentence)."""
    if not conclusion:
        return True
    t = conclusion.strip().rstrip(".!?").strip()
    if len(t) < 15:                      # too short, e.g. "Mayport Ferry"
        return True
    if t.lower() in _BARE_WORDS:         # bare Yes/No
        return True
    if len(t.split()) <= 3:              # too few words, e.g. "Option B" / "Marshal of branches"
        return True
    # No verb / relation / stance word at all -> most likely a bare entity-name phrase, e.g. "Marshal of the branch"
    toks = re.findall(r"[a-zA-Z]+", t.lower())
    if not any(tok in _CLAIM_WORDS for tok in toks):
        return True
    return False


def _last_substantive_sentence(answer: str) -> str:
    """Take the last substantive conclusion sentence from the end of the answer (fallback when the LLM emits a bare token).

    Conclusions usually sit at the end of the answer; split on sentence-ending punctuation / newlines, then scan
    backwards for the first sentence that is long enough and not a question.
    """
    if not answer:
        return ""
    parts = re.split(r'(?<=[.!?])\s+|\n+', answer.strip())
    parts = [p.strip().strip('*').strip() for p in parts if p.strip()]
    for s in reversed(parts):
        if len(s) >= 15 and not s.endswith("?"):
            return s[:300]
    return parts[-1][:300] if parts else ""


class LLMConclusionExtractor:
    """URL-based LLM conclusion extractor (defaults to the local Qwen3-8B)"""

    def __init__(self,
                 base_url: str = 'http://localhost:8001/v1',
                 api_key: str = 'your-api-key',
                 model_name: str = 'Qwen3-8B',
                 protocol: str = 'openai',
                 max_tokens: int = 128,
                 temperature: float = 0.0,
                 max_concurrent: int = 32,
                 rate_limit: float = 200.0,
                 timeout: int = 120,
                 enable_thinking: bool = False,
                 tail_chars: int = 4000):
        """
        Args:
            base_url: OpenAI-compatible service address (local vLLM defaults to localhost:8001)
            api_key: API key (any value works for a local service)
            model_name: Model name (defaults to Qwen3-8B)
            protocol: Protocol (use 'openai' for local vLLM)
            max_tokens: Output upper bound per extraction
            temperature: 0 to ensure determinism
            max_concurrent: Thread concurrency for batch extraction
            rate_limit: Maximum requests per second (can be set high for a local service)
            timeout: Per-request timeout (seconds)
            enable_thinking: Whether to enable thinking mode (off by default for Qwen3 to speed up)
            tail_chars: Tail-truncation length for long answers
        """
        from honest.llm import create_sync_llm_client

        self.tail_chars = tail_chars
        self.max_concurrent = max_concurrent
        self.client = create_sync_llm_client(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            protocol=protocol,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limit=rate_limit,
            timeout=timeout,
            enable_thinking=enable_thinking,
        )
        logger.info(f"LLMConclusionExtractor initialized: {model_name} @ {base_url}")

    def _build_prompt(self, answer: str, question: Optional[str]) -> str:
        q = (question or "(not provided)").strip()
        a = _truncate_tail(answer, self.tail_chars)
        return _EXTRACT_USER_TMPL.format(question=q[:800], answer=a)

    def _extract_one(self, answer: str, question: Optional[str]):
        if not answer or not answer.strip():
            return "", 0.0
        prompt = self._build_prompt(answer, question)
        try:
            raw = self.client.generate_answer(prompt, max_retries=2,
                                              system_prompt=_EXTRACT_SYSTEM)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return "", 0.0
        if isinstance(raw, str) and raw.startswith("ERROR"):
            logger.error(f"LLM extraction returned error: {raw[:120]}")
            raw = ""
        conclusion = _clean_conclusion(raw, answer)
        # Bare-token fallback: the LLM occasionally emits only "Yes." / a bare entity name; in that case grab the complete conclusion sentence from the end of the answer
        if _is_bare_conclusion(conclusion):
            fb = _last_substantive_sentence(answer)
            if fb:
                logger.debug(f"Bare conclusion {conclusion!r} -> fallback to last sentence")
                conclusion = fb
        # Heuristic confidence: locatable in the original text -> high confidence; otherwise medium
        if conclusion and conclusion.lower() in answer.lower():
            conf = 0.95
        elif conclusion:
            conf = 0.8
        else:
            conf = 0.0
        return conclusion, conf

    @staticmethod
    def _locate(answer: str, conclusion: str):
        if not conclusion:
            return 0, 0
        sp = answer.lower().find(conclusion.lower())
        if sp != -1:
            return sp, sp + len(conclusion)
        return max(0, len(answer) - len(conclusion) * 2), len(answer)

    def extract(self, answer: str, question: str = None,
                question_type: str = None) -> LLMExtractionResult:
        """Single-item extraction"""
        conclusion, conf = self._extract_one(answer, question)
        sp, ep = self._locate(answer or "", conclusion)
        return LLMExtractionResult(
            conclusion=conclusion,
            confidence=conf,
            start_position=sp,
            end_position=ep,
            method='llm',
        )

    def batch_extract(self, answers: List[str], batch_size: int = 32) -> List[LLMExtractionResult]:
        """
        Batch extraction. batch_size is reused here as the thread concurrency upper bound (taking min(batch_size, max_concurrent)).
        Order corresponds one-to-one with answers; empty answers return an llm_empty result.
        """
        answers = list(answers)
        results: List[Optional[LLMExtractionResult]] = [None] * len(answers)

        # Empty answers get a placeholder directly
        nonempty_idxs = [i for i, a in enumerate(answers) if a and str(a).strip()]
        if not nonempty_idxs:
            return [self._empty_result() for _ in answers]

        workers = max(1, min(batch_size, self.max_concurrent))

        def _work(i):
            a = answers[i]
            conclusion, conf = self._extract_one(a, None)
            sp, ep = self._locate(a, conclusion)
            return i, LLMExtractionResult(conclusion=conclusion, confidence=conf,
                                          start_position=sp, end_position=ep, method='llm')

        # Thread-pool concurrency; SyncLLMClient already does locking and rate limiting internally, so it is thread-safe
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, res in pool.map(_work, nonempty_idxs):
                results[i] = res

        for i in range(len(results)):
            if results[i] is None:
                results[i] = self._empty_result()
        return results

    @staticmethod
    def _empty_result() -> LLMExtractionResult:
        return LLMExtractionResult(conclusion="", confidence=0.0,
                                   start_position=0, end_position=0, method='llm_empty')
