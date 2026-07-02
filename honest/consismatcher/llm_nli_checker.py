# -*- coding: utf-8 -*-
"""
LLM NLI Consistency Checker: uses a remote/local LLM (e.g., DeepSeek-V4-Flash) over an OpenAI/Anthropic-compatible URL to perform NLI judging

Its interface (check_consistency / batch_check_consistency) is fully compatible with NLIConsistencyChecker and can
serve as a parallel backend to the cross-encoder NLI, used for the ablation comparison "LLM judges consistency" vs
"NLI small model judges consistency".

Robustness strategy:
  - Force the LLM to output JSON (allows wrapping in a markdown ```json code block)
  - Robust parsing: strip fence -> json.loads -> regex-extract the first {...} -> if all fails, retry once
  - Label canonicalization: accept entailment/contradiction/neutral and common abbreviations / casing
  - Score synthesis: the selected label = confidence, the other two = (1-confidence)/2, for downstream rq3 score analysis reuse
  - Consistency policy aligned with NLIConsistencyChecker: neutral is treated as consistent (conservative, fewer false negatives)
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from .nli_checker import ConsistencyResult, ConsistencyLevel

logger = logging.getLogger(__name__)


# ---- Prompt (Premise/Hypothesis standard NLI terminology to avoid triggering truth/falsity judging) ----
_NLI_SYSTEM = (
    "You are an NLI (Natural Language Inference) classifier. Given a PREMISE and a HYPOTHESIS, "
    "output their relation as a single JSON object with the EXACT keys: label, confidence, reason.\n\n"
    "label must be exactly one of:\n"
    "- \"entailment\": the hypothesis follows from the premise, or they mean the same thing "
    "(paraphrase; identical sentences are always entailment).\n"
    "- \"contradiction\": the hypothesis asserts the opposite of the premise.\n"
    "- \"neutral\": neither entails nor contradicts (unrelated / not enough info).\n\n"
    "Judge ONLY from the literal meaning of the two sentences; do NOT use world knowledge.\n"
    "Respond with ONLY the JSON object, starting with `{`. Example:\n"
    "{\"label\": \"entailment\", \"confidence\": 0.95, \"reason\": \"same claim\"}"
)

_NLI_USER_TMPL = """Premise: {conc_a}
Hypothesis: {conc_b}

Respond with ONLY the JSON object (keys: label, confidence, reason):"""


# ---- JSON parsing (robust) ----
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", re.DOTALL)
# Greedy match of the first balanced {...} (non-greedy across the whole block; then validated by json.loads)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_nli_json(raw: str) -> Optional[dict]:
    """Parse JSON from the LLM output. Returns {label, confidence, reason} or None.

    Order: strip markdown fence -> json.loads -> regex-extract the first {...} then loads.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()

    # 1) Try the whole thing first (may already be a bare JSON)
    candidates = [text]

    # 2) Strip the ```json ... ``` fence and take the inner content
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())

    # 3) Regex-extract the first {...}
    bm = _BRACE_RE.search(text)
    if bm:
        candidates.append(bm.group(0).strip())

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# Label canonicalization mapping
_LABEL_ALIASES = {
    # entailment
    "entailment": "entailment", "entail": "entailment", "entails": "entailment",
    "yes": "entailment", "same": "entailment", "consistent": "entailment",
    "paraphrase": "entailment", "equivalent": "entailment", "e": "entailment",
    # contradiction
    "contradiction": "contradiction", "contradict": "contradiction",
    "contradicts": "contradiction", "no": "contradiction", "opposite": "contradiction",
    "inconsistent": "contradiction", "conflict": "contradiction", "c": "contradiction",
    # neutral
    "neutral": "neutral", "neu": "neutral", "unknown": "neutral", "unrelated": "neutral",
    "insufficient": "neutral", "n": "neutral",
}


def _canonical_label(obj: dict) -> str:
    """Extract and canonicalize the label from the parsed dict. Invalid/missing -> neutral.

    Tolerance strategy:
      1) First look in common label fields (label/relation/judgment/result/answer), case-insensitive;
      2) Accommodate the case where the model uses the task name as the key and the label as the value
         (e.g., {"entailment": "contradiction"}): scan all keys and string values, take the first one
         that hits _LABEL_ALIASES (and exclude the case where the key itself is a misused task name);
      3) Still none -> neutral.
    """
    if not isinstance(obj, dict) or not obj:
        return "neutral"

    def _norm_tok(s: str) -> str:
        s = re.sub(r"^(label\s*[:：]\s*)?([a-zA-Z]\.\s*)", "", str(s).strip().lower()).strip()
        return s.split()[0].rstrip(".,;:!") if s else ""

    lower_obj = {k.lower(): v for k, v in obj.items()}

    # 1) Standard label fields
    for k in ("label", "relation", "judgment", "result", "answer", "prediction"):
        v = lower_obj.get(k)
        if v:
            tok = _norm_tok(v)
            if tok in _LABEL_ALIASES:
                return _LABEL_ALIASES[tok]

    # 2) Scan all values (strings), take the first legal label
    for v in obj.values():
        if not isinstance(v, str):
            continue
        tok = _norm_tok(v)
        if tok in _LABEL_ALIASES:
            return _LABEL_ALIASES[tok]

    # 3) Scan all keys (fallback, e.g., {"entailment": ...} but the value is invalid)
    for k in obj.keys():
        tok = _norm_tok(k)
        if tok in _LABEL_ALIASES:
            return _LABEL_ALIASES[tok]

    return "neutral"


def _safe_confidence(obj: dict, default: float = 0.9) -> float:
    for k in ("confidence", "conf", "score"):
        if k in obj:
            try:
                v = float(obj[k])
                return max(0.0, min(1.0, v))
            except (TypeError, ValueError):
                continue
    return default


def _scores_from_label(label: str, conf: float):
    """Selected label = conf; the other two share the remainder equally. Returns (entail, contra, neutral)."""
    rest = (1.0 - conf) / 2.0
    if label == "entailment":
        return conf, rest, rest
    if label == "contradiction":
        return rest, conf, rest
    return rest, rest, conf  # neutral


def _result_from_label(label: str, conf: float, reason: str, method_suffix: str) -> ConsistencyResult:
    ent, con, neu = _scores_from_label(label, conf)
    if label == "entailment":
        return ConsistencyResult(
            is_consistent=True, level=ConsistencyLevel.CONSISTENT, confidence=conf,
            entailment_score=ent, contradiction_score=con, neutral_score=neu,
            explanation=reason or f"LLM judged entailment ({conf:.2f})",
            method=f"nli_llm_entailment_{method_suffix}".rstrip("_"),
        )
    if label == "contradiction":
        return ConsistencyResult(
            is_consistent=False, level=ConsistencyLevel.INCONSISTENT, confidence=conf,
            entailment_score=ent, contradiction_score=con, neutral_score=neu,
            explanation=reason or f"LLM judged contradiction ({conf:.2f})",
            method=f"nli_llm_contradiction_{method_suffix}".rstrip("_"),
        )
    # neutral -> treat as consistent (conservative, aligned with NLIConsistencyChecker)
    return ConsistencyResult(
        is_consistent=True, level=ConsistencyLevel.NEUTRAL, confidence=conf,
        entailment_score=ent, contradiction_score=con, neutral_score=neu,
        explanation=reason or f"LLM judged neutral ({conf:.2f}), treating as consistent",
        method=f"nli_llm_neutral_{method_suffix}".rstrip("_"),
    )


def _error_result(reason: str) -> ConsistencyResult:
    return ConsistencyResult(
        is_consistent=False, level=ConsistencyLevel.ERROR, confidence=0.0,
        entailment_score=0.0, contradiction_score=0.0, neutral_score=0.0,
        explanation=reason, method="error",
    )


class LLMNLIConsistencyChecker:
    """URL-based LLM NLI consistency checker (defaults to DeepSeek-V4-Flash, anthropic protocol)"""

    def __init__(self,
                 base_url: str = 'https://api.deepseek.com/anthropic',
                 api_key: str = None,
                 model_name: str = 'deepseek-v4-flash',
                 protocol: str = 'anthropic',
                 max_tokens: int = 256,
                 temperature: float = 0.0,
                 max_concurrent: int = 20,
                 rate_limit: float = 20.0,
                 timeout: int = 120,
                 enable_thinking: bool = False,
                 method_suffix: str = ""):
        """
        Args:
            base_url: API service address (DeepSeek anthropic: https://api.deepseek.com/anthropic)
            api_key: API key (read from env DEEPSEEK_API_KEY by default)
            model_name: Model name (defaults to deepseek-v4-flash)
            protocol: Protocol (DeepSeek uses anthropic; openai is also supported)
            max_tokens: Output upper bound per call (JSON is short, 256 is enough)
            temperature: 0 to ensure determinism
            max_concurrent: Thread concurrency for batch judging (conservative recommended for remote APIs)
            rate_limit: Maximum requests per second
            timeout: Per-request timeout (seconds)
            enable_thinking: Whether to enable thinking (off by default for DeepSeek to speed up / save cost)
            method_suffix: Suffix written into ConsistencyResult.method (used to tag the extractor source, e.g., "llm")
        """
        import os
        from honest.llm import create_sync_llm_client

        if api_key is None:
            api_key = os.environ.get('DEEPSEEK_API_KEY', 'your-api-key')

        self.max_concurrent = max_concurrent
        self.method_suffix = method_suffix or ""
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
        logger.info(f"LLMNLIConsistencyChecker initialized: {model_name} @ {base_url} ({protocol})")

    def _build_user(self, conc_a: str, conc_b: str) -> str:
        return _NLI_USER_TMPL.format(conc_a=(conc_a or "")[:1500], conc_b=(conc_b or "")[:1500])

    def _judge_one(self, conc_a: str, conc_b: str, strict_retry: bool = True) -> ConsistencyResult:
        """Judge a single pair. On parse failure, retry once with an 'only JSON' instruction appended."""
        if not (conc_a and conc_a.strip() and conc_b and conc_b.strip()):
            return _error_result("Empty conclusion input")

        user = self._build_user(conc_a, conc_b)
        try:
            raw = self.client.generate_answer(user, max_retries=2, system_prompt=_NLI_SYSTEM)
        except Exception as e:
            logger.error(f"LLM NLI request failed: {e}")
            return _error_result(f"Request error: {e}")

        if isinstance(raw, str) and raw.startswith("ERROR"):
            logger.error(f"LLM NLI returned error: {raw[:120]}")
            return _error_result(raw[:200])

        obj = _parse_nli_json(raw)
        if obj is None and strict_retry:
            logger.debug(f"JSON parse failed, retrying strict. raw={raw[:120]!r}")
            try:
                raw2 = self.client.generate_answer(
                    user + "\n\nIMPORTANT: output ONLY a valid JSON object, no other text.",
                    max_retries=2, system_prompt=_NLI_SYSTEM)
                if not (isinstance(raw2, str) and raw2.startswith("ERROR")):
                    obj = _parse_nli_json(raw2)
            except Exception as e:
                logger.error(f"LLM NLI strict retry failed: {e}")

        if obj is None:
            logger.warning(f"LLM NLI JSON unparseable, defaulting neutral. raw={raw[:160]!r}")
            return _result_from_label("neutral", 0.5, "Unparseable LLM output, defaulted neutral",
                                      self.method_suffix)

        label = _canonical_label(obj)
        conf = _safe_confidence(obj)
        reason = str(obj.get("reason") or obj.get("explanation") or "").strip()
        return _result_from_label(label, conf, reason, self.method_suffix)

    def check_consistency(self, text1: str, text2: str) -> ConsistencyResult:
        """Judge consistency for a single pair (interface aligned with NLIConsistencyChecker)."""
        return self._judge_one(text1, text2)

    def batch_check_consistency(self, pairs: List[Tuple[str, str]]) -> List[ConsistencyResult]:
        """Batch judging: thread-pool concurrency, in one-to-one order with pairs."""
        if not pairs:
            return []
        results: List[Optional[ConsistencyResult]] = [None] * len(pairs)
        workers = max(1, min(len(pairs), self.max_concurrent))

        def _work(i):
            t1, t2 = pairs[i]
            return i, self._judge_one(t1, t2)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, res in pool.map(_work, range(len(pairs))):
                results[i] = res

        for i in range(len(results)):
            if results[i] is None:
                results[i] = _error_result("Worker returned None")
        return results
