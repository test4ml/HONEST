"""
MetaQA baseline implementation (Yang et al., FSE 2025).

Paper-faithful adaptation of MetaQA's hallucination-detection approach:
MetaQA generates **synonym** and **antonym** mutations of the LLM's base
response, verifies each mutated statement against facts, and computes a
hallucination score.

Reference:
    Yang et al., "Hallucination Detection in Large Language Models with
    Metamorphic Relations", FSE 2025 (arXiv 2502.15844).
    Local copy: ``papers/MetaQA_2502.15844.md``

Workflow (per question):
    1. Get the LLM's concise answer, then **summarize** it into one short,
       accurate sentence (paper Section 3.1, line ~112) -> ``base_response``.
    2. Generate N synonym mutations (semantically equivalent paraphrases).
    3. Generate M antonym mutations (negations of the answer).
    4. Verify each mutation with the **single, MR-type-agnostic** fact-check
       verifier released with the official MetaQA code
       (``FACT_VERIFICATION_PROMPT``). The synonym/antonym distinction enters
       only in the scoring step (Step 5): a synonym of a true answer should
       verify "Yes", an antonym should verify "No".
    5. Calculate the hallucination score (paper Section 3.4) and classify with
       threshold theta (default 0.5).

------------------------------------------------------------------------------
Paper-vs-implementation provenance table
------------------------------------------------------------------------------
| Item                                   | Paper loc        | Source    | Notes |
|----------------------------------------|------------------|-----------|-------|
| MR types = synonym + antonym           | line 44, 70      | verbatim  |       |
| SYNONYM_MUTATION_PROMPT (Query block)  | Table 1 ~175     | verbatim  |       |
| ANTONYM_MUTATION_PROMPT (Query block)  | Table 1 ~175     | verbatim  |       |
| Instruction: lines (synonym/antonym)   | Table 1 ~175     | verbatim  | restored (see P1-1) |
| Verification prompt                    | released code    | verbatim  | single MR-agnostic FACT_VERIFICATION_PROMPT; MR only affects scoring |
| SYN_SCORE / ANT_SCORE                  | Eq. ~189-197     | verbatim  |       |
| Not Sure = 0.5                         | line ~185        | verbatim  |       |
| Total score = sum / (N + M)            | Eq.5 ~202        | verbatim  |       |
| N = M = 5 (total 10)                   | line ~368 (RQ4)  | verbatim  | default |
| Step 1 summarize                       | line ~112        | authored prompt | see P2-1 |
| Threshold theta = 0.5                  | line ~205, 227   | verbatim  | see P2-2 |
| Concise-answer system prompt           | NOT released     | authored  |       |
------------------------------------------------------------------------------
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Enums & data-classes
# ---------------------------------------------------------------------------

class MetaQAMutationType(str, Enum):
    """The two metamorphic relation types from MetaQA."""
    SYNONYM = "Synonym"
    ANTONYM = "Antonym"


@dataclass
class MetaQAMutation:
    """A single mutation (synonym or antonym) of the base response."""
    mr_type: str                       # "Synonym" or "Antonym"
    original_question: str             # the original question
    base_response: str                 # the LLM's original answer
    mutation_text: str                 # the mutated statement text
    verification_result: str = ""      # "Yes" / "No" / "Not Sure" (filled after verification)
    hallucination_contribution: float = 0.0  # score contribution (filled after verification)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
# Provenance notes:
#   - The ``Instruction:`` / ``Query:`` blocks below come verbatim from
#     MetaQA Table 1 (line ~175). The ``Instruction:`` lines were previously
#     dropped and are now retained (P1-1).
#   - The concise-answer system prompt and the verification prompts are NOT
#     released in the paper and are authored here (documented inline).

# Table 1 ``Instruction:`` lines (verbatim) — paper Table 1, line ~175.
SYNONYM_INSTRUCTION = "Generate synonym mutations of the base response"
ANTONYM_INSTRUCTION = "Generate antonym mutations of the base response"

# Step 1: Guide LLM to produce a concise answer. [AUTHORED — not in paper]
CONCISE_ANSWER_SYSTEM = (
    "You are a factual question-answering assistant. "
    "Provide a concise, fact-based answer. "
    "Do not add explanations or extra context. "
    "Just give the direct answer."
)

# Step 1b: Summarize the concise answer into one short sentence.
# Paper Section 3.1 (line ~112): "summarize its responses into short, accurate
# sentences". The exact summarize prompt is not released; this is [AUTHORED].
SUMMARIZE_PROMPT = (
    "Summarize the following answer into one short, accurate sentence that "
    "preserves the full meaning. Do not add any new information.\n\n"
    "Answer: {answer}\n"
    "Summarized sentence:"
)

# Step 2a: Synonym mutation generation.
# The Query block is verbatim from Table 1; the Instruction line is prepended
# (P1-1). The few-shot example is part of the paper's Query block.
SYNONYM_MUTATION_PROMPT = (
    "Instruction: " + SYNONYM_INSTRUCTION + "\n\n"
    "Generate synonym mutations of the answer based on the context of the "
    "question and return a numbered list to me. Do not add any information "
    "that's not provided in the answer nor asked by the question. Make sure "
    "the generated synonyms are meaningful sentences.\n\n"
    "For example:\n"
    "Question: What is the most popular sport in Japan?\n"
    "Answer: Baseball is the most popular sport in Japan.\n"
    "Mutations:\n"
    "1. Japan holds baseball as its most widely embraced sport.\n"
    "2. The sport with the highest popularity in Japan is baseball.\n"
    "3. Baseball reigns as Japan's most favored sport among the populace.\n\n"
    "Notice how the full context is included in each generated synonym. "
    "If you generated just 'baseball', it would not make a meaningful sentence.\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Mutations:"
)

# Step 2b: Antonym mutation generation (Table 1 Query block + Instruction).
ANTONYM_MUTATION_PROMPT = (
    "Instruction: " + ANTONYM_INSTRUCTION + "\n\n"
    "Generate negations of the answer based on the context of the question "
    "and return a numbered list to me. Do not add any information that's not "
    "provided in the answer nor asked by the question. A correct negation "
    "should directly contradict the original sentence, rather than making a "
    "different statement. Make sure the generated antonyms are meaningful "
    "sentences.\n\n"
    "For example:\n"
    "Question: What is the most popular sport in Japan?\n"
    "Answer: Baseball is the most popular sport in Japan.\n"
    "Mutations:\n"
    "1. The most popular sport in Japan is not baseball.\n"
    "2. Baseball is not the most popular sport in Japan.\n"
    "3. Japan does not consider baseball as the most popular sport.\n\n"
    "Be careful about double negations which make the sentence semantically "
    "same to the provided one. The context of the question is really important.\n\n"
    "Notice how the negations are meaningful sentences in the example. "
    "You should negate the meaning of the sentence based on the question.\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Mutations:"
)

# Step 3: Fact-verification prompt.
# [PAPER-VERBATIM from the released MetaQA code] The official verifier is a
# SINGLE, MR-type-agnostic fact-check prompt
# (``baseline_opensource/MetaQA/llm_prompts/prompts.py:FACT_VERIFICATION_PROMPT``):
# the synonym/antonym distinction enters only through the SCORING function
# (SynScore vs. AntScore, paper Section 3.4), NOT through the verifier prompt.
# Earlier versions of this module used two MR-aware prompts that telegraphed the
# expected outcome ("should be true"/"should be false") into the verifier — that
# was an unfaithful deviation (it biases the oracle and contradicts MetaQA's
# principle of not telling the model what to expect, Section 2.1). We now use the
# single official prompt for all mutations.
#
# The released code passes FACT_VERIFICATION_PROMPT as a system prompt and the
# mutation as the user input. Our pipeline feeds a single combined string to the
# LLM, so the mutation is appended as a ``Sentence:`` line below the official
# instruction (the instruction text itself is verbatim).
FACT_VERIFICATION_PROMPT = (
    "For the sentence, you should check whether it is correct ground truth or "
    "not. Answer YES or NO. If you are NOT SURE, answer NOT SURE. "
    "Don't return anything else except YES, NO, or NOT SURE.\n\n"
    "Sentence: {mutation_text}"
)

# Backward-compatible alias (kept so existing imports keep working).
VERIFICATION_PROMPT = FACT_VERIFICATION_PROMPT

# Deprecated MR-aware prompts (kept for import-compat only; no longer selected).
VERIFICATION_PROMPT_SYNONYM = FACT_VERIFICATION_PROMPT
VERIFICATION_PROMPT_ANTONYM = FACT_VERIFICATION_PROMPT


# ---------------------------------------------------------------------------
# Score mapping (paper Section 3.4)
# ---------------------------------------------------------------------------

# Synonym mutation score: if synonym is verified as factual -> 0.0 (no hallucination)
SYN_SCORE = {"Yes": 0.0, "No": 1.0, "Not Sure": 0.5}

# Antonym mutation score: if antonym is verified as factual -> 1.0 (hallucination)
ANT_SCORE = {"Yes": 1.0, "No": 0.0, "Not Sure": 0.5}

# Paper Section 3.4 (line ~205, 227): a response is a hallucination iff
# S_QB >= theta, with default theta = 0.5.
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetaQA:
    """
    MetaQA baseline – generates synonym/antonym mutations of the LLM's answer
    and verifies each to detect hallucinations.
    """

    def __init__(
        self,
        n_synonym: int = 5,
        n_antonym: int = 5,
        summarize_llm_fn: Optional[Callable[[str], Any]] = None,
    ):
        """
        Args:
            n_synonym: number of synonym mutations to generate (paper RQ4
                optimal total = N + M = 10, line ~368).
            n_antonym: number of antonym mutations to generate.
            summarize_llm_fn: optional async callable ``fn(prompt) -> text``
                used by the Step-1 summarize sub-step (paper Section 3.1). When
                ``None`` the concise answer is used directly as the base
                response and ``summarized`` is flagged ``False``.
        """
        self.n_synonym = n_synonym
        self.n_antonym = n_antonym
        self.summarize_llm_fn = summarize_llm_fn

    # ----- Step 1: concise answer + summarize -----

    @staticmethod
    def build_summarize_prompt(answer: str) -> str:
        """Build the Step-1 summarize prompt ([AUTHORED], paper Section 3.1)."""
        return SUMMARIZE_PROMPT.format(answer=answer)

    async def summarize_answer(self, concise_answer: str) -> Dict[str, Any]:
        """
        Step-1b: summarize the concise answer into one short sentence.

        Returns a dict ``{base_response, summarized}``. When no
        ``summarize_llm_fn`` is wired, the concise answer is used as-is and
        ``summarized`` is ``False`` (documented deviation).
        """
        if not concise_answer or not concise_answer.strip():
            return {"base_response": concise_answer, "summarized": False}
        if self.summarize_llm_fn is None:
            return {"base_response": concise_answer, "summarized": False}
        try:
            prompt = self.build_summarize_prompt(concise_answer)
            summarized = await self.summarize_llm_fn(prompt)
            summarized = str(summarized).strip() if summarized else ""
            if summarized:
                return {"base_response": summarized, "summarized": True}
        except Exception as exc:  # noqa: BLE001 – degrade gracefully
            import logging
            logging.getLogger(__name__).warning(
                "MetaQA summarize failed, using concise answer: %s", exc,
            )
        return {"base_response": concise_answer, "summarized": False}

    # ----- Step 2: mutation generation -----

    @staticmethod
    def build_synonym_prompt(question: str, answer: str) -> str:
        """Build the synonym mutation-generation prompt (Table 1 + Instruction)."""
        return SYNONYM_MUTATION_PROMPT.format(question=question, answer=answer)

    @staticmethod
    def build_antonym_prompt(question: str, answer: str) -> str:
        """Build the antonym mutation-generation prompt (Table 1 + Instruction)."""
        return ANTONYM_MUTATION_PROMPT.format(question=question, answer=answer)

    # ----- Step 3: verification -----

    @staticmethod
    def build_verification_prompt(
        mutation_text: str,
        mr_type: Optional[str] = None,
    ) -> str:
        """
        Build the verification prompt for a single mutation.

        Uses the **single, MR-type-agnostic** fact-verification prompt released
        with the official MetaQA code (``FACT_VERIFICATION_PROMPT``). The
        ``mr_type`` argument is accepted for backward compatibility and
        provenance only — it does NOT change the prompt. Per the paper
        (Section 3.3-3.4) the synonym/antonym distinction is applied solely in
        the scoring function (``SYN_SCORE`` / ``ANT_SCORE``), not in the
        verifier. The released code applies this exact same prompt to both
        synonym and antonym mutations.
        """
        return FACT_VERIFICATION_PROMPT.format(mutation_text=mutation_text)

    @staticmethod
    def parse_numbered_list(text: str) -> List[str]:
        """
        Parse a numbered-list response from the LLM into individual items.
        Handles formats like "1. item", "1) item", "- item", etc.
        """
        if not text or not text.strip():
            return []

        lines = text.strip().split("\n")
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip the numeric prefix
            m = re.match(r"^(\d+[\.\)\-:]\s*)", line)
            if m:
                line = line[m.end():]
            # Strip the bullet prefix
            if line.startswith("- "):
                line = line[2:]
            if line:
                items.append(line.strip())
        return items

    @staticmethod
    def parse_verification(text: str) -> Optional[str]:
        """
        Parse the verification response into one of ``"Yes"`` / ``"No"`` /
        ``"Not Sure"``. Returns ``None`` when the label cannot be determined
        (paper Algorithm 1 expects exact labels; unparseable outputs are now
        observable rather than silently mapped to "Not Sure" -> 0.5).

        Matching strategy: exact (case-insensitive) label first, then a
        conservative fuzzy fallback.
        """
        if not text or not text.strip():
            return None

        raw = text.strip()
        text_lower = raw.lower()

        # 1. Exact-label match (paper Algorithm 1 expects exact {Yes, No, Not Sure}).
        exact_map = {
            "yes": "Yes", "no": "No", "not sure": "Not Sure",
            "not sure.": "Not Sure", "unsure": "Not Sure",
        }
        if text_lower in exact_map:
            return exact_map[text_lower]

        # 2. Look at the leading token(s).
        first_words = text_lower.split()[:5]
        first_chunk = " ".join(first_words)
        if first_chunk.startswith("not sure"):
            return "Not Sure"
        if first_chunk.startswith("yes"):
            return "Yes"
        if first_chunk.startswith("no") and "not sure" not in first_chunk:
            return "No"

        # 3. Conservative fuzzy fallback over the whole text.
        if "not sure" in text_lower:
            return "Not Sure"
        if "yes" in text_lower:
            return "Yes"
        if "no" in text_lower:
            return "No"

        # 4. Unparseable — let the caller decide (counted as ``unparseable``).
        return None

    # ----- Step 4: score calculation + classification -----

    @staticmethod
    def calculate_hallucination_score(
        synonym_results: List[Optional[str]],
        antonym_results: List[Optional[str]],
    ) -> float:
        """
        Calculate the hallucination score following the paper's formula
        (Eq. 5, Section 3.4). ``None`` entries (unparseable verifications) are
        treated as "Not Sure" (0.5), matching the paper's Not-Sure mapping, but
        are also counted in :meth:`calculate_hallucination_detail` for
        observability.

        Returns:
            Hallucination score in [0, 1]. Higher = more likely hallucination.
        """
        detail = MetaQA.calculate_hallucination_detail(
            synonym_results, antonym_results,
        )
        return detail["score"]

    @staticmethod
    def calculate_hallucination_detail(
        synonym_results: List[Optional[str]],
        antonym_results: List[Optional[str]],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Full scoring detail (paper Section 3.4):

        - ``score``: ``sum(SynScore + AntScore) / (N + M)``
        - ``n_synonym`` / ``n_antonym``: counts used in the denominator.
        - ``unparseable``: number of verifications that returned ``None``
          (observability for prompt-quality diagnosis; these contribute 0.5
          each, matching Not-Sure).
        - ``threshold`` and ``is_hallucination``: binary label per
          ``S_QB >= theta`` (default 0.5).
        """
        def _resolve(result: Optional[str], table: Dict[str, float]) -> float:
            if result is None:
                return 0.5
            return table.get(result, 0.5)

        n_syn = len(synonym_results)
        n_ant = len(antonym_results)
        total = n_syn + n_ant
        if total == 0:
            return {
                "score": 0.0,
                "n_synonym": 0,
                "n_antonym": 0,
                "unparseable": 0,
                "threshold": threshold,
                "is_hallucination": False,
            }

        score = 0.0
        unparseable = 0
        for result in synonym_results:
            if result is None:
                unparseable += 1
            score += _resolve(result, SYN_SCORE)
        for result in antonym_results:
            if result is None:
                unparseable += 1
            score += _resolve(result, ANT_SCORE)

        score = score / total
        return {
            "score": score,
            "n_synonym": n_syn,
            "n_antonym": n_ant,
            "unparseable": unparseable,
            "threshold": threshold,
            "is_hallucination": score >= threshold,
        }

    @staticmethod
    def classify_hallucination(
        score: float,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> bool:
        """
        Binary hallucination label per paper Section 3.4: a response is a
        hallucination iff ``S_QB >= theta`` (default theta = 0.5).
        """
        return score >= threshold

    # ----- Serialization helpers -----

    @staticmethod
    def mutation_to_dict(m: MetaQAMutation) -> Dict[str, Any]:
        return {
            "mr_type": m.mr_type,
            "original_question": m.original_question,
            "base_response": m.base_response,
            "mutation_text": m.mutation_text,
            "verification_result": m.verification_result,
            "hallucination_contribution": m.hallucination_contribution,
            "metadata": m.metadata,
        }

    @staticmethod
    def mutations_to_json(mutations: List[MetaQAMutation]) -> str:
        return json.dumps(
            [MetaQA.mutation_to_dict(m) for m in mutations],
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_get(row: Optional[pd.Series], key: str, default: str = "") -> str:
    if row is None:
        return default
    val = row.get(key, default)
    if pd.isna(val):
        return default
    return str(val)
