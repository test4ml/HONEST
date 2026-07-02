#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 DrHall Mutation Generation Script

Reads the golden dataset and generates DrHall follow-up questions for each
original question using the metamorphic relations QMR1/QMR2 (offline) plus
QMR3/AMR1/AMR2 (deferred to the answering phase). QMR4 is intentionally
EXCLUDED from the experiments (see baselines/drhall/IMPLEMENTATION_VS_PAPER.md).

QMR1 and (degraded) QMR2 are generated synchronously (no LLM needed).
QMR2 faithful translation uses an LLM (deepseek-v4-flash by default).
QMR3 (paraphrase), AMR2 (distractor generation) and AMR1 (negation of the
LLM's own answer) require LLM data and are generated during the answering phase.

Output directory: data/examples/golden_dataset_drhall_mutation/

Usage:
    conda activate karma
    # QMR1/QMR2, with faithful QMR2 translation via deepseek-v4-flash
    # (export DEEPSEEK_API_KEY=... first). QMR2 questions are translated into
    # es/de/nl; QMR1 is template-based (no LLM).
    python scripts/rqs/rq4_drhall_mutations.py

    # Use a different translator model
    python scripts/rqs/rq4_drhall_mutations.py --translate-model glm-5-turbo

    # Disable faithful QMR2 translation (old degraded behaviour: untranslated)
    python scripts/rqs/rq4_drhall_mutations.py --no-translate

    # Also generate QMR3 with LLM-based paraphrase (LLM answering phase)
    python scripts/rqs/rq4_drhall_mutations.py --with-llm-paraphrase

    # Limit to N files for testing
    python scripts/rqs/rq4_drhall_mutations.py --limit 3
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import get_config
from honest.llm import SUPPORTED_PROTOCOLS, create_llm_client

from baselines.drhall import DrHall, DrHallMR, DrHallMutation

# Default model used for faithful QMR2 translation (paper QMR2 translates the
# question into es/de/nl).
DEFAULT_TRANSLATE_MODEL = "deepseek-v4-flash"

# Model config aliases (shared with other rq4 scripts)
MODEL_CONFIG_ALIASES = {
    "deepseek-v4-flash": "deepseek_v4_flash",
    "deepseek_v4_flash": "deepseek_v4_flash",
    "glm-5-turbo": "glm_5_turbo",
    "GLM-5-Turbo": "glm_5_turbo",
    "glm_5_turbo": "glm_5_turbo",
}


def resolve_model_config(
    model_config: str, args
) -> Tuple[Optional[str], Optional[str], str, str]:
    """Resolve (base_url, api_key, model_name, protocol) from config/CLI overrides.

    Identical resolution logic to rq4_metaqa_mutations.py.
    """
    config_key = MODEL_CONFIG_ALIASES.get(model_config, model_config)
    if config_key == "local":
        llm_config = get_config("llm.local", {})
        return (
            args.base_url,
            args.api_key,
            args.model or llm_config.get("default_model", "Qwen2.5-7B-Instruct"),
            args.protocol or llm_config.get("protocol", "openai"),
        )
    if config_key == "deepseek":
        llm_config = get_config("llm.deepseek", {})
        return (
            args.base_url or llm_config.get("base_url"),
            args.api_key or llm_config.get("api_key"),
            args.model or llm_config.get("default_model", "deepseek-chat"),
            args.protocol or llm_config.get("protocol", "openai"),
        )
    if config_key in {"deepseek_v4_flash", "glm_5_turbo", "anthropic", "gemini"}:
        llm_config = get_config(f"llm.{config_key}", {})
        return (
            args.base_url or llm_config.get("base_url"),
            args.api_key or llm_config.get("api_key"),
            args.model or llm_config.get("default_model", model_config),
            args.protocol or llm_config.get("protocol", "openai"),
        )
    return (
        args.base_url,
        args.api_key,
        args.model or model_config,
        args.protocol or "openai",
    )

# Paths
GOLDEN_DATASET_PATH = Path("data/examples/golden_dataset/golden_dataset_full.csv")
OUTPUT_DIR = Path("data/examples/golden_dataset_drhall_mutation")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _SuppressHttpRequestLogs(logging.Filter):
    """Precisely suppress httpx's per-request INFO logs ("HTTP Request: POST ... 200 OK").

    Only drops these records, keeping httpx's WARNING/ERROR and other INFO; and
    does not touch any other logger. The source of the log spam has been
    confirmed to be exactly the ``httpx`` logger.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("HTTP Request:")


logging.getLogger("httpx").addFilter(_SuppressHttpRequestLogs())

# Mutation columns – one JSON column per MR
# NOTE: QMR4 (Adding External Knowledge) is intentionally EXCLUDED from our
# experiments. The paper retrieves external evidence from Wikipedia; our dataset
# has no comparable independent external source, and using the same KG rule that
# generated the question leaks the answer and neutralizes QMR4's discriminative
# power. See baselines/drhall/IMPLEMENTATION_VS_PAPER.md. The faithful wrapper
# method DrHall.generate_qmr4 is retained as library code but not used here.
MR_COLUMNS = {
    DrHallMR.QMR1: "drhall_qmr1_mutations",
    DrHallMR.QMR2: "drhall_qmr2_mutations",
    DrHallMR.QMR3: "drhall_qmr3_mutations",
    DrHallMR.AMR1: "drhall_amr1_mutations",
    DrHallMR.AMR2: "drhall_amr2_mutations",
    # Composite MR (paper §3.4). Requires LLM calls (paraphrase + translation),
    # so it is NOT generated offline here — the column is initialized so the
    # answering phase (rq4_drhall_llm_answer.py --with-cmr3) can populate it,
    # exactly like QMR3/AMR2. NOTE: CMR3 stacks QMR4's evidence step internally,
    # so it inherits the same external-knowledge caveat (not used by default).
    DrHallMR.CMR3: "drhall_cmr3_mutations",
}


def generate_sync_mutations(
    row: pd.Series, drhall: DrHall, qmr2_degraded: bool = True
) -> Dict[str, str]:
    """Generate QMR1 mutations (no LLM required).

    QMR2 is generated here synchronously in degraded mode ONLY when
    ``qmr2_degraded=True`` (no translator wired, --no-translate); otherwise QMR2 is
    generated faithfully by :func:`run_qmr2_async_phase` via
    ``generate_qmr2_async`` + ``translate_llm_fn``.

    QMR4 (Adding External Knowledge) **is NOT generated here and is excluded from
    the experimental pipeline**: the paper's QMR4 retrieves independent external
    evidence from Wikipedia, which our dataset has no equivalent of. Reusing the
    same KG rule that generated the question as evidence would leak the answer and
    weaken discriminative power. See baselines/drhall/IMPLEMENTATION_VS_PAPER.md.

    AMR1 (General Question) **is NOT generated offline here**: paper §3.3 requires
    negating *the LLM's own answer* ``a = P(q)`` to form the follow-up, which is
    not available at the offline stage. Also, our dataset's answers are short
    labels (True/False, A/B/C/D), so directly applying the paper's template would
    degenerate into meaningless questions like
    ``[True] Is this statement not true?``. AMR1 is therefore deferred to the
    answering phase (see ``generate_amr1_from_llm_answer_async`` in
    rq4_drhall_llm_answer.py), where the LLM answer is declarativized per question
    type before negation.
    """
    result = {}
    question = str(row.get("original_question", "")).strip()

    if not question:
        return result

    # QMR1: Chain of Thought
    qmr1 = drhall.generate_qmr1(question, row)
    result[MR_COLUMNS[DrHallMR.QMR1]] = DrHall.mutations_to_json([qmr1])

    # QMR2: synchronous degraded path (only when --no-translate). Faithful
    # translation runs in the async phase.
    if qmr2_degraded:
        qmr2_list = drhall.generate_qmr2(question, row)
        result[MR_COLUMNS[DrHallMR.QMR2]] = DrHall.mutations_to_json(qmr2_list)

    return result



async def _qmr2_for_row(row: pd.Series, drhall: DrHall) -> List[DrHallMutation]:
    """Faithful QMR2 for one row: translate the question into es/de/nl.

    Requires ``translate_llm_fn`` wired on ``drhall``. Returns one mutation per
    language (paper §3.2 QMR2); translation failures degrade gracefully (the
    per-language warning is emitted only once — see ``_warn_once``).
    """
    question = str(row.get("original_question", "")).strip()
    if not question:
        return []
    return await drhall.generate_qmr2_async(question, row)


async def run_qmr2_async_phase(df: pd.DataFrame, drhall: DrHall) -> None:
    """Populate the QMR2 column for all rows via faithful LLM translation.

    Process rows serially (no concurrency): each row is translated into es/de/nl
    in turn (see ``generate_qmr2_async``); the progress bar advances by one per
    finished row. This is the simplest, most predictable, and easiest to monitor
    and debug; the cost is that it is overall slower
    (1452 rows x 3 language calls). To speed it up, switch back to submitting
    concurrently with ``async_tqdm.gather``.
    """
    for idx, row in tqdm(list(df.iterrows()), total=len(df), desc="QMR2 translation"):
        mutations = await _qmr2_for_row(row, drhall)
        df.at[idx, MR_COLUMNS[DrHallMR.QMR2]] = DrHall.mutations_to_json(mutations)


async def generate_async_mutations(
    row: pd.Series,
    drhall: DrHall,
    paraphrase_llm_fn=None,
    distractor_llm_fn=None,
) -> Dict[str, str]:
    """Generate QMR3 and AMR2 mutations (require LLM)."""
    result = {}
    question = str(row.get("original_question", "")).strip()
    answer = row.get("original_correct_answer", "")

    if not question:
        return result

    # QMR3: Problem Optimization (paraphrase)
    if paraphrase_llm_fn is not None:
        try:
            qmr3 = await drhall.generate_qmr3_async(question, row)
            if qmr3 is not None:
                result[MR_COLUMNS[DrHallMR.QMR3]] = DrHall.mutations_to_json([qmr3])
        except Exception as e:
            logger.warning(f"QMR3 generation failed for row: {e}")

    # AMR2: Multi-Choice (distractor generation)
    if distractor_llm_fn is not None and pd.notna(answer) and str(answer).strip():
        try:
            amr2 = await drhall.generate_amr2_async(question, str(answer), row)
            if amr2 is not None:
                result[MR_COLUMNS[DrHallMR.AMR2]] = DrHall.mutations_to_json([amr2])
        except Exception as e:
            logger.warning(f"AMR2 generation failed for row: {e}")

    return result


def save_by_kg_rule(df: pd.DataFrame, output_dir: Path) -> None:
    """Save mutations split by kg_rule and mutation_type, following golden dataset convention."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    df.to_csv(output_dir / "golden_dataset_full.csv", index=False)

    # Also save per kg_rule / mutation_type
    for kg_rule, rule_df in df.groupby("kg_rule"):
        rule_dir = output_dir / str(kg_rule)
        rule_dir.mkdir(parents=True, exist_ok=True)
        for mut_type, group_df in rule_df.groupby("mutation_type"):
            group_df.to_csv(rule_dir / f"{mut_type}.csv", index=False)

    # Save summary
    total_rows = len(df)
    mr_counts = {}
    for mr, col in MR_COLUMNS.items():
        if col in df.columns:
            count = df[col].notna().sum()
            mr_counts[mr.value] = int(count)
        else:
            mr_counts[mr.value] = 0

    summary_path = output_dir / "mutation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DrHall Mutation Generation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total rows: {total_rows}\n")
        for mr_name, count in mr_counts.items():
            f.write(f"  {mr_name}: {count} mutations\n")
        f.write(f"\nOutput directory: {output_dir}\n")

    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"  Total rows: {total_rows}")
    for mr_name, count in mr_counts.items():
        logger.info(f"  {mr_name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DrHall mutations for RQ4")
    parser.add_argument("--input", type=str, default=str(GOLDEN_DATASET_PATH))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--with-llm-paraphrase", action="store_true", help="Generate QMR3 with LLM")
    parser.add_argument("--with-llm-distractors", action="store_true", help="Generate AMR2 with LLM")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")

    # QMR2 faithful translation (paper §3.2): translate the question into
    # es/de/nl. Uses deepseek-v4-flash by default; --no-translate falls back to
    # the old degraded (untranslated) mode.
    parser.add_argument(
        "--translate-model", type=str, default=DEFAULT_TRANSLATE_MODEL,
        help=f"LLM used for QMR2 translation (default: {DEFAULT_TRANSLATE_MODEL}). "
             "Set via --no-translate to disable faithful translation.",
    )
    parser.add_argument(
        "--no-translate", action="store_true",
        help="Disable faithful QMR2 translation; produce degraded (untranslated) "
             "mutations like the old behavior.",
    )
    # LLM client overrides (mirror rq4_metaqa_mutations.py)
    parser.add_argument("--model", type=str, default=None, help="Single model name override")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--protocol", type=str, choices=sorted(SUPPORTED_PROTOCOLS), default=None)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--rate-limit", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if output_dir.exists() and not args.force:
        existing = output_dir / "golden_dataset_full.csv"
        if existing.exists():
            logger.info(f"Output already exists at {output_dir}. Use --force to overwrite.")
            return

    print("=" * 70)
    print("RQ4 DrHall Mutation Generation")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")

    # --- QMR2 translation client (optional) ---
    translate_llm_fn = None
    translate_enabled = not args.no_translate
    if translate_enabled:
        base_url, api_key, model_name, protocol = resolve_model_config(
            args.translate_model, args,
        )
        print(f"\nQMR2 translation: {model_name} ({protocol}) @ {base_url}")
        try:
            translate_client = create_llm_client(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                protocol=protocol,
                max_concurrent=args.max_concurrent,
                rate_limit=args.rate_limit,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enable_thinking=False,
            )

            # translate_llm_fn signature: async (prompt, lang) -> translated_question
            async def translate_llm_fn(prompt: str, lang: str) -> str:
                return await translate_client.generate_answer(prompt)
        except ValueError as exc:
            logger.warning(
                "Failed to initialize translate client (%s); "
                "falling back to degraded QMR2 (--no-translate). %s",
                args.translate_model, exc,
            )
            translate_enabled = False
    if not translate_enabled:
        print("\nQMR2: degraded mode (no faithful translation).")

    # Load golden dataset
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} rows")

    # Initialize DrHall — wire translate_llm_fn for faithful QMR2
    drhall = DrHall(translate_llm_fn=translate_llm_fn)

    # Initialize mutation columns
    for col in MR_COLUMNS.values():
        df[col] = None

    # Generate synchronous mutations (QMR1; QMR2 here only if degraded/no-translate)
    print("\nGenerating synchronous mutations (QMR1)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sync mutations"):
        sync_result = generate_sync_mutations(row, drhall, qmr2_degraded=not translate_enabled)
        for col, value in sync_result.items():
            df.at[idx, col] = value

    # QMR2 faithful translation phase (es/de/nl via translate_llm_fn)
    if translate_enabled:
        print("\nGenerating QMR2 mutations (faithful multilingual translation)...")
        asyncio.run(run_qmr2_async_phase(df, drhall))

    # Generate async mutations if requested
    if args.with_llm_paraphrase or args.with_llm_distractors:
        print("\nNote: LLM-based mutations (QMR3, AMR2) are generated during the "
              "LLM answering phase instead, since they require LLM calls that "
              "can be combined with the answering step for efficiency.")
        print("Skipping inline LLM generation. QMR3 and AMR2 will be generated "
              "during the rq4_drhall_llm_answer.py step.")

    # Save
    save_by_kg_rule(df, output_dir)

    print(f"\n{'=' * 70}")
    print("Mutation generation complete!")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
