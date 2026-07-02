#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ4 DrHall LLM Answering Script

Reads DrHall mutations from data/examples/golden_dataset_drhall_mutation,
generates LLM answers for original questions and all follow-up questions,
and also generates QMR3 (paraphrase) and AMR2 (distractor-based) mutations
on-the-fly using the LLM.

Output: data/examples/golden_dataset_drhall_answer_<model>/

Usage:
    conda activate karma
    # Answer with default LLM
    python scripts/rqs/rq4_drhall_llm_answer.py

    # Resume from interruption
    python scripts/rqs/rq4_drhall_llm_answer.py --skip-processed

    # Specific model
    python scripts/rqs/rq4_drhall_llm_answer.py --models Qwen2.5-7B-Instruct

    # Generate QMR3 and AMR2 on-the-fly (requires extra LLM calls)
    python scripts/rqs/rq4_drhall_llm_answer.py --with-qmr3 --with-amr2

    # Limit for testing
    python scripts/rqs/rq4_drhall_llm_answer.py --limit 3
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from openai import APIConnectionError, APITimeoutError
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs import get_config
from honest.llm import LLMClient, SUPPORTED_PROTOCOLS, create_llm_client
from baselines.drhall import DrHall
from baselines.drhall.wrapper import AMR2_DISTRACTOR_PROMPT

# Paths
MUTATION_DIR = Path("data/examples/golden_dataset_drhall_mutation")
OUTPUT_BASE_DIR = Path("data/examples")

# MR columns
# NOTE: QMR4 (Adding External Knowledge) is intentionally EXCLUDED — the paper
# retrieves external evidence from Wikipedia, which our dataset lacks; reusing
# the KG rule that generated the question leaks the answer. See
# baselines/drhall/IMPLEMENTATION_VS_PAPER.md.
MR_COLUMNS = [
    "drhall_qmr1_mutations",
    "drhall_qmr2_mutations",
    "drhall_qmr3_mutations",
    "drhall_amr1_mutations",
    "drhall_amr2_mutations",
    # Composite MR (paper §3.4, Table 2): only processed with --with-cmr3.
    # NOTE: CMR3 internally stacks QMR4's evidence step and inherits the same
    # external-knowledge caveat; it is opt-in and not used by default.
    "drhall_cmr3_mutations",
]

# NOTE: QMR3 (paraphrase) and AMR2 (distractor) prompts now live in the DrHall
# wrapper (``baselines.drhall.wrapper``) as the single source of truth and are
# used via ``make_drhall_with_llm``. The previous local prompt templates have
# been removed to avoid divergence from the paper.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


def load_csv_files(input_dir: Path) -> Dict[str, pd.DataFrame]:
    csv_files = {}
    for csv_file in input_dir.rglob("*.csv"):
        if "summary" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file)
            csv_files[str(csv_file.relative_to(input_dir))] = df
        except Exception as exc:
            logger.warning(f"Failed to read {csv_file}: {exc}")
    return csv_files


def parse_mutations(mutations_str: str) -> List[Dict]:
    if pd.isna(mutations_str) or not mutations_str:
        return []
    try:
        mutations = json.loads(str(mutations_str).strip())
        if isinstance(mutations, list):
            return mutations
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def answer_is_reusable(answer_str: str) -> bool:
    """Check if an existing answer can be reused (not ERROR, not empty)."""
    if not answer_str or pd.isna(answer_str):
        return False
    return not str(answer_str).startswith("ERROR:")


def make_drhall_with_llm(
    llm_client: LLMClient,
    with_qmr3: bool,
    with_amr2: bool,
    with_qmr2_translate: bool = True,
    with_amr1_statement: bool = True,
    with_amr2_simplify: bool = True,
    with_cmr3: bool = False,
) -> DrHall:
    """
    Build a DrHall wrapper instance with the paper-required LLM callables bound
    to the LLM client, so every MR uses the **paper-verbatim prompts** defined
    in the wrapper (single source of truth).

    Wired callables:
    * ``paraphrase_llm_fn``      — QMR3 paraphrase (also used by CMR3).
    * ``distractor_llm_fn``      — AMR2 distractor generation.
    * ``translate_llm_fn``       — QMR2 multilingual translation of the *question*
      (paper Section 3.2). Without this, QMR2 degrades to three identical
      English questions and the multi-language execution path collapses.
      Also used by CMR3's outer translation step.
    * ``statement_llm_fn``       — AMR1 declarativization of the answer (paper
      Table 1 uses a full declarative sentence, not a bare short answer).
    * ``simplify_llm_fn``        — AMR2 Step-1 answer simplification.

    ``with_cmr3`` implies paraphrase + translation (CMR3 stacks QMR3+QMR1+QMR4
    and wraps the result with QMR2 translation), so both callables are wired
    when either ``with_qmr3``/``with_qmr2_translate`` or ``with_cmr3`` is set.
    """
    paraphrase_fn = None
    distractor_fn = None
    translate_fn = None
    statement_fn = None
    simplify_fn = None

    # CMR3 needs paraphrase (QMR3 step) + translation (QMR2 step).
    need_paraphrase = with_qmr3 or with_cmr3
    need_translate = with_qmr2_translate or with_cmr3

    if need_paraphrase:
        async def paraphrase_fn(prompt: str) -> str:
            return await llm_client.generate_answer(prompt)

    if with_amr2:
        async def distractor_fn(question: str, simplified: str) -> List[str]:
            # Wrapper's paper-verbatim distractor prompt (uses {simplified_answer}).
            prompt = AMR2_DISTRACTOR_PROMPT.format(simplified_answer=simplified)
            text = await llm_client.generate_answer(prompt)
            if not text or not text.strip():
                return []
            return [d.strip() for d in text.strip().split("\n") if d.strip()]

    if need_translate:
        # The wrapper builds the TRANSLATE_PROMPT and passes (prompt, lang).
        async def translate_fn(prompt: str, lang: str) -> str:
            return await llm_client.generate_answer(prompt)

    if with_amr1_statement:
        # The wrapper builds AMR1_STATEMENT_PROMPT and passes the formatted prompt.
        async def statement_fn(prompt: str) -> str:
            return await llm_client.generate_answer(prompt)

    if with_amr2_simplify:
        # The wrapper builds AMR2_SIMPLIFY_PROMPT and passes the formatted prompt.
        async def simplify_fn(prompt: str) -> str:
            return await llm_client.generate_answer(prompt)

    return DrHall(
        paraphrase_llm_fn=paraphrase_fn,
        distractor_llm_fn=distractor_fn,
        translate_llm_fn=translate_fn,
        statement_llm_fn=statement_fn,
        simplify_llm_fn=simplify_fn,
    )


async def generate_qmr3_on_the_fly(
    llm_client: LLMClient, question: str, drhall: Optional[DrHall] = None
) -> Optional[str]:
    """Use the DrHall wrapper to paraphrase the question for QMR3 (paper prompt).

    ``drhall`` should be a wrapper instance with ``paraphrase_llm_fn`` wired to
    ``llm_client`` (see :func:`make_drhall_with_llm`). When ``None``, a wrapper
    is built on the fly from ``llm_client``.
    """
    try:
        if drhall is None:
            drhall = make_drhall_with_llm(llm_client, with_qmr3=True, with_amr2=False)
        mutation = await drhall.generate_qmr3_async(question, None)
        if mutation is not None and mutation.follow_up_question:
            return mutation.follow_up_question.strip()
    except Exception as e:
        logger.warning(f"QMR3 paraphrase failed: {e}")
    return None


async def generate_amr2_on_the_fly(
    llm_client: LLMClient,
    question: str,
    answer: str,
    drhall: Optional[DrHall] = None,
) -> Optional[str]:
    """Use the DrHall wrapper to build the AMR2 multi-choice question (paper prompt).

    Uses the wrapper's paper-faithful AMR2 (3 named options + "none of the
    above") rather than the previous local 4-option (A/B/C/D) template.
    """
    try:
        if drhall is None:
            drhall = make_drhall_with_llm(llm_client, with_qmr3=False, with_amr2=True)
        mutation = await drhall.generate_amr2_async(question, str(answer), None)
        if mutation is not None and mutation.follow_up_question:
            return mutation.follow_up_question.strip()
    except Exception as e:
        logger.warning(f"AMR2 generation failed: {e}")
    return None


async def process_row_with_llm(
    row_idx: int,
    row: pd.Series,
    llm_client: LLMClient,
    model_name: str,
    mr_columns: List[str],
    df_columns: List[str],
    with_qmr3: bool = False,
    with_amr2: bool = False,
    with_qmr2_translate: bool = True,
    with_cmr3: bool = False,
    debug: bool = False,
    existing_answers: Optional[Dict[str, str]] = None,
    drhall: Optional[DrHall] = None,
) -> Dict:
    """Process a single row: get original answer + follow-up answers for each MR."""
    result = {}
    question = str(row.get("original_question", "")).strip()
    correct_answer = row.get("original_correct_answer", "")

    if not question:
        return result

    # 1. Get LLM answer to the ORIGINAL question
    original_answer_key = f"original_question_{model_name}_answer"
    existing_original = None
    if existing_answers and original_answer_key in existing_answers:
        val = existing_answers[original_answer_key]
        if pd.notna(val) and isinstance(val, str) and not val.startswith("ERROR:"):
            existing_original = val

    if existing_original:
        result[original_answer_key] = existing_original
    else:
        try:
            original_answer = await llm_client.generate_answer(question)
            result[original_answer_key] = original_answer
        except Exception as e:
            logger.error(f"Error getting original answer for row {row_idx}: {e}")
            result[original_answer_key] = f"ERROR: {e}"

    # 2. Process each MR column
    for mr_col in mr_columns:
        if mr_col not in df_columns:
            continue

        answer_key = f"{mr_col}_{model_name}_answers"

        # Check for existing answers (resume support)
        if existing_answers and answer_key in existing_answers:
            existing_val = existing_answers[answer_key]
            if pd.notna(existing_val) and existing_val and not str(existing_val).startswith("ERROR:"):
                # Parse existing answers and check for errors inside
                try:
                    existing_list = json.loads(str(existing_val))
                    all_good = all(
                        isinstance(a, dict)
                        and not str(a.get("follow_up_answer", "")).startswith("ERROR:")
                        for a in existing_list
                    )
                    if all_good:
                        result[answer_key] = existing_val
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

        mutations_str = row.get(mr_col, "")
        mutations = parse_mutations(mutations_str)

        # QMR2: regenerate the three per-language questions with actual
        # translation (paper Section 3.2). The persisted offline mutations are
        # produced in degraded mode (identical English copies); translating the
        # question into es/de/nl here realizes the paper's multilingual
        # execution path. Only reached when this row is (re)answered — rows with
        # reusable existing answers are short-circuited by the resume check above.
        if (
            mr_col == "drhall_qmr2_mutations"
            and with_qmr2_translate
            and drhall is not None
            and getattr(drhall, "translate_llm_fn", None) is not None
        ):
            try:
                qmr2_muts = await drhall.generate_qmr2_async(question, None)
                if qmr2_muts:
                    mutations = [DrHall.mutation_to_dict(m) for m in qmr2_muts]
            except Exception as e:
                logger.warning(f"QMR2 translation failed for row {row_idx}: {e}")

        # CMR3 (paper §3.4, Table 2): build the composed question on-the-fly
        # (paraphrase -> CoT -> evidence wrap -> translate to es/de/nl) since it
        # requires LLM calls and is not persisted offline. Mirrors the QMR2
        # regeneration above; produces 3 per-language follow-up questions.
        if (
            mr_col == "drhall_cmr3_mutations"
            and with_cmr3
            and drhall is not None
            and getattr(drhall, "paraphrase_llm_fn", None) is not None
        ):
            try:
                cmr3_muts = await drhall.generate_cmr3_async(question, row)
                if cmr3_muts:
                    mutations = [DrHall.mutation_to_dict(m) for m in cmr3_muts]
            except Exception as e:
                logger.warning(f"CMR3 composition failed for row {row_idx}: {e}")

        # AMR1 (paper §3.3): the follow-up MUST be built from the LLM's OWN
        # answer a = P(q), never the golden label — DrHall tests whether the
        # LLM stays consistent when its own answer is re-questioned. The offline
        # AMR1 column is therefore ignored; we regenerate the declarativized
        # negation probe here from the original answer (boolean verdict / MC
        # option text / WH answer), see generate_amr1_from_llm_answer_async.
        if mr_col == "drhall_amr1_mutations" and drhall is not None:
            llm_answer = result.get(original_answer_key, "")
            if llm_answer and not str(llm_answer).startswith("ERROR:"):
                try:
                    amr1_mut = await drhall.generate_amr1_from_llm_answer_async(
                        question, str(llm_answer), row,
                    )
                    if amr1_mut is not None:
                        mutations = [DrHall.mutation_to_dict(amr1_mut)]
                except Exception as e:
                    logger.warning(f"AMR1 regeneration failed for row {row_idx}: {e}")

        if not mutations:
            # For QMR3 and AMR2, generate on-the-fly via the DrHall wrapper
            # (paper-verbatim prompts) if requested.
            if mr_col == "drhall_qmr3_mutations" and with_qmr3:
                follow_up_q = await generate_qmr3_on_the_fly(llm_client, question, drhall)
                if follow_up_q:
                    mutations = [{"follow_up_question": follow_up_q, "mr_type": "QMR3_ProblemOptimization"}]

            elif mr_col == "drhall_amr2_mutations" and with_amr2:
                if pd.notna(correct_answer) and str(correct_answer).strip():
                    follow_up_q = await generate_amr2_on_the_fly(
                        llm_client, question, str(correct_answer), drhall,
                    )
                    if follow_up_q:
                        mutations = [{"follow_up_question": follow_up_q, "mr_type": "AMR2_MultiChoice"}]

            if not mutations:
                result[answer_key] = None
                continue

        # Get LLM answer for each follow-up question
        answers = []
        for mutation in mutations:
            follow_up_q = mutation.get("follow_up_question", "")
            if not follow_up_q:
                answers.append({
                    "mr_type": mutation.get("mr_type", ""),
                    "follow_up_question": "",
                    "follow_up_answer": "",
                })
                continue

            try:
                follow_up_answer = await llm_client.generate_answer(follow_up_q)
                answers.append({
                    "mr_type": mutation.get("mr_type", ""),
                    "follow_up_question": follow_up_q,
                    "follow_up_answer": follow_up_answer,
                })
            except Exception as e:
                logger.error(f"Error answering MR for row {row_idx}: {e}")
                answers.append({
                    "mr_type": mutation.get("mr_type", ""),
                    "follow_up_question": follow_up_q,
                    "follow_up_answer": f"ERROR: {e}",
                })

        result[answer_key] = json.dumps(answers, ensure_ascii=False) if answers else None

    return result


async def process_dataframe(
    df: pd.DataFrame,
    llm_client: LLMClient,
    model_name: str,
    mr_columns: List[str],
    with_qmr3: bool = False,
    with_amr2: bool = False,
    with_qmr2_translate: bool = True,
    with_cmr3: bool = False,
    debug: bool = False,
    existing_df: Optional[pd.DataFrame] = None,
    checkpoint_every: int = 0,
    checkpoint_fn: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """Process entire DataFrame with LLM.

    Checkpoint large files against interruption: persist every ``checkpoint_every``
    rows (``checkpoint_fn`` writes to the same file as the final output), and do one
    final save in the ``finally`` block on any abnormal exit (network drop
    ``APIConnectionError`` / ctrl-C / other exceptions). Combined with per-row reuse
    under ``--skip-processed``, reruns only redo the unfinished rows instead of
    losing the whole file when it is not fully written.
    """
    result_df = df.copy()

    # Initialize answer columns
    result_df[f"original_question_{model_name}_answer"] = None
    for mr_col in mr_columns:
        if mr_col in df.columns:
            result_df[f"{mr_col}_{model_name}_answers"] = None

    df_columns = df.columns.tolist()

    # One wrapper instance (LLM fns bound) reused across all rows.
    drhall = make_drhall_with_llm(
        llm_client,
        with_qmr3=with_qmr3,
        with_amr2=with_amr2,
        with_qmr2_translate=with_qmr2_translate,
        with_cmr3=with_cmr3,
    )

    tasks = []
    for idx, row in df.iterrows():
        row_existing = None
        if existing_df is not None and idx in existing_df.index:
            row_existing = {}
            original_key = f"original_question_{model_name}_answer"
            if original_key in existing_df.columns:
                row_existing[original_key] = existing_df.at[idx, original_key]
            for mr_col in mr_columns:
                answer_key = f"{mr_col}_{model_name}_answers"
                if answer_key in existing_df.columns:
                    row_existing[answer_key] = existing_df.at[idx, answer_key]

        task = process_row_with_llm(
            idx, row, llm_client, model_name, mr_columns, df_columns,
            with_qmr3, with_amr2, with_qmr2_translate, with_cmr3, debug, row_existing, drhall,
        )
        tasks.append((idx, task))

    total = len(tasks)
    completed = 0
    aborted = True
    try:
        for idx, task in async_tqdm(tasks, total=total, desc=f"Answering with {model_name}"):
            try:
                row_result = await task
                for key, value in row_result.items():
                    result_df.at[idx, key] = value
            except (APIConnectionError, APITimeoutError):
                raise
            except Exception as exc:
                logger.error(f"Unexpected error processing row {idx}: {exc}")
                result_df.at[idx, f"original_question_{model_name}_answer"] = f"ERROR: {exc}"
            completed += 1
            # Periodic checkpoint: persist every checkpoint_every rows.
            if (checkpoint_fn is not None and checkpoint_every > 0
                    and completed % checkpoint_every == 0):
                try:
                    checkpoint_fn(result_df)
                except Exception as ce:  # noqa: BLE001
                    logger.warning(f"Checkpoint save failed at {completed}/{total}: {ce}")
        aborted = False
    finally:
        # On abnormal exit, persist the completed rows (completed only counts
        # rows fully written back to result_df).
        if checkpoint_fn is not None and aborted and completed > 0:
            try:
                checkpoint_fn(result_df)
                logger.info(f"Checkpoint saved on abort: {completed}/{total} rows persisted.")
            except Exception as ce:  # noqa: BLE001
                logger.warning(f"Final checkpoint save failed: {ce}")

    return result_df


def _atomic_to_csv(df: pd.DataFrame, output_file: Path) -> None:
    """Write ``df`` to ``output_file`` atomically (temp file + ``os.replace``).

    Prevent a checkpoint write from being interrupted halfway (network drop /
    ctrl-C) and corrupting the output file: write to a temp file first, then
    atomically rename, so on disk there is always either the previous complete
    file or the new complete file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    df.to_csv(tmp_file, index=False)
    os.replace(tmp_file, output_file)


def save_single_file(df: pd.DataFrame, output_dir: Path, rel_path: str, model_name: str) -> Dict:
    output_file = output_dir / rel_path
    _atomic_to_csv(df, output_file)

    answer_cols = [c for c in df.columns if c.endswith(f"_{model_name}_answers")]
    return {
        "file": rel_path,
        "rows": len(df),
        "answer_cols": len(answer_cols),
    }


def save_summary(stats_list: List[Dict], output_dir: Path, model_name: str) -> None:
    total_rows = sum(s["rows"] for s in stats_list)
    summary_path = output_dir / "answering_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DrHall LLM Answering Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total files: {len(stats_list)}\n")
        f.write(f"Total rows: {total_rows}\n\n")
        for s in stats_list:
            f.write(f"  {s['file']}: {s['rows']} rows, {s['answer_cols']} answer cols\n")
    logger.info(f"Summary saved to {summary_path}")


def check_file_already_processed(output_dir: Path, rel_path: str, model_name: str, mr_columns: List[str]) -> str:
    """Classify an output file as not / partially / fully processed.

    Key fix: the old logic marked a file as fully_processed as long as the answer
    column had no ERROR, which misclassified files whose checkpoint had only
    written some rows as "done" and skipped them, leaving the remaining rows
    forever unprocessed. Now: an unfilled file (filled < total) also counts as
    partially_processed, triggering per-row resume.
    """
    output_file = output_dir / rel_path
    if not output_file.exists():
        return "not_processed"
    try:
        df = pd.read_csv(output_file)
        original_col = f"original_question_{model_name}_answer"
        if original_col not in df.columns:
            return "not_processed"
        total = len(df)
        filled = int(df[original_col].notna().sum())
        if filled == 0:
            return "not_processed"
        has_errors = any("ERROR:" in str(v) for v in df[original_col].dropna())
        # Unfilled or with ERROR both count as "partially processed" to drive
        # per-row reuse and resume.
        if has_errors or filled < total:
            return "partially_processed"
        return "fully_processed"
    except Exception:
        return "not_processed"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Answer DrHall mutations with LLMs")
    parser.add_argument("--input-dir", type=str, default=str(MUTATION_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE_DIR))
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--rate-limit", type=float, default=10.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None, help="Display/alias name for output folder and column naming; defaults to --model")
    parser.add_argument("--protocol", type=str, choices=sorted(SUPPORTED_PROTOCOLS), default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-processed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Periodically checkpoint the in-progress file every N rows so an "
             "interruption (network drop / ctrl-C) only loses the last <N rows. "
             "0 disables. Combined with --skip-processed this gives per-row resume. "
             "(default: 50)",
    )
    parser.add_argument("--with-qmr3", action="store_true", help="Generate QMR3 paraphrase on-the-fly with LLM")
    parser.add_argument("--with-amr2", action="store_true", help="Generate AMR2 distractors on-the-fly with LLM")
    parser.add_argument(
        "--with-cmr3", action="store_true",
        help="Generate the CMR3 composite MR (paper §3.4, Table 2) on-the-fly: "
             "stack QMR3 paraphrase + QMR1 CoT + QMR4 evidence + QMR2 translation "
             "into one composed follow-up question per language (voted like QMR2). "
             "This is DrHall's strongest detector (paper F1=0.836).",
    )
    parser.add_argument(
        "--no-qmr2-translate", action="store_true",
        help="Disable QMR2 multilingual translation of the question (paper "
             "Section 3.2). By default the question is translated into es/de/nl "
             "via the LLM during answering; this flag keeps the degraded "
             "(untranslated) behavior. NOT paper-faithful.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)

    # Determine which MR columns to process
    active_mr_columns = []
    for col in MR_COLUMNS:
        # Always include QMR1, QMR2, AMR1 (QMR4 excluded — see IMPLEMENTATION_VS_PAPER.md)
        if col in [
            "drhall_qmr1_mutations",
            "drhall_qmr2_mutations",
            "drhall_amr1_mutations",
        ]:
            active_mr_columns.append(col)
        elif col == "drhall_qmr3_mutations" and args.with_qmr3:
            active_mr_columns.append(col)
        elif col == "drhall_amr2_mutations" and args.with_amr2:
            active_mr_columns.append(col)
        elif col == "drhall_cmr3_mutations" and args.with_cmr3:
            active_mr_columns.append(col)

    print("=" * 70)
    print("RQ4 DrHall LLM Answering")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Active MRs: {[c.replace('drhall_', '').replace('_mutations', '').upper() for c in active_mr_columns]}")

    csv_files = load_csv_files(input_dir)
    print(f"Loaded {len(csv_files)} CSV files")
    if not csv_files:
        logger.error("No CSV files found")
        sys.exit(1)

    if args.limit:
        csv_files = dict(list(csv_files.items())[:args.limit])
        print(f"Limited to {args.limit} files")

    if args.models:
        model_configs = args.models.split(",")
    elif args.model:
        model_configs = [args.model]
    else:
        model_configs = ["local"]

    llm_clients = {}
    for mc in model_configs:
        mc = mc.strip()
        base_url, api_key, model_name, protocol = resolve_model_config(mc, args)
        column_model_name = (args.model_name or model_name).replace("/", "_").replace("-", "_")
        try:
            llm_clients[column_model_name] = create_llm_client(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                protocol=protocol,
                max_concurrent=args.max_concurrent,
                rate_limit=args.rate_limit,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(f"  {column_model_name}: {model_name} ({protocol})")
        except ValueError as exc:
            logger.warning(f"Failed to initialize {mc}: {exc}")

    if not llm_clients:
        logger.error("No LLM clients initialized")
        sys.exit(1)

    for model_key, llm_client in llm_clients.items():
        print(f"\n{'=' * 70}")
        print(f"Processing with model: {model_key}")
        print(f"{'=' * 70}")

        model_output_dir = output_dir / f"golden_dataset_drhall_answer_{model_key}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        stats_list = []
        skipped_count = 0
        skip_processed = args.skip_processed and not args.force

        for rel_path, df in csv_files.items():
            try:
                existing_df = None
                if skip_processed:
                    status = check_file_already_processed(model_output_dir, rel_path, model_key, active_mr_columns)
                    if status == "fully_processed":
                        logger.info(f"Skipping {rel_path} (fully processed)")
                        skipped_count += 1
                        continue
                    elif status == "partially_processed":
                        try:
                            existing_df = pd.read_csv(model_output_dir / rel_path)
                            logger.info(f"Resuming {rel_path}")
                        except Exception as exc:
                            logger.warning(f"Failed to load existing: {exc}")

                logger.info(f"Processing {rel_path} with {model_key}...")
                # Checkpoint to the same file as the final output; on resume,
                # check_file_already_processed classifies it as partially_processed
                # and reuses rows per-row.
                output_file = model_output_dir / rel_path

                def _checkpoint(frame: pd.DataFrame, _of: Path = output_file) -> None:
                    _atomic_to_csv(frame, _of)

                processed_df = await process_dataframe(
                    df.copy(), llm_client, model_key, active_mr_columns,
                    with_qmr3=args.with_qmr3,
                    with_amr2=args.with_amr2,
                    with_qmr2_translate=not args.no_qmr2_translate,
                    with_cmr3=args.with_cmr3,
                    debug=args.debug,
                    existing_df=existing_df,
                    checkpoint_every=args.checkpoint_every,
                    checkpoint_fn=_checkpoint,
                )

                stats = save_single_file(processed_df, model_output_dir, rel_path, model_key)
                stats_list.append(stats)
                logger.info(f"Saved: {rel_path} ({stats['rows']} rows)")

            except (APIConnectionError, APITimeoutError) as exc:
                logger.error(f"Critical error: {exc}")
                if stats_list:
                    save_summary(stats_list, model_output_dir, model_key)
                break
            except Exception as exc:
                logger.error(f"Error processing {rel_path}: {exc}")
                traceback.print_exc()

        if stats_list:
            save_summary(stats_list, model_output_dir, model_key)
        if skipped_count:
            logger.info(f"Skipped {skipped_count} already processed files")
        print(f"Completed {model_key}: {llm_client.request_count} requests made")

    for llm_client in llm_clients.values():
        await llm_client.close()

    print(f"\n{'=' * 70}")
    print("Answering complete!")
    print(f"{'=' * 70}")
    for model_key in llm_clients:
        print(f"  - {output_dir / f'golden_dataset_drhall_answer_{model_key}'}")


if __name__ == "__main__":
    asyncio.run(main())
