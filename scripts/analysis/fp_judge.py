#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP-Judge: use an LLM judge to estimate the false positive (FP) rate of the NLI consistency pipeline.

For each specified mutation operator, sample N cases judged "inconsistent" by NLI
from the consistency results directory, and use an LLM judge per case to decide
whether it is a genuine inconsistency (GENUINE) or a false positive (FP).

Uses a v2 judging strategy (validated acc~80%, balanced GENUINE/FP recall):
  - Neutral phrasing: do not disclose the NLI verdict, to avoid anchoring bias
  - few-shot: 4 hard cases (including Yes/No flips, different entities) calibrate the GENUINE boundary
  - stance-first: restate each stance first before judging, enforcing an extract-then-compare reasoning order
  - entity_rename: inject the inferred placeholder->entity mapping into the prompt (inject as text only,
    no substitution, to avoid false consistency from mismapping)

Example usage:
  python scripts/analysis/fp_judge.py \
      --results_dir data/examples/consistency_results_nli_gpt-5.5_llmfix \
      --n 100 --mutations entity_rename body_augmentation body_permutation \
      --output data/examples/fp_judge_gpt-5.5_llmfix.json
"""

import os
import sys
import re
import ast
import json
import glob
import random
import argparse
import asyncio
from typing import List, Dict, Optional

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from honest.llm import create_llm_client  # noqa: E402

random.seed(123)

# ---- Shared stop words (excluded from word-level diff; cannot be entities/placeholders) ----
_DIFF_STOP = set(
    "a an the of is are was were to in on at from and or that this it for with by as be "
    "been being has have had do does did not no yes next crossing upstream downstream "
    "first second rank higher lower level below above".split()
)


def parse_entity_labels(raw) -> List[str]:
    """entity_labels field -> list of entity-label strings (take the part after ':' and before the first '|')."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = [raw]
    else:
        parsed = raw
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    out = []
    for it in parsed:
        s = str(it)
        if ":" in s:
            s = s.split(":", 1)[1]
        lab = s.split("|")[0].strip()
        if len(lab) >= 2:
            out.append(lab)
    return out


def compute_rename_map(original_question: str, mutated_question: str,
                       labels: List[str]) -> Dict[str, str]:
    """Infer the placeholder -> real entity-label mapping (entity_rename only).

    Idea: a word appearing in the mutated question but not in the original is a placeholder;
          an entity label present in the original but missing from the mutated question is the renamed entity.
          Pair them in order of appearance (best-effort).
    """
    oq, mq = str(original_question or ""), str(mutated_question or "")
    o_toks = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", oq))
    m_toks = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", mq))
    placeholders = [t for t in (m_toks - o_toks) if len(t) >= 2 and t.lower() not in _DIFF_STOP]
    disappeared = [lab for lab in labels if lab and lab in oq and lab not in mq]
    mapping = {}
    for i, p in enumerate(placeholders):
        if i < len(disappeared):
            mapping[p] = disappeared[i]
    return mapping


# ---- v2 judge prompt (English, neutral, few-shot, stance-first) ----
_FEWSHOT = """
Study these labeled examples first to calibrate the decision boundary:

[Example 1] mutation=entity_rename, mapping {Object_A = major}
 Original answer conclusion: "No, captain does not have the next higher rank of major."
 Mutated  answer conclusion: "Yes, captain has the next higher rank of Object_A."
 Label: GENUINE. Reason: original says No, mutated says Yes — the stance flipped, the model gave different answers.

[Example 2] mutation=entity_rename
 Original answer conclusion: "the next crossing downstream from Monostor Bridge is the Komárom rail bridge."
 Mutated  answer conclusion: "the next crossing downstream from Monostor Bridge is the Elisabeth Bridge."
 Label: GENUINE. Reason: the two answers name different bridges — different substantive answers.

[Example 3] mutation=entity_rename, mapping {Entity_2 = modulation}
 Original answer conclusion: "modulation is the opposite of demodulation."
 Mutated  answer conclusion: "Entity_2 is modulation; Entity_2 is the opposite of demodulation."
 Label: FP. Reason: Entity_2 is just the placeholder for modulation. Both conclusions are identical in substance.

[Example 4] mutation=body_augmentation
 Original answer conclusion: "captain has the next higher rank of major."
 Mutated  answer conclusion: "captain has the next higher rank of major."
 Label: FP. Reason: after adding context, the answer is unchanged.
"""

_PROMPT = """You are a strict logic judge. In a metamorphic test, the same model answered an ORIGINAL question and a MUTATED question. The mutation is {mut}.{mapping}

Your task: decide whether the model's two answers have the SAME or a DIFFERENT substantive conclusion.
Rules (apply strictly):
- If the two answers differ in ANY of these dimensions (stance, chosen option, named entity/value, final conclusion, or direction of a relationship), label GENUINE. Typical GENUINE cases:
   * One answers Yes and the other No; or one says "holds" while the other says "does not hold / cannot be determined".
   * Different multiple-choice options are selected (e.g. B vs A).
   * Different entities/values/names are given (different bridge, different league, different person).
- Only label FP when both answers express the SAME substantive conclusion and differ only because of an entity rename or rephrasing.
- Do NOT default to FP just because "the mutation should be equivalent". Focus on whether the model ACTUALLY gave the same answer.
{fewshot}

Now do two things IN ORDER — first restate each stance, then decide:
[ORIGINAL QUESTION] {oq}
[MUTATED QUESTION] {mq}
[ORIGINAL ANSWER]
{oa}
[MUTATED ANSWER]
{ma}

Output JSON only:
{{"orig_stance":"<one short line: the original answer's actual conclusion/stance>",
  "mut_stance":"<one short line: the mutated answer's actual conclusion/stance>",
  "verdict":"FP" or "GENUINE"}}"""


def build_prompt(row: dict, mut: str) -> str:
    mapping_str = ""
    if mut == "entity_rename":
        labels = parse_entity_labels(row.get("entity_labels"))
        m = compute_rename_map(row.get("original_question", ""), row.get("mutated_question", ""), labels)
        if m:
            mapping_str = "\nPlaceholder mapping (placeholder = real entity it replaced): " + \
                "; ".join(f"{k} = {v}" for k, v in m.items()) + "."
    return _PROMPT.format(
        mut=mut, mapping=mapping_str, fewshot=_FEWSHOT,
        oq=str(row.get("original_question", ""))[:400],
        mq=str(row.get("mutated_question", ""))[:400],
        oa=str(row.get("original_llm_answer", ""))[:650],
        ma=str(row.get("mutated_llm_answer", ""))[:650],
    )


async def judge_one(client, row: dict, mut: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        p = build_prompt(row, mut)
        try:
            r = await client.generate_answer(p, max_retries=2)
            s, e = r.find("{"), r.rfind("}")
            if s >= 0 and e > s:
                return json.loads(r[s:e + 1])
            return {"verdict": "PARSE_FAIL", "raw": r[:120]}
        except Exception as ex:
            return {"verdict": "ERR", "reason": str(ex)[:120]}


def sample_inconsistent(results_dir: str, mut: str, n: int,
                        rule_set: Optional[set] = None) -> pd.DataFrame:
    """Collect all rows judged inconsistent by NLI for a given mutation operator, then sample n of them."""
    frames = []
    for f in glob.glob(os.path.join(results_dir, "kg_rule_*", f"{mut}_llm_answers.csv")):
        if rule_set is not None:
            m = re.search(r"kg_rule_(\d+)", f)
            if not m or int(m.group(1)) not in rule_set:
                continue
        d = pd.read_csv(f)
        frames.append(d[d["answers_consistent"] == False])
    if not frames:
        return pd.DataFrame()
    b = pd.concat(frames, ignore_index=True)
    b = b[b["original_llm_answer"].fillna("").str.len() >= 3]
    b = b[b["mutated_llm_answer"].fillna("").str.len() >= 3]
    return b.sample(min(n, len(b)), random_state=123).reset_index(drop=True)


async def run_judge(args):
    client = create_llm_client(
        base_url=args.base_url, api_key=args.api_key,
        model_name=args.model, protocol=args.protocol,
        max_tokens=160, temperature=0.0,
        max_concurrent=args.concurrency, rate_limit=args.rate_limit, timeout=120,
    )
    sem = asyncio.Semaphore(args.concurrency)
    rule_set = None
    if args.rules:
        rule_set = set()
        for part in args.rules.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                rule_set.update(range(int(lo), int(hi) + 1))
            elif part:
                rule_set.add(int(part))

    report = {"model": args.model, "results_dir": args.results_dir, "n_target": args.n, "per_mutation": {}}
    print(f"{'mutation':<20}{'sampled':>9}{'FP':>6}{'GENUINE':>9}{'err':>5}{'FP%':>8}")
    for mut in args.mutations:
        samp = sample_inconsistent(args.results_dir, mut, args.n, rule_set)
        if len(samp) == 0:
            print(f"{mut:<20}{'0':>9}{'-':>6}{'-':>9}{'-':>5}{'-':>8}  (no inconsistent samples found)")
            report["per_mutation"][mut] = {"sampled": 0, "fp": 0, "genuine": 0, "err": 0, "fp_rate": None, "cases": []}
            continue
        rows = samp.to_dict("records")
        verdicts = await asyncio.gather(*[judge_one(client, r, mut, sem) for r in rows])
        fp = sum(1 for v in verdicts if v.get("verdict") == "FP")
        gen = sum(1 for v in verdicts if v.get("verdict") == "GENUINE")
        err = sum(1 for v in verdicts if v.get("verdict") not in ("FP", "GENUINE"))
        n = len(samp)
        fp_rate = fp / n if n else 0.0
        print(f"{mut:<20}{n:>9}{fp:>6}{gen:>9}{err:>5}{fp_rate*100:>7.1f}%")
        report["per_mutation"][mut] = {
            "sampled": n, "fp": fp, "genuine": gen, "err": err,
            "fp_rate": fp_rate,
            "cases": [{"verdict": v.get("verdict"),
                       "orig_stance": v.get("orig_stance", "")[:200],
                       "mut_stance": v.get("mut_stance", "")[:200],
                       "extracted_original_conclusion": str(r.get("extracted_original_conclusion", ""))[:200],
                       "extracted_mutated_conclusion": str(r.get("extracted_mutated_conclusion", ""))[:200],
                       "original_answer_tail": str(r.get("original_llm_answer", ""))[-220:],
                       "mutated_answer_tail": str(r.get("mutated_llm_answer", ""))[-220:]}
                      for r, v in zip(rows, verdicts)],
        }
    await client.close()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump(report, open(args.output, "w"), ensure_ascii=False, indent=1)
        print(f"\nreport saved -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Estimate the false positive rate of the NLI consistency pipeline using an LLM judge (v2 strategy)")
    parser.add_argument("--results_dir", required=True, help="Consistency results directory (containing kg_rule_*/{mut}_llm_answers.csv)")
    parser.add_argument("--mutations", nargs="+", default=["entity_rename", "body_augmentation", "body_permutation"],
                        help="Mutation operators to evaluate")
    parser.add_argument("--n", type=int, default=100, help="Number of inconsistent samples to sample per operator")
    parser.add_argument("--rules", default=None, help="Optional: sample only within specified rules, e.g. '1-40'")
    parser.add_argument("--base_url", default="https://api.deepseek.com/anthropic")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api_key", default="<YOUR-API-KEY>")
    parser.add_argument("--protocol", default="anthropic")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--rate_limit", type=float, default=60.0)
    parser.add_argument("--output", default=None, help="JSON report output path")
    args = parser.parse_args()
    asyncio.run(run_judge(args))


if __name__ == "__main__":
    main()
