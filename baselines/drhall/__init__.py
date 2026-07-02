"""
DrHall-inspired baseline implementation.

Adapts the metamorphic relations from:
  "Detecting and Reducing the Factual Hallucinations of Large Language Models
   with Metamorphic Testing" (Wu et al., FSE 2025)

Six base MRs are implemented (QMR4 retained as library code but EXCLUDED from
experiments — see IMPLEMENTATION_VS_PAPER.md):
  QMR1 – Chain of Thought
  QMR2 – Multilingual Voting
  QMR3 – Problem Optimization (paraphrase)
  QMR4 – Adding External Knowledge        ⚠ not used (no independent evidence source)
  AMR1 – General Question Construction (negation of original answer)
  AMR2 – Multi-Choice Question Construction (distractor-based)
"""

from .wrapper import DrHall, DrHallMR, DrHallMutation

__all__ = ["DrHall", "DrHallMR", "DrHallMutation"]
