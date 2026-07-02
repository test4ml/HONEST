"""
QAQA Baseline Implementation

This module provides a clean wrapper around the original QAQA implementation
from ASE 2022. It implements five metamorphic relations (MRs):
- EC (Extra Context): Add similar sentences to context
- EQ (Extra Question): Add similar sentences to question
- EQC (Extra Question + Context): Add similar sentences to both
- ETI (Extra Two Inputs): Add redundant QA pairs as context
- TI (Two Inputs): Combine two QA pairs

Based on: "Natural Test Generation for Precise Testing of Question Answering Software"
https://github.com/yichuan-cs/QAQA
"""

from .wrapper import QAQA, MutationResult, QA, QAList

__all__ = ['QAQA', 'MutationResult', 'QA', 'QAList']
