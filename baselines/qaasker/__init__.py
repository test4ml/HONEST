"""
QAAskeR Baseline Implementation

This module provides a clean wrapper around the original QAAskeR implementation
from ASE 2021. It implements three metamorphic relations (MRs):
- MR1 (Wh→New Wh): Transform one wh-question to another type of wh-question
- MR2 (Wh→General): Transform wh-question to a general statement (Q2S)
- MR3 (General→Wh): Transform statement to wh-question (S2G)

Based on: "Testing Your Question Answering Software via Asking Recursively"
https://github.com/imcsq/ASE21-QAAskeR
"""

from .wrapper import QAAskeR, MutationResult

__all__ = ['QAAskeR', 'MutationResult']
