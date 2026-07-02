#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Question type enum definition."""

from enum import Enum


class QuestionType(Enum):
    """Question type enum."""
    YES_NO = "yes_no"           # Yes/No question: Can you infer that...?
    WH_QUESTION = "wh_question"  # WH-question: What can you infer about...?
    TRUE_FALSE = "true_false"    # True/False question: Is it true that...?
    MULTIPLE_CHOICE = "multiple_choice"  # Multiple choice question
