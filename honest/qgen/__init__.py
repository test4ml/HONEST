#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural language question generation module.

Supports generating several types of natural language questions from inference rules:
- Yes/No questions
- WH-questions
- True/False questions
- Multiple choice questions

You can specify the core entities or relations to ask about.
"""

from .types import QuestionType
from .property_manager import PropertyManager
from .grammar_analyzer import GrammarAnalyzer
from .question_generator import QuestionGenerator

__all__ = ['QuestionType', 'PropertyManager', 'GrammarAnalyzer', 'QuestionGenerator']