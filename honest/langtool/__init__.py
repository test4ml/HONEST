#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Analysis Toolkit (LangTool)

A comprehensive toolkit for linguistic analysis supporting multiple languages.
Currently supports English with plans for other languages.
"""

# Export simplified class names (drop the English prefix)
from .english.sentence_analyzer import EnglishSentenceAnalyzer as SentenceAnalyzer
from .english.negation_transformer import EnglishNegationTransformer as NegationTransformer
from .english.yesno_question_transformer import EnglishYesNoQuestionTransformer as YesNoQuestionTransformer
from .english.wh_transformer import EnglishWhTransformer as WhTransformer
from .english.article_analyzer import EnglishArticleAnalyzer as ArticleAnalyzer
from .english.number_analyzer import EnglishNumberAnalyzer as NumberAnalyzer

# Other components
from .english.tense_analyzer import EnglishTenseAnalyzer
from .english.pos_analyzer import EnglishPOSAnalyzer
from .core.language_factory import LanguageFactory
from .core.types import (
    TenseInfo, POSInfo, SentenceTypeInfo, Language,
    SentenceType, QuestionType, Polarity
)

__all__ = [
    # Main components - simplified names
    'SentenceAnalyzer',
    'NegationTransformer',
    'YesNoQuestionTransformer',
    'WhTransformer',
    'ArticleAnalyzer',
    'NumberAnalyzer',

    # Other components
    'EnglishTenseAnalyzer',
    'EnglishPOSAnalyzer',
    'LanguageFactory',
    'TenseInfo',
    'POSInfo',
    'SentenceTypeInfo',
    'Language',
    'SentenceType',
    'QuestionType',
    'Polarity'
]

__version__ = '0.1.0'