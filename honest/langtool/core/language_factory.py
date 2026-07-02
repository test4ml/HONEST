#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language factory for creating language-specific analyzers
"""

from typing import Protocol
from .types import Language

class TenseAnalyzer(Protocol):
    """Protocol for tense analyzers"""
    def analyze_tense(self, sentence: str): ...

class POSAnalyzer(Protocol):
    """Protocol for POS analyzers"""
    def analyze_pos(self, sentence: str): ...

class NumberAnalyzer(Protocol):
    """Protocol for number analyzers"""
    def analyze_number(self, word: str, context: str = ""): ...

class SentenceAnalyzer(Protocol):
    """Protocol for sentence type analyzers"""
    def analyze_sentence_type(self, sentence: str): ...

class WhTransformer(Protocol):
    """Protocol for WH-question transformers"""
    def transform(self, statement: str, question_word: str = "what", focus_entity: str = None): ...

class NegationTransformer(Protocol):
    """Protocol for negation transformers"""
    def transform(self, statement: str): ...

class ArticleAnalyzer(Protocol):
    """Protocol for article analyzers"""
    def add_article(self, noun_phrase: str) -> str: ...
    def is_subject_plural(self, subject: str) -> bool: ...

class YesNoQuestionTransformer(Protocol):
    """Protocol for yes-no question transformers"""
    def transform(self, statement: str): ...

class LanguageFactory:
    """Factory for creating language-specific analyzers"""

    @staticmethod
    def get_tense_analyzer(language: Language):
        """Get a tense analyzer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.tense_analyzer import EnglishTenseAnalyzer
            return EnglishTenseAnalyzer()
        else:
            raise ValueError(f"Tense analyzer for {language.value} is not yet supported")

    @staticmethod
    def get_pos_analyzer(language: Language):
        """Get a POS analyzer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.pos_analyzer import EnglishPOSAnalyzer
            return EnglishPOSAnalyzer()
        else:
            raise ValueError(f"POS analyzer for {language.value} is not yet supported")

    @staticmethod
    def get_number_analyzer(language: Language):
        """Get a number analyzer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.number_analyzer import EnglishNumberAnalyzer
            return EnglishNumberAnalyzer()
        else:
            raise ValueError(f"Number analyzer for {language.value} is not yet supported")

    @staticmethod
    def get_sentence_analyzer(language: Language):
        """Get a sentence type analyzer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.sentence_analyzer import EnglishSentenceAnalyzer
            return EnglishSentenceAnalyzer()
        else:
            raise ValueError(f"Sentence analyzer for {language.value} is not yet supported")

    @staticmethod
    def get_wh_transformer(language: Language):
        """Get a WH-question transformer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.wh_transformer import EnglishWhTransformer
            return EnglishWhTransformer()
        else:
            raise ValueError(f"WH transformer for {language.value} is not yet supported")

    @staticmethod
    def get_negation_transformer(language: Language):
        """Get a negation transformer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.negation_transformer import EnglishNegationTransformer
            return EnglishNegationTransformer()
        else:
            raise ValueError(f"Negation transformer for {language.value} is not yet supported")

    @staticmethod
    def get_yesno_question_transformer(language: Language):
        """Get a yes-no question transformer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.yesno_question_transformer import EnglishYesNoQuestionTransformer
            return EnglishYesNoQuestionTransformer()
        else:
            raise ValueError(f"Yes-no question transformer for {language.value} is not yet supported")

    @staticmethod
    def get_article_analyzer(language: Language):
        """Get an article analyzer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.article_analyzer import EnglishArticleAnalyzer
            return EnglishArticleAnalyzer()
        else:
            raise ValueError(f"Article analyzer for {language.value} is not yet supported")

    @staticmethod
    def get_full_analyzer(language: Language):
        """Get a complete analyzer for the specified language"""
        if language == Language.ENGLISH:
            from ..english.analyzer import EnglishAnalyzer
            return EnglishAnalyzer()
        else:
            raise ValueError(f"Analyzer for {language.value} is not yet supported")