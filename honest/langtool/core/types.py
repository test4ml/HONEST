#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core types and enums for the language analysis toolkit
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    # Future languages can be added here
    # CHINESE = "zh"
    # SPANISH = "es"
    # FRENCH = "fr"

class SentenceType(Enum):
    """Basic sentence types"""
    DECLARATIVE = "declarative"  # declarative sentence
    INTERROGATIVE = "interrogative"  # interrogative sentence
    IMPERATIVE = "imperative"  # imperative sentence
    EXCLAMATORY = "exclamatory"  # exclamatory sentence

class QuestionType(Enum):
    """Types of questions"""
    YES_NO = "yes_no"  # yes/no question (Can you...? Are you...?)
    WH = "wh"  # WH-question (What, Where, When, How...)
    CHOICE = "choice"  # alternative question (Do you want A or B?)
    TAG = "tag"  # tag question (You are coming, aren't you?)
    RHETORICAL = "rhetorical"  # rhetorical question (Who knows?)

class Polarity(Enum):
    """Sentence polarity (positive/negative)"""
    POSITIVE = "positive"  # affirmative sentence
    NEGATIVE = "negative"  # negative sentence

class TenseTime(Enum):
    """The four time categories in English"""
    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"
    FUTURE_IN_PAST = "future_in_past"

class TenseAspect(Enum):
    """The four aspect categories in English"""
    SIMPLE = "simple"
    PROGRESSIVE = "progressive"  # Also called continuous
    PERFECT = "perfect"
    PERFECT_PROGRESSIVE = "perfect_progressive"

class Tense(Enum):
    """All 16 English tenses"""
    # Simple tenses
    SIMPLE_PRESENT = "simple_present"
    SIMPLE_PAST = "simple_past"
    SIMPLE_FUTURE = "simple_future"
    SIMPLE_FUTURE_IN_PAST = "simple_future_in_past"

    # Progressive tenses
    PRESENT_PROGRESSIVE = "present_progressive"
    PAST_PROGRESSIVE = "past_progressive"
    FUTURE_PROGRESSIVE = "future_progressive"
    FUTURE_PROGRESSIVE_IN_PAST = "future_progressive_in_past"

    # Perfect tenses
    PRESENT_PERFECT = "present_perfect"
    PAST_PERFECT = "past_perfect"
    FUTURE_PERFECT = "future_perfect"
    FUTURE_PERFECT_IN_PAST = "future_perfect_in_past"

    # Perfect progressive tenses
    PRESENT_PERFECT_PROGRESSIVE = "present_perfect_progressive"
    PAST_PERFECT_PROGRESSIVE = "past_perfect_progressive"
    FUTURE_PERFECT_PROGRESSIVE = "future_perfect_progressive"
    FUTURE_PERFECT_PROGRESSIVE_IN_PAST = "future_perfect_progressive_in_past"

@dataclass
class SentenceTypeInfo:
    """Information about sentence type analysis"""
    sentence: str
    sentence_type: SentenceType
    question_type: Optional[QuestionType]
    polarity: Polarity
    confidence: float  # 0.0 to 1.0
    explanation: str
    detected_features: List[str]  # Features that led to this classification

@dataclass
class TenseInfo:
    """Information about a detected tense"""
    tense: Tense
    time: TenseTime
    aspect: TenseAspect
    confidence: float  # 0.0 to 1.0
    main_verb: str
    auxiliary_verbs: List[str]
    verb_phrase: str
    explanation: str

    @property
    def tense_name(self) -> str:
        """Human-readable tense name"""
        return self.tense.value.replace('_', ' ').title()

@dataclass
class POSInfo:
    """Part-of-speech information"""
    token: str
    pos_tag: str
    lemma: str
    is_verb: bool
    is_auxiliary: bool
    is_modal: bool
    dependencies: Dict[str, Any]

@dataclass
class SentenceAnalysis:
    """Complete analysis of a sentence"""
    text: str
    language: Language
    tense_info: Optional[TenseInfo]
    pos_info: List[POSInfo]
    tokens: List[str]
    verb_phrases: List[str]
    subject: Optional[str]
    main_verb: Optional[str]