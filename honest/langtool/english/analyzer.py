#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete English language analyzer combining tense and POS analysis
"""

from typing import Optional
import spacy
from .tense_analyzer import EnglishTenseAnalyzer
from .pos_analyzer import EnglishPOSAnalyzer
from ..core.types import SentenceAnalysis, Language, TenseInfo, POSInfo
from honest.constants import SPACY_MODEL_NAME

class EnglishAnalyzer:
    """Complete English language analyzer"""

    def __init__(self):
        """Initialize the complete analyzer"""
        self.tense_analyzer = EnglishTenseAnalyzer()
        self.pos_analyzer = EnglishPOSAnalyzer()

        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            self.nlp = None

    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        """
        Complete analysis of an English sentence

        Args:
            sentence: The sentence to analyze

        Returns:
            SentenceAnalysis object with complete linguistic information
        """
        # Basic tokenization
        if self.nlp:
            doc = self.nlp(sentence)
            tokens = [token.text for token in doc]
        else:
            import re
            tokens = re.findall(r'\w+', sentence)

        # Tense analysis
        tense_info = self.tense_analyzer.analyze_tense(sentence)

        # POS analysis
        pos_info = self.pos_analyzer.analyze_pos(sentence)

        # Extract additional information
        verb_phrases = self._extract_verb_phrases(pos_info)
        subject = self._extract_subject(sentence) if self.nlp else None
        main_verb = tense_info.main_verb if tense_info else None

        return SentenceAnalysis(
            text=sentence,
            language=Language.ENGLISH,
            tense_info=tense_info,
            pos_info=pos_info,
            tokens=tokens,
            verb_phrases=verb_phrases,
            subject=subject,
            main_verb=main_verb
        )

    def _extract_verb_phrases(self, pos_info: list) -> list:
        """Extract verb phrases from POS information"""
        verb_phrases = []
        current_phrase = []

        for pos in pos_info:
            if pos.is_verb or pos.is_auxiliary:
                current_phrase.append(pos.token)
            else:
                if current_phrase:
                    verb_phrases.append(' '.join(current_phrase))
                    current_phrase = []

        # Add final phrase if exists
        if current_phrase:
            verb_phrases.append(' '.join(current_phrase))

        return verb_phrases

    def _extract_subject(self, sentence: str) -> Optional[str]:
        """Extract the subject of the sentence using spaCy"""
        if not self.nlp:
            return None

        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == 'nsubj':  # Nominal subject
                # Include the token and its dependencies
                subject_tokens = [token.text]
                for child in token.children:
                    if child.dep_ in ['det', 'amod', 'compound']:
                        subject_tokens.insert(0, child.text)
                return ' '.join(subject_tokens)

        return None