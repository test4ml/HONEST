#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English POS analyzer using spaCy
"""

from typing import List, Optional
import spacy
from ..core.types import POSInfo
from honest.constants import SPACY_MODEL_NAME

class EnglishPOSAnalyzer:
    """English Part-of-Speech analyzer using spaCy"""

    def __init__(self):
        """Initialize the analyzer with spaCy English model"""
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Auxiliary and modal verb sets
        self.auxiliary_verbs = {'be', 'have', 'do', 'will', 'shall', 'would', 'should', 'could', 'might', 'may', 'must', 'can'}
        self.modal_verbs = {'will', 'shall', 'would', 'should', 'could', 'might', 'may', 'must', 'can'}

    def analyze_pos(self, sentence: str) -> List[POSInfo]:
        """
        Analyze part-of-speech tags for each token in the sentence

        Args:
            sentence: The sentence to analyze

        Returns:
            List of POSInfo objects with detailed POS information
        """
        if not self.nlp:
            return self._fallback_pos_analysis(sentence)

        doc = self.nlp(sentence)
        pos_info_list = []

        for token in doc:
            # Check if it's a verb
            is_verb = token.pos_ in ['VERB', 'AUX']

            # Check if it's auxiliary
            is_auxiliary = (token.lemma_.lower() in self.auxiliary_verbs or
                          token.pos_ == 'AUX' or
                          token.dep_ in ['aux', 'auxpass'])

            # Check if it's modal
            is_modal = token.lemma_.lower() in self.modal_verbs

            # Extract dependency information
            dependencies = {
                'dep': token.dep_,
                'head': token.head.text,
                'children': [child.text for child in token.children]
            }

            pos_info = POSInfo(
                token=token.text,
                pos_tag=token.tag_,
                lemma=token.lemma_,
                is_verb=is_verb,
                is_auxiliary=is_auxiliary,
                is_modal=is_modal,
                dependencies=dependencies
            )

            pos_info_list.append(pos_info)

        return pos_info_list

    def _fallback_pos_analysis(self, sentence: str) -> List[POSInfo]:
        """Fallback POS analysis when spaCy is not available"""
        # Simple tokenization fallback
        import re
        tokens = re.findall(r'\w+', sentence)

        pos_info_list = []
        for token in tokens:
            pos_info = POSInfo(
                token=token,
                pos_tag='UNKNOWN',
                lemma=token.lower(),
                is_verb=False,
                is_auxiliary=False,
                is_modal=False,
                dependencies={'dep': 'unknown', 'head': '', 'children': []}
            )
            pos_info_list.append(pos_info)

        return pos_info_list