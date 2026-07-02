#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English tense analyzer

Analyzes English sentences to identify one of the 16 English tenses:
- 4 time categories: Present, Past, Future, Future-in-Past
- 4 aspect categories: Simple, Progressive, Perfect, Perfect Progressive
"""

import re
from typing import Optional, List, Dict, Tuple
import spacy
from ..core.types import TenseInfo, Tense, TenseTime, TenseAspect
from honest.constants import SPACY_MODEL_NAME

class EnglishTenseAnalyzer:
    """English tense analyzer using spaCy for linguistic analysis"""

    def __init__(self):
        """Initialize the analyzer with spaCy English model"""
        try:
            # Try to load the English model
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Tense mapping: (time, aspect) -> Tense
        self.tense_map = {
            (TenseTime.PRESENT, TenseAspect.SIMPLE): Tense.SIMPLE_PRESENT,
            (TenseTime.PAST, TenseAspect.SIMPLE): Tense.SIMPLE_PAST,
            (TenseTime.FUTURE, TenseAspect.SIMPLE): Tense.SIMPLE_FUTURE,
            (TenseTime.FUTURE_IN_PAST, TenseAspect.SIMPLE): Tense.SIMPLE_FUTURE_IN_PAST,

            (TenseTime.PRESENT, TenseAspect.PROGRESSIVE): Tense.PRESENT_PROGRESSIVE,
            (TenseTime.PAST, TenseAspect.PROGRESSIVE): Tense.PAST_PROGRESSIVE,
            (TenseTime.FUTURE, TenseAspect.PROGRESSIVE): Tense.FUTURE_PROGRESSIVE,
            (TenseTime.FUTURE_IN_PAST, TenseAspect.PROGRESSIVE): Tense.FUTURE_PROGRESSIVE_IN_PAST,

            (TenseTime.PRESENT, TenseAspect.PERFECT): Tense.PRESENT_PERFECT,
            (TenseTime.PAST, TenseAspect.PERFECT): Tense.PAST_PERFECT,
            (TenseTime.FUTURE, TenseAspect.PERFECT): Tense.FUTURE_PERFECT,
            (TenseTime.FUTURE_IN_PAST, TenseAspect.PERFECT): Tense.FUTURE_PERFECT_IN_PAST,

            (TenseTime.PRESENT, TenseAspect.PERFECT_PROGRESSIVE): Tense.PRESENT_PERFECT_PROGRESSIVE,
            (TenseTime.PAST, TenseAspect.PERFECT_PROGRESSIVE): Tense.PAST_PERFECT_PROGRESSIVE,
            (TenseTime.FUTURE, TenseAspect.PERFECT_PROGRESSIVE): Tense.FUTURE_PERFECT_PROGRESSIVE,
            (TenseTime.FUTURE_IN_PAST, TenseAspect.PERFECT_PROGRESSIVE): Tense.FUTURE_PERFECT_PROGRESSIVE_IN_PAST,
        }

        # Auxiliary verb patterns
        self.be_verbs = {'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been'}
        self.have_verbs = {'have', 'has', 'had', 'having'}
        self.will_verbs = {'will', 'shall'}
        self.would_verbs = {'would', 'should', 'could', 'might'}
        self.modal_verbs = self.will_verbs | self.would_verbs | {'can', 'may', 'must'}

    def analyze_tense(self, sentence: str) -> Optional[TenseInfo]:
        """
        Analyze the tense of an English sentence

        Args:
            sentence: The sentence to analyze

        Returns:
            TenseInfo object with detected tense information, or None if no tense detected
        """
        if not self.nlp:
            return self._fallback_analysis(sentence)

        # Process with spaCy
        doc = self.nlp(sentence)

        # Find the main verb and its dependencies
        main_verb, verb_phrase, auxiliaries = self._extract_verb_info(doc)

        if not main_verb:
            return None

        # Analyze time and aspect
        time = self._analyze_time(auxiliaries, main_verb, doc)
        aspect = self._analyze_aspect(auxiliaries, main_verb, doc)

        # Get the tense
        tense = self.tense_map.get((time, aspect))
        if not tense:
            return None

        # Calculate confidence based on pattern matches
        confidence = self._calculate_confidence(auxiliaries, main_verb, time, aspect)

        # Generate explanation
        explanation = self._generate_explanation(tense, auxiliaries, main_verb)

        return TenseInfo(
            tense=tense,
            time=time,
            aspect=aspect,
            confidence=confidence,
            main_verb=main_verb.text,
            auxiliary_verbs=[aux.text for aux in auxiliaries],
            verb_phrase=verb_phrase,
            explanation=explanation
        )

    def _extract_verb_info(self, doc) -> Tuple[Optional[object], str, List[object]]:
        """Extract main verb and auxiliary verbs from spaCy doc"""
        main_verb = None
        auxiliaries = []
        verb_tokens = []

        # Find all verb-related tokens
        for token in doc:
            if token.pos_ in ['VERB', 'AUX']:
                verb_tokens.append(token)
                if token.dep_ == 'ROOT' or (token.pos_ == 'VERB' and not main_verb):
                    main_verb = token
                elif token.pos_ == 'AUX' or token.lemma_.lower() in self.modal_verbs:
                    auxiliaries.append(token)

        # If no main verb found, try to find one from verb tokens
        if not main_verb and verb_tokens:
            # Look for the rightmost content verb
            for token in reversed(verb_tokens):
                if token.pos_ == 'VERB':
                    main_verb = token
                    break

        # Create verb phrase
        if verb_tokens:
            verb_phrase = ' '.join([token.text for token in sorted(verb_tokens, key=lambda x: x.i)])
        else:
            verb_phrase = ""

        return main_verb, verb_phrase, auxiliaries

    def _analyze_time(self, auxiliaries: List[object], main_verb: object, doc) -> TenseTime:
        """Analyze the time category of the verb phrase"""
        aux_lemmas = [aux.lemma_.lower() for aux in auxiliaries]
        aux_texts = [aux.text.lower() for aux in auxiliaries]

        # Future in past: would/should/could + base form
        if any(aux in self.would_verbs for aux in aux_lemmas):
            return TenseTime.FUTURE_IN_PAST

        # Future: will/shall + base form
        if any(aux in self.will_verbs for aux in aux_lemmas):
            return TenseTime.FUTURE

        # Past: past auxiliary verbs or past main verb
        if any(aux in ['was', 'were', 'had'] for aux in aux_texts):
            return TenseTime.PAST

        if main_verb and main_verb.tag_ in ['VBD', 'VBN'] and not auxiliaries:
            return TenseTime.PAST

        # Present: present auxiliaries or present main verb
        return TenseTime.PRESENT

    def _analyze_aspect(self, auxiliaries: List[object], main_verb: object, doc) -> TenseAspect:
        """Analyze the aspect category of the verb phrase"""
        aux_lemmas = [aux.lemma_.lower() for aux in auxiliaries]
        aux_texts = [aux.text.lower() for aux in auxiliaries]

        has_be = any(aux in self.be_verbs for aux in aux_lemmas)
        has_have = any(aux in self.have_verbs for aux in aux_lemmas)

        # Perfect Progressive: have + been + V-ing
        if has_have and 'been' in aux_texts and main_verb and main_verb.tag_ == 'VBG':
            return TenseAspect.PERFECT_PROGRESSIVE

        # Progressive: be + V-ing
        if has_be and main_verb and main_verb.tag_ == 'VBG':
            return TenseAspect.PROGRESSIVE

        # Perfect: have + V-ed/V-en (past participle)
        if has_have and main_verb and main_verb.tag_ in ['VBN']:
            return TenseAspect.PERFECT

        # Simple: base form, past form, or future without other aspects
        return TenseAspect.SIMPLE

    def _calculate_confidence(self, auxiliaries: List[object], main_verb: object,
                            time: TenseTime, aspect: TenseAspect) -> float:
        """Calculate confidence score for the tense detection"""
        confidence = 0.5  # Base confidence

        # Higher confidence for clear auxiliary patterns
        if auxiliaries:
            confidence += 0.3

        # Higher confidence for clear main verb patterns
        if main_verb and main_verb.tag_ in ['VBD', 'VBG', 'VBN', 'VBZ', 'VBP']:
            confidence += 0.2

        # Cap at 1.0
        return min(confidence, 1.0)

    def _generate_explanation(self, tense: Tense, auxiliaries: List[object], main_verb: object) -> str:
        """Generate human-readable explanation of the tense"""
        explanations = {
            Tense.SIMPLE_PRESENT: "Uses base form or present form (s/es) of the verb",
            Tense.SIMPLE_PAST: "Uses past form of the verb",
            Tense.SIMPLE_FUTURE: "Uses will/shall + base form",
            Tense.SIMPLE_FUTURE_IN_PAST: "Uses would/should + base form",

            Tense.PRESENT_PROGRESSIVE: "Uses am/is/are + verb-ing",
            Tense.PAST_PROGRESSIVE: "Uses was/were + verb-ing",
            Tense.FUTURE_PROGRESSIVE: "Uses will be + verb-ing",
            Tense.FUTURE_PROGRESSIVE_IN_PAST: "Uses would be + verb-ing",

            Tense.PRESENT_PERFECT: "Uses have/has + past participle",
            Tense.PAST_PERFECT: "Uses had + past participle",
            Tense.FUTURE_PERFECT: "Uses will have + past participle",
            Tense.FUTURE_PERFECT_IN_PAST: "Uses would have + past participle",

            Tense.PRESENT_PERFECT_PROGRESSIVE: "Uses have/has been + verb-ing",
            Tense.PAST_PERFECT_PROGRESSIVE: "Uses had been + verb-ing",
            Tense.FUTURE_PERFECT_PROGRESSIVE: "Uses will have been + verb-ing",
            Tense.FUTURE_PERFECT_PROGRESSIVE_IN_PAST: "Uses would have been + verb-ing",
        }

        base_explanation = explanations.get(tense, "Unknown tense pattern")

        # Add specific verb information
        aux_text = ', '.join([aux.text for aux in auxiliaries]) if auxiliaries else "no auxiliaries"
        main_text = main_verb.text if main_verb else "no main verb"

        return f"{base_explanation} (auxiliaries: {aux_text}, main verb: {main_text})"

    def _fallback_analysis(self, sentence: str) -> Optional[TenseInfo]:
        """Fallback analysis when spaCy is not available"""
        # Simple regex-based fallback
        sentence = sentence.lower().strip()

        # Very basic pattern matching
        if re.search(r'\b(will|shall)\b.*\bgoing\s+to\b', sentence):
            return TenseInfo(
                tense=Tense.FUTURE_PROGRESSIVE,
                time=TenseTime.FUTURE,
                aspect=TenseAspect.PROGRESSIVE,
                confidence=0.3,
                main_verb="going",
                auxiliary_verbs=["will"],
                verb_phrase=sentence,
                explanation="Simple regex fallback analysis"
            )

        # Add more fallback patterns as needed
        return None

    def get_all_tenses(self) -> List[Tense]:
        """Get list of all supported tenses"""
        return list(Tense)

    def get_tense_examples(self, tense: Tense) -> List[str]:
        """Get example sentences for a specific tense"""
        examples = {
            Tense.SIMPLE_PRESENT: [
                "I go to school every day.",
                "She works at the hospital.",
                "The sun rises in the east."
            ],
            Tense.SIMPLE_PAST: [
                "I went to school yesterday.",
                "She worked at the hospital last year.",
                "The sun rose at 6 AM."
            ],
            Tense.SIMPLE_FUTURE: [
                "I will go to school tomorrow.",
                "She will work at the hospital next year.",
                "The sun will rise at 6 AM tomorrow."
            ],
            Tense.SIMPLE_FUTURE_IN_PAST: [
                "I said I would go to school the next day.",
                "She mentioned she would work at the hospital.",
                "We knew the sun would rise at 6 AM."
            ],
            Tense.PRESENT_PROGRESSIVE: [
                "I am going to school now.",
                "She is working at the hospital.",
                "The sun is rising."
            ],
            Tense.PAST_PROGRESSIVE: [
                "I was going to school at 8 yesterday.",
                "She was working at the hospital when I called.",
                "The sun was rising when we woke up."
            ],
            Tense.FUTURE_PROGRESSIVE: [
                "I will be going to school at 8 tomorrow.",
                "She will be working at the hospital next week.",
                "The sun will be rising when we wake up."
            ],
            Tense.FUTURE_PROGRESSIVE_IN_PAST: [
                "I said I would be going to school at 8.",
                "She mentioned she would be working late.",
                "We knew the sun would be rising early."
            ],
            Tense.PRESENT_PERFECT: [
                "I have gone to school already.",
                "She has worked at the hospital for 5 years.",
                "The sun has risen."
            ],
            Tense.PAST_PERFECT: [
                "I had gone to school before 8.",
                "She had worked there before moving.",
                "The sun had risen before we woke up."
            ],
            Tense.FUTURE_PERFECT: [
                "I will have gone to school by 8 tomorrow.",
                "She will have worked there for 10 years by then.",
                "The sun will have risen by the time we wake up."
            ],
            Tense.FUTURE_PERFECT_IN_PAST: [
                "I said I would have gone to school by 8.",
                "She mentioned she would have worked there for 10 years.",
                "We knew the sun would have risen by then."
            ],
            Tense.PRESENT_PERFECT_PROGRESSIVE: [
                "I have been going to school for 3 years.",
                "She has been working at the hospital since 2020.",
                "The sun has been rising earlier lately."
            ],
            Tense.PAST_PERFECT_PROGRESSIVE: [
                "I had been going to school for 3 years before graduation.",
                "She had been working there for 5 years before quitting.",
                "The sun had been rising later before spring."
            ],
            Tense.FUTURE_PERFECT_PROGRESSIVE: [
                "I will have been going to school for 4 years by 2025.",
                "She will have been working there for 10 years by then.",
                "The sun will have been rising earlier for months."
            ],
            Tense.FUTURE_PERFECT_PROGRESSIVE_IN_PAST: [
                "I said I would have been going there for 4 years by 2025.",
                "She mentioned she would have been working there for 10 years.",
                "We knew the sun would have been rising earlier."
            ]
        }
        return examples.get(tense, [])