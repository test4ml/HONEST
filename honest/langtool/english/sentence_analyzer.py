#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English sentence type analyzer using spaCy

Analyzes English sentences to determine:
1. Sentence type (declarative, interrogative, imperative, exclamatory)
2. Question type (yes/no, wh, choice, tag, rhetorical)
3. Polarity (positive/negative)
"""

from typing import List, Optional, Tuple
import spacy
import re
from ..core.types import SentenceType, QuestionType, Polarity, SentenceTypeInfo
from honest.constants import SPACY_MODEL_NAME

class EnglishSentenceAnalyzer:
    """English sentence type analyzer using spaCy and pattern matching"""

    def __init__(self):
        """Initialize the analyzer with spaCy English model"""
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # WH-words for special questions
        self.wh_words = {
            'what', 'where', 'when', 'why', 'who', 'whom', 'whose', 'which',
            'how', 'whatever', 'wherever', 'whenever', 'however', 'whichever'
        }

        # Modal verbs and auxiliaries for yes/no questions
        self.question_starters = {
            'do', 'does', 'did', 'will', 'would', 'can', 'could', 'may', 'might',
            'shall', 'should', 'must', 'ought', 'have', 'has', 'had', 'am', 'is',
            'are', 'was', 'were', 'being', 'been'
        }

        # Negative words and contractions
        self.negative_words = {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither',
            'none', 'hardly', 'scarcely', 'barely', 'rarely', 'seldom'
        }

        self.negative_contractions = {
            "n't", "won't", "can't", "shouldn't", "wouldn't", "couldn't",
            "mustn't", "needn't", "haven't", "hasn't", "hadn't", "don't",
            "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"
        }

        # Choice indicators
        self.choice_indicators = {'or', 'either'}

        # Tag question patterns
        self.tag_patterns = [
            r",\s+(aren't|isn't|wasn't|weren't|haven't|hasn't|hadn't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't)\s+(you|I|we|they|he|she|it)\?",
            r",\s+(are|is|was|were|have|has|had|do|does|did|will|would|can|could|should|must)\s+(you|I|we|they|he|she|it)\?",
            r",\s+right\?", r",\s+ok\?", r",\s+okay\?"
        ]

        # Imperative indicators
        self.imperative_starters = {
            'please', 'let', "let's", 'come', 'go', 'stop', 'wait', 'listen',
            'look', 'see', 'hear', 'take', 'give', 'put', 'get', 'make'
        }

        # Exclamatory indicators
        self.exclamatory_starters = {
            'what', 'how', 'such', 'so', 'oh', 'wow', 'amazing', 'incredible'
        }

    def analyze_sentence_type(self, sentence: str) -> SentenceTypeInfo:
        """
        Analyze the type of a sentence

        Args:
            sentence: The sentence to analyze

        Returns:
            SentenceTypeInfo object with detailed analysis
        """
        if not sentence.strip():
            return self._create_fallback_analysis(sentence, "Empty sentence")

        if not self.nlp:
            return self._create_fallback_analysis(sentence, "spaCy not available")

        # Clean and prepare sentence
        cleaned_sentence = sentence.strip()
        doc = self.nlp(cleaned_sentence)

        # Analyze sentence
        sentence_type = self._determine_sentence_type(cleaned_sentence, doc)
        question_type = None
        if sentence_type == SentenceType.INTERROGATIVE:
            question_type = self._determine_question_type(cleaned_sentence, doc)

        polarity = self._determine_polarity(cleaned_sentence, doc)
        confidence, features = self._calculate_confidence_and_features(
            cleaned_sentence, doc, sentence_type, question_type, polarity
        )

        explanation = self._generate_explanation(
            sentence_type, question_type, polarity, features
        )

        return SentenceTypeInfo(
            sentence=sentence,
            sentence_type=sentence_type,
            question_type=question_type,
            polarity=polarity,
            confidence=confidence,
            explanation=explanation,
            detected_features=features
        )

    def _determine_sentence_type(self, sentence: str, doc) -> SentenceType:
        """Determine the basic sentence type"""
        sentence_lower = sentence.lower().strip()

        # Check for question mark
        if sentence.endswith('?'):
            return SentenceType.INTERROGATIVE

        # Check for exclamation mark
        if sentence.endswith('!'):
            # Could be exclamatory or imperative
            first_word = sentence_lower.split()[0] if sentence_lower.split() else ""
            if first_word in self.exclamatory_starters or first_word in self.wh_words:
                return SentenceType.EXCLAMATORY
            else:
                return SentenceType.IMPERATIVE

        # Check for imperative patterns (no subject, starts with verb)
        if self._is_imperative(sentence_lower, doc):
            return SentenceType.IMPERATIVE

        # Check for question patterns without question mark
        if self._is_question_pattern(sentence_lower, doc):
            return SentenceType.INTERROGATIVE

        # Default to declarative
        return SentenceType.DECLARATIVE

    def _determine_question_type(self, sentence: str, doc) -> QuestionType:
        """Determine the type of question"""
        sentence_lower = sentence.lower().strip()

        # Check for tag questions
        for pattern in self.tag_patterns:
            if re.search(pattern, sentence_lower):
                return QuestionType.TAG

        # Check for WH-questions
        first_word = sentence_lower.split()[0] if sentence_lower.split() else ""
        if first_word in self.wh_words:
            # Check if it might be rhetorical
            if self._is_rhetorical_question(sentence_lower, doc):
                return QuestionType.RHETORICAL
            return QuestionType.WH

        # Check for choice questions
        if any(word in sentence_lower for word in self.choice_indicators):
            return QuestionType.CHOICE

        # Check for yes/no questions
        if first_word in self.question_starters:
            return QuestionType.YES_NO

        # Default for questions without clear type
        return QuestionType.YES_NO

    def _determine_polarity(self, sentence: str, doc) -> Polarity:
        """Determine if sentence is positive or negative"""
        sentence_lower = sentence.lower()

        # Check for negative contractions
        for contraction in self.negative_contractions:
            if contraction in sentence_lower:
                return Polarity.NEGATIVE

        # Check for negative words
        words = sentence_lower.split()
        for word in words:
            if word in self.negative_words:
                return Polarity.NEGATIVE

        # Use spaCy to check for negative dependencies
        if self.nlp:
            for token in doc:
                if token.dep_ == 'neg':  # Negation dependency
                    return Polarity.NEGATIVE

        return Polarity.POSITIVE

    def _is_imperative(self, sentence_lower: str, doc) -> bool:
        """Check if sentence is imperative"""
        words = sentence_lower.split()
        if not words:
            return False

        first_word = words[0]

        # Check for common imperative starters
        if first_word in self.imperative_starters:
            return True

        # Check for "you" implied imperative (verb at start)
        if self.nlp and doc:
            first_token = doc[0]
            # Check if first word is a verb and no explicit subject
            if first_token.pos_ == 'VERB' and first_token.tag_ in ['VB', 'VBP']:
                # Look for explicit subject
                has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
                if not has_subject:
                    return True

        return False

    def _is_question_pattern(self, sentence_lower: str, doc) -> bool:
        """Check for question patterns without question mark"""
        words = sentence_lower.split()
        if not words:
            return False

        first_word = words[0]

        # Starts with question word
        if first_word in self.wh_words:
            return True

        # Starts with auxiliary/modal for yes/no question
        if first_word in self.question_starters:
            return True

        return False

    def _is_rhetorical_question(self, sentence_lower: str, doc) -> bool:
        """Heuristic to detect rhetorical questions"""
        # Common rhetorical question patterns
        rhetorical_patterns = [
            r"who knows\?", r"who cares\?", r"what\s+difference\s+does\s+it\s+make\?",
            r"why\s+bother\?", r"what\s+else\s+is\s+new\?", r"how\s+should\s+i\s+know\?",
            r"what\s+do\s+you\s+think\?", r"don't\s+you\s+think\?"
        ]

        for pattern in rhetorical_patterns:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _calculate_confidence_and_features(
        self, sentence: str, doc, sentence_type: SentenceType,
        question_type: Optional[QuestionType], polarity: Polarity
    ) -> Tuple[float, List[str]]:
        """Calculate confidence score and detected features"""
        features = []
        confidence = 0.0

        # Punctuation features
        if sentence.endswith('?'):
            features.append("question_mark")
            confidence += 0.4
        elif sentence.endswith('!'):
            features.append("exclamation_mark")
            confidence += 0.3
        elif sentence.endswith('.'):
            features.append("period")
            confidence += 0.2

        # Word order features
        words = sentence.lower().split()
        if words:
            first_word = words[0]
            if first_word in self.wh_words:
                features.append(f"wh_word_{first_word}")
                confidence += 0.3
            elif first_word in self.question_starters:
                features.append(f"auxiliary_start_{first_word}")
                confidence += 0.3
            elif first_word in self.imperative_starters:
                features.append(f"imperative_start_{first_word}")
                confidence += 0.3

        # Negative features
        if polarity == Polarity.NEGATIVE:
            if any(c in sentence.lower() for c in self.negative_contractions):
                features.append("negative_contraction")
                confidence += 0.2
            if any(w in sentence.lower() for w in self.negative_words):
                features.append("negative_word")
                confidence += 0.2

        # Tag question features
        if question_type == QuestionType.TAG:
            features.append("tag_question_pattern")
            confidence += 0.4

        # Choice question features
        if question_type == QuestionType.CHOICE:
            features.append("choice_indicators")
            confidence += 0.3

        return min(confidence, 1.0), features

    def _generate_explanation(
        self, sentence_type: SentenceType, question_type: Optional[QuestionType],
        polarity: Polarity, features: List[str]
    ) -> str:
        """Generate explanation for the analysis"""
        explanations = []

        # Basic type
        type_explanations = {
            SentenceType.DECLARATIVE: "This is a declarative sentence (statement)",
            SentenceType.INTERROGATIVE: "This is an interrogative sentence (question)",
            SentenceType.IMPERATIVE: "This is an imperative sentence (command)",
            SentenceType.EXCLAMATORY: "This is an exclamatory sentence"
        }
        explanations.append(type_explanations.get(sentence_type, "Unknown sentence type"))

        # Question subtype
        if question_type:
            question_explanations = {
                QuestionType.YES_NO: "specifically a yes/no question",
                QuestionType.WH: "specifically a WH-question (special question)",
                QuestionType.CHOICE: "specifically a choice question",
                QuestionType.TAG: "specifically a tag question",
                QuestionType.RHETORICAL: "specifically a rhetorical question"
            }
            explanations.append(question_explanations.get(question_type, "unknown question type"))

        # Polarity
        polarity_explanations = {
            Polarity.POSITIVE: "with positive polarity",
            Polarity.NEGATIVE: "with negative polarity"
        }
        explanations.append(polarity_explanations.get(polarity, "unknown polarity"))

        # Features
        if features:
            explanations.append(f"Features detected: {', '.join(features)}")

        return ". ".join(explanations) + "."

    def _create_fallback_analysis(self, sentence: str, reason: str) -> SentenceTypeInfo:
        """Create fallback analysis when advanced analysis fails"""
        # Simple heuristics
        if sentence.strip().endswith('?'):
            sentence_type = SentenceType.INTERROGATIVE
            question_type = QuestionType.YES_NO
        elif sentence.strip().endswith('!'):
            sentence_type = SentenceType.EXCLAMATORY
            question_type = None
        else:
            sentence_type = SentenceType.DECLARATIVE
            question_type = None

        polarity = Polarity.NEGATIVE if 'not' in sentence.lower() else Polarity.POSITIVE

        return SentenceTypeInfo(
            sentence=sentence,
            sentence_type=sentence_type,
            question_type=question_type,
            polarity=polarity,
            confidence=0.3,
            explanation=f"Fallback analysis ({reason})",
            detected_features=["fallback_analysis"]
        )

    # Convenience methods
    def is_question(self, sentence: str) -> bool:
        """Check if sentence is a question"""
        result = self.analyze_sentence_type(sentence)
        return result.sentence_type == SentenceType.INTERROGATIVE

    def is_declarative(self, sentence: str) -> bool:
        """Check if sentence is declarative"""
        result = self.analyze_sentence_type(sentence)
        return result.sentence_type == SentenceType.DECLARATIVE

    def is_imperative(self, sentence: str) -> bool:
        """Check if sentence is imperative"""
        result = self.analyze_sentence_type(sentence)
        return result.sentence_type == SentenceType.IMPERATIVE

    def is_exclamatory(self, sentence: str) -> bool:
        """Check if sentence is exclamatory"""
        result = self.analyze_sentence_type(sentence)
        return result.sentence_type == SentenceType.EXCLAMATORY

    def is_negative(self, sentence: str) -> bool:
        """Check if sentence is negative"""
        result = self.analyze_sentence_type(sentence)
        return result.polarity == Polarity.NEGATIVE

    def get_question_type(self, sentence: str) -> Optional[QuestionType]:
        """Get question type if sentence is a question"""
        result = self.analyze_sentence_type(sentence)
        return result.question_type