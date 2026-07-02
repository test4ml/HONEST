#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English number analyzer using spaCy

Analyzes English nouns to determine singular/plural forms and provides
intelligent number detection and conversion capabilities, including
advanced uncountable noun detection.
"""

from typing import List, Dict, Optional, Tuple, Union, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import spacy
import re
from honest.constants import SPACY_MODEL_NAME

class NumberType(Enum):
    """Number types for nouns"""
    SINGULAR = "singular"
    PLURAL = "plural"
    UNCOUNTABLE = "uncountable"
    BOTH = "both"  # Can be both singular and plural (e.g., "sheep")
    UNKNOWN = "unknown"

class PluralType(Enum):
    """Types of plural formation"""
    REGULAR = "regular"  # Add -s or -es
    IRREGULAR = "irregular"  # Completely different form
    UNCHANGED = "unchanged"  # Same form for singular and plural
    COMPOUND = "compound"  # Compound words
    FOREIGN = "foreign"  # Foreign plurals (Latin, Greek, etc.)

@dataclass
class NumberInfo:
    """Information about noun number analysis"""
    word: str
    number_type: NumberType
    plural_type: Optional[PluralType]
    singular_form: Optional[str]
    plural_form: Optional[str]
    confidence: float  # 0.0 to 1.0
    pos_tag: str
    lemma: str
    explanation: str

class EnglishNumberAnalyzer:
    """English number analyzer for singular/plural detection and conversion

    Includes advanced uncountable noun detection based on multiple methods:
    - Core uncountable word list
    - Morphological analysis (suffixes)
    - Context analysis (quantifiers, verb agreement)
    - spaCy linguistic features
    - WordNet semantic categories
    - Pattern-based heuristics
    """

    def __init__(self):
        """Initialize the analyzer with spaCy English model"""
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Initialize uncountable detection
        self._init_uncountable_detection()

        # Irregular plural mappings
        self.irregular_plurals = {
            'child': 'children',
            'foot': 'feet',
            'tooth': 'teeth',
            'goose': 'geese',
            'man': 'men',
            'woman': 'women',
            'mouse': 'mice',
            'louse': 'lice',
            'ox': 'oxen',
            'person': 'people',
            'datum': 'data',
            'phenomenon': 'phenomena',
            'criterion': 'criteria',
            'analysis': 'analyses',
            'basis': 'bases',
            'crisis': 'crises',
            'thesis': 'theses',
            'hypothesis': 'hypotheses',
            'axis': 'axes',
            'appendix': 'appendices',
            'index': 'indices',
            'matrix': 'matrices',
            'vertex': 'vertices',
            'radius': 'radii',
            'nucleus': 'nuclei',
            'stimulus': 'stimuli',
            'cactus': 'cacti',
            'focus': 'foci',
            'fungus': 'fungi',
            'alumnus': 'alumni',
            'syllabus': 'syllabi'
        }

        # Create reverse mapping
        self.irregular_singulars = {v: k for k, v in self.irregular_plurals.items()}

        # Nouns that are same in singular and plural
        self.unchanged_nouns = {
            'sheep', 'deer', 'fish', 'species', 'series', 'means', 'aircraft',
            'spacecraft', 'offspring', 'headquarters', 'crossroads', 'scissors',
            'glasses', 'pants', 'trousers', 'jeans', 'shorts', 'clothes',
            'police', 'cattle', 'people', 'staff'
        }

        # Foreign plural patterns
        self.foreign_patterns = [
            (r'us$', 'i'),      # cactus -> cacti
            (r'is$', 'es'),     # analysis -> analyses
            (r'on$', 'a'),      # criterion -> criteria
            (r'um$', 'a'),      # datum -> data
            (r'ex$', 'ices'),   # vertex -> vertices
            (r'ix$', 'ices'),   # matrix -> matrices
        ]

    def _init_uncountable_detection(self):
        """Initialize advanced uncountable noun detection configuration"""
        # Core uncountable nouns (high confidence)
        self.core_uncountable = {
            # Abstract concepts
            'information', 'advice', 'knowledge', 'wisdom', 'intelligence',
            'love', 'happiness', 'anger', 'fear', 'joy', 'sadness', 'peace',
            'freedom', 'democracy', 'justice', 'truth', 'beauty', 'courage',

            # Materials and substances
            'water', 'air', 'oxygen', 'hydrogen', 'oil', 'gas', 'electricity',
            'gold', 'silver', 'iron', 'steel', 'plastic', 'glass', 'wood',
            'paper', 'cloth', 'leather', 'cotton', 'wool', 'silk',

            # Food and drinks (mass nouns)
            'bread', 'butter', 'cheese', 'milk', 'coffee', 'tea', 'wine',
            'beer', 'rice', 'flour', 'sugar', 'salt', 'pepper', 'meat',

            # Activities and processes
            'homework', 'research', 'work', 'progress', 'development',
            'education', 'training', 'experience', 'practice', 'exercise',

            # Phenomena and states
            'weather', 'climate', 'traffic', 'pollution', 'noise',
            'silence', 'darkness', 'light', 'heat', 'cold',

            # Fields and subjects
            'music', 'art', 'literature', 'poetry', 'drama', 'history',
            'mathematics', 'physics', 'chemistry', 'biology', 'geography',

            # Business and economics
            'money', 'currency', 'cash', 'business', 'trade', 'commerce',
            'industry', 'tourism', 'advertising', 'marketing',

            # Daily items (collective)
            'furniture', 'equipment', 'machinery', 'luggage', 'baggage',
            'clothing', 'jewelry', 'makeup', 'software', 'hardware',

            # News and media
            'news', 'media', 'propaganda', 'publicity',

            # Others
            'fun', 'luck', 'help', 'time', 'space', 'energy', 'power'
        }

        # Semantic category indicators
        self.semantic_indicators = {
            # Abstract noun suffixes
            'abstract_suffixes': {'-ness', '-ity', '-tion', '-sion', '-ment', '-ence', '-ance'},
            # Material noun suffixes
            'material_suffixes': {'-ing'},  # Some -ing endings indicate materials or activities
            # Subject suffixes
            'subject_suffixes': {'-ics', '-ology', '-ography'}
        }

        # Context patterns
        self.uncountable_patterns = {
            # Quantifiers for uncountable nouns
            'mass_quantifiers': {'much', 'little', 'some', 'any', 'a lot of', 'lots of',
                               'a great deal of', 'a large amount of', 'plenty of'},
            # Quantifiers for countable nouns
            'count_quantifiers': {'many', 'few', 'several', 'a few', 'a number of',
                                'a large number of', 'numerous'},
            # Indefinite articles (suggest countable)
            'indefinite_articles': {'a', 'an'}
        }

        # WordNet hypernyms for uncountable categories
        self.wordnet_uncountable_hypernyms = {
            'substance.n.01', 'material.n.01', 'matter.n.03',  # Substances
            'abstraction.n.06', 'concept.n.01', 'idea.n.01',   # Abstract concepts
            'knowledge.n.01', 'information.n.02',              # Knowledge/information
            'feeling.n.01', 'emotion.n.01',                    # Emotions
            'activity.n.01', 'work.n.01',                      # Activities
            'phenomenon.n.01', 'process.n.06'                  # Phenomena/processes
        }

    def detect_uncountability(self, word: str, context: str = "") -> Tuple[bool, float, str]:
        """
        Comprehensive uncountability detection using multiple methods

        Args:
            word: The word to check
            context: Optional context sentence for better analysis

        Returns:
            Tuple of (is_uncountable, confidence, explanation)
        """
        if not word.strip():
            return False, 0.0, "Empty word"

        word_lower = word.lower().strip()
        evidences = []
        confidence_score = 0.0

        # 1. Core uncountable list (highest weight)
        if word_lower in self.core_uncountable:
            evidences.append("core_uncountable_list")
            confidence_score += 0.9

        # 2. Context analysis (high weight)
        if context and self.nlp:
            context_evidence, context_confidence = self._analyze_context_uncountability(word, context)
            if context_evidence:
                evidences.append(f"context_{context_evidence}")
                confidence_score += context_confidence * 0.8

        # 3. Morphological analysis (medium weight)
        morphological_evidence = self._analyze_morphology_uncountability(word_lower)
        if morphological_evidence:
            evidences.append(f"morphology_{morphological_evidence}")
            confidence_score += 0.6

        # 4. spaCy analysis (medium weight)
        if self.nlp:
            spacy_evidence, spacy_confidence = self._analyze_spacy_uncountability(word)
            if spacy_evidence:
                evidences.append(f"spacy_{spacy_evidence}")
                confidence_score += spacy_confidence * 0.5

        # 5. WordNet analysis (lower weight)
        try:
            wordnet_evidence = self._analyze_wordnet_uncountability(word_lower)
            if wordnet_evidence:
                evidences.append(f"wordnet_{wordnet_evidence}")
                confidence_score += 0.3
        except ImportError:
            pass

        # 6. Pattern-based analysis (lower weight)
        pattern_evidence = self._analyze_pattern_uncountability(word_lower)
        if pattern_evidence:
            evidences.append(f"pattern_{pattern_evidence}")
            confidence_score += 0.2

        # Final decision
        is_uncountable = confidence_score > 0.5
        final_confidence = min(confidence_score, 1.0)

        explanation = f"Evidence: {', '.join(evidences) if evidences else 'none'}"

        return is_uncountable, final_confidence, explanation

    def _analyze_context_uncountability(self, word: str, context: str) -> Tuple[Optional[str], float]:
        """Analyze context for uncountability clues"""
        if not self.nlp:
            return None, 0.0

        doc = self.nlp(context.lower())
        word_lower = word.lower()

        # Find target word in context
        target_token = None
        for token in doc:
            if token.text.lower() == word_lower or token.lemma_.lower() == word_lower:
                target_token = token
                break

        if not target_token:
            return None, 0.0

        confidence = 0.0
        evidence_type = None

        # Check quantifier modification
        for child in target_token.children:
            child_text = child.text.lower()
            if child_text in self.uncountable_patterns['mass_quantifiers']:
                evidence_type = "mass_quantifier"
                confidence = 0.8
                break
            elif child_text in self.uncountable_patterns['count_quantifiers']:
                evidence_type = "count_quantifier"
                confidence = -0.8  # Negative indicates countable
                break
            elif child_text in self.uncountable_patterns['indefinite_articles']:
                evidence_type = "indefinite_article"
                confidence = -0.9  # Strongly suggests countable
                break

        # Check verb agreement
        if target_token.dep_ == 'nsubj':  # Subject
            verb = target_token.head
            if verb.tag_ in ['VBZ', 'VBD']:  # Third person singular verb
                # If word looks plural but uses singular verb, might be uncountable
                if word_lower.endswith('s') and not target_token.tag_.startswith('NNS'):
                    evidence_type = "singular_verb_agreement"
                    confidence = 0.6

        return evidence_type, abs(confidence) if confidence != 0 else 0.0

    def _analyze_morphology_uncountability(self, word: str) -> Optional[str]:
        """Morphological analysis for uncountability"""
        # Check abstract noun suffixes
        for suffix in self.semantic_indicators['abstract_suffixes']:
            if word.endswith(suffix.lstrip('-')):
                return f"abstract_suffix_{suffix}"

        # Check subject suffixes
        for suffix in self.semantic_indicators['subject_suffixes']:
            if word.endswith(suffix.lstrip('-')):
                return f"subject_suffix_{suffix}"

        # Special patterns
        if word.endswith('ing') and len(word) > 5:
            # Some -ing words are uncountable (e.g., training, learning)
            return "ing_activity"

        return None

    def _analyze_spacy_uncountability(self, word: str) -> Tuple[Optional[str], float]:
        """spaCy-based uncountability analysis"""
        if not self.nlp:
            return None, 0.0

        doc = self.nlp(word)
        if not doc:
            return None, 0.0

        token = doc[0]

        # Check POS tagging
        if token.pos_ == 'NOUN':
            morph_dict = token.morph.to_dict()

            # Check number information
            if 'Number' in morph_dict:
                if morph_dict['Number'] == 'Sing':
                    # Singular form, need further analysis
                    return "singular_form", 0.3
                elif morph_dict['Number'] == 'Plur':
                    # Plural form, usually countable
                    return "plural_form", -0.8

        return None, 0.0

    def _analyze_wordnet_uncountability(self, word: str) -> Optional[str]:
        """WordNet semantic analysis for uncountability"""
        try:
            import nltk
            from nltk.corpus import wordnet as wn

            synsets = wn.synsets(word, pos=wn.NOUN)
            if not synsets:
                return None

            # Check hypernyms
            for synset in synsets[:2]:  # Check first two most common meanings
                for hypernym in synset.hypernyms():
                    if hypernym.name() in self.wordnet_uncountable_hypernyms:
                        return f"semantic_category_{hypernym.name().split('.')[0]}"

                # Check definition keywords
                definition = synset.definition().lower()
                if any(indicator in definition for indicator in
                       ['abstract', 'concept', 'substance', 'material', 'activity', 'knowledge']):
                    return "definition_indicator"

        except ImportError:
            pass

        return None

    def _analyze_pattern_uncountability(self, word: str) -> Optional[str]:
        """Pattern-based uncountability analysis"""
        # Chemical elements
        if re.match(r'^[a-z]+ium$', word) or re.match(r'^[a-z]+ine$', word):
            return "chemical_element"

        # Words ending in -ware (usually uncountable)
        if word.endswith('ware'):
            return "ware_suffix"

        # Certain -ing gerunds
        if word.endswith('ing') and word in {'learning', 'teaching', 'training', 'reading', 'writing'}:
            return "activity_gerund"

        return None

    def batch_detect_uncountability(self, words: List[str]) -> Dict[str, Tuple[bool, float, str]]:
        """Batch analyze multiple words for uncountability"""
        results = {}
        for word in words:
            results[word] = self.detect_uncountability(word)
        return results

    def analyze_text_uncountability(self, text: str) -> Dict[str, Tuple[bool, float, str]]:
        """Analyze all nouns in text for uncountability"""
        if not self.nlp:
            return {}

        doc = self.nlp(text)
        results = {}

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and token.is_alpha:
                result = self.detect_uncountability(token.lemma_.lower(), text)
                results[token.text] = result

        return results

    def analyze_number(self, word: str, context: str = "") -> NumberInfo:
        """
        Analyze the number (singular/plural) of a noun

        Args:
            word: The word to analyze
            context: Optional context sentence for better analysis

        Returns:
            NumberInfo object with detailed number analysis
        """
        if not self.nlp:
            return self._fallback_analysis(word)

        # Process with spaCy
        if context:
            doc = self.nlp(context)
            # Find the word in context
            target_token = None
            for token in doc:
                if token.text.lower() == word.lower():
                    target_token = token
                    break
        else:
            doc = self.nlp(word)
            target_token = doc[0] if doc else None

        if not target_token:
            return self._fallback_analysis(word)

        # Analyze number type
        number_type = self._determine_number_type(target_token)
        plural_type = self._determine_plural_type(word.lower())

        # Get forms
        singular_form, plural_form = self._get_singular_plural_forms(word.lower(), number_type)

        # Calculate confidence
        confidence = self._calculate_confidence(target_token, number_type)

        # Generate explanation
        explanation = self._generate_explanation(word, number_type, plural_type)

        return NumberInfo(
            word=word,
            number_type=number_type,
            plural_type=plural_type,
            singular_form=singular_form,
            plural_form=plural_form,
            confidence=confidence,
            pos_tag=target_token.tag_,
            lemma=target_token.lemma_,
            explanation=explanation
        )

    def _determine_number_type(self, token) -> NumberType:
        """Determine if the token is singular or plural"""
        word_lower = token.text.lower()
        lemma_lower = token.lemma_.lower()

        # Use advanced uncountable detection
        is_uncountable, uncountable_confidence, _ = self.detect_uncountability(
            lemma_lower, ""
        )

        # High confidence uncountable detection
        if is_uncountable and uncountable_confidence > 0.7:
            return NumberType.UNCOUNTABLE

        # Check unchanged nouns
        if lemma_lower in self.unchanged_nouns:
            return NumberType.BOTH

        # Use POS tags for number detection
        if token.tag_ in ['NN', 'NNP']:  # Singular noun tags
            # Double-check with moderate confidence uncountable detection
            if is_uncountable and uncountable_confidence > 0.5:
                return NumberType.UNCOUNTABLE
            return NumberType.SINGULAR
        elif token.tag_ in ['NNS', 'NNPS']:  # Plural noun tags
            return NumberType.PLURAL

        # Check if word and lemma are different (often indicates plural)
        if word_lower != lemma_lower:
            return NumberType.PLURAL

        # Final check for uncountable with lower confidence
        if is_uncountable and uncountable_confidence > 0.3:
            return NumberType.UNCOUNTABLE

        # Default to singular if unsure
        return NumberType.SINGULAR

    def _determine_plural_type(self, word: str) -> PluralType:
        """Determine the type of plural formation"""
        # Check irregular plurals
        if word in self.irregular_plurals or word in self.irregular_singulars:
            return PluralType.IRREGULAR

        # Check unchanged nouns
        if word in self.unchanged_nouns:
            return PluralType.UNCHANGED

        # Check foreign patterns
        for pattern, _ in self.foreign_patterns:
            if re.search(pattern, word):
                return PluralType.FOREIGN

        # Check compound words (basic heuristic)
        if '-' in word or ' ' in word:
            return PluralType.COMPOUND

        # Default to regular
        return PluralType.REGULAR

    def _get_singular_plural_forms(self, word: str, number_type: NumberType) -> Tuple[Optional[str], Optional[str]]:
        """Get both singular and plural forms of a word"""
        if number_type == NumberType.UNCOUNTABLE:
            return word, None

        if number_type == NumberType.BOTH:
            return word, word

        # Check irregular forms first
        if word in self.irregular_plurals:
            return word, self.irregular_plurals[word]
        if word in self.irregular_singulars:
            return self.irregular_singulars[word], word

        # If we know it's plural, try to get singular
        if number_type == NumberType.PLURAL:
            singular = self._pluralize_to_singular(word)
            return singular, word

        # If we know it's singular, try to get plural
        if number_type == NumberType.SINGULAR:
            plural = self._singularize_to_plural(word)
            return word, plural

        return word, None

    def _singularize_to_plural(self, word: str) -> str:
        """Convert singular to plural form"""
        # Irregular plurals
        if word in self.irregular_plurals:
            return self.irregular_plurals[word]

        # Foreign patterns
        for pattern, replacement in self.foreign_patterns:
            if re.search(pattern, word):
                return re.sub(pattern, replacement, word)

        # Regular patterns
        if word.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return word + 'es'
        elif word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
            return word[:-1] + 'ies'
        elif word.endswith('f'):
            return word[:-1] + 'ves'
        elif word.endswith('fe'):
            return word[:-2] + 'ves'
        elif word.endswith('o') and len(word) > 1 and word[-2] not in 'aeiou':
            return word + 'es'
        else:
            return word + 's'

    def _pluralize_to_singular(self, word: str) -> str:
        """Convert plural to singular form"""
        # Irregular singulars
        if word in self.irregular_singulars:
            return self.irregular_singulars[word]

        # Foreign patterns (reverse)
        if word.endswith('i') and word[:-1] + 'us' in self.irregular_plurals.values():
            return word[:-1] + 'us'
        if word.endswith('es') and word[:-2] + 'is' in self.irregular_plurals.values():
            return word[:-2] + 'is'
        if word.endswith('a') and word[:-1] + 'on' in self.irregular_plurals.values():
            return word[:-1] + 'on'

        # Regular patterns (reverse)
        if word.endswith('ies') and len(word) > 3:
            return word[:-3] + 'y'
        elif word.endswith('ves'):
            if word[:-3] + 'f' in self.irregular_plurals:
                return word[:-3] + 'f'
            elif word[:-3] + 'fe' in self.irregular_plurals:
                return word[:-3] + 'fe'
            else:
                return word[:-3] + 'f'  # Default to 'f'
        elif word.endswith('es') and len(word) > 2:
            base = word[:-2]
            if base.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return base
            elif base.endswith('o'):
                return base
            else:
                return word[:-1]  # Just remove 's'
        elif word.endswith('s') and len(word) > 1:
            return word[:-1]
        else:
            return word

    def _calculate_confidence(self, token, number_type: NumberType) -> float:
        """Calculate confidence score for number detection"""
        confidence = 0.5  # Base confidence

        # Higher confidence for clear POS tags
        if token.tag_ in ['NN', 'NNP', 'NNS', 'NNPS']:
            confidence += 0.3

        # Higher confidence for known patterns
        word_lower = token.text.lower()
        if (word_lower in self.irregular_plurals or
            word_lower in self.irregular_singulars or
            word_lower in self.core_uncountable or
            word_lower in self.unchanged_nouns):
            confidence += 0.3

        return min(confidence, 1.0)

    def _generate_explanation(self, word: str, number_type: NumberType, plural_type: Optional[PluralType]) -> str:
        """Generate explanation for the number analysis"""
        explanations = {
            NumberType.SINGULAR: f"'{word}' is a singular noun",
            NumberType.PLURAL: f"'{word}' is a plural noun",
            NumberType.UNCOUNTABLE: f"'{word}' is an uncountable noun",
            NumberType.BOTH: f"'{word}' has the same form for singular and plural",
            NumberType.UNKNOWN: f"'{word}' number type is unclear"
        }

        base_explanation = explanations.get(number_type, "Unknown number type")

        if plural_type:
            type_explanations = {
                PluralType.REGULAR: "follows regular pluralization rules",
                PluralType.IRREGULAR: "has an irregular plural form",
                PluralType.UNCHANGED: "does not change form",
                PluralType.COMPOUND: "is a compound word",
                PluralType.FOREIGN: "follows foreign language patterns"
            }
            plural_explanation = type_explanations.get(plural_type, "")
            if plural_explanation:
                base_explanation += f" and {plural_explanation}"

        return base_explanation

    def _fallback_analysis(self, word: str) -> NumberInfo:
        """Fallback analysis when spaCy is not available"""
        word_lower = word.lower()

        # Basic heuristics
        if word_lower in self.core_uncountable:
            number_type = NumberType.UNCOUNTABLE
        elif word_lower in self.unchanged_nouns:
            number_type = NumberType.BOTH
        elif word.endswith('s') and len(word) > 1:
            number_type = NumberType.PLURAL
        else:
            number_type = NumberType.SINGULAR

        plural_type = self._determine_plural_type(word_lower)
        singular_form, plural_form = self._get_singular_plural_forms(word_lower, number_type)

        return NumberInfo(
            word=word,
            number_type=number_type,
            plural_type=plural_type,
            singular_form=singular_form,
            plural_form=plural_form,
            confidence=0.3,  # Low confidence for fallback
            pos_tag='UNKNOWN',
            lemma=word_lower,
            explanation=f"Fallback analysis for '{word}'"
        )

    def is_singular(self, word: str, context: str = "") -> bool:
        """Check if a word is singular"""
        result = self.analyze_number(word, context)
        return result.number_type == NumberType.SINGULAR

    def is_plural(self, word: str, context: str = "") -> bool:
        """Check if a word is plural"""
        result = self.analyze_number(word, context)
        return result.number_type == NumberType.PLURAL

    def is_uncountable(self, word: str, context: str = "") -> bool:
        """Check if a word is uncountable"""
        result = self.analyze_number(word, context)
        return result.number_type == NumberType.UNCOUNTABLE

    def to_singular(self, word: str) -> str:
        """Convert word to singular form"""
        result = self.analyze_number(word)
        return result.singular_form if result.singular_form else word

    def to_plural(self, word: str) -> str:
        """Convert word to plural form"""
        result = self.analyze_number(word)
        return result.plural_form if result.plural_form else word

    def get_both_forms(self, word: str) -> Tuple[str, Optional[str]]:
        """Get both singular and plural forms"""
        result = self.analyze_number(word)
        return result.singular_form or word, result.plural_form

    def analyze_sentence_nouns(self, sentence: str) -> List[NumberInfo]:
        """Analyze all nouns in a sentence for their number"""
        if not self.nlp:
            return []

        doc = self.nlp(sentence)
        noun_analyses = []

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:  # Nouns and proper nouns
                analysis = self.analyze_number(token.text, sentence)
                noun_analyses.append(analysis)

        return noun_analyses
