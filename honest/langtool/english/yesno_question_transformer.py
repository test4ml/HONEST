#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English Yes-No Question Transformer using spaCy

Transforms English declarative sentences into yes-no questions using advanced
spaCy-based natural language processing for POS tagging, dependency parsing,
and grammatical analysis.
"""

import re
import logging
from typing import Dict, Optional, Tuple, List
import spacy
from ..core.types import Language
from honest.constants import SPACY_MODEL_NAME

logger = logging.getLogger(__name__)

class EnglishYesNoQuestionTransformer:
    """English declarative to yes-no question transformer using spaCy"""

    def __init__(self):
        """Initialize the transformer with spaCy English model"""
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Verb classifications for question transformation
        self.be_verbs = {'be', 'am', 'is', 'are', 'was', 'were', 'being', 'been'}
        self.modal_verbs = {
            'can', 'could', 'will', 'would', 'shall', 'should',
            'may', 'might', 'must', 'ought'
        }
        self.auxiliary_verbs = {
            'do', 'does', 'did', 'have', 'has', 'had'
        }

        # All verbs that can start questions
        self.question_starters = self.be_verbs | self.modal_verbs | self.auxiliary_verbs

    def transform(self, statement: str) -> str:
        """
        Transform declarative sentence into yes-no question

        Args:
            statement: Declarative sentence

        Returns:
            Transformed yes-no question
        """
        if not self.nlp:
            return self._fallback_transformation(statement)

        # Check if already a question
        if self.is_already_question(statement):
            return statement

        try:
            # Step 1: spaCy analysis of statement
            doc = self.nlp(statement)

            # Step 2: Analyze sentence structure
            analysis = self._analyze_sentence_structure(doc)

            # Step 3: Apply appropriate transformation strategy
            return self._apply_transformation_strategy(doc, analysis)

        except (AttributeError, ValueError, RuntimeError, IndexError) as e:
            logger.warning(f"Question transformation failed, using fallback: {e}")
            return self._fallback_transformation(statement)

    def _analyze_sentence_structure(self, doc) -> Dict:
        """Analyze sentence structure using spaCy"""
        analysis = {
            'main_verb': None,
            'auxiliary_verb': None,
            'subject': None,
            'verb_type': 'regular',  # 'be', 'modal', 'auxiliary', 'regular'
            'tense': 'present',
            'person': 'third_singular',
            'has_contraction': False,
            'contraction_info': None  # Store contraction details
        }

        # Check for contractions in the original text and tokens
        original_text = doc.text
        contraction_found = False
        for token in doc:
            # Only consider 's as a contraction if it's not a possessive marker
            if "'s" in token.text and token.tag_ != 'POS':
                analysis['has_contraction'] = True
                analysis['contraction_info'] = {
                    'token': token,
                    'position': token.i,
                    'text': token.text
                }
                contraction_found = True
                break
            elif "'re" in token.text or "'m" in token.text:
                analysis['has_contraction'] = True
                analysis['contraction_info'] = {
                    'token': token,
                    'position': token.i,
                    'text': token.text
                }
                contraction_found = True
                break

        # Find main verb (ROOT)
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ in ['VERB', 'AUX']:
                analysis['main_verb'] = token
                break

        # If no ROOT verb found, find any verb
        if not analysis['main_verb']:
            for token in doc:
                if token.pos_ in ['VERB', 'AUX']:
                    analysis['main_verb'] = token
                    break

        # Find subject
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                analysis['subject'] = token
                break

        # Find auxiliary verb
        for token in doc:
            if token.pos_ == 'AUX' and token != analysis['main_verb']:
                analysis['auxiliary_verb'] = token
                break

        # Determine verb type and characteristics
        if analysis['main_verb']:
            verb_lemma = analysis['main_verb'].lemma_.lower()
            verb_text = analysis['main_verb'].text.lower()
            verb_tag = analysis['main_verb'].tag_

            # Handle contractions for be verbs
            if contraction_found:
                contraction_text = analysis['contraction_info']['text']
                if "'s" in contraction_text:
                    # This is likely "She's", "He's", etc. - could be "is" or possessive
                    # Check if it's followed by a verb (progressive) or noun/adjective (be verb)
                    contraction_token = analysis['contraction_info']['token']
                    next_tokens = [t for t in contraction_token.doc if t.i > contraction_token.i]
                    if next_tokens and next_tokens[0].pos_ in ['VERB', 'ADJ', 'NOUN']:
                        if next_tokens[0].tag_ == 'VBG':  # Progressive: "She's working"
                            analysis['verb_type'] = 'be'
                        elif next_tokens[0].pos_ in ['ADJ', 'NOUN']:  # "She's happy", "She's a teacher"
                            analysis['verb_type'] = 'be'
                        else:
                            analysis['verb_type'] = 'be'  # Default to be for 's contractions
                    else:
                        analysis['verb_type'] = 'be'  # Default to be
                elif "'re" in contraction_text:
                    analysis['verb_type'] = 'be'
                elif "'m" in contraction_text:
                    analysis['verb_type'] = 'be'
            # Classify verb type
            elif verb_lemma in self.be_verbs:
                analysis['verb_type'] = 'be'
            elif verb_lemma in self.modal_verbs:
                analysis['verb_type'] = 'modal'
            elif analysis['auxiliary_verb'] and verb_lemma in self.auxiliary_verbs:
                analysis['verb_type'] = 'auxiliary'
            elif verb_lemma == 'have' and analysis['auxiliary_verb']:
                # "have" as auxiliary in perfect tenses
                analysis['verb_type'] = 'auxiliary'
            else:
                analysis['verb_type'] = 'regular'

            # Determine tense and person
            if verb_tag in ['VBD']:
                analysis['tense'] = 'past'
            elif verb_tag in ['VBZ']:
                analysis['tense'] = 'present'
                analysis['person'] = 'third_singular'
            elif verb_tag in ['VBP']:
                analysis['tense'] = 'present'
                analysis['person'] = 'other'
            elif verb_tag in ['VBG']:
                analysis['tense'] = 'continuous'

        return analysis

    def _apply_transformation_strategy(self, doc, analysis) -> str:
        """Apply the appropriate transformation strategy based on sentence structure"""
        tokens = [token.text for token in doc]

        if not analysis['main_verb']:
            return self._fallback_transformation(' '.join(tokens))

        main_verb = analysis['main_verb']
        verb_type = analysis['verb_type']

        if verb_type == 'be':
            return self._transform_be_verb_sentence(tokens, main_verb, analysis)
        elif verb_type == 'modal':
            return self._transform_modal_sentence(tokens, main_verb, analysis)
        elif analysis['auxiliary_verb']:
            return self._transform_auxiliary_sentence(tokens, analysis)
        else:
            return self._transform_regular_verb_sentence(tokens, main_verb, analysis)

    def _transform_be_verb_sentence(self, tokens: List[str], be_verb, analysis) -> str:
        """Transform sentences with be verbs"""
        # For be verbs: move be verb to front
        # "John is happy" -> "Is John happy?"
        tokens_copy = tokens.copy()

        # Handle contractions (e.g., "She's working" -> "Is she working?")
        if analysis.get('has_contraction', False) and analysis.get('contraction_info'):
            contraction_info = analysis['contraction_info']
            contraction_text = contraction_info['text']
            contraction_pos = contraction_info['position']

            if "'s" in contraction_text:
                # Split contraction: "She's" -> "She" + "is"
                subject_part = contraction_text.split("'s")[0]
                tokens_copy[contraction_pos] = subject_part  # Preserve original capitalization
                be_verb_capitalized = "Is"
                result_tokens = [be_verb_capitalized] + tokens_copy
                return self._clean_question(' '.join(result_tokens))
            elif "'re" in contraction_text:
                # Split contraction: "They're" -> "They" + "are"
                subject_part = contraction_text.split("'re")[0]
                tokens_copy[contraction_pos] = subject_part  # Preserve original capitalization
                be_verb_capitalized = "Are"
                result_tokens = [be_verb_capitalized] + tokens_copy
                return self._clean_question(' '.join(result_tokens))
            elif "'m" in contraction_text:
                # Split contraction: "I'm" -> "I" + "am"
                subject_part = contraction_text.split("'m")[0]
                tokens_copy[contraction_pos] = subject_part  # "I" stays capitalized
                be_verb_capitalized = "Am"
                result_tokens = [be_verb_capitalized] + tokens_copy
                return self._clean_question(' '.join(result_tokens))

        # Normal case: no contractions
        be_verb_text = be_verb.text
        be_verb_pos = be_verb.i

        # Remove be verb from current position
        tokens_copy.pop(be_verb_pos)

        # Add be verb at the beginning (capitalize first letter)
        be_verb_capitalized = be_verb_text.capitalize()
        result_tokens = [be_verb_capitalized] + tokens_copy

        return self._clean_question(' '.join(result_tokens))

    def _transform_modal_sentence(self, tokens: List[str], modal_verb, analysis) -> str:
        """Transform sentences with modal verbs"""
        # For modal verbs: move modal to front
        # "John can swim" -> "Can John swim?"
        tokens_copy = tokens.copy()
        modal_text = modal_verb.text
        modal_pos = modal_verb.i

        # Remove modal from current position
        tokens_copy.pop(modal_pos)

        # Add modal at the beginning (capitalize first letter)
        modal_capitalized = modal_text.capitalize()
        result_tokens = [modal_capitalized] + tokens_copy

        return self._clean_question(' '.join(result_tokens))

    def _transform_auxiliary_sentence(self, tokens: List[str], analysis) -> str:
        """Transform sentences with auxiliary verbs"""
        # For auxiliary verbs: move auxiliary to front
        # "John has finished" -> "Has John finished?"
        # "John is eating" -> "Is John eating?"

        aux_verb = analysis['auxiliary_verb']
        tokens_copy = tokens.copy()
        aux_text = aux_verb.text
        aux_pos = aux_verb.i

        # Remove auxiliary from current position
        tokens_copy.pop(aux_pos)

        # Add auxiliary at the beginning (capitalize first letter)
        aux_capitalized = aux_text.capitalize()
        result_tokens = [aux_capitalized] + tokens_copy

        return self._clean_question(' '.join(result_tokens))

    def _transform_regular_verb_sentence(self, tokens: List[str], main_verb, analysis) -> str:
        """Transform sentences with regular verbs by adding do/does/did"""
        tokens_copy = tokens.copy()
        verb_tag = main_verb.tag_
        verb_text = main_verb.text
        verb_lemma = main_verb.lemma_
        verb_pos = main_verb.i

        # Get the subject to determine correct auxiliary
        subject = analysis.get('subject')
        subject_text = subject.text.lower() if subject else ""

        # Determine appropriate auxiliary question word based on subject and verb
        if verb_tag == 'VBZ':  # Third person singular present
            question_aux = 'Does'
            new_verb = verb_lemma  # Use base form
        elif verb_tag == 'VBD':  # Past tense
            question_aux = 'Did'
            new_verb = verb_lemma  # Use base form
        else:  # Other present forms (VBP, VB)
            # Check if subject is plural or singular
            if subject and self._is_plural_subject(subject, tokens):
                question_aux = 'Do'
            else:
                # For base verb forms with singular subjects, still use "Do"
                # unless it's clearly third person singular
                question_aux = 'Do'
            new_verb = verb_text  # Keep as is for base forms

        # Replace the main verb with base form
        tokens_copy[verb_pos] = new_verb

        # Add auxiliary at the beginning
        result_tokens = [question_aux] + tokens_copy

        return self._clean_question(' '.join(result_tokens))

    def _is_plural_subject(self, subject_token, tokens: List[str]) -> bool:
        """Determine if subject is plural"""
        subject_text = subject_token.text.lower()

        # Plural pronouns
        if subject_text in ['they', 'we', 'you']:
            return True

        # Compound subjects (with "and")
        if 'and' in tokens:
            subject_start = subject_token.i
            # Look for "and" near the subject
            for i, token in enumerate(tokens):
                if token.lower() == 'and' and abs(i - subject_start) <= 3:
                    return True

        # Check morphological features if available
        if hasattr(subject_token, 'morph') and subject_token.morph:
            number = subject_token.morph.get('Number')
            if number and 'Plur' in number:
                return True

        # Check if subject ends with 's' (simple heuristic for plural nouns)
        if subject_text.endswith('s') and subject_text not in ['this', 'his', 'yes', 'us']:
            return True

        return False

    def _clean_question(self, question: str) -> str:
        """Clean up the generated question while preserving proper nouns and entities"""
        # Remove extra spaces
        question = re.sub(r'\s+', ' ', question.strip())

        # Fix separated contractions and possessives (e.g., "John 's" -> "John's")
        # Be more careful to preserve proper noun capitalization in compound names
        question = re.sub(r"\b([A-Za-z]+)\s+'s\b", r"\1's", question)
        question = re.sub(r"\b([A-Za-z]+)\s+'re\b", r"\1're", question)
        question = re.sub(r"\b([A-Za-z]+)\s+'m\b", r"\1'm", question)
        question = re.sub(r"\b([A-Za-z]+)\s+'ll\b", r"\1'll", question)
        question = re.sub(r"\b([A-Za-z]+)\s+'ve\b", r"\1've", question)
        question = re.sub(r"\b([A-Za-z]+)\s+'d\b", r"\1'd", question)

        # Fix compound proper nouns that may have been split (e.g., "Maritime Jingan" stays as "Maritime Jingan")
        # This preserves capitalization in multi-word proper names

        # Split into words for case handling
        words = question.split()
        if not words:
            return question

        # Ensure first word is capitalized
        words[0] = words[0].capitalize()

        # Words that should be lowercase (unless first word or "I")
        lowercase_words = {
            'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'in', 'on', 'at', 'by', 'for', 'with', 'without', 'of', 'to', 'from'
        }

        for i in range(1, len(words)):  # Skip first word
            word = words[i]
            word_lower = word.lower()

            if word_lower == 'i':
                words[i] = 'I'  # "I" is always uppercase
            elif word_lower in lowercase_words:
                words[i] = word_lower
            else:
                # Preserve original capitalization for:
                # 1. Proper nouns (start with uppercase)
                # 2. Contractions (containing apostrophes)
                # 3. Other content words
                words[i] = word  # Keep original case

        question = ' '.join(words)

        # Ensure question ends with question mark
        if not question.endswith('?'):
            question += '?'

        return question

    def is_already_question(self, statement: str) -> bool:
        """Check if statement is already a question"""
        statement = statement.strip()

        # Simple checks for question indicators
        if statement.endswith('?'):
            return True

        # Check for question words at the beginning
        question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
        first_word = statement.split()[0].lower() if statement.split() else ''
        if first_word in question_words:
            return True

        # Check for inverted auxiliary verbs (typical of questions)
        if self.nlp:
            doc = self.nlp(statement)
            tokens = [token.text.lower() for token in doc]
            if len(tokens) >= 2:
                first_token = tokens[0]
                if first_token in self.question_starters:
                    return True

        return False

    def _fallback_transformation(self, statement: str) -> str:
        """Fallback transformation when spaCy is not available"""
        if self.is_already_question(statement):
            return statement

        statement = statement.strip()

        # Simple pattern-based transformation
        patterns = [
            # Be verbs
            (r'^(\w+)\s+(is|are|was|were|am)\s+(.+)$', r'\2 \1 \3?'),

            # Modal verbs
            (r'^(\w+)\s+(can|could|will|would|shall|should|may|might|must)\s+(.+)$', r'\2 \1 \3?'),

            # Have auxiliaries
            (r'^(\w+)\s+(has|have|had)\s+(.+)$', r'\2 \1 \3?'),

            # Simple present third person singular
            (r'^(\w+)\s+(\w+s)\s+(.*)$', r'Does \1 \2 \3?'),

            # Simple past
            (r'^(\w+)\s+(\w+ed)\s+(.*)$', r'Did \1 \2 \3?'),
        ]

        for pattern, replacement in patterns:
            match = re.match(pattern, statement, re.IGNORECASE)
            if match:
                result = re.sub(pattern, replacement, statement, flags=re.IGNORECASE)
                return self._clean_question(result)

        # Default: add "Is it true that" for complex cases
        return f"Is it true that {statement.lower()}?"

    # Convenience methods
    def to_question(self, statement: str) -> str:
        """Alias for transform method"""
        return self.transform(statement)

    def analyze_statement(self, statement: str) -> Dict:
        """Analyze statement structure for debugging"""
        if not self.nlp:
            return {"error": "spaCy not available"}

        doc = self.nlp(statement)

        analysis = {
            "tokens": [(token.text, token.pos_, token.tag_, token.dep_) for token in doc],
            "sentence_analysis": self._analyze_sentence_structure(doc),
            "is_question": self.is_already_question(statement)
        }

        return analysis