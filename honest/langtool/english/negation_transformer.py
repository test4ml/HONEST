#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English Negation Transformer using spaCy

Transforms English affirmative sentences into negative sentences using advanced
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

class EnglishNegationTransformer:
    """English affirmative to negative sentence transformer using spaCy"""

    def __init__(self):
        """Initialize the transformer with spaCy English model"""
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Verb classifications for negation handling
        self.be_verbs = {
            'be': 'not be',
            'am': 'am not',
            'is': 'is not',
            'are': 'are not',
            'was': 'was not',
            'were': 'were not',
            'being': 'not being',
            'been': 'not been'
        }

        self.modal_verbs = {
            'can': 'cannot',  # Special case: cannot is one word
            'could': 'could not',
            'will': 'will not',
            'would': 'would not',
            'shall': 'shall not',
            'should': 'should not',
            'may': 'may not',
            'might': 'might not',
            'must': 'must not',
            'ought': 'ought not'
        }

        self.auxiliary_verbs = {
            'do': 'do not',
            'does': 'does not',
            'did': 'did not',
            'have': 'have not',
            'has': 'has not',
            'had': 'had not'
        }

        # All verbs that can be directly negated
        self.direct_negation_verbs = {**self.be_verbs, **self.modal_verbs, **self.auxiliary_verbs}

        # Negative indicators for checking if sentence is already negative
        self.negative_indicators = {
            'not', 'no', 'never', 'nothing', 'nobody', 'nowhere',
            'neither', 'none', 'cannot', "can't", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
            "hadn't", "mustn't", "shan't", "mayn't", "mightn't"
        }

    def transform(self, statement: str) -> str:
        """
        Transform affirmative sentence into negative sentence

        Args:
            statement: Affirmative sentence

        Returns:
            Transformed negative sentence
        """
        if not self.nlp:
            return self._fallback_transformation(statement)

        # Check if already negative
        if self.is_already_negative(statement):
            return statement

        try:
            # Step 1: spaCy analysis of statement
            doc = self.nlp(statement)

            # Step 2: Analyze sentence structure
            analysis = self._analyze_sentence_structure(doc)

            # Step 3: Apply appropriate negation strategy
            return self._apply_negation_strategy(doc, analysis)

        except (AttributeError, ValueError, RuntimeError, IndexError) as e:
            logger.warning(f"Negation transformation failed, using fallback: {e}")
            return self._fallback_transformation(statement)

    def _analyze_sentence_structure(self, doc) -> Dict:
        """Analyze sentence structure using spaCy"""
        analysis = {
            'main_verb': None,
            'auxiliary_verb': None,
            'subject': None,
            'verb_type': 'regular',  # 'be', 'modal', 'auxiliary', 'regular'
            'tense': 'present',
            'person': 'third_singular'
        }

        # Find subject first (important for identifying correct verb)
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                analysis['subject'] = token
                break

        # Find main verb with improved logic
        # Priority 1: Find finite verb (VBZ, VBP, VBD, VB) that is ROOT or has subject dependency
        main_verb_candidates = []

        for token in doc:
            if token.pos_ in ['VERB', 'AUX']:
                score = 0

                # ROOT verb gets highest priority
                if token.dep_ == 'ROOT':
                    score += 100

                # Finite verbs (not participles/gerunds) get higher priority
                if token.tag_ in ['VBZ', 'VBP', 'VBD', 'VB', 'MD']:
                    score += 50

                # Verbs with subjects get higher priority
                if analysis['subject'] and analysis['subject'].head == token:
                    score += 30

                # Avoid participles used as modifiers (amod, nmod, etc.)
                if token.dep_ in ['amod', 'acl', 'relcl', 'advcl']:
                    score -= 50

                # Gerunds (VBG) used as modifiers should have low priority
                if token.tag_ == 'VBG' and token.dep_ in ['amod', 'nmod']:
                    score -= 100

                main_verb_candidates.append((token, score))

        # Special handling: If no good VERB candidates found, check for common verbs misclassified as NOUN
        # This happens with words like "repeals", "amends", etc. in complex titles
        if not main_verb_candidates or max(c[1] for c in main_verb_candidates) < 50:
            for token in doc:
                # Look for words that look like third-person singular verbs
                # Pattern: word ends with 's' or 'es', tagged as NOUN, between proper nouns
                if (token.pos_ == 'NOUN' and
                    token.tag_ in ['NNS', 'NN'] and
                    token.text.endswith(('s', 'es'))):

                    # Check if this could be a verb based on context
                    # Common verb endings: repeals, amends, modifies, extends, etc.
                    lemma = token.lemma_.lower()

                    # Common verbs that might be misclassified
                    common_verbs = {
                        'repeal', 'amend', 'modify', 'extend', 'replace', 'supersede',
                        'establish', 'create', 'define', 'regulate', 'prohibit',
                        'restrict', 'allow', 'permit', 'require', 'mandate'
                    }

                    if lemma in common_verbs:
                        # This is likely a misclassified verb
                        # Check position: should be after subject (proper noun) and before object
                        if token.i > 0:
                            prev_tokens = [t for t in doc if t.i < token.i]
                            next_tokens = [t for t in doc if t.i > token.i]

                            # If there are proper nouns before and after, this is likely the verb
                            has_noun_before = any(t.pos_ in ['PROPN', 'NOUN'] for t in prev_tokens[-3:])
                            has_noun_after = any(t.pos_ in ['PROPN', 'NOUN'] for t in next_tokens[:3])

                            if has_noun_before and has_noun_after:
                                # Add as high-priority candidate
                                main_verb_candidates.append((token, 75))

        # Select the candidate with the highest score
        if main_verb_candidates:
            main_verb_candidates.sort(key=lambda x: x[1], reverse=True)
            analysis['main_verb'] = main_verb_candidates[0][0]

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

            # Classify verb type
            if verb_lemma in self.be_verbs:
                analysis['verb_type'] = 'be'
            elif verb_lemma in self.modal_verbs:
                analysis['verb_type'] = 'modal'
            elif verb_lemma in self.auxiliary_verbs and analysis['auxiliary_verb']:
                # Only classify as auxiliary if there's actually an auxiliary relationship
                analysis['verb_type'] = 'auxiliary'
            elif verb_lemma == 'have' and verb_tag in ['VBZ', 'VBP', 'VBD'] and not analysis['auxiliary_verb']:
                # "have" as main verb (possession) vs auxiliary
                # Check if followed by past participle
                main_verb_token = analysis['main_verb']
                next_tokens = [token for token in main_verb_token.doc if token.i > main_verb_token.i]
                has_past_participle = any(token.tag_ == 'VBN' for token in next_tokens[:2])

                if has_past_participle:
                    analysis['verb_type'] = 'auxiliary'
                else:
                    analysis['verb_type'] = 'regular'  # Possession: "I have money"
            else:
                analysis['verb_type'] = 'regular'

            # Determine tense
            if verb_tag in ['VBD', 'VBN']:
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

    def _apply_negation_strategy(self, doc, analysis) -> str:
        """Apply the appropriate negation strategy based on sentence structure"""
        tokens = [token.text for token in doc]

        if not analysis['main_verb']:
            return self._fallback_transformation(' '.join(tokens))

        main_verb = analysis['main_verb']
        verb_type = analysis['verb_type']

        # Special handling for imperatives
        if self._is_imperative(doc, analysis):
            return self._negate_imperative(tokens, main_verb)

        if verb_type == 'be':
            return self._negate_be_verb(tokens, main_verb)
        elif verb_type == 'modal':
            return self._negate_modal_verb(tokens, main_verb)
        elif analysis['auxiliary_verb']:
            return self._negate_auxiliary_verb(tokens, analysis['auxiliary_verb'])
        else:
            return self._negate_regular_verb(tokens, main_verb, analysis)

    def _is_imperative(self, doc, analysis) -> bool:
        """Check if sentence is imperative (command)"""
        # Imperative sentences usually:
        # 1. Start with a base form verb (VB)
        # 2. Have no explicit subject
        # 3. Verb is at position 0 or after interjections/discourse markers

        if not analysis['subject'] and analysis['main_verb']:
            first_token = next(iter(doc), None)
            main_verb = analysis['main_verb']

            # Check if the main verb is a base form (VB) - characteristic of imperatives
            if main_verb.tag_ != 'VB':
                return False

            # Check if main verb is at or near the start of the sentence
            # Allow for discourse markers like "Please", "Now", etc.
            if first_token and first_token.pos_ in ['VERB', 'AUX'] and first_token.tag_ == 'VB':
                return True

            # Check for "Please + verb" pattern
            if first_token and first_token.text.lower() in ['please', 'now', 'then']:
                if main_verb.i <= 1 and main_verb.tag_ == 'VB':
                    return True

        return False

    def _negate_imperative(self, tokens: List[str], main_verb) -> str:
        """Negate imperative sentences"""
        verb_text = main_verb.text.lower()

        # Special case for "be" in imperatives
        if verb_text == 'be':
            tokens_copy = tokens.copy()
            tokens_copy[main_verb.i] = 'do not be'
            return self._clean_result(' '.join(tokens_copy))
        else:
            # General imperatives: "Do not" + verb (preserve original case for first word)
            sentence = ' '.join(tokens)
            # Check if starts with "Let"
            if sentence.lower().startswith('let'):
                return self._clean_result(f"Do not {sentence.lower()}")
            else:
                return self._clean_result(f"Do not {sentence.lower()}")

    def _negate_be_verb(self, tokens: List[str], be_verb) -> str:
        """Negate sentences with be verbs"""
        tokens_copy = tokens.copy()
        verb_text = be_verb.text.lower()

        if verb_text in self.be_verbs:
            tokens_copy[be_verb.i] = self.be_verbs[verb_text]
        else:
            # Handle contractions and other forms
            tokens_copy[be_verb.i] = f"{be_verb.text} not"

        return self._clean_result(' '.join(tokens_copy))

    def _negate_modal_verb(self, tokens: List[str], modal_verb) -> str:
        """Negate sentences with modal verbs"""
        tokens_copy = tokens.copy()
        verb_text = modal_verb.text.lower()

        if verb_text in self.modal_verbs:
            negation = self.modal_verbs[verb_text]
            tokens_copy[modal_verb.i] = negation
        else:
            # Handle edge cases
            tokens_copy[modal_verb.i] = f"{modal_verb.text} not"

        return self._clean_result(' '.join(tokens_copy))

    def _negate_auxiliary_verb(self, tokens: List[str], aux_verb) -> str:
        """Negate sentences with auxiliary verbs"""
        tokens_copy = tokens.copy()
        verb_text = aux_verb.text.lower()

        if verb_text in self.auxiliary_verbs:
            tokens_copy[aux_verb.i] = self.auxiliary_verbs[verb_text]
        else:
            # Handle edge cases
            tokens_copy[aux_verb.i] = f"{aux_verb.text} not"

        return self._clean_result(' '.join(tokens_copy))

    def _negate_regular_verb(self, tokens: List[str], main_verb, analysis) -> str:
        """Negate sentences with regular verbs by adding do/does/did + not"""
        tokens_copy = tokens.copy()
        verb_pos = main_verb.i
        verb_tag = main_verb.tag_
        verb_text = main_verb.text
        verb_lemma = main_verb.lemma_

        # Special handling for verbs misclassified as NOUN
        if main_verb.pos_ == 'NOUN' and verb_text.endswith(('s', 'es')):
            # This is likely a third-person singular verb misclassified as noun
            # e.g., "repeals" -> "repeal"
            aux_negation = 'does not'
            new_verb = verb_lemma  # Use base form
        # Determine appropriate auxiliary for regular verbs
        elif verb_tag == 'VBZ':  # Third person singular present
            aux_negation = 'does not'
            new_verb = verb_lemma  # Use base form
        elif verb_tag == 'VBD':  # Past tense
            aux_negation = 'did not'
            new_verb = verb_lemma  # Use base form
        else:  # Other present forms
            aux_negation = 'do not'
            new_verb = verb_text  # Keep as is for base forms

        # Insert auxiliary negation before the main verb
        tokens_copy.insert(verb_pos, aux_negation)
        # Replace the main verb with base form
        tokens_copy[verb_pos + 1] = new_verb

        return self._clean_result(' '.join(tokens_copy))

    def _clean_result(self, result: str) -> str:
        """Clean up the negation result"""
        # Fix common contraction spacing issues
        contractions = {
            " 't": "'t",
            " 's": "'s",
            " 're": "'re",
            " 'll": "'ll",
            " 've": "'ve",
            " 'd": "'d"
        }

        for wrong, correct in contractions.items():
            result = result.replace(wrong, correct)

        # Fix punctuation spacing issues
        # Remove space before punctuation
        result = re.sub(r'\s+([,;:.!?)])', r'\1', result)
        # Remove space after opening parenthesis
        result = re.sub(r'([(])\s+', r'\1', result)

        # Fix hyphen spacing (e.g., "Non - Fatal" -> "Non-Fatal")
        # This pattern handles hyphens that were tokenized separately by spaCy
        result = re.sub(r'\s+-\s+', '-', result)

        # Remove extra spaces
        result = re.sub(r'\s+', ' ', result.strip())

        # Ensure proper capitalization
        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        return result

    def is_already_negative(self, statement: str) -> bool:
        """Check if statement is already negative"""
        if not self.nlp:
            # Fallback: simple string checking
            statement_lower = statement.lower()
            return any(indicator in statement_lower for indicator in self.negative_indicators)

        # Use spaCy for more sophisticated checking
        doc = self.nlp(statement)

        # Check for negative words
        for token in doc:
            if token.text.lower() in self.negative_indicators:
                return True
            if token.lemma_.lower() in self.negative_indicators:
                return True

        # Check for negative dependencies
        for token in doc:
            if token.dep_ == 'neg':  # Negation dependency
                return True

        return False

    def _fallback_transformation(self, statement: str) -> str:
        """Fallback transformation when spaCy is not available"""
        if self.is_already_negative(statement):
            return statement

        statement = statement.strip()

        # Simple pattern-based negation
        patterns = [
            (r'\b(is)\b', r'is not'),
            (r'\b(are)\b', r'are not'),
            (r'\b(was)\b', r'was not'),
            (r'\b(were)\b', r'were not'),
            (r'\b(am)\b', r'am not'),
            (r'\b(can)\b', r'cannot'),
            (r'\b(will)\b', r'will not'),
            (r'\b(would)\b', r'would not'),
            (r'\b(should)\b', r'should not'),
            (r'\b(could)\b', r'could not'),
            (r'\b(may)\b', r'may not'),
            (r'\b(might)\b', r'might not'),
            (r'\b(must)\b', r'must not'),
            (r'\b(have)\b', r'have not'),
            (r'\b(has)\b', r'has not'),
            (r'\b(had)\b', r'had not'),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, statement, re.IGNORECASE):
                return re.sub(pattern, replacement, statement, count=1, flags=re.IGNORECASE)

        # If no auxiliary/modal/be verb found, add "do not" before main verb
        # This is a simplified approach
        return f"It is not true that {statement.lower()}"

    # Convenience methods
    def negate(self, statement: str) -> str:
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
            "is_negative": self.is_already_negative(statement)
        }

        return analysis