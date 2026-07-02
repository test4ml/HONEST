#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English WH-Question Transformer Core Module

Contains the main EnglishWhTransformer class with transformation logic
for converting declarative sentences into WH-questions.
"""

import re
import logging
from typing import Dict, Optional, Tuple, List
import spacy
from ...core.types import Language
from honest.constants import SPACY_MODEL_NAME
from .entity_matcher import EntityMatcher
from .utils import (
    find_main_verb, find_subject, join_tokens_with_punct,
    find_matching_parenthesis, clean_question
)

logger = logging.getLogger(__name__)


class EnglishWhTransformer:
    """English statement to WH-question transformer using spaCy"""

    def __init__(self):
        """Initialize the transformer with spaCy English model"""
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Initialize entity matcher
        self.entity_matcher = EntityMatcher(self.nlp) if self.nlp else None

        # Verb classifications for auxiliary verb handling
        self.be_verbs = {'be', 'am', 'is', 'are', 'was', 'were', 'being', 'been'}
        self.do_verbs = {'do', 'does', 'did'}
        self.have_verbs = {'have', 'has', 'had'}
        self.modal_verbs = {
            'can', 'could', 'will', 'would', 'shall', 'should',
            'may', 'might', 'must', 'ought'
        }

        # All auxiliary verbs
        self.auxiliary_verbs = self.be_verbs | self.do_verbs | self.have_verbs | self.modal_verbs

        # Question words mapping
        self.question_words = {
            'what', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'which'
        }

        # Preposition classifications
        self.common_prepositions = ['as', 'at', 'in', 'on', 'by', 'for', 'with', 'to', 'of', 'from', 'about']
        self.semantic_prepositions = {'as', 'for', 'about'}  # Prepositions that add semantic meaning
        self.location_prepositions = {'in', 'at', 'on', 'to', 'from'}  # Location-related prepositions

        # Common noun+preposition collocations where preposition is essential
        # Format: {preposition: set_of_nouns}
        self.noun_preposition_collocations = {
            'on': {'account', 'reliance', 'dependence', 'emphasis', 'focus', 'impact', 'effect'},
            'in': {'interest', 'belief', 'confidence', 'faith', 'trust', 'participation', 'candidate', 'taxon'},  # Added 'taxon' for "found in taxon" pattern
            'of': {'kind', 'type', 'sort', 'piece', 'member', 'part', 'collection', 'diocese', 'universe', 'group', 'category', 'class', 'family', 'set', 'series', 'factor'},  # Added 'factor' for "prime factor of" pattern
            'at': {'look', 'glance', 'attempt'},
            'for': {'need', 'demand', 'search', 'quest', 'reason', 'cause', 'candidate'},  # Added 'candidate' for "is a candidate for"
            'to': {'approach', 'access', 'reference', 'response', 'reaction'},
            'with': {'agreement', 'disagreement', 'connection', 'association'},
            'from': {'difference', 'distance', 'departure', 'escape'},
        }

        # Sentence complexity thresholds
        self.MAX_SIMPLE_PREDICATE_LENGTH = 4  # Predicates longer than this are considered complex

    def transform(self, statement: str, question_word: str = "what", focus_entity: str = None) -> str:
        """
        Transform declarative sentence into WH-question

        Args:
            statement: Declarative sentence
            question_word: Question word (what, who, where, when, which, etc.)
            focus_entity: Entity to be replaced with question word

        Returns:
            Transformed WH-question
        """
        if not self.nlp:
            return self._fallback_transformation(statement, question_word, focus_entity)

        try:
            # Step 1: spaCy analysis of original statement
            doc = self.nlp(statement)

            # Step 2: Determine if focus entity is the subject (before replacement)
            is_focus_entity_subject = False
            if focus_entity:
                # Use spaCy entity matching to find the correct occurrence
                # (in case the entity appears multiple times in the statement)
                matched_entity, start_char, end_char, match_info = self.entity_matcher.spacy_entity_matching(
                    focus_entity, doc
                )

                if matched_entity and start_char != -1:
                    # FIX: Only check if the CORE entity (before any parenthetical description)
                    # is the subject. Do NOT check tokens inside parentheses, as they may contain
                    # relative clause subjects (e.g., "that" in "Bridge (thing that does X)")
                    # which should not affect whether the entire focus entity is the sentence subject.
                    is_focus_entity_subject = self.entity_matcher.is_core_entity_subject(
                        doc, start_char, end_char, focus_entity
                    )
                else:
                    # Fallback: spaCy matching failed (e.g., "Sharks and Little Fish")
                    # Use simple string matching to find the entity position
                    entity_start_pos = statement.find(focus_entity)
                    if entity_start_pos != -1:
                        # FIX: Use the same improved logic for fallback case
                        is_focus_entity_subject = self.entity_matcher.is_core_entity_subject(
                            doc, entity_start_pos, entity_start_pos + len(focus_entity), focus_entity
                        )

            # Step 3: Handle entity replacement if focus entity specified
            if focus_entity:
                processed_statement = self._replace_entity_with_question_word(
                    statement, focus_entity, question_word, doc
                )
                # If entity replacement failed, fall back to general question
                if processed_statement is None:
                    return self._construct_general_question(statement, question_word, doc)
            else:
                # No focus entity - generate general question without entity replacement
                return self._construct_general_question(statement, question_word, doc)

            # Step 4: Transform to question form, passing the subject info
            return self._transform_to_question_form(processed_statement, question_word, doc, is_focus_entity_subject, focus_entity)

        except (AttributeError, ValueError, RuntimeError, IndexError) as e:
            logger.warning(f"Grammar transformation failed, using fallback: {e}")
            return self._fallback_transformation(statement, question_word, focus_entity)

    def _replace_entity_with_question_word(self, statement: str, focus_entity: str, question_word: str, doc) -> str:
        """Replace focus entity with question word"""
        # Find the best match using spaCy analysis
        matched_entity, start_char, end_char, match_info = self.entity_matcher.spacy_entity_matching(
            focus_entity, doc
        )

        if matched_entity is None:
            # Fallback: try simple string replacement with word boundaries
            pattern = r'\b' + re.escape(focus_entity) + r'\b'
            if re.search(pattern, statement, re.IGNORECASE):
                return re.sub(pattern, question_word, statement, flags=re.IGNORECASE)
            else:
                # No match found - return None to indicate failure
                return None

        # Check for article before the entity that should be removed
        # For "where" and "when" questions, we should remove articles like "the", "a", "an"
        # before the entity to avoid patterns like "competes in the where"
        adjusted_start_char = start_char

        # Check if there's an article immediately before the entity
        before_entity = statement[:start_char].rstrip()
        articles = ['the', 'a', 'an']

        for article in articles:
            # Check if the text before entity ends with this article (as a separate word)
            if before_entity.lower().endswith(' ' + article):
                # For "where" and "when" questions, remove the article
                if question_word.lower() in ['where', 'when']:
                    # Calculate the position to remove the article
                    article_start = len(before_entity) - len(article)
                    adjusted_start_char = article_start
                    break

        # Handle possessive case and parentheses after the entity
        # Important: Process in the correct order:
        # 1. First remove parentheses (e.g., "(researcher ORCID ID = ...)")
        # 2. Then handle possessive markers (e.g., "'s")
        # This handles cases like "Allanah Kenny (researcher ORCID ID = 0000-0002-2643-9404)'s doctoral thesis"

        after_text = statement[end_char:]

        # Step 1: Handle parentheses after the entity
        # Check if there are parentheses immediately after the entity that should be removed
        # This handles cases like "Frederik Bernard Albinus ( Dutch university teacher ) is..."
        # or "Allanah Kenny (researcher ORCID ID = 0000-0002-2643-9404)'s doctoral thesis"
        #
        # FIX: Only remove parentheses if they are NOT part of the original focus_entity
        # This prevents incorrectly removing entity descriptions like "Name (description)"
        if after_text.strip().startswith('('):
            # Check if the parentheses are part of the original focus_entity
            should_remove_parentheses = False

            if focus_entity and matched_entity:
                # Check if focus_entity contains the matched text followed by '('
                if matched_entity in focus_entity:
                    # Find where matched_entity appears in focus_entity
                    match_pos = focus_entity.find(matched_entity)
                    remaining_focus = focus_entity[match_pos + len(matched_entity):]

                    # If focus_entity has more content after matched_entity,
                    # and it starts with '(', then the parentheses are part of the entity
                    if remaining_focus.strip().startswith('('):
                        # Parentheses are part of the entity label - DO NOT REMOVE
                        should_remove_parentheses = False
                    else:
                        # Parentheses are additional notes - CAN REMOVE
                        should_remove_parentheses = True
                else:
                    # Be conservative: if matched_entity not in focus_entity, don't remove
                    should_remove_parentheses = False
            else:
                # If no focus_entity provided, use old behavior (remove parentheses)
                should_remove_parentheses = True

            # Only remove parentheses if determined safe to do so
            if should_remove_parentheses:
                # Find the matching closing parenthesis
                open_paren_pos = after_text.find('(')
                close_paren_pos = find_matching_parenthesis(after_text, open_paren_pos)

                if close_paren_pos != -1:
                    # Remove the entire parenthetical content
                    # Keep the text before '(' and after ')'
                    after_text = after_text[:open_paren_pos] + after_text[close_paren_pos + 1:]

        # Step 2: Handle possessive markers
        # For example: "Tim David's doctoral thesis" -> "whose doctoral thesis"
        # or after removing parentheses: " 's doctoral thesis" -> "whose doctoral thesis"
        possessive_markers = ["'s", "'"]
        for marker in possessive_markers:
            if after_text.strip().startswith(marker):
                # For "who" questions with possessive, use "whose"
                if question_word.lower() == "who":
                    question_word = "whose"
                # Remove the possessive marker
                marker_pos = after_text.find(marker)
                after = after_text[marker_pos + len(marker):]
                break
        else:
            # No possessive marker found
            after = after_text

        # Replace the matched entity (and possibly article) with question word
        before = statement[:adjusted_start_char]
        result = f"{before}{question_word}{after}"

        # Clean up any double spaces
        result = re.sub(r'\s+', ' ', result)

        return result

    def _transform_to_question_form(self, statement: str, question_word: str, original_doc, is_focus_entity_subject: bool = False, focus_entity: str = None) -> str:
        """Transform statement to proper question form

        Args:
            statement: Statement to transform (possibly with entity already replaced)
            question_word: The question word to use
            original_doc: spaCy doc of the original statement (before replacement)
            is_focus_entity_subject: Whether the replaced entity was the subject
            focus_entity: The entity being replaced with question word
        """
        # Re-analyze the statement after entity replacement
        doc = self.nlp(statement)

        # Find key sentence components
        main_verb = find_main_verb(doc)
        subject = find_subject(doc)
        auxiliary = self._find_auxiliary(doc)

        if not main_verb:
            # Fallback for sentences without clear main verb
            return f"{question_word} {statement}?"

        # Check if question word is already in the sentence
        tokens_text = [token.text for token in doc]

        # Handle special case where "who" was changed to "whose"
        actual_question_word = question_word
        if question_word.lower() == "who" and "whose" in [token.lower() for token in tokens_text]:
            actual_question_word = "whose"
            question_word_positions = [i for i, token in enumerate(tokens_text) if token.lower() == "whose"]
        else:
            # FIX: Handle multi-word question words (e.g., "how many", "how much")
            # Split question word into tokens and search for matching sequences
            question_word_tokens = question_word.lower().split()

            if len(question_word_tokens) > 1:
                # Multi-word question word - search for token sequences
                question_word_positions = []
                for i in range(len(tokens_text) - len(question_word_tokens) + 1):
                    # Check if tokens at position i match the question word sequence
                    if all(tokens_text[i + j].lower() == question_word_tokens[j]
                           for j in range(len(question_word_tokens))):
                        question_word_positions.append(i)
            else:
                # Single-word question word - use original logic
                question_word_positions = [i for i, token in enumerate(tokens_text)
                                          if token.lower() == question_word.lower()]


        if question_word_positions:
            # Question word is in the sentence, need to move it to front and rearrange
            return self._rearrange_with_question_word_in_sentence(doc, tokens_text, main_verb, subject, auxiliary, actual_question_word, question_word_positions[0], is_focus_entity_subject, original_doc, focus_entity)
        else:
            # Question word not in sentence, add it and rearrange
            return self._construct_question_from_statement(main_verb, subject, auxiliary, question_word, doc)

    def _handle_subject_replacement(self, question_word: str, tokens_without_q: List[str], q_word_pos: int, auxiliary) -> str:
        """Handle question formation when question word replaced the subject

        Args:
            question_word: The question word (what, who, etc.)
            tokens_without_q: Token list with question word already removed
            q_word_pos: Original position of question word
            auxiliary: Auxiliary verb token (if present)

        Returns:
            Formatted question string
        """
        # Also remove determiners that might be left hanging
        cleaned_tokens = []
        for i, token in enumerate(tokens_without_q):
            # Skip determiners at the beginning if they preceded the replaced subject
            # The determiner should be at position q_word_pos - 1 before replacement
            if (i == 0 and token.lower() in ['the', 'a', 'an'] and
                q_word_pos == 1):  # Question word was at position 1, so determiner was at 0
                continue
            cleaned_tokens.append(token)

        if auxiliary:
            # Keep auxiliary structure: "What is eating"
            result = f"{question_word} {join_tokens_with_punct(cleaned_tokens)}"
        else:
            # For subject replacement, we generally don't need to add auxiliary
            # "John loves Mary" -> "Who loves Mary?" (not "Who does love Mary?")
            # However, for certain question words like "when" and "where",
            # we may need to add auxiliary for natural-sounding questions
            # Example: "When does February 11, 2018 occur?" vs "When occurs February 11, 2018?"
            if question_word.lower() in ['when', 'where']:
                # For "when" and "where" questions with subject replacement,
                # add auxiliary and convert verb to base form for natural sound
                if cleaned_tokens and len(cleaned_tokens) > 0:
                    # Find the main verb position and convert to base form
                    # This is a simplified approach - we assume the first verb is the main verb
                    aux_word = 'does'  # Default auxiliary for present tense

                    # Convert verbs to base form
                    converted_tokens = []
                    for token in cleaned_tokens:
                        # Simple heuristic: if it looks like a verb in present tense, convert to base form
                        if token.endswith('s') and token not in ['is', 'has', 'was', 'does']:
                            # Remove 's' ending for third person singular
                            base_form = token[:-1]
                            converted_tokens.append(base_form)
                        else:
                            converted_tokens.append(token)

                    result = f"{question_word} {aux_word} {join_tokens_with_punct(converted_tokens)}"
                else:
                    result = f"{question_word} {join_tokens_with_punct(cleaned_tokens)}"
            else:
                # Keep the verb form as is for other subject questions
                result = f"{question_word} {join_tokens_with_punct(cleaned_tokens)}"

        return clean_question(result, self.question_words)

    def _extract_subject_text_and_indices(self, doc, subject, tokens_text: List[str], question_word: str, keep_determiner: bool = False) -> Tuple[str, int, int]:
        """Extract complete subject text and its token indices

        Args:
            doc: spaCy Doc object
            subject: Subject token from spaCy
            tokens_text: List of token strings
            question_word: Question word for article handling
            keep_determiner: If True, keep determiners (a, an, the, this, that) in subject

        Returns:
            Tuple of (subject_text, subject_start_idx, subject_end_idx)
        """
        # Find subject span in original doc (using spaCy tokens)
        # Strategy: find the complete noun phrase that contains the subject
        subject_start_idx = subject.i
        subject_end_idx = subject.i

        # FIX for kg_rule_102: If subject is an npadvmod (Category: pattern),
        # include the determiner before it if present
        if subject.dep_ == 'npadvmod' and subject.i > 0:
            prev_token = doc[subject.i - 1]
            if prev_token.pos_ == 'DET':
                subject_start_idx = prev_token.i

        # Method 1: Use spaCy's noun_chunks to find complete noun phrase
        subject_chunk = None
        for chunk in doc.noun_chunks:
            if chunk.start <= subject.i <= chunk.end:
                subject_chunk = chunk
                subject_start_idx = chunk.start
                subject_end_idx = chunk.end - 1  # end is exclusive
                break

        # Method 2: If noun_chunks didn't work, manually find compound nouns
        if not subject_chunk:
            # Look backwards for compound parts (before subject)
            for token in doc:
                if token.i < subject.i and token.head.i == subject.i and token.dep_ in ['compound']:
                    subject_start_idx = min(subject_start_idx, token.i)

            # Look forward for compound parts (after subject)
            for token in doc:
                if token.i > subject.i and (token.head.i == subject.i and token.dep_ in ['compound', 'flat']) or (token.i == subject.i + 1 and token.pos_ in ['PROPN', 'NOUN']):
                    subject_end_idx = max(subject_end_idx, token.i)

            # Also check for consecutive proper nouns (like "Main Street Bridge")
            current_idx = subject.i
            while current_idx + 1 < len(doc):
                next_token = doc[current_idx + 1]
                if next_token.pos_ in ['PROPN', 'NOUN']:
                    subject_end_idx = next_token.i
                    current_idx = next_token.i
                else:
                    break

        # Method 3 (FIX for kg_rule_10): Include prepositional phrases attached to subject
        # Example: "quasi-national park of Japan" should include "of Japan"
        # Check if subject has prep children and include their objects
        #
        # FIX for kg_rule_1: Also recursively check descendants for prepositions
        # Example: "the next crossing downstream from X"
        # Structure: crossing -> downstream (advmod) -> from (prep) -> X (pobj)
        # We need to include "from X" as part of the subject
        subject_token = doc[subject.i] if hasattr(subject, 'i') else subject

        def include_prep_descendants(token, current_end_idx):
            """Recursively include prepositional phrases and appositional phrases from descendants

            Handles both:
            - Prepositional phrases (e.g., "from the city of London")
            - Appositional phrases (e.g., "(fictional character in the Star Trek Universe)")

            Also includes punctuation (especially closing parentheses) that are direct children
            of the token, to handle cases like "Alice (famous scientist)" where ')' is a child
            of 'Alice' rather than a child of the appos 'scientist'.
            """
            max_end_idx = current_end_idx
            for child in token.children:
                # Handle prepositional phrases
                if child.dep_ == 'prep' and child.i > current_end_idx:
                    # Found a preposition - include it and its objects
                    max_end_idx = max(max_end_idx, child.i)
                    # Also include the object of the preposition (pobj)
                    for grandchild in child.children:
                        if grandchild.dep_ == 'pobj':
                            max_end_idx = max(max_end_idx, grandchild.i)
                            # Recursively check if pobj has compounds
                            for ggchild in grandchild.children:
                                if ggchild.dep_ in ['compound', 'amod']:
                                    max_end_idx = max(max_end_idx, ggchild.i)

                # Handle appositional phrases (e.g., parenthetical descriptions)
                # Example: "Jankom Pog (fictional character in the Star Trek Universe)"
                # The appos is "character" with children including '(' and ')' as punct
                elif child.dep_ == 'appos':
                    # Include the appos itself
                    max_end_idx = max(max_end_idx, child.i)
                    # Include all descendants of appos (modifiers, preps, and punctuation)
                    for grandchild in child.children:
                        max_end_idx = max(max_end_idx, grandchild.i)
                        # Recursively include grandchild's descendants (e.g., prep objects)
                        for ggchild in grandchild.children:
                            max_end_idx = max(max_end_idx, ggchild.i)
                            # Go one more level for compounds in prep objects
                            for gggchild in ggchild.children:
                                if gggchild.dep_ in ['compound', 'amod']:
                                    max_end_idx = max(max_end_idx, gggchild.i)

                # Handle punctuation that's a direct child (e.g., closing parenthesis)
                # Example: "Alice (famous scientist)" - ')' is a child of 'Alice', not of 'scientist'
                # The '(' is typically a child of the appos, not of the main noun
                elif child.dep_ == 'punct' and child.text == ')' and child.i > current_end_idx:
                    # Check if there's an appos sibling (which would have the opening paren)
                    # This indicates a parenthetical phrase structure
                    has_appos = False
                    for sibling in token.children:
                        if sibling.dep_ == 'appos' and sibling.i < child.i:
                            has_appos = True
                            break
                    if has_appos:
                        max_end_idx = max(max_end_idx, child.i)

                # Recursively check this child's descendants
                max_end_idx = max(max_end_idx, include_prep_descendants(child, max_end_idx))
            return max_end_idx

        subject_end_idx = include_prep_descendants(subject_token, subject_end_idx)

        # FIX for kg_rule_102: Handle Wikidata entities with colon (Category:, Template:)
        # spaCy incorrectly splits "the Category:Horse breeds" into two noun chunks:
        #   1. "the Category"
        #   2. "Horse breeds"
        # We need to merge them back together to avoid word order issues
        #
        # Check if the subject contains "Category:" or "Template:" pattern
        if subject_end_idx + 2 < len(doc):
            # Check if next token is ":" and the current subject is "Category" or "Template"
            current_subject_text = tokens_text[subject_start_idx:subject_end_idx + 1]
            next_token = doc[subject_end_idx + 1]

            # Check if we have a pattern like "Category" or "Template" followed by ":"
            subject_contains_meta = any(
                token.lower() in ['category', 'template']
                for token in current_subject_text
            )

            if subject_contains_meta and next_token.text == ':':
                # Found "Category:" or "Template:" pattern
                # Now look for the next noun phrase that should be part of this entity
                # This is typically identified as another nsubjpass by spaCy
                next_noun_idx = subject_end_idx + 2

                # Find the end of the continuation noun phrase
                # Look for tokens that are part of the entity name (e.g., "Horse breeds")
                continuation_end = next_noun_idx
                for i in range(next_noun_idx, len(doc)):
                    token = doc[i]
                    # Include tokens that are part of the entity name
                    # Stop at verbs or prepositions (that's where the entity name ends)
                    if token.pos_ in ['VERB', 'AUX', 'ADP'] and token.dep_ not in ['compound', 'amod']:
                        break
                    if token.pos_ in ['NOUN', 'PROPN'] or token.dep_ in ['compound', 'amod']:
                        continuation_end = i
                    else:
                        # If we hit a non-noun/non-adjective that's not a compound,
                        # check if it's part of a multi-nsubj pattern
                        if token.dep_ in ['nsubj', 'nsubjpass'] and i == next_noun_idx:
                            # This is the continuation noun (e.g., "breeds" in "Horse breeds")
                            continuation_end = i
                        else:
                            break

                # Extend subject to include the continuation
                if continuation_end >= next_noun_idx:
                    subject_end_idx = continuation_end

        # Extract subject text from tokens_text
        subject_tokens_text = tokens_text[subject_start_idx:subject_end_idx + 1]

        # Check if there are parentheses after the subject that should be included
        # For example: "Nick Brown (researcher in New Zealand)" should extract the full name with parentheses
        # This is important for preserving descriptive information in the subject
        if subject_end_idx + 1 < len(tokens_text) and tokens_text[subject_end_idx + 1] == '(':
            # Find the matching closing parenthesis
            paren_level = 0
            paren_end_idx = subject_end_idx + 1

            for i in range(subject_end_idx + 1, len(tokens_text)):
                if tokens_text[i] == '(':
                    paren_level += 1
                elif tokens_text[i] == ')':
                    paren_level -= 1
                    if paren_level == 0:
                        paren_end_idx = i
                        break

            # If we found the matching closing parenthesis, include all tokens up to and including it
            if paren_level == 0:
                # Update subject_end_idx to include the closing parenthesis
                original_subject_end_idx = subject_end_idx
                subject_end_idx = paren_end_idx
                # Re-extract subject tokens to include parenthetical content
                subject_tokens_text = tokens_text[subject_start_idx:subject_end_idx + 1]

        # Remove leading articles (a, an, the) from subject
        # But keep them if:
        # 1. "where" questions need the article
        # 2. keep_determiner is True (e.g., for "be + noun + of" constructions)
        # 3. FIX for kg_rule_1: Complex noun phrases like "the next X from Y"
        #    Pattern: "the next [NOUN] [ADV/PREP] from [ENTITY]"
        #    Example: "the next crossing downstream from Kanmon Roadway Tunnel"
        #    In these cases, "the" is essential and should not be removed
        # 4. FIX for kg_rule_102: Wikidata entities (Category:, Template:) need "the"
        #    Example: "the Category:Horse breeds" should keep "the"
        is_complex_next_pattern = False
        if (subject_start_idx < len(doc) and
            doc[subject_start_idx].text.lower() == 'the' and
            subject_start_idx + 1 < len(doc) and
            doc[subject_start_idx + 1].text.lower() == 'next'):
            # This is "the next..." pattern - keep "the"
            is_complex_next_pattern = True

        # Check if this is a Wikidata entity (Category:, Template:)
        is_wikidata_entity = False
        for token in subject_tokens_text:
            if token.lower() in ['category', 'template'] and ':' in subject_tokens_text:
                is_wikidata_entity = True
                break

        if (subject_start_idx < len(doc) and
            doc[subject_start_idx].pos_ == 'DET' and
            question_word.lower() != 'where' and
            not keep_determiner and
            not is_complex_next_pattern and
            not is_wikidata_entity):  # Don't remove "the" for Wikidata entities
            # Skip the article in the text
            subject_tokens_text = subject_tokens_text[1:]

        # Join subject tokens, handling punctuation properly
        subject_text = join_tokens_with_punct(subject_tokens_text)

        return subject_text, subject_start_idx, subject_end_idx

    def _handle_whose_question(self, doc, tokens_text: List[str], main_verb, subject, auxiliary, q_word_pos: int, original_doc) -> str:
        """Handle special case for 'whose' questions

        For sentences like "Allanah Kenny supervised whose doctoral thesis?"
        we want to generate "Whose doctoral thesis did Allanah Kenny supervise?"
        """
        # Extract the complete noun phrase that follows "whose" including possessive markers
        # In "Allanah Kenny supervised whose doctoral thesis", extract "doctoral thesis"
        # In "Peter borrowed his friend's brother's laptop", extract "brother's laptop"
        noun_phrase_tokens = []
        current_pos = q_word_pos + 1  # Start after "whose"

        # Look for nouns, adjectives, and possessive markers after "whose"
        while current_pos < len(tokens_text):
            token = tokens_text[current_pos]
            # Check if token is a noun, proper noun, adjective, or possessive marker
            if (current_pos < len(doc) and
                (doc[current_pos].pos_ in ['NOUN', 'PROPN', 'ADJ'] or
                 token in ["'s", "'"])):
                noun_phrase_tokens.append(token)
                current_pos += 1
            else:
                break

        if not noun_phrase_tokens:
            # Fallback: just use the remaining tokens
            return f"Whose {' '.join(tokens_text[q_word_pos + 1:])}?"

        # Extract the noun phrase
        noun_phrase = join_tokens_with_punct(noun_phrase_tokens)

        # Get the tokens before "whose" (subject and verb)
        before_whose = tokens_text[:q_word_pos]

        # Handle pronouns like "his" that should be removed
        # In "Peter borrowed his friend's brother's laptop", remove "his"
        cleaned_before_whose = []
        for i, token in enumerate(before_whose):
            # Skip possessive pronouns that precede the "whose" noun phrase
            if (token.lower() in ['his', 'her', 'their', 'my', 'your'] and
                i == len(before_whose) - 1):  # Only if it's immediately before "whose"
                continue
            cleaned_before_whose.append(token)

        before_whose = cleaned_before_whose

        # Find the main verb and convert to base form if needed
        if main_verb and main_verb.lemma_.lower() not in self.be_verbs | self.have_verbs:
            main_verb_base = self._get_base_form(main_verb)
            # Find the position of the main verb in before_whose
            for i, token in enumerate(before_whose):
                if token.lower() == main_verb.text.lower():
                    before_whose[i] = main_verb_base
                    break

        # Construct the question: "Whose [noun phrase] [auxiliary] [subject and verb]"
        if auxiliary:
            aux_text = auxiliary.text
            # Remove auxiliary from before_whose
            aux_pos = auxiliary.i
            if aux_pos < len(before_whose):
                before_whose = before_whose[:aux_pos] + before_whose[aux_pos + 1:]
            result = f"Whose {noun_phrase} {aux_text} {' '.join(before_whose)}"
        else:
            # Add appropriate auxiliary
            aux_word = self._determine_auxiliary(main_verb, subject)
            result = f"Whose {noun_phrase} {aux_word} {' '.join(before_whose)}"

        return clean_question(result, self.question_words)

    def _rearrange_with_question_word_in_sentence(self, doc, tokens_text: List[str], main_verb, subject, auxiliary, question_word: str, q_word_pos: int, is_focus_entity_subject: bool, original_doc, focus_entity: str = None) -> str:
        """Rearrange when question word is already in the sentence

        Args:
            doc: spaCy Doc object with full token information (after replacement)
            tokens_text: List of token strings (for easy manipulation)
            main_verb, subject, auxiliary: spaCy Token objects
            question_word: The question word string
            q_word_pos: Position of question word in tokens_text (for multi-word, this is the start position)
            is_focus_entity_subject: Whether the replaced entity was the subject in original sentence
            original_doc: spaCy Doc object of the original statement (before replacement)
            focus_entity: The entity being replaced with question word
        """
        # Special handling for "whose" questions
        if question_word.lower() == "whose":
            result = self._handle_whose_question(doc, tokens_text, main_verb, subject, auxiliary, q_word_pos, original_doc)
            return result

        # FIX: Handle multi-word question words (e.g., "how many" takes 2 tokens)
        # Remove the question word from its current position
        question_word_tokens = question_word.split()
        num_tokens_to_remove = len(question_word_tokens)

        # Remove all tokens that make up the question word
        tokens_without_q = tokens_text[:q_word_pos] + tokens_text[q_word_pos + num_tokens_to_remove:]

        # Handle subject replacement case
        if is_focus_entity_subject:
            return self._handle_subject_replacement(question_word, tokens_without_q, q_word_pos, auxiliary)

        # Handle object replacement case
        if auxiliary and subject:
            return self._handle_object_replacement_with_auxiliary(
                doc, tokens_text, tokens_without_q, main_verb, subject, auxiliary,
                question_word, q_word_pos, original_doc, focus_entity
            )
        else:
            return self._handle_object_replacement_without_auxiliary(
                main_verb, subject, tokens_without_q, question_word, q_word_pos,
                tokens_text, original_doc, focus_entity
            )

    def _handle_object_replacement_with_auxiliary(self, doc, tokens_text: List[str], tokens_without_q: List[str], main_verb, subject, auxiliary, question_word: str, q_word_pos: int, original_doc, focus_entity: str) -> str:
        """Handle object replacement when auxiliary verb is present"""
        aux_text = auxiliary.text
        aux_pos = auxiliary.i

        # FIX for kg_rule_10: Convert "has/have" to "is/are" for "has X of" pattern
        # When auxiliary is "has/have" (from our special detection), convert to be-verb
        # Example: "X has the next higher rank of Y" -> "What is X the next higher rank of?"
        # NOT: "What has X the next higher rank of?"
        if auxiliary.lemma_.lower() in self.have_verbs:
            # Check if there's "of" in the sentence (our special pattern)
            has_of_pattern = any(
                token.lemma_.lower() == 'of' and token.dep_ == 'prep'
                for token in doc
            )
            if has_of_pattern:
                # Convert "has" -> "is", "have" -> "are"
                if auxiliary.tag_ in ['VBZ', 'VBP']:  # present tense
                    # VBZ = third person singular (has -> is)
                    # VBP = non-third person (have -> are)
                    aux_text = 'is' if auxiliary.tag_ == 'VBZ' else 'are'
                elif auxiliary.tag_ in ['VBD']:  # past tense (had)
                    # Check subject number for was/were
                    aux_text = 'was'  # default to singular
                    # Could enhance this with subject plurality check if needed
                else:
                    aux_text = 'is'  # fallback

        # Adjust auxiliary position if it comes after removed question word
        if aux_pos > q_word_pos:
            aux_pos -= 1

        # Check if this is a "be + noun + of" construction
        # For such constructions, we should keep determiners in the subject
        # Example: "This book is a part of X" -> "What is this book a part of?"
        #          (keep "this" in the subject)
        preposition_before_qword = self._find_preposition_before_position(tokens_text, q_word_pos)
        is_be_noun_of = (
            aux_text.lower() in ['is', 'are', 'was', 'were', 'am'] and
            preposition_before_qword and preposition_before_qword.lower() == 'of'
        )

        # Extract complete subject text and its position
        # Keep determiners for "be + noun + of" constructions
        subject_text, subject_start_idx, subject_end_idx = self._extract_subject_text_and_indices(
            doc, subject, tokens_text, question_word, keep_determiner=is_be_noun_of
        )

        # Extract remaining tokens after removing auxiliary and subject
        rest_tokens = self._extract_remaining_tokens(
            tokens_without_q, aux_pos, subject_start_idx, subject_end_idx, q_word_pos
        )

        # Process preposition logic and generate result
        # Pass tokens_text and q_word_pos for category noun pattern detection
        return self._process_preposition_logic_with_auxiliary(
            question_word, aux_text, subject_text, rest_tokens, preposition_before_qword, original_doc, tokens_text, q_word_pos
        )

    def _handle_object_replacement_without_auxiliary(self, main_verb, subject, tokens_without_q: List[str], question_word: str, q_word_pos: int, tokens_text: List[str], original_doc, focus_entity: str) -> str:
        """Handle object replacement when no auxiliary verb is present"""
        # Check for special patterns first
        special_result = self._check_special_patterns(original_doc, focus_entity)
        if special_result:
            return clean_question(special_result, self.question_words)

        # Add appropriate auxiliary
        aux_word = self._determine_auxiliary(main_verb, subject)

        # Convert main verb to base form if needed
        tokens_without_q = self._convert_main_verb_to_base_form(
            tokens_without_q, main_verb, q_word_pos
        )

        # Handle preposition before question word
        preposition_before_qword, tokens_without_q = self._handle_preposition_removal(
            tokens_text, tokens_without_q, q_word_pos
        )

        # FIX: Check for "prep + category_noun + qword" pattern (e.g., "in taxon what")
        # If we have this pattern, we need to extract the category noun and format the question properly
        category_noun = None
        if preposition_before_qword and q_word_pos > 1:
            category_nouns = {'taxon', 'category', 'class', 'group', 'type', 'kind', 'genus', 'family', 'order'}
            potential_noun = tokens_text[q_word_pos - 1]
            if potential_noun.lower() in category_nouns:
                category_noun = potential_noun
                # Remove the category noun from tokens_without_q
                # It was at position q_word_pos - 1 in original, so in tokens_without_q it's at q_word_pos - 1
                noun_pos_in_tokens_without_q = q_word_pos - 1
                if noun_pos_in_tokens_without_q < len(tokens_without_q):
                    tokens_without_q = tokens_without_q[:noun_pos_in_tokens_without_q] + tokens_without_q[noun_pos_in_tokens_without_q + 1:]

                # Also remove the preposition (it's at q_word_pos - 2 originally, but after removing qword, it's at q_word_pos - 2)
                prep_pos_in_tokens_without_q = q_word_pos - 2
                if prep_pos_in_tokens_without_q < len(tokens_without_q) and prep_pos_in_tokens_without_q >= 0:
                    tokens_without_q = tokens_without_q[:prep_pos_in_tokens_without_q] + tokens_without_q[prep_pos_in_tokens_without_q + 1:]

        # Decide whether to keep the preposition
        should_keep_preposition = self._should_keep_preposition(
            preposition_before_qword, question_word, original_doc
        )

        # Generate final result
        if category_noun:
            # FIX: For "prep + category_noun + qword" pattern, format as "Prep qword noun aux subject verb?"
            # Example: "In what taxon is Gene X found?"
            result = f"{preposition_before_qword.capitalize()} {question_word} {category_noun} {aux_word} {join_tokens_with_punct(tokens_without_q)}"
        elif should_keep_preposition:
            result = f"{question_word} {aux_word} {join_tokens_with_punct(tokens_without_q)} {preposition_before_qword}"
        else:
            result = f"{question_word} {aux_word} {join_tokens_with_punct(tokens_without_q)}"

        return clean_question(result, self.question_words)

    def _check_special_patterns(self, original_doc, focus_entity: str) -> Optional[str]:
        """Check for special patterns that need template-based generation"""
        if self._is_track_gauge_pattern(original_doc, focus_entity):
            subject_text = self.entity_matcher.extract_complete_subject_from_original_doc(original_doc)
            return f"What is the track gauge of {subject_text}?"

        if self._is_hold_office_pattern(original_doc, focus_entity):
            subject_text = self.entity_matcher.extract_complete_subject_from_original_doc(original_doc)
            return f"What office does {subject_text} hold?"

        return None

    def _extract_remaining_tokens(self, tokens_without_q: List[str], aux_pos: int, subject_start_idx: int, subject_end_idx: int, q_word_pos: int) -> List[str]:
        """Extract tokens that remain after removing auxiliary and subject"""
        tokens_to_remove_indices = set()

        # Mark auxiliary for removal
        tokens_to_remove_indices.add(aux_pos)

        # Mark subject tokens for removal
        for i in range(subject_start_idx, subject_end_idx + 1):
            adjusted_idx = i if i < q_word_pos else i - 1
            tokens_to_remove_indices.add(adjusted_idx)

        # Build remaining tokens
        rest_tokens = []
        for i, token in enumerate(tokens_without_q):
            if i not in tokens_to_remove_indices:
                rest_tokens.append(token)

        return rest_tokens

    def _find_preposition_before_position(self, tokens_text: List[str], position: int) -> Optional[str]:
        """Find preposition before the given position"""
        for i in range(position - 1, -1, -1):
            if tokens_text[i].lower() in self.common_prepositions:
                return tokens_text[i]
        return None

    def _process_preposition_logic_with_auxiliary(self, question_word: str, aux_text: str, subject_text: str, rest_tokens: List[str], preposition_before_qword: Optional[str], original_doc=None, tokens_text: List[str] = None, q_word_pos: int = -1) -> str:
        """Process preposition logic for auxiliary verb cases

        Args:
            question_word: The question word
            aux_text: Auxiliary verb text
            subject_text: Subject text
            rest_tokens: Remaining tokens
            preposition_before_qword: Preposition before question word
            original_doc: Original spaCy doc (before entity replacement) for passive voice detection
            tokens_text: Original tokens before removing question word (for detecting category noun patterns)
            q_word_pos: Position of question word in original tokens
        """
        # FIX: Check for "prep + category_noun + qword" pattern (e.g., "in taxon what")
        category_noun = None
        if preposition_before_qword and tokens_text and q_word_pos > 1:
            category_nouns = {'taxon', 'category', 'class', 'group', 'type', 'kind', 'genus', 'family', 'order'}
            potential_noun = tokens_text[q_word_pos - 1]
            if potential_noun.lower() in category_nouns:
                category_noun = potential_noun
                # Remove the category noun from rest_tokens if present
                if category_noun in rest_tokens:
                    rest_tokens = [t for t in rest_tokens if t != category_noun]
                # Remove the preposition from rest_tokens if present
                if preposition_before_qword in rest_tokens:
                    rest_tokens = [t for t in rest_tokens if t != preposition_before_qword]

                # FIX: For "where" questions, don't use "In where taxon" format
                # Use simple "Where is X found?" instead
                # Only use "Prep + qword + noun" format for "what", "which", etc.
                if question_word.lower() in ['where', 'when']:
                    # For "where" and "when" questions, use simple format without category noun
                    # "Where is Gene X found?" (not "In where taxon is Gene X found?")
                    result = f"{question_word} {aux_text} {subject_text} {join_tokens_with_punct(rest_tokens)}"
                    return clean_question(result, self.question_words)
                else:
                    # For "what", "which" questions, use "Prep qword noun" format
                    # Example: "In what taxon is Gene X found?"
                    result = f"{preposition_before_qword.capitalize()} {question_word} {category_noun} {aux_text} {subject_text} {join_tokens_with_punct(rest_tokens)}"
                    return clean_question(result, self.question_words)

        if not preposition_before_qword:
            # No preposition case
            if len(rest_tokens) > self.MAX_SIMPLE_PREDICATE_LENGTH:
                result = f"{question_word} {aux_text} {subject_text} {join_tokens_with_punct(rest_tokens)}"
            else:
                result = f"{question_word} {aux_text} {join_tokens_with_punct(rest_tokens)} {subject_text}"
            return clean_question(result, self.question_words)

        # Preposition case
        q_word_lower = question_word.lower()
        prep_lower = preposition_before_qword.lower()

        # Check if this is a noun+preposition collocation BEFORE removing anything
        is_collocation = self._is_noun_prep_collocation(rest_tokens, preposition_before_qword)

        # Detect passive voice from original document
        is_passive_voice = False
        if original_doc is not None:
            is_passive_voice = any(token.dep_ == 'auxpass' for token in original_doc)

        # Find the preposition position
        prep_index = -1
        if preposition_before_qword in rest_tokens:
            prep_index = rest_tokens.index(preposition_before_qword)

        # Decide whether to keep preposition
        should_keep_prep = self._should_keep_preposition_with_auxiliary(
            q_word_lower, prep_lower, rest_tokens, is_collocation, is_passive_voice
        )

        # Handle collocation case (e.g., "part of")
        if should_keep_prep and prep_index != -1:
            # For collocations like "part of", keep the preposition immediately after the noun
            # Format: "What is subject part of?" (NOT "part of the collection")
            # The preposition should stay with its noun, not move to the end

            # FIX: Don't truncate if the preposition is inside a parenthetical entity description
            # Check if there's an unclosed opening parenthesis before the preposition
            open_paren_count = 0
            for i in range(prep_index):
                if rest_tokens[i] == '(':
                    open_paren_count += 1
                elif rest_tokens[i] == ')':
                    open_paren_count -= 1

            if open_paren_count > 0:
                # We're inside a parenthetical description - don't truncate
                # The preposition is part of an entity description like "(sculpture by Hans von Worms at Liebieghaus...)"
                rest_tokens_cleaned = rest_tokens
            else:
                # Normal case - remove tokens AFTER the preposition (like "the collection" in "part of the collection")
                # Keep tokens up to and including the preposition
                rest_tokens_cleaned = rest_tokens[:prep_index + 1]

            # Format: "What is subject part of?"
            result = f"{question_word} {aux_text} {subject_text} {join_tokens_with_punct(rest_tokens_cleaned)}"
            return clean_question(result, self.question_words)

        # Non-collocation case - need to remove preposition
        if prep_index != -1:
            rest_tokens.pop(prep_index)
            # Remove article after preposition if needed
            if (prep_index < len(rest_tokens) and
                rest_tokens[prep_index].lower() in ['the', 'a', 'an'] and
                q_word_lower != 'where'):
                rest_tokens.pop(prep_index)

        # Format result based on question word and preposition type
        # CRITICAL FIX: For "be + noun + of" constructions (e.g., "be a student of", "be a member of"),
        # the subject should come AFTER the noun, not after the preposition
        # Correct: "Who was Nick Brown a student of?" (subject after "student")
        # Wrong: "Who was a student of Nick Brown?" (subject after "of")

        # Check if this is a "be + noun + of" construction
        # (main verb is be, with a noun + preposition "of" structure)
        is_be_noun_of_construction = (
            aux_text.lower() in ['is', 'are', 'was', 'were', 'am'] and  # be-verb
            prep_lower == 'of' and  # preposition is "of"
            len(rest_tokens) > 0 and  # has remaining tokens (noun, etc.)
            # rest_tokens should contain a noun (e.g., "student", "member", "part", etc.)
            any(token.lower() in ['student', 'member', 'part', 'candidate', 'collection', 'piece', 'type', 'kind', 'sort']
                for token in rest_tokens)
        )

        # Check if preposition should be added at the end for non-be verbs
        # For "who/what/whom" questions with prepositions like "from", "to", "of", "with",
        # the preposition should be moved to the end of the question
        # Example: "John received a gift from Mary" -> "Who did John receive a gift from?"
        should_add_prep_at_end = (
            q_word_lower in ['who', 'what', 'whom', 'which'] and
            prep_lower in ['from', 'to', 'with', 'of', 'by', 'about', 'for'] and
            prep_index != -1  # Preposition was found and removed
        )

        if is_be_noun_of_construction:
            # For "be + noun + of" constructions, format: {q} {aux} {subj} {noun} {prep}
            # Example: "Who was Nick Brown a student of?"
            result = f"{question_word} {aux_text} {subject_text} {join_tokens_with_punct(rest_tokens)} {preposition_before_qword}"
        elif should_add_prep_at_end:
            # For non-be verbs with prepositions, add preposition at end
            # Example: "Who did John receive a gift from?"
            result = f"{question_word} {aux_text} {subject_text} {join_tokens_with_punct(rest_tokens)} {preposition_before_qword}"
        else:
            result = f"{question_word} {aux_text} {subject_text} {join_tokens_with_punct(rest_tokens)}"

        return clean_question(result, self.question_words)

    def _should_keep_preposition_with_auxiliary(self, q_word_lower: str, prep_lower: str, rest_tokens: List[str] = None, is_collocation: bool = False, is_passive_voice: bool = False) -> bool:
        """Determine whether to keep preposition in auxiliary verb cases

        Args:
            q_word_lower: Question word in lowercase
            prep_lower: Preposition in lowercase
            rest_tokens: Remaining tokens (optional, used for collocation detection)
            is_collocation: Whether this is a noun+prep collocation (precomputed)
            is_passive_voice: Whether the original sentence is in passive voice

        Returns:
            True if preposition should be kept, False otherwise
        """
        # If this is a recognized noun+preposition collocation, always keep it
        if is_collocation:
            return True

        # CRITICAL FIX for Error #720: Passive voice 'by' should always be preserved
        # In passive voice (e.g., "X is studied by Y"), the "by" marks the agent
        # and must be kept in questions like "What is X studied by?"
        if is_passive_voice and prep_lower == 'by':
            return True

        # Semantic prepositions ('as', 'for', 'about') should be kept for all question types
        # This fixes the issue where "as" was being removed in "has as their home venue"
        if prep_lower in self.semantic_prepositions:
            return True

        # For 'where' questions, keep location prepositions
        if q_word_lower == 'where' and prep_lower in self.location_prepositions:
            return True

        # For 'when' questions, we generally don't keep prepositions
        if q_word_lower == 'when':
            return False

        # For other question types, keep semantic prepositions
        if q_word_lower in ['what', 'who', 'which', 'whom']:
            return prep_lower in self.semantic_prepositions

        return False

    def _is_noun_prep_collocation(self, tokens: List[str], preposition: str) -> bool:
        """Check if the preposition is part of a noun+preposition collocation

        Args:
            tokens: List of tokens (should include the preposition)
            preposition: The preposition to check

        Returns:
            True if this is a recognized collocation, False otherwise
        """
        if not tokens or not preposition:
            return False

        prep_lower = preposition.lower()

        # Check if this preposition has registered collocations
        if prep_lower not in self.noun_preposition_collocations:
            return False

        # Find the position of the preposition
        try:
            prep_index = tokens.index(preposition)
        except ValueError:
            return False

        # Look for nouns before the preposition (within a window of 3 tokens)
        window_size = 3
        start_idx = max(0, prep_index - window_size)

        for i in range(start_idx, prep_index):
            token = tokens[i].lower()
            # Remove articles and check if it's a noun in our collocation set
            if token in ['a', 'an', 'the']:
                continue
            # Check if this noun forms a collocation with the preposition
            if token in self.noun_preposition_collocations[prep_lower]:
                return True

        return False

    def _convert_main_verb_to_base_form(self, tokens_without_q: List[str], main_verb, q_word_pos: int) -> List[str]:
        """Convert main verb to base form if needed"""
        if main_verb:
            # For have verbs used as main verbs (not auxiliaries), convert to base form
            # Example: "has" -> "have" in "What does she have?"
            if main_verb.lemma_.lower() not in self.be_verbs | self.have_verbs:
                main_verb_pos = main_verb.i
                if main_verb_pos > q_word_pos:
                    main_verb_pos -= 1

                if main_verb_pos < len(tokens_without_q):
                    main_verb_base = self._get_base_form(main_verb)
                    tokens_without_q[main_verb_pos] = main_verb_base
            elif main_verb.lemma_.lower() in self.have_verbs:
                # Handle have verbs that are main verbs (not auxiliaries)
                # Check if this "have" is being used as a main verb
                if (main_verb.dep_ == 'ROOT' and
                    any(child.dep_ in ['dobj', 'attr', 'pobj', 'prep'] for child in main_verb.children)):
                    # This "have" is a main verb, convert to base form
                    main_verb_pos = main_verb.i
                    if main_verb_pos > q_word_pos:
                        main_verb_pos -= 1

                    if main_verb_pos < len(tokens_without_q):
                        main_verb_base = self._get_base_form(main_verb)
                        tokens_without_q[main_verb_pos] = main_verb_base

        return tokens_without_q

    def _handle_preposition_removal(self, tokens_text: List[str], tokens_without_q: List[str], q_word_pos: int) -> Tuple[Optional[str], List[str]]:
        """Handle preposition removal from tokens

        FIX: Also detect "prep + noun + qword" patterns (e.g., "in taxon what")
        where the preposition is not directly before the question word,
        but before a noun that modifies the question word.
        """
        preposition_before_qword = None

        # Case 1: Preposition directly before question word (original logic)
        # Pattern: "... prep what"
        if q_word_pos > 0 and tokens_text[q_word_pos - 1].lower() in self.common_prepositions:
            preposition_before_qword = tokens_text[q_word_pos - 1]
            # Remove preposition from tokens_without_q
            prep_pos_in_tokens_without_q = q_word_pos - 1
            if prep_pos_in_tokens_without_q < len(tokens_without_q):
                tokens_without_q = tokens_without_q[:prep_pos_in_tokens_without_q] + tokens_without_q[prep_pos_in_tokens_without_q + 1:]

        # Case 2: FIX for "prep + noun + qword" pattern (e.g., "in taxon what")
        # Pattern: "... prep noun what"
        # This handles cases like "found in taxon Y" -> "found in taxon what"
        elif q_word_pos > 1 and tokens_text[q_word_pos - 2].lower() in self.common_prepositions:
            # Check if token at q_word_pos - 1 is a noun (category word)
            # Common category nouns: taxon, category, class, group, type, kind, etc.
            category_nouns = {'taxon', 'category', 'class', 'group', 'type', 'kind', 'genus', 'family', 'order'}
            potential_noun = tokens_text[q_word_pos - 1]

            if potential_noun.lower() in category_nouns:
                # This is "prep + category_noun + qword" pattern
                preposition_before_qword = tokens_text[q_word_pos - 2]
                # Don't remove the preposition or noun yet - they form a unit with the question word
                # The question should be "In what taxon..." not "What taxon..."
                # We'll handle this in the question construction logic

        return preposition_before_qword, tokens_without_q

    def _should_keep_preposition(self, preposition_before_qword: Optional[str], question_word: str, original_doc) -> bool:
        """Determine whether to keep the preposition"""
        if not preposition_before_qword:
            return False

        q_word_lower = question_word.lower()
        prep_lower = preposition_before_qword.lower()

        # Check for passive voice
        is_passive_voice = any(token.dep_ == 'auxpass' for token in original_doc)

        # FIX: Check for noun+preposition collocations FIRST, before any blanket removal
        # This ensures essential collocations like "factor of", "found in taxon" are preserved
        # even for question words like "when", "where", "why"
        tokens_text = [token.text for token in original_doc]
        is_collocation = self._is_noun_prep_collocation(tokens_text, preposition_before_qword)
        if is_collocation:
            return True

        if is_passive_voice and prep_lower in self.location_prepositions:
            return True
        elif q_word_lower in ['where', 'when', 'why']:
            # Only remove preposition if it's NOT part of a collocation (already checked above)
            return False
        elif q_word_lower in ['what', 'who', 'which', 'whom']:
            # Keep semantic prepositions ('as', 'for', 'about') and 'of'
            # 'of' is commonly used in possessive/attributive constructions that need to be preserved
            # Examples: "books of X" -> "Who does John read books of?"
            #           "student of X" -> "Who is John a student of?"
            # Also keep directional prepositions ('from', 'to', 'with', 'by', 'in') for grammatical correctness
            # Examples: "received from X" -> "Who did John receive a gift from?"
            #           "sent to X" -> "Who did Mary send a message to?"
            #           "competes in X" -> "What does Y compete in?" (FIX: added 'in')
            directional_prepositions = {'from', 'to', 'with', 'by', 'in'}
            return prep_lower in self.semantic_prepositions or prep_lower == 'of' or prep_lower in directional_prepositions

        return False

    def _construct_question_from_statement(self, main_verb, subject, auxiliary, question_word: str, doc) -> str:
        """Construct question from statement without question word"""
        tokens = [token.text for token in doc]

        if auxiliary:
            # Move auxiliary after question word
            aux_pos = auxiliary.i
            aux_text = auxiliary.text
            tokens_without_aux = tokens[:aux_pos] + tokens[aux_pos + 1:]
            result = f"{question_word} {aux_text} {join_tokens_with_punct(tokens_without_aux)}"
        else:
            # Add appropriate auxiliary
            aux_word = self._determine_auxiliary(main_verb, subject)

            # Convert main verb to base form if needed
            if main_verb and main_verb.lemma_.lower() not in self.be_verbs | self.have_verbs:
                main_verb_base = self._get_base_form(main_verb)
                tokens[main_verb.i] = main_verb_base

            result = f"{question_word} {aux_word} {join_tokens_with_punct(tokens)}"

        # Clean up hanging articles and prepositions
        return clean_question(result, self.question_words)

    def _construct_general_question(self, statement: str, question_word: str, doc) -> str:
        """Construct general question without entity replacement

        For cases where no focus entity is specified, generate a general question
        like "What does John read?" instead of "What does John read books?"
        """
        main_verb = find_main_verb(doc)
        subject = find_subject(doc)
        auxiliary = self._find_auxiliary(doc)

        if auxiliary:
            # Move auxiliary after question word
            tokens = [token.text for token in doc]
            aux_pos = auxiliary.i
            aux_text = auxiliary.text
            tokens_without_aux = tokens[:aux_pos] + tokens[aux_pos + 1:]
            result = f"{question_word} {aux_text} {join_tokens_with_punct(tokens_without_aux)}"
        else:
            # Add appropriate auxiliary
            aux_word = self._determine_auxiliary(main_verb, subject)

            # Convert main verb to base form if needed
            tokens = [token.text for token in doc]
            if main_verb and main_verb.lemma_.lower() not in self.be_verbs | self.have_verbs:
                main_verb_base = self._get_base_form(main_verb)
                tokens[main_verb.i] = main_verb_base

            # For general questions, we typically ask about the object, not the whole statement
            # So we should remove the object and leave the question open-ended
            # "John reads books" -> "What does John read?" (not "What does John read books?")

            # Find the object position
            object_pos = -1
            for i, token in enumerate(doc):
                if token.dep_ in ['dobj', 'attr', 'pobj']:
                    object_pos = i
                    break

            # If we found an object, remove it from the tokens
            if object_pos != -1:
                tokens = tokens[:object_pos]

            result = f"{question_word} {aux_word} {join_tokens_with_punct(tokens)}"

        return clean_question(result, self.question_words)

    def _is_track_gauge_pattern(self, original_doc, focus_entity: str) -> bool:
        """
        Detect if the sentence follows the "X has a track gauge of Y" pattern

        Args:
            original_doc: spaCy doc of the original statement
            focus_entity: The entity being replaced

        Returns:
            True if this is a track gauge pattern
        """
        if not original_doc:
            return False

        # Check for the specific pattern: "has a track gauge of"
        tokens_text = [token.text.lower() for token in original_doc]

        # Look for the pattern: [has, a, track, gauge, of]
        pattern = ['has', 'a', 'track', 'gauge', 'of']

        # Check if all pattern words appear in order
        pattern_found = False
        for i in range(len(tokens_text) - len(pattern) + 1):
            if tokens_text[i:i+len(pattern)] == pattern:
                pattern_found = True
                break

        if not pattern_found:
            return False

        # Additional check: the focus entity should be the object of "of"
        # In "X has a track gauge of Y", Y is the focus entity
        # So if focus_entity appears after "of", it's likely this pattern
        full_text = original_doc.text.lower()
        of_pos = full_text.find(' of ')
        if of_pos != -1:
            entity_pos = full_text.find(focus_entity.lower())
            if entity_pos > of_pos:
                return True

        return False

    def _is_hold_office_pattern(self, original_doc, focus_entity: str) -> bool:
        """
        Detect if the sentence follows the "X holds the office of Y" pattern

        Args:
            original_doc: spaCy doc of the original statement
            focus_entity: The entity being replaced

        Returns:
            True if this is a hold office pattern
        """
        if not original_doc:
            return False

        # Check for the specific patterns: "holds the office" or "hold the office"
        tokens_text = [token.text.lower() for token in original_doc]

        # Look for patterns like: [hold/holds, the, office] or [hold/holds, office]
        hold_patterns = [
            ['holds', 'the', 'office'],
            ['hold', 'the', 'office'],
            ['holds', 'office'],
            ['hold', 'office']
        ]

        for pattern in hold_patterns:
            pattern_found = True
            for i in range(len(tokens_text) - len(pattern) + 1):
                if tokens_text[i:i+len(pattern)] == pattern:
                    # Found the pattern, check if focus entity is the object
                    # The focus entity should appear after "office"
                    # In "X holds the office of Y", Y is the focus entity
                    entity_tokens = focus_entity.lower().split()
                    office_pos = i + len(pattern) - 1  # Position of "office"

                    # Check if focus entity appears after the office pattern
                    for j in range(office_pos + 1, len(tokens_text)):
                        if tokens_text[j:j+len(entity_tokens)] == entity_tokens:
                            return True
                    return False

        return False

    def _find_auxiliary(self, doc):
        """Find auxiliary verb if present

        FIX for kg_rule_10: Special handling for "has X of" pattern.
        When main verb is "has" followed by a nominal phrase and "of",
        treat "has" as an auxiliary verb for question formation.
        Example: "X has the next higher rank of Y" -> "What is X the next higher rank of?"
        NOT: "What does X have the next higher rank of?"
        """
        for token in doc:
            if token.pos_ == 'AUX':
                return token
            # For have verbs, only treat as auxiliary if they're not the main verb
            elif (token.lemma_.lower() in self.have_verbs and
                  token.dep_ != 'ROOT' and
                  not any(child.dep_ in ['dobj', 'attr', 'pobj', 'prep'] for child in token.children)):
                return token

        # FIX: Special case for "has X of" pattern
        # Check if main verb is "has/have" followed by nominal + "of"
        # In this case, treat "has" as auxiliary for question formation
        for token in doc:
            if token.lemma_.lower() in self.have_verbs and token.dep_ == 'ROOT':
                # Check if there's a prepositional phrase with "of" in the sentence
                has_of_prep = False
                for child in token.children:
                    if child.dep_ == 'prep' and child.lemma_.lower() == 'of':
                        has_of_prep = True
                        break
                    # Also check deeper: "has the X of" where "X" has "of" as child
                    for grandchild in child.children:
                        if grandchild.dep_ == 'prep' and grandchild.lemma_.lower() == 'of':
                            has_of_prep = True
                            break

                if has_of_prep:
                    # This is "has X of" pattern - treat "has" as auxiliary
                    return token

        return None

    def _determine_auxiliary(self, main_verb, subject) -> str:
        """Determine appropriate auxiliary verb"""
        if not main_verb:
            return 'do'

        verb_lemma = main_verb.lemma_.lower()
        verb_tag = main_verb.tag_

        # Be verbs are their own auxiliary
        if verb_lemma in self.be_verbs:
            return main_verb.text.lower()

        # Have verbs in perfect tenses - only treat as auxiliary if they're actually auxiliaries
        # For main verb "have" (like "has as their home venue"), use do/does
        if verb_lemma in self.have_verbs:
            # Check if this "have" is being used as a main verb or auxiliary
            # If it's the ROOT verb and has objects/complements, it's likely a main verb
            if (main_verb.dep_ == 'ROOT' and
                any(child.dep_ in ['dobj', 'attr', 'pobj', 'prep'] for child in main_verb.children)):
                # This "have" is a main verb, use do/does
                if verb_tag in ['VBD']:  # Past tense
                    return 'did'
                elif verb_tag in ['VBZ']:  # Third person singular present
                    return 'does'
                else:
                    return 'do'
            else:
                # This "have" is likely an auxiliary, keep it
                return main_verb.text.lower()

        # Modal verbs
        if verb_lemma in self.modal_verbs:
            return main_verb.text.lower()

        # Regular verbs need do/does/did
        if verb_tag in ['VBD']:  # Past tense
            return 'did'
        elif verb_tag in ['VBZ']:  # Third person singular present
            return 'does'
        else:  # Base form, present tense - consider subject for agreement
            # For plural subjects, use 'do'
            if subject and (subject.tag_ in ['NNS', 'NNPS'] or subject.text.lower() in ['they', 'we', 'you']):
                return 'do'
            else:
                return 'do'  # Default to 'do' for questions

    def _get_base_form(self, verb_token) -> str:
        """Get base form of verb"""
        if not verb_token:
            return ""
        return verb_token.lemma_

    def _fallback_transformation(self, statement: str, question_word: str, focus_entity: str = None) -> str:
        """Fallback transformation when spaCy is not available"""
        if focus_entity:
            # Simple replacement
            pattern = r'\b' + re.escape(focus_entity) + r'\b'
            result = re.sub(pattern, question_word, statement, flags=re.IGNORECASE)
            if not result.endswith('?'):
                result += '?'
            return result
        else:
            return f"{question_word} {statement}?"

    # Convenience methods
    def transform_to_what(self, statement: str, focus_entity: str = None) -> str:
        """Transform to what-question"""
        return self.transform(statement, "what", focus_entity)

    def transform_to_who(self, statement: str, focus_entity: str = None) -> str:
        """Transform to who-question"""
        return self.transform(statement, "who", focus_entity)

    def transform_to_where(self, statement: str, focus_entity: str = None) -> str:
        """Transform to where-question"""
        return self.transform(statement, "where", focus_entity)

    def transform_to_when(self, statement: str, focus_entity: str = None) -> str:
        """Transform to when-question"""
        return self.transform(statement, "when", focus_entity)

    def transform_to_why(self, statement: str, focus_entity: str = None) -> str:
        """Transform to why-question"""
        return self.transform(statement, "why", focus_entity)

    def transform_to_how(self, statement: str, focus_entity: str = None) -> str:
        """Transform to how-question"""
        return self.transform(statement, "how", focus_entity)

    def analyze_statement(self, statement: str) -> Dict:
        """Analyze statement structure for debugging"""
        if not self.nlp:
            return {"error": "spaCy not available"}

        doc = self.nlp(statement)

        analysis = {
            "tokens": [(token.text, token.pos_, token.tag_, token.dep_) for token in doc],
            "entities": [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents],
            "noun_chunks": [(chunk.text, chunk.start_char, chunk.end_char) for chunk in doc.noun_chunks],
            "main_verb": None,
            "subject": None,
            "auxiliary": None
        }

        main_verb = find_main_verb(doc)
        if main_verb:
            analysis["main_verb"] = (main_verb.text, main_verb.lemma_, main_verb.tag_)

        subject = find_subject(doc)
        if subject:
            analysis["subject"] = (subject.text, subject.pos_, subject.tag_)

        auxiliary = self._find_auxiliary(doc)
        if auxiliary:
            analysis["auxiliary"] = (auxiliary.text, auxiliary.lemma_, auxiliary.tag_)

        return analysis


# Alias for backward compatibility
WHTransformer = EnglishWhTransformer
