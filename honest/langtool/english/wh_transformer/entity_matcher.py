#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity Matcher for WH-Question Transformation

This module handles entity matching and replacement logic for transforming
declarative sentences into WH-questions. It uses spaCy's NLP features for
advanced entity recognition and matching.
"""

import re
from typing import List, Tuple, Optional


class EntityMatcher:
    """Handles entity matching and subject detection for WH-question transformation"""

    def __init__(self, nlp):
        """Initialize the entity matcher

        Args:
            nlp: spaCy language model
        """
        self.nlp = nlp

        # Entity matching priority order (from highest to lowest priority)
        # Used in _spacy_entity_matching to select the best match
        self.entity_match_priority_order = [
            'possessive_entity',  # Entities in possessive context (highest priority for 'who' questions)
            'direct_text_match',  # Direct text match
            'date_exact_match',   # Date entity exact match (very high priority)
            'date_ner_match',     # Date entity NER match (high priority)
            'ner_PERSON',         # Named entities - person
            'ner_ORG',            # Named entities - organization
            'ner_GPE',            # Named entities - geopolitical entity
            'compound_proper_noun',  # Compound proper nouns
            'noun',               # Regular nouns
            'noun_phrase',        # Noun phrases
            'proper_noun'         # Individual proper nouns (lowest priority)
        ]

    def spacy_entity_matching(self, target_entity: str, doc) -> Tuple[Optional[str], int, int, str]:
        """
        Advanced entity matching using spaCy features

        Returns:
            (matched_text, start_char, end_char, match_type) or (None, -1, -1, 'no_match')
        """
        candidates = []

        # 0. Direct text matching for complex entities that spaCy might miss
        # This handles cases like "Szajol–Lőkösháza railway line" and date entities like "February 11, 2018"
        # FIX: Find ALL occurrences, not just the first one (to avoid matching substrings)
        target_lower = target_entity.lower()
        doc_text_lower = doc.text.lower()

        if target_lower in doc_text_lower:
            # Find all occurrences of the target entity
            all_positions = []
            pos = 0
            while pos < len(doc_text_lower):
                pos = doc_text_lower.find(target_lower, pos)
                if pos == -1:
                    break
                all_positions.append(pos)
                pos += 1  # Move forward to find next occurrence

            # For each occurrence, check if it's a standalone match (word boundary check)
            # Priority: prefer matches that are NOT substrings of other words
            standalone_positions = []
            for start_pos in all_positions:
                end_pos = start_pos + len(target_lower)
                # Check word boundaries
                is_standalone = True
                # Check left boundary
                if start_pos > 0:
                    left_char = doc_text_lower[start_pos - 1]
                    # If left char is alphanumeric or hyphen, it's part of another word
                    if left_char.isalnum() or left_char == '-':
                        is_standalone = False
                # Check right boundary
                if end_pos < len(doc_text_lower):
                    right_char = doc_text_lower[end_pos]
                    if right_char.isalnum() or right_char == '-':
                        is_standalone = False

                if is_standalone:
                    standalone_positions.append((start_pos, end_pos))

            # Prefer standalone matches. If multiple standalone, choose the LAST one (rightmost)
            # because in "X has relation Y" patterns, Y (the answer) is typically at the end
            if standalone_positions:
                start_char, end_char = standalone_positions[-1]  # Use rightmost standalone match
                candidates.append((target_entity, start_char, end_char, 'direct_text_match_standalone'))
            elif all_positions:
                # Fallback: if no standalone match, use rightmost occurrence
                start_char = all_positions[-1]
                end_char = start_char + len(target_lower)
                candidates.append((target_entity, start_char, end_char, 'direct_text_match_rightmost'))

        # 0.1. Enhanced date entity matching
        # Specifically handle date patterns like "February 11, 2018" which may be partially matched
        if self._is_date_like_entity(target_entity):
            date_matches = self._find_date_entity_matches(target_entity, doc)
            candidates.extend(date_matches)

        # 1. Named Entity Recognition
        for ent in doc.ents:
            if self._entity_matches(ent.text, target_entity):
                candidates.append((ent.text, ent.start_char, ent.end_char, f'ner_{ent.label_}'))

        # 2. Noun Phrases
        for noun_phrase in doc.noun_chunks:
            if self._entity_matches(noun_phrase.text, target_entity):
                candidates.append((
                    noun_phrase.text,
                    noun_phrase.start_char,
                    noun_phrase.end_char,
                    'noun_phrase'
                ))

        # 3. Proper Nouns (individual tokens)
        for token in doc:
            if token.pos_ == 'PROPN' and self._entity_matches(token.text, target_entity):
                candidates.append((token.text, token.idx, token.idx + len(token.text), 'proper_noun'))

        # 4. Regular nouns (individual tokens) - for cases like "cat" in "The cat"
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and self._entity_matches(token.text, target_entity):
                candidates.append((token.text, token.idx, token.idx + len(token.text), 'noun'))

        # 5. Compound proper nouns (consecutive PROPN tokens)
        compound_nouns = self._extract_compound_proper_nouns(doc)
        for compound_text, start_char, end_char in compound_nouns:
            if self._entity_matches(compound_text, target_entity):
                candidates.append((compound_text, start_char, end_char, 'compound_proper_noun'))

        # 6. Special handling for possessive context
        possessive_candidates = self._find_possessive_entities(doc, target_entity)
        candidates.extend(possessive_candidates)

        if not candidates:
            return None, -1, -1, 'no_match'

        # First, check for exact matches across all candidates
        # This prevents choosing "Beirut Shahnameh" when we want "Beirut"
        exact_matches = [c for c in candidates if c[0].lower().strip() == target_entity.lower().strip()]
        if exact_matches:
            # Among exact matches, prefer higher priority types
            for priority in self.entity_match_priority_order:
                for candidate in exact_matches:
                    if candidate[3].startswith(priority.split('_')[0]):
                        return candidate
            # If no priority match, return the first exact match
            return exact_matches[0]

        # No exact matches found, use priority-based matching
        # Prioritize matches: direct text > NER > compound proper nouns > individual nouns > noun phrases > proper nouns
        for priority in self.entity_match_priority_order:
            matching_candidates = [c for c in candidates if c[3].startswith(priority.split('_')[0])]
            if matching_candidates:
                # Return the longest match for better precision
                return max(matching_candidates, key=lambda x: len(x[0]))

        # Return the longest match if no priority match found
        return max(candidates, key=lambda x: len(x[0]))

    def _entity_matches(self, candidate: str, target: str) -> bool:
        """Check if candidate entity matches target using various methods

        Returns True only if candidate is a reasonable match for target.
        Prioritizes exact matches and avoids partial substring matches that could
        leave trailing words unmatched.
        """
        candidate_clean = candidate.strip().lower()
        target_clean = target.strip().lower()

        # Exact match - highest priority
        if candidate_clean == target_clean:
            return True

        # Handle hyphenated names and complex entities
        # For cases like "Szajol–Lőkösháza railway line" vs "Lőkösháza railway line"
        candidate_normalized = candidate_clean.replace('–', '-').replace('_', ' ')
        target_normalized = target_clean.replace('–', '-').replace('_', ' ')

        candidate_words = candidate_normalized.split()
        target_words = target_normalized.split()

        # For single target word, check if it's a meaningful part of candidate
        if len(target_words) == 1:
            # Check if target word appears as a complete word in candidate
            if target_normalized in candidate_words:
                return True
            # Check if candidate is a single word and contains target
            if len(candidate_words) == 1 and target_normalized in candidate_normalized:
                return True

        # For multi-word targets, use more flexible matching
        if len(target_words) > 1:
            # Check if candidate contains all target words in order
            # This handles cases like "Lőkösháza railway line" matching "Szajol–Lőkösháza railway line"
            if self._contains_ordered_words(candidate_normalized, target_normalized):
                return True

            # Check if candidate is a meaningful subset of target
            # For example, "railway line" should match "Szajol–Lőkösháza railway line"
            if len(candidate_words) < len(target_words):
                # Check if candidate appears as consecutive words in target
                if candidate_normalized in target_normalized:
                    # Only accept if it's a meaningful part (not just random substring)
                    candidate_start = target_normalized.find(candidate_normalized)
                    candidate_end = candidate_start + len(candidate_normalized)
                    # Check if it's at word boundaries
                    if (candidate_start == 0 or target_normalized[candidate_start-1] == ' ') and \
                       (candidate_end == len(target_normalized) or target_normalized[candidate_end] == ' '):
                        return True

        return False

    def _contains_ordered_words(self, text: str, pattern: str) -> bool:
        """Check if text contains all words from pattern in the same order"""
        text_words = text.split()
        pattern_words = pattern.split()

        if len(pattern_words) > len(text_words):
            return False

        # Find the starting position where pattern might match
        for i in range(len(text_words) - len(pattern_words) + 1):
            if text_words[i:i+len(pattern_words)] == pattern_words:
                return True

        return False

    def extract_complete_subject_from_original_doc(self, original_doc) -> str:
        """
        Extract complete subject phrase from original document (before entity replacement)

        This helper method is used for special pattern matching (like track gauge)
        where we need to extract the subject from the original statement.

        Args:
            original_doc: spaCy Doc of the original statement

        Returns:
            Complete subject text as string
        """
        from .utils import find_subject, join_tokens_with_punct

        original_subject = find_subject(original_doc)
        if not original_subject:
            return ""

        # Extract complete noun phrase containing the subject from original doc
        subject_chunk = None
        for chunk in original_doc.noun_chunks:
            if chunk.start <= original_subject.i <= chunk.end:
                subject_chunk = chunk
                break

        if subject_chunk:
            subject_text = subject_chunk.text
        else:
            # Fallback: use the subject token and its compound parts
            subject_start_idx = original_subject.i
            subject_end_idx = original_subject.i

            # Look for compound parts
            for token in original_doc:
                if token.i < original_subject.i and token.head.i == original_subject.i and token.dep_ in ['compound']:
                    subject_start_idx = min(subject_start_idx, token.i)
                if token.i > original_subject.i and (token.head.i == original_subject.i and token.dep_ in ['compound', 'flat']) or (token.i == original_subject.i + 1 and token.pos_ in ['PROPN', 'NOUN']):
                    subject_end_idx = max(subject_end_idx, token.i)

            # Special handling for hyphenated compound nouns
            # Check if we have a hyphenated pattern like "Szajol–Lőkösháza railway line"
            # Look for hyphenated patterns that should be treated as single entities
            if subject_start_idx > 0:
                # Check for hyphen before subject
                prev_token = original_doc[subject_start_idx - 1]
                if prev_token.text in ['–', '-'] and subject_start_idx - 1 > 0:
                    # Include the hyphen and the word before it
                    subject_start_idx = subject_start_idx - 2

            subject_tokens = [token.text for token in original_doc[subject_start_idx:subject_end_idx + 1]]
            subject_text = join_tokens_with_punct(subject_tokens)

        # Special case: if we have multiple subjects (like "Szajol" and "line" both as nsubj)
        # Try to find the complete phrase by looking for the earliest subject
        all_subjects = [token for token in original_doc if token.dep_ in ['nsubj', 'nsubjpass']]
        if len(all_subjects) > 1:
            # Find the earliest subject
            earliest_subject = min(all_subjects, key=lambda x: x.i)
            # Find the latest subject
            latest_subject = max(all_subjects, key=lambda x: x.i)

            # If subjects are separated by other tokens, try to extract the complete phrase
            if latest_subject.i - earliest_subject.i > 1:
                # Extract tokens between earliest and latest subject
                complete_subject_tokens = [token.text for token in original_doc[earliest_subject.i:latest_subject.i + 1]]
                complete_subject_text = join_tokens_with_punct(complete_subject_tokens)

                # Use the complete phrase if it makes sense
                if len(complete_subject_text.split()) > len(subject_text.split()):
                    subject_text = complete_subject_text

        # Handle articles properly - ensure "the" is lowercase
        if subject_text.lower().startswith('the '):
            subject_text = 'the ' + subject_text[4:].lstrip()

        return subject_text

    def _find_date_entity_matches(self, target_entity: str, doc) -> List[Tuple[str, int, int, str]]:
        """Find matches for date-like entities with enhanced precision"""
        matches = []

        # Strategy 1: Look for exact date matches using character positions
        doc_text = doc.text
        target_lower = target_entity.lower()
        doc_text_lower = doc_text.lower()

        # Find all exact occurrences
        start = 0
        while True:
            pos = doc_text_lower.find(target_lower, start)
            if pos == -1:
                break

            end_pos = pos + len(target_entity)

            # Verify it's a word boundary match to avoid partial matches
            if self._is_word_boundary_match(doc_text, pos, end_pos):
                actual_text = doc_text[pos:end_pos]
                matches.append((actual_text, pos, end_pos, 'date_exact_match'))

            start = pos + 1

        # Strategy 2: Use spaCy's DATE entities as fallback
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                if self._date_entities_equivalent(ent.text, target_entity):
                    matches.append((ent.text, ent.start_char, ent.end_char, 'date_ner_match'))

        return matches

    def _is_word_boundary_match(self, text: str, start: int, end: int) -> bool:
        """Check if the match is at word boundaries"""
        # Check character before start
        if start > 0:
            prev_char = text[start - 1]
            if prev_char.isalnum():
                return False

        # Check character after end
        if end < len(text):
            next_char = text[end]
            if next_char.isalnum():
                return False

        return True

    def _date_entities_equivalent(self, candidate: str, target: str) -> bool:
        """Check if two date entities are equivalent"""
        # Simple equivalence check for now
        # Could be enhanced with date parsing for more complex comparisons
        return candidate.lower().strip() == target.lower().strip()

    def _is_date_like_entity(self, entity: str) -> bool:
        """Check if an entity looks like a date"""
        import re

        # Pattern for dates like "February 11, 2018", "April 21, 2018", etc.
        date_patterns = [
            r'\b\w+\s+\d{1,2},\s+\d{4}\b',  # Month Day, Year (e.g., "February 11, 2018")
            r'\b\d{1,2}\s+\w+\s+\d{4}\b',   # Day Month Year (e.g., "11 February 2018")
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',   # YYYY-MM-DD format
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',   # MM/DD/YYYY format
        ]

        for pattern in date_patterns:
            if re.search(pattern, entity):
                return True

        # Also check for month names
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']

        entity_lower = entity.lower()
        has_month = any(month in entity_lower for month in months)
        has_year = re.search(r'\b\d{4}\b', entity)
        has_day = re.search(r'\b\d{1,2}\b', entity)

        # If it has month + year + day, it's likely a date
        return has_month and has_year and has_day

    def _extract_compound_proper_nouns(self, doc) -> List[Tuple[str, int, int]]:
        """Extract compound proper nouns (consecutive PROPN tokens)"""
        compounds = []
        current_compound = []

        for token in doc:
            if token.pos_ == 'PROPN':
                current_compound.append(token)
            else:
                if len(current_compound) > 1:  # Only consider compounds of 2+ words
                    start_char = current_compound[0].idx
                    end_char = current_compound[-1].idx + len(current_compound[-1].text)
                    compound_text = doc.text[start_char:end_char]
                    compounds.append((compound_text, start_char, end_char))
                current_compound = []

        # Handle compound at end of sentence
        if len(current_compound) > 1:
            start_char = current_compound[0].idx
            end_char = current_compound[-1].idx + len(current_compound[-1].text)
            compound_text = doc.text[start_char:end_char]
            compounds.append((compound_text, start_char, end_char))

        return compounds

    def _find_possessive_entities(self, doc, target_entity: str) -> List[Tuple[str, int, int, str]]:
        """Find entities in possessive context (e.g., "John's story")

        Args:
            doc: spaCy Doc object
            target_entity: Target entity to match

        Returns:
            List of candidate tuples (text, start_char, end_char, match_type)
        """
        candidates = []

        for i, token in enumerate(doc):
            # Check if this token matches the target entity
            if self._entity_matches(token.text, target_entity):
                # Check if the next token is a possessive marker
                if i + 1 < len(doc):
                    next_token = doc[i + 1]
                    if next_token.text in ["'s", "'"]:
                        # This is a possessive entity - give it higher priority
                        start_char = token.idx
                        end_char = token.idx + len(token.text)
                        candidates.append((token.text, start_char, end_char, 'possessive_entity'))

        return candidates

    def is_core_entity_subject(self, doc, start_char: int, end_char: int, focus_entity: str) -> bool:
        """Determine if the CORE part of the focus entity is the sentence subject

        This method fixes a bug where relative clause subjects inside parenthetical
        descriptions (e.g., "that" in "Bridge (thing that does X)") were incorrectly
        causing the entire focus entity to be identified as the sentence subject.

        The fix is to only consider tokens BEFORE the first opening parenthesis
        when determining if the entity is the subject. Tokens inside parentheses
        are part of the entity description, not the main sentence structure.

        Args:
            doc: spaCy Doc object
            start_char: Start character position of the focus entity
            end_char: End character position of the focus entity
            focus_entity: The focus entity string

        Returns:
            True if the core entity (before parentheses) is the subject, False otherwise
        """
        # Find the position of the first opening parenthesis in the focus entity
        # to determine where the core entity ends
        paren_pos = focus_entity.find('(')
        if paren_pos != -1:
            # Core entity ends at the first parenthesis
            # Adjust end_char to only include the core entity text
            core_end_char = start_char + paren_pos
        else:
            # No parenthesis, use the full entity
            core_end_char = end_char

        # Check if any token in the CORE entity range is the subject
        for token in doc:
            # Only check tokens within the core entity range (before parentheses)
            if token.idx >= start_char and token.idx < core_end_char:
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    return True

        return False
