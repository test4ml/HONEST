#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for WH-Question Transformation

Shared utility functions used across the WH-transformer module.
"""

import re
from typing import List


def find_main_verb(doc):
    """Find the main verb in the sentence"""
    # First try to find ROOT verb
    for token in doc:
        if token.dep_ == 'ROOT' and token.pos_ in ['VERB', 'AUX']:
            return token

    # If no ROOT verb found (common after entity replacement), find any verb
    for token in doc:
        if token.pos_ in ['VERB', 'AUX']:
            return token

    return None


def find_subject(doc):
    """Find the subject of the sentence

    Handles special case for Wikidata entities like "Category:X" or "Template:X"
    where spaCy (especially core_web_lg) may parse "Category" as npadvmod
    instead of nsubjpass when followed by a colon.

    Example:
        "the Category:Horse breeds" may be parsed as:
        - "Category" (npadvmod) + ":" + "Horse" (compound) + "breeds" (nsubjpass)

        In this case, we should return "Category" (the leftmost token)
        as it's the start of the complete subject phrase.
    """
    # Find standard subjects
    subjects = [token for token in doc if token.dep_ in ['nsubj', 'nsubjpass']]

    if not subjects:
        return None

    # Get the first (leftmost) subject
    primary_subject = min(subjects, key=lambda t: t.i)

    # Check if there's an npadvmod token followed by ":" before the subject
    # This handles Wikidata entities like "Category:Horse breeds"
    # Look backwards from the subject to find the pattern
    for i in range(primary_subject.i - 1, -1, -1):
        token = doc[i]

        # If we find an npadvmod token
        if token.dep_ == 'npadvmod':
            # Check if there's a colon after it (between npadvmod and subject)
            # The colon might be immediately after npadvmod or a few tokens later
            found_colon = False
            for j in range(token.i + 1, primary_subject.i):
                if doc[j].text == ':':
                    found_colon = True
                    break

            if found_colon:
                # Found "Category:" or "Template:" pattern
                # Return the npadvmod token as it's the real start of the subject
                return token

        # Stop searching if we hit a verb or auxiliary (subject can't extend beyond these)
        if token.pos_ in ['VERB', 'AUX'] and token.dep_ == 'ROOT':
            break

    return primary_subject


def join_tokens_with_punct(tokens: List[str]) -> str:
    """Join tokens with proper punctuation handling

    Hyphens, parentheses, and other punctuation should not have spaces around them.
    """
    if not tokens:
        return ""

    result = []
    for i, token in enumerate(tokens):
        if token in ['-', '–', ',', '.', ':', ';', '!', '?', "'", '"', ')']:
            # Punctuation - no space before (including closing parenthesis)
            if result:
                result[-1] = result[-1] + token
        elif token == '(':
            # Opening parenthesis - no space after
            result.append(token)
        elif token == "'s":
            # Possessive marker - attach to previous word
            if result:
                result[-1] = result[-1] + token
            else:
                result.append(token)
        elif i > 0 and tokens[i-1] in ['-', '–', '(', ':']:
            # After hyphen, en dash, opening parenthesis, or colon - no space
            # FIX for kg_rule_102: Colon is used in Wikidata entities like "Category:Horse breeds"
            if result:
                result[-1] = result[-1] + token
            else:
                result.append(token)
        else:
            result.append(token)

    return ' '.join(result)


def find_matching_parenthesis(text: str, open_pos: int) -> int:
    """Find the matching closing parenthesis position

    Args:
        text: Text to search in
        open_pos: Position of the opening parenthesis

    Returns:
        Position of the matching closing parenthesis, or -1 if not found
    """
    if open_pos < 0 or open_pos >= len(text) or text[open_pos] != '(':
        return -1

    stack = []
    for i in range(open_pos, len(text)):
        if text[i] == '(':
            stack.append(i)
        elif text[i] == ')':
            if stack:
                stack.pop()
                if not stack:  # Found the matching closing parenthesis
                    return i
    return -1


def clean_hanging_words_before_punctuation(text: str) -> str:
    """Clean hanging words before any punctuation is added

    Note: We preserve prepositions like 'as', 'at', 'in' when they appear to be
    part of a wh-question pattern (e.g., "What is X classified as?")
    """
    words = text.split()

    # Hanging words that should be removed, but NOT prepositions that are part of questions
    # like "What is X classified as?" or "Where does X live at?"
    hanging_words_to_remove = ['a', 'an', 'the']

    # Only remove determiners from the end, preserve prepositions for question patterns
    while len(words) >= 1 and words[-1].lower() in hanging_words_to_remove:
        words.pop()

    return ' '.join(words)


def is_likely_proper_noun(word: str, words: List[str], index: int) -> bool:
    """Check if a word is likely a proper noun that should keep its capitalization"""
    # Single uppercase letters are likely entity identifiers/variables (e.g., "A", "B", "X")
    # and should keep their capitalization - check this FIRST before common words check
    if len(word) == 1 and word.isupper():
        return True

    # Common words that should NOT be treated as proper nouns even if capitalized
    common_words = {'this', 'that', 'these', 'those', 'the', 'a', 'an', 'is', 'are', 'was', 'were'}
    if word.lower() in common_words:
        return False

    # Check if word contains hyphens or en dashes (common in proper nouns)
    if any(char in word for char in ['-', '–']):
        return True

    # Check if word is part of a multi-word proper noun
    # Look at surrounding context
    if index > 0:
        prev_word = words[index - 1]
        # If previous word is capitalized and current word is capitalized,
        # it's likely a proper noun phrase
        if prev_word[0].isupper() and word[0].isupper():
            return True

    if index < len(words) - 1:
        next_word = words[index + 1]
        # If next word is capitalized and current word is capitalized,
        # it's likely a proper noun phrase
        if next_word[0].isupper() and word[0].isupper():
            return True

    # Words that start with capital and are longer than 2 chars are likely proper nouns
    if word[0].isupper() and len(word) > 2:
        return True

    return False


def clean_question(question: str, question_words: set) -> str:
    """Clean up the generated question"""
    # Remove extra spaces
    question = re.sub(r'\s+', ' ', question.strip())

    # Clean hanging words BEFORE adding question mark
    question = clean_hanging_words_before_punctuation(question)

    # Fix capitalization issues - ensure proper sentence case
    # First, ensure first letter is capitalized
    if question and question[0].islower():
        question = question[0].upper() + question[1:]

    # Fix specific capitalization issues in the middle of sentences
    # For example: "What does The cat eat?" -> "What does the cat eat?"
    words = question.split()
    for i in range(1, len(words)):  # Start from 1 to skip the first word
        word = words[i]
        # If a word starts with capital letter but isn't a proper noun or the start of a sentence,
        # and it's not a question word, and it's not part of a hyphenated proper noun, lowercase it
        if (word[0].isupper() and
            i > 0 and
            word.lower() not in question_words and
            not word.endswith('?') and
            not is_likely_proper_noun(word, words, i)):
            words[i] = word[0].lower() + word[1:]

    question = ' '.join(words)

    # Ensure question ends with question mark
    if not question.endswith('?'):
        question += '?'

    return question
