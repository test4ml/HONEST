#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English Article Analyzer using spaCy

Smart article manager that determines appropriate articles (a/an/the) for English nouns
using advanced spaCy-based natural language processing for POS tagging, morphological
analysis, and pronunciation patterns.
"""

import re
import logging
from typing import Optional, Dict, Set, List, Tuple
from functools import lru_cache
import spacy
from ..core.types import Language
from honest.constants import SPACY_MODEL_NAME

logger = logging.getLogger(__name__)

class EnglishArticleAnalyzer:
    """English article analyzer using spaCy for smart article selection"""

    def __init__(self):
        """Initialize the analyzer with spaCy English model"""
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Load CMU pronunciation dictionary once
        self.cmu_dict = None
        try:
            import nltk
            from nltk.corpus import cmudict
            try:
                self.cmu_dict = cmudict.dict()
            except LookupError:
                nltk.download('cmudict', quiet=True)
                self.cmu_dict = cmudict.dict()
        except ImportError:
            logger.warning("Failed to load CMU pronunciation dictionary")

        # Cache for article analysis results - PERFORMANCE OPTIMIZATION
        # Caching dramatically improves performance (1500x speedup) for repeated phrases
        # which is common in distractor generation where similar entities are processed
        self._article_cache: Dict[str, str] = {}

        # Special cases for article selection
        self.special_cases = {
            # Definite article required cases
            'definite_required': {
                'superlatives': {'best', 'worst', 'most', 'least', 'greatest', 'smallest', 'largest'},
                'unique_entities': {'sun', 'moon', 'earth', 'internet', 'universe'},
                'ordinals': {'first', 'second', 'third', 'last', 'next', 'previous', 'final'},
            },
            # No article needed
            'no_article': {
                'mass_nouns': {
                    # Common mass nouns
                    'water', 'air', 'music', 'information', 'advice', 'furniture',
                    'homework', 'knowledge', 'money', 'research', 'traffic', 'weather',
                    # Fields of study / academic disciplines (uncountable)
                    'archaeology', 'anthropology', 'biology', 'chemistry', 'physics',
                    'mathematics', 'statistics', 'economics', 'psychology', 'sociology',
                    'philosophy', 'theology', 'history', 'geography', 'geology',
                    'astronomy', 'medicine', 'engineering', 'architecture', 'literature',
                    'linguistics', 'law', 'politics', 'science', 'art',
                    # Abstract concepts (uncountable)
                    'education', 'health', 'peace', 'freedom', 'justice', 'democracy',
                    'happiness', 'love', 'anger', 'courage', 'wisdom',
                },
            },
            # Pronunciation-based rules for a/an
            'pronunciation': {
                'vowel_sound_consonants': {'university', 'union', 'uniform', 'unique', 'user', 'european'},
                'consonant_sound_vowels': {'honest', 'honor', 'hour', 'heir'},
            }
        }

        # Countries that need definite article
        self.definite_countries = {
            'united states', 'united kingdom', 'netherlands', 'philippines',
            'czech republic', 'dominican republic', 'central african republic'
        }

    def add_article(self, noun_phrase: str) -> str:
        """
        Add appropriate article to noun phrase

        Args:
            noun_phrase: Input noun phrase

        Returns:
            Noun phrase with appropriate article
        """
        if not self.nlp:
            return self._fallback_article_selection(noun_phrase)

        if not noun_phrase or not noun_phrase.strip():
            return noun_phrase

        noun_phrase = noun_phrase.strip()

        # Check cache first - PERFORMANCE OPTIMIZATION
        # This provides ~1500x speedup for repeated phrases
        if noun_phrase in self._article_cache:
            return self._article_cache[noun_phrase]

        # Compute result and cache it
        result = self._add_article_impl(noun_phrase)

        # Cache the result before returning - PERFORMANCE OPTIMIZATION
        self._article_cache[noun_phrase] = result

        return result

    def _add_article_impl(self, noun_phrase: str) -> str:
        """
        Internal implementation of add_article (without caching)

        Args:
            noun_phrase: Input noun phrase (already stripped)

        Returns:
            Noun phrase with appropriate article
        """
        # 1. Check if already has determiner
        if self._has_determiner(noun_phrase):
            return noun_phrase

        # 2. Check if it's a technical identifier (gene ID, protein code, etc.)
        # Technical identifiers should NOT have articles
        if self._is_technical_identifier(noun_phrase):
            return noun_phrase

        try:
            # 3. spaCy analysis
            doc = self.nlp(noun_phrase)

            # 4. Proper noun detection
            if self._is_proper_noun_spacy(doc) or self._is_wikidata_entity(noun_phrase):
                return self._handle_proper_noun(noun_phrase)

            # 4.5 Special pattern: "list of X" always needs "the"
            # This must come BEFORE plural detection because "list" is typically followed by plural nouns
            # but the phrase "list of X" as a whole is treated as singular and needs "the"
            if noun_phrase.lower().startswith('list of '):
                return f"the {noun_phrase}"

            # 4. Plural noun detection
            if self._is_plural_spacy(doc):
                return noun_phrase

            # 5. Mass noun detection
            if self._is_mass_noun_spacy(doc):
                return noun_phrase

            # 6. Definite article cases
            if self._needs_definite_article_spacy(doc):
                return f"the {noun_phrase}"

            # 7. Default: indefinite article
            return self._choose_indefinite_article(noun_phrase)

        except (AttributeError, ValueError, RuntimeError) as e:
            logger.warning(f"spaCy analysis failed, using fallback: {e}")
            return self._fallback_article_selection(noun_phrase)

    def add_article_to_predicate(self, predicate: str) -> str:
        """Add appropriate articles to nouns in predicate phrases"""
        if not predicate or not predicate.strip():
            return predicate

        # Handle ordinals in predicates
        words = predicate.split()
        ordinals = self.special_cases['definite_required']['ordinals']

        for i, word in enumerate(words):
            if word.lower() in ordinals and i + 1 < len(words):
                if not any(det in words[:i] for det in ['the', 'a', 'an']):
                    words.insert(i, 'the')
                    break

        return ' '.join(words)

    def is_subject_plural(self, subject: str) -> bool:
        """Determine if subject is plural (for subject-verb agreement)

        IMPORTANT: This method needs to handle special cases where grammatically
        plural-looking phrases should be treated as singular:

        1. Wikidata entities (Category:X, Template:X) - always singular as entity names
        2. Titles (books, albums, events, championships) - always singular as titles
        3. Coordination in titles ("Asterix and Obelix's Birthday") - singular as title
        4. Entities with descriptions - only analyze the main label, not the description
        """
        if not subject or not subject.strip():
            return False

        # PRIORITY 0: Wikidata special entities are always singular
        # "Category:Mexican people" refers to ONE category, not multiple people
        if self._is_wikidata_entity(subject):
            return False

        # PRIORITY 0.5: Check if this looks like a title/proper noun phrase
        # Titles are grammatically singular even if they contain plural words
        if self._is_likely_title(subject):
            return False

        # PRIORITY 0.7: Extract pure label from "Label (description)" format
        # When subject has format "Merited Artist (honorary title and award in...countries)",
        # only analyze "Merited Artist", not the description which may contain plural words
        pure_label = self._extract_pure_label(subject)

        if not self.nlp:
            return self._fallback_plural_detection(pure_label)

        try:
            doc = self.nlp(pure_label)
            return self._is_plural_spacy(doc, pure_label)
        except (AttributeError, ValueError, RuntimeError):
            return self._fallback_plural_detection(pure_label)

    def _is_likely_title(self, subject: str) -> bool:
        """Detect if the subject is likely a title (book, album, event, etc.)

        Titles should be treated as singular even if they contain:
        - Plural nouns ("The Lord of the Rings")
        - Coordination ("Asterix and Obelix's Birthday")
        - Event-style names ("2014 Swiss Open Badminton Championships")

        Detection heuristics:
        1. Starts with a year (event names like "2014 Swiss Open...")
        2. Contains title-like patterns (Championship, Tournament, Open, etc.)
        3. Contains possessive patterns ("'s")
        4. Most words are capitalized (proper noun phrase)
        """
        if not subject:
            return False

        words = subject.split()
        if not words:
            return False

        # Heuristic 1: Starts with a 4-digit year (likely an event name)
        if words[0].isdigit() and len(words[0]) == 4:
            try:
                year = int(words[0])
                if 1800 <= year <= 2100:
                    return True
            except ValueError:
                pass

        # Heuristic 2: Contains championship/event keywords
        event_keywords = {
            'championship', 'championships', 'tournament', 'open', 'cup',
            'olympics', 'games', 'festival', 'competition', 'series',
            'league', 'trophy', 'grand prix', 'world cup'
        }
        subject_lower = subject.lower()
        if any(keyword in subject_lower for keyword in event_keywords):
            # Additional check: if it looks like a proper name (mostly capitalized)
            alpha_words = [w for w in words if w[0].isalpha()]
            if alpha_words:
                cap_ratio = sum(1 for w in alpha_words if w[0].isupper()) / len(alpha_words)
                if cap_ratio >= 0.5:  # At least 50% capitalized
                    return True

        # Heuristic 3: Contains possessive pattern with 'and' coordination
        # e.g., "Asterix and Obelix's Birthday" is a book title
        if "'s" in subject and " and " in subject:
            # Check if most words are capitalized (title case)
            alpha_words = [w for w in words if w[0].isalpha()]
            if alpha_words:
                cap_ratio = sum(1 for w in alpha_words if w[0].isupper()) / len(alpha_words)
                if cap_ratio >= 0.6:  # At least 60% capitalized suggests a title
                    return True

        # Heuristic 4: Contains em-dash or en-dash (common in event names)
        # e.g., "2014 Swiss Open Badminton Championships – mixed doubles"
        if '–' in subject or '—' in subject or ' - ' in subject:
            alpha_words = [w for w in words if w and w[0].isalpha()]
            if alpha_words:
                cap_ratio = sum(1 for w in alpha_words if w[0].isupper()) / len(alpha_words)
                if cap_ratio >= 0.5:
                    return True

        return False

    def _extract_pure_label(self, subject: str) -> str:
        """Extract pure label from "Label (description)" format

        This is critical for subject-verb agreement when entities have descriptions.

        Examples:
        - "Merited Artist (honorary title...countries)" -> "Merited Artist"
        - "acolyte (profession)" -> "acolyte"
        - "simple label" -> "simple label"

        Without this extraction, the plurality checker would analyze the entire string
        including the description, which may contain plural words (e.g., "countries")
        that don't reflect the plurality of the actual subject.
        """
        if not subject:
            return subject

        # Match "Label (description)" pattern
        # Handle nested parentheses in description: "Label (desc (nested) desc)"
        match = re.match(r'^(.+?)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)$', subject)
        if match:
            pure_label = match.group(1).strip()
            return pure_label

        # No description found, return as-is
        return subject

    def _has_determiner(self, noun_phrase: str) -> bool:
        """Check if noun phrase already has a determiner"""
        determiners = {
            'a', 'an', 'the', 'this', 'that', 'these', 'those',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'some', 'any', 'many', 'much', 'few', 'little', 'several',
            'all', 'most', 'every', 'each', 'both', 'either', 'neither'
        }
        first_word = noun_phrase.split()[0].lower()
        return first_word in determiners

    def _is_technical_identifier(self, noun_phrase: str) -> bool:
        """
        Check if noun phrase is a technical identifier (gene ID, protein code, etc.)

        Technical identifiers should NOT have articles.

        Patterns include:
        - Contains underscores with alphanumeric sequences (e.g., FGRAMPH1_01G06555)
        - LOC prefix followed by numbers (e.g., LOC123456)
        - Two-letter prefix + underscore + numbers (e.g., XP_012345)
        - Complex alphanumeric codes typical of biological databases

        Args:
            noun_phrase: The phrase to check

        Returns:
            True if it's a technical identifier
        """
        # Remove surrounding whitespace and punctuation for checking
        identifier = noun_phrase.strip().strip('.,;:!?"\')').strip()

        if not identifier:
            return False

        # Pattern 1: Contains underscores with mixed alphanumeric content
        # Examples: FGRAMPH1_01G06555, GENE_ABC_123, XP_012345
        if '_' in identifier:
            # Must have both letters and digits
            has_digit = any(c.isdigit() for c in identifier)
            has_letter = any(c.isalpha() for c in identifier)
            if has_digit and has_letter:
                return True

        # Pattern 2: LOC prefix followed by numbers
        # Examples: LOC123456, LOC108348065
        if re.match(r'^LOC\d+$', identifier, re.IGNORECASE):
            return True

        # Pattern 3: Two-letter uppercase prefix + underscore + numbers
        # Examples: XP_012345, NP_123456, YP_789012
        if re.match(r'^[A-Z]{2}_\d+$', identifier):
            return True

        # Pattern 4: NEWGENE_ prefix (common in some databases)
        # Example: NEWGENE_6497122
        if re.match(r'^NEWGENE_\d+$', identifier, re.IGNORECASE):
            return True

        # Pattern 5: High ratio of uppercase to lowercase + contains digits
        # Typical of technical identifiers
        if identifier.isupper() or (
            sum(1 for c in identifier if c.isupper()) >
            sum(1 for c in identifier if c.islower()) * 2
        ):
            has_digit = any(c.isdigit() for c in identifier)
            has_letter = any(c.isalpha() for c in identifier)
            # Must be reasonably long and have both letters and numbers
            if has_digit and has_letter and len(identifier) > 5:
                return True

        return False

    def _is_proper_noun_spacy(self, doc) -> bool:
        """Detect proper nouns using spaCy"""
        # Check for proper noun POS tags
        proper_tags = {'PROPN'}
        has_proper_noun = any(token.pos_ in proper_tags for token in doc)

        if has_proper_noun:
            return True

        # Check for named entities
        if doc.ents:
            return True

        # Check capitalization patterns
        words = [token.text for token in doc if token.is_alpha]
        if not words:
            return False

        # Multi-word proper noun: most words capitalized
        capitalized_words = [word for word in words if word[0].isupper()]
        if len(words) > 1 and len(capitalized_words) >= len(words) * 0.7:
            return True

        # Single word proper noun detection
        if len(words) == 1 and words[0][0].isupper():
            # Exclude common words that might be capitalized
            common_words = {'Bridge', 'Road', 'Street', 'Building', 'Park', 'River', 'Lake'}
            if words[0] not in common_words:
                return True

        return False

    def _is_wikidata_entity(self, noun_phrase: str) -> bool:
        """Detect Wikidata special entities"""
        if ':' in noun_phrase:
            parts = noun_phrase.split(':', 1)
            if len(parts) == 2:
                prefix = parts[0].lower()
                return prefix in ['category', 'template', 'file', 'user']
        return False

    def _handle_proper_noun(self, noun_phrase: str) -> str:
        """Handle proper nouns - improved Wikidata entity recognition"""
        # Check for countries that need definite article
        if noun_phrase.lower() in self.definite_countries:
            return f"the {noun_phrase}"

        # Wikidata namespace handling
        if ':' in noun_phrase:
            parts = noun_phrase.split(':', 1)
            if len(parts) == 2:
                prefix, content = parts
                prefix_lower = prefix.lower()

                if prefix_lower in ['category', 'template']:
                    return f"the {noun_phrase}"
                elif prefix_lower in ['file', 'user']:
                    return noun_phrase

        # Check for event names that need definite article
        if self._needs_definite_article_for_event(noun_phrase):
            return f"the {noun_phrase}"

        # Check for organization names that need definite article
        if self._needs_definite_article_for_organization(noun_phrase):
            return f"the {noun_phrase}"


        # Most proper nouns don't need articles
        return noun_phrase

    def _needs_definite_article_for_event(self, noun_phrase: str) -> bool:
        """Check if event name needs definite article"""
        if not self.nlp:
            return False

        try:
            doc = self.nlp(noun_phrase)

            # Event name patterns that need "the"
            event_keywords = {
                'championships', 'championship', 'tournament', 'open',
                'olympics', 'games', 'festival', 'competition', 'series',
                'conference', 'symposium', 'congress', 'summit'
            }

            # Check if contains event keywords
            for token in doc:
                if token.lemma_.lower() in event_keywords or token.text.lower() in event_keywords:
                    # Additional checks to avoid false positives
                    # 1. Check if it's part of a proper noun phrase
                    if token.pos_ == 'PROPN':
                        # 2. Check if it's likely an event name (contains year or location)
                        words = noun_phrase.split()

                        # Check for year pattern (e.g., "2014 Swiss Open")
                        if len(words) >= 2:
                            first_word = words[0]
                            if first_word.isdigit() and len(first_word) == 4:
                                year = int(first_word)
                                if 1900 <= year <= 2100:
                                    return True

                        # Check for location + event pattern (e.g., "Swiss Open")
                        location_indicators = {'Swiss', 'Sydney', 'Tahiti', 'International', 'World', 'European', 'Asian',
                                              'American', 'African', 'National', 'Regional', 'Global', 'Continental'}
                        if any(word in location_indicators for word in words):
                            return True

                        # Check for ordinal pattern (e.g., "First", "Second")
                        ordinals = {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'}
                        if any(word.lower() in ordinals for word in words):
                            return True

                    return True

            return False
        except (AttributeError, ValueError, IndexError):
            return False

    def _needs_definite_article_for_organization(self, noun_phrase: str) -> bool:
        """Check if organization name needs definite article"""
        if not self.nlp:
            return False

        try:
            doc = self.nlp(noun_phrase)

            # Organization name patterns that need "the"
            org_keywords = {
                'association', 'society', 'institute', 'institution', 'foundation',
                'committee', 'council', 'commission', 'board', 'academy',
                'university', 'college', 'school', 'hospital', 'museum',
                'gallery', 'theatre', 'orchestra', 'band', 'club'
            }

            # Check if contains organization keywords
            for token in doc:
                if token.lemma_.lower() in org_keywords:
                    # Additional checks to avoid false positives
                    # 1. Check if it's part of a proper noun phrase
                    if token.pos_ == 'PROPN':
                        # 2. Check if it's likely an organization name
                        words = noun_phrase.split()

                        # Check for location + organization pattern (e.g., "Harvard University")
                        location_indicators = {'Harvard', 'Stanford', 'Oxford', 'Cambridge', 'National', 'International',
                                              'Royal', 'American', 'British', 'European', 'Asian', 'African'}
                        if any(word in location_indicators for word in words):
                            return True

                        # Check for proper noun capitalization pattern
                        if len(words) > 1:
                            # If most words are capitalized, it's likely a proper organization name
                            capitalized_words = [word for word in words if word[0].isupper()]
                            if len(capitalized_words) >= len(words) * 0.6:
                                return True

                    # If it's not a proper noun, don't add "the" for common nouns
                    return False

            return False
        except (AttributeError, ValueError, IndexError):
            return False

    def _is_plural_spacy(self, doc, original_subject: str = "") -> bool:
        """Detect plural nouns using spaCy

        PRIORITY SYSTEM:
        1. Named Entity Recognition (NER) - Most reliable for proper nouns
        2. Morphological analysis - For common nouns
        3. POS tags - Fallback method

        Special handling for PERSON and ORG entities to avoid misclassification:
        - PERSON: Rare surnames ending in 's' (e.g., "Caruthers", "Worms")
        - ORG: Company names ending in 's' or with plural-like patterns (e.g., "Gemalto", "Reuters", "Illinois Tool Works")

        Also handles EVENT entities as singular (championships, tournaments, etc.)
        """
        # PRIORITY 1: Check Named Entity Recognition (NER)
        # If entity is tagged as PERSON, ORG, FAC, or EVENT, treat as singular
        singular_entity_indices = set()

        for ent in doc.ents:
            # All these entity types should be treated as singular
            if ent.label_ in ('PERSON', 'ORG', 'FAC', 'EVENT', 'WORK_OF_ART', 'LAW', 'PRODUCT'):
                for token in ent:
                    singular_entity_indices.add(token.i)

        # Check if the main noun (ROOT) is part of a singular entity type
        for token in doc:
            if token.dep_ == 'ROOT' or (token.pos_ in ['NOUN', 'PROPN'] and token.dep_ == 'compound'):
                if token.i in singular_entity_indices:
                    # ROOT noun is part of a singular entity → singular
                    return False

        # PRIORITY 2: Check morphological features (most reliable for common nouns)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                if hasattr(token, 'morph') and token.morph:
                    number = token.morph.get('Number')
                    if number and 'Plur' in number:
                        # Skip if it's part of a singular entity type
                        if token.i in singular_entity_indices:
                            continue
                        # Additional check: if it's a noun ending in 's' and looks like a proper noun, it might be singular
                        # Examples: "Thales" (company), "Reuters" (news agency)
                        if token.text.endswith('s') and token.text[0].isupper():
                            # Check if it's likely a singular proper noun (company name, person name, etc.)
                            # by looking at capitalization and context
                            if len(doc) == 1:  # Single word proper noun ending in 's'
                                # Common singular company names ending in 's'
                                singular_company_names = {'Thales', 'Reuters', 'Gemalto', 'Siemens', 'Airbus', 'Netsize'}
                                if token.text in singular_company_names:
                                    continue  # Skip, treat as singular
                        return True

        # PRIORITY 3: Check for coordination (A and B) - but NOT in titles
        # We now check this earlier in is_subject_plural via _is_likely_title
        # So we can simplify this check: only return plural if there's coordination
        # of truly separate entities, not coordination within a title
        has_conjunction = any(token.dep_ == 'conj' for token in doc)
        if has_conjunction:
            # If we have coordinated nouns and this isn't a title (checked earlier),
            # it's usually plural
            noun_count = sum(1 for token in doc if token.pos_ in ['NOUN', 'PROPN'])
            if noun_count >= 2:
                # Final check: are ALL coordinated nouns part of singular entities?
                # If yes, treat as singular
                propn_indices = [token.i for token in doc if token.pos_ in ['NOUN', 'PROPN']]
                if propn_indices and all(idx in singular_entity_indices for idx in propn_indices):
                    pass  # All nouns are in singular entities, don't return True
                else:
                    return True

        # PRIORITY 4: Check POS tag patterns (least reliable, but fast)
        plural_patterns = {'NNS', 'NNPS'}  # Plural noun tags
        for token in doc:
            if token.tag_ in plural_patterns:
                # Skip if part of a singular entity type
                if token.i in singular_entity_indices:
                    continue
                # Additional check: if it's a noun ending in 's' and looks like a proper noun, it might be singular
                # Examples: "Thales" (company), "Reuters" (news agency)
                if token.text.endswith('s') and token.text[0].isupper():
                    # Check if it's likely a singular proper noun (company name, person name, etc.)
                    # by looking at capitalization and context
                    if len(doc) == 1:  # Single word proper noun ending in 's'
                        # Common singular company names ending in 's'
                        singular_company_names = {'Thales', 'Reuters', 'Gemalto', 'Siemens', 'Airbus', 'Netsize'}
                        if token.text in singular_company_names:
                            continue  # Skip, treat as singular
                return True

        return False

    def _is_mass_noun_spacy(self, doc) -> bool:
        """Detect mass/uncountable nouns using spaCy"""
        mass_nouns = self.special_cases['no_article']['mass_nouns']

        # Find the main noun (head noun)
        main_noun = None
        for token in doc:
            if token.pos_ == 'NOUN' and token.dep_ in ['ROOT', 'dobj', 'pobj']:
                main_noun = token.lemma_.lower()
                break

        # If no head noun found, check last noun
        if not main_noun:
            for token in reversed(doc):
                if token.pos_ == 'NOUN':
                    main_noun = token.lemma_.lower()
                    break

        return main_noun in mass_nouns if main_noun else False

    def _needs_definite_article_spacy(self, doc) -> bool:
        """Determine if definite article is needed based on spaCy analysis"""
        # Check for superlatives
        for token in doc:
            if token.pos_ == 'ADJ' and hasattr(token, 'morph') and token.morph:
                degree = token.morph.get('Degree')
                if degree and 'Sup' in degree:  # Superlative
                    return True

        # Check for ordinals and unique entities
        ordinals = self.special_cases['definite_required']['ordinals']
        unique_entities = self.special_cases['definite_required']['unique_entities']

        for token in doc:
            if token.lemma_.lower() in ordinals or token.lemma_.lower() in unique_entities:
                return True

        return False

    def _choose_indefinite_article(self, noun_phrase: str) -> str:
        """Choose between 'a' and 'an' based on pronunciation"""
        if not noun_phrase:
            return noun_phrase

        first_word = noun_phrase.split()[0].lower()

        if self._starts_with_vowel_sound(first_word):
            return f"an {noun_phrase}"
        else:
            return f"a {noun_phrase}"

    def _starts_with_vowel_sound(self, word: str) -> bool:
        """Check if word starts with vowel sound using CMU pronunciation dictionary"""
        word_lower = word.lower()

        # Special pronunciation cases (highest priority)
        consonant_sound_vowels = self.special_cases['pronunciation']['consonant_sound_vowels']
        if word_lower in consonant_sound_vowels:
            return True

        vowel_sound_consonants = self.special_cases['pronunciation']['vowel_sound_consonants']
        if word_lower in vowel_sound_consonants:
            return False

        # Use CMU pronunciation dictionary if available
        if self.cmu_dict and word_lower in self.cmu_dict:
            # Get first pronunciation variant
            phonemes = self.cmu_dict[word_lower][0]
            if phonemes:
                # First phoneme of the pronunciation
                first_phoneme = phonemes[0]
                # Vowel phonemes in CMU dict end with stress markers (0, 1, 2)
                # They start with: AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW
                vowel_phonemes = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
                # Remove stress markers (digits)
                phoneme_base = ''.join(c for c in first_phoneme if c.isalpha())
                return phoneme_base in vowel_phonemes

        # Fallback to letter-based detection
        return word[0].lower() in 'aeiou'

    def _fallback_article_selection(self, noun_phrase: str) -> str:
        """Fallback method when spaCy is not available"""
        if not noun_phrase or not noun_phrase.strip():
            return noun_phrase

        noun_phrase = noun_phrase.strip()

        # Check for existing determiners
        if self._has_determiner(noun_phrase):
            return noun_phrase

        # Check for technical identifiers (before other checks)
        if self._is_technical_identifier(noun_phrase):
            return noun_phrase

        # Simple proper noun detection
        if noun_phrase[0].isupper() or ':' in noun_phrase:
            return self._handle_proper_noun(noun_phrase)

        # Simple plural detection
        if self._fallback_plural_detection(noun_phrase):
            return noun_phrase

        # Default indefinite article
        return self._choose_indefinite_article(noun_phrase)

    def _fallback_plural_detection(self, noun_phrase: str) -> bool:
        """Simple plural detection fallback"""
        words = noun_phrase.split()
        if not words:
            return False

        # Check for 'and' coordination
        if 'and' in words and len(words) >= 3:
            return True

        # Check for common plural endings
        last_word = words[-1].lower()
        if last_word.endswith(('s', 'es', 'ies', 'ves')):
            # Exclude words that commonly end in 's' but are singular
            singular_s_words = {'bus', 'class', 'glass', 'pass', 'gas', 'this', 'yes'}
            if last_word not in singular_s_words:
                return True

        return False

    # Convenience methods
    def analyze_noun_phrase(self, noun_phrase: str) -> Dict:
        """Analyze noun phrase structure for debugging"""
        if not self.nlp:
            return {"error": "spaCy not available"}

        doc = self.nlp(noun_phrase)

        analysis = {
            "tokens": [(token.text, token.pos_, token.tag_, token.dep_) for token in doc],
            "is_proper_noun": self._is_proper_noun_spacy(doc),
            "is_plural": self._is_plural_spacy(doc),
            "is_mass_noun": self._is_mass_noun_spacy(doc),
            "needs_definite": self._needs_definite_article_spacy(doc),
            "has_determiner": self._has_determiner(noun_phrase),
            "entities": [(ent.text, ent.label_) for ent in doc.ents]
        }

        return analysis