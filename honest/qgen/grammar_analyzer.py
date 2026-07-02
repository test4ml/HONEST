#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grammar analyzer: uses NLTK for intelligent grammatical analysis."""

from .property_manager import PropertyManager
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    # Ensure required NLTK data is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available, using simple rule-based grammar analysis")
    NLTK_AVAILABLE = False


class GrammarAnalyzer:
    """Grammar analyzer: uses NLTK for intelligent grammatical analysis."""

    def __init__(self):
        self.property_manager = PropertyManager()
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
        # Import article analyzer for consistent plurality detection
        from ..langtool.english.article_analyzer import EnglishArticleAnalyzer
        self._article_analyzer = EnglishArticleAnalyzer()

    def analyze_sentence_structure(self, subject: str, predicate: str, object_: str, property_id=None):
        """
        Analyze the full sentence structure and generate correct grammar from
        subject, predicate, and object.

        Args:
            subject: Subject
            predicate: Predicate
            object_: Object
            property_id: Property ID (e.g. P1435)

        Returns:
            Dict: Grammar analysis results
        """
        # First, retrieve the full property label
        if property_id:
            predicate = self.property_manager.get_property_label(property_id)

        # Analyze the number (singular/plural) of the subject
        subject_analysis = self._analyze_subject_number(subject)

        # Analyze the predicate structure
        predicate_analysis = self._analyze_predicate_structure(predicate, property_id)

        # Select an appropriate verb based on subject number
        suggested_verb = self._suggest_verb_with_agreement(
            subject, predicate, subject_analysis, predicate_analysis
        )

        # Analyze preposition requirements
        suggested_preposition = self._suggest_preposition_for_context(predicate, object_)

        return {
            'subject_analysis': subject_analysis,
            'predicate_analysis': predicate_analysis,
            'suggested_verb': suggested_verb,
            'suggested_preposition': suggested_preposition,
            'original_predicate': predicate,
            'property_id': property_id
        }

    def _analyze_subject_number(self, subject: str):
        """Analyze the number (singular/plural) of the subject

        IMPORTANT: Use the article_analyzer for consistent plurality detection,
        which correctly handles:
        - Wikidata entities (Category:X, Template:X) - always singular
        - Titles (books, events, championships) - always singular
        - Coordination in titles - singular as a title
        """
        # Use article_analyzer for authoritative plurality detection
        is_plural = self._article_analyzer.is_subject_plural(subject)

        # Return in the expected format for compatibility
        return {
            'is_plural': is_plural,
            'main_noun': subject,
            'method': 'article_analyzer'
        }

    def _analyze_subject_number_nltk(self, subject: str):
        """Analyze subject number (singular/plural) using NLTK."""
        tokens = nltk.word_tokenize(subject)
        pos_tags = nltk.pos_tag(tokens)

        # Find the main nouns
        main_nouns = [word for word, pos in pos_tags if pos.startswith('NN')]

        if not main_nouns:
            return {'is_plural': False, 'main_noun': subject, 'method': 'nltk_fallback'}

        main_noun = main_nouns[-1]  # Typically the last noun is the head word

        # Determine singular/plural from the POS tag
        main_noun_pos = [pos for word, pos in pos_tags if word == main_noun][-1]

        is_plural = main_noun_pos in ['NNS', 'NNPS']  # Plural nouns

        return {
            'is_plural': is_plural,
            'main_noun': main_noun,
            'pos_tag': main_noun_pos,
            'all_nouns': main_nouns,
            'method': 'nltk'
        }

    def _analyze_subject_number_simple(self, subject: str):
        """Analyze subject number (singular/plural) with simple rules."""
        words = subject.split()
        if not words:
            return {'is_plural': False, 'main_noun': subject, 'method': 'simple_fallback'}

        last_word = words[-1].lower()

        # Plural rules
        plural_patterns = {
            'regular': ['s', 'es'],
            'irregular': ['children', 'people', 'men', 'women', 'feet', 'teeth', 'mice', 'geese'],
            'invariant': ['sheep', 'deer', 'fish', 'species', 'series'],
            'latin': ['data', 'criteria', 'phenomena', 'alumni']
        }

        is_plural = False
        if last_word in plural_patterns['irregular'] + plural_patterns['invariant'] + plural_patterns['latin']:
            is_plural = True
        elif last_word.endswith(('s', 'es')) and not last_word.endswith(('ss', 'us', 'is', 'ous', 'ious')):
            is_plural = True
        elif last_word.endswith('ies') and len(last_word) > 3:
            is_plural = True
        elif last_word.endswith('ves') and len(last_word) > 3:
            is_plural = True

        return {
            'is_plural': is_plural,
            'main_noun': last_word,
            'method': 'simple_rules'
        }

    def _suggest_verb_with_agreement(self, subject: str, predicate: str, subject_analysis: dict, predicate_analysis: dict):
        """Suggest a verb according to subject-verb agreement."""
        is_plural = subject_analysis.get('is_plural', False)

        # Check for special verb patterns
        predicate_lower = predicate.lower()

        # Verb mappings for special properties
        special_verb_patterns = {
            'heritage designation': 'is' if not is_plural else 'are',
            'designation': 'is' if not is_plural else 'are',
            'born': 'was' if not is_plural else 'were',
            'died': 'was' if not is_plural else 'were',
            'located': 'is' if not is_plural else 'are',
            'situated': 'is' if not is_plural else 'are',
            'written': 'was' if not is_plural else 'were',
            'created': 'was' if not is_plural else 'were',
            'built': 'was' if not is_plural else 'were',
            'founded': 'was' if not is_plural else 'were',
            'established': 'was' if not is_plural else 'were',
        }

        # Check for special patterns
        for pattern, verb_template in special_verb_patterns.items():
            if pattern in predicate_lower:
                return verb_template

        # Default to the correct form of the copula "be"
        if is_plural:
            return 'are'
        else:
            return 'is'

    def _analyze_predicate_structure(self, predicate: str, property_id=None):
        """Analyze the predicate structure."""
        tokens = predicate.split()

        return {
            'original_text': predicate,
            'tokens': tokens,
            'is_noun_phrase': self._is_noun_phrase_simple(tokens),
            'is_verb_phrase': self._is_verb_phrase_simple(tokens),
            'starts_with_preposition': self._starts_with_preposition(tokens),
            'ends_with_preposition': self._ends_with_preposition(tokens),
            'contains_preposition': self._contains_preposition_simple(tokens),
            'main_noun': self._get_main_noun_simple(tokens)
        }

    def _suggest_preposition_for_context(self, predicate: str, object_: str):
        """Suggest a preposition based on context."""
        text = predicate.lower()

        preposition_rules = {
            'place': 'in',
            'location': 'in',
            'country': 'in',
            'city': 'in',
            'date': 'on',
            'time': 'at',
            'birth': 'in',
            'death': 'in',
            'from': 'from',
            'to': 'to',
            'of': 'of',
            'upstream': 'from',
            'downstream': 'from'
        }

        for keyword, prep in preposition_rules.items():
            if keyword in text:
                return prep

        return None

    # Keep old method names for backward compatibility
    def analyze_predicate_structure(self, predicate_text, property_id=None):
        """Backward-compatible method, redirects to the new analyze_sentence_structure."""
        # Use default subject and object for backward compatibility
        default_subject = "entity"
        default_object = "object"

        result = self.analyze_sentence_structure(default_subject, predicate_text, default_object, property_id)

        # Return results in the old format
        return {
            'original_text': predicate_text,
            'tokens': result['predicate_analysis']['tokens'],
            'is_noun_phrase': result['predicate_analysis']['is_noun_phrase'],
            'is_verb_phrase': result['predicate_analysis']['is_verb_phrase'],
            'starts_with_preposition': result['predicate_analysis']['starts_with_preposition'],
            'ends_with_preposition': result['predicate_analysis']['ends_with_preposition'],
            'contains_preposition': result['predicate_analysis']['contains_preposition'],
            'main_noun': result['predicate_analysis']['main_noun'],
            'suggested_verb': result['suggested_verb'],
            'suggested_preposition': result['suggested_preposition']
        }

    def _is_noun_phrase_simple(self, tokens):
        """Simple check for whether this is a noun phrase."""
        if not tokens:
            return False

        # Check whether it starts with a common noun
        common_nouns = ['next', 'instance', 'subclass', 'country', 'place', 'date', 'occupation', 'location']
        return tokens[0].lower() in common_nouns

    def _is_verb_phrase_simple(self, tokens):
        """Simple check for whether this is a verb phrase."""
        # Check whether it contains a common verb
        common_verbs = ['born', 'died', 'occurred', 'has', 'have', 'is', 'are', 'was', 'were']
        return any(token.lower() in common_verbs for token in tokens)

    def _starts_with_preposition(self, tokens):
        """Check whether it starts with a preposition."""
        if not tokens:
            return False
        common_prepositions = ['in', 'on', 'at', 'from', 'to', 'of', 'for', 'with', 'by', 'about']
        return tokens[0].lower() in common_prepositions

    def _ends_with_preposition(self, tokens):
        """Check whether it ends with a preposition."""
        if not tokens:
            return False
        common_prepositions = ['in', 'on', 'at', 'from', 'to', 'of', 'for', 'with', 'by', 'about']
        return tokens[-1].lower() in common_prepositions

    def _contains_preposition_simple(self, tokens):
        """Simple check for whether it contains a preposition."""
        common_prepositions = ['in', 'on', 'at', 'from', 'to', 'of', 'for', 'with', 'by', 'about']
        return any(token.lower() in common_prepositions for token in tokens)

    def _get_main_noun_simple(self, tokens):
        """Simply get the main noun."""
        # Find the last non-preposition noun
        for i in range(len(tokens)-1, -1, -1):
            token = tokens[i].lower()
            if token not in ['in', 'on', 'at', 'from', 'to', 'of', 'for', 'with', 'by', 'about']:
                return tokens[i]
        return tokens[-1] if tokens else ""

    def _suggest_verb_simple(self, tokens):
        """Simply suggest an appropriate verb."""
        text = ' '.join(tokens).lower()

        # Mapping of specific verb patterns
        verb_patterns = {
            'born': 'was',
            'died': 'was',
            'located': 'is',
            'situated': 'is',
            'written': 'was',
            'created': 'was',
            'built': 'was',
            'founded': 'was',
            'established': 'was',
            'heritage designation': 'is',  # Heritage designation case
            'designation': 'is',
        }

        # Check whether it contains a specific verb
        for pattern, verb in verb_patterns.items():
            if pattern in text:
                return verb

        # Suggest a verb based on the phrase structure
        if self._is_noun_phrase_simple(tokens):
            return 'is'  # Noun phrases use the copula
        elif self._is_verb_phrase_simple(tokens):
            return 'is'  # Changed to default to "is" instead of "has"
        else:
            return 'is'  # Default to the copula

    def _suggest_preposition_simple(self, tokens):
        """Simply suggest an appropriate preposition."""
        text = ' '.join(tokens).lower()

        preposition_rules = {
            'place': 'in',
            'location': 'in',
            'country': 'in',
            'city': 'in',
            'date': 'on',
            'time': 'at',
            'birth': 'in',
            'death': 'in',
            'from': 'from',
            'to': 'to',
            'of': 'of',
            'upstream': 'from',
            'downstream': 'from'
        }

        for keyword, prep in preposition_rules.items():
            if keyword in text:
                return prep

        # Default preposition
        if self._ends_with_preposition(tokens):
            return ''  # Preposition already present
        else:
            return 'of'  # Default preposition
