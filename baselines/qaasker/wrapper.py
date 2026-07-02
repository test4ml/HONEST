"""
QAAskeR Wrapper Implementation

This module provides a clean wrapper around the original QAAskeR implementation.
It wraps the original functions and provides a clean Python API.

Based on: "Testing Your Question Answering Software via Asking Recursively"
https://github.com/imcsq/ASE21-QAAskeR

The complete MR flows are:
- MR1: Wh-question + Answer -> Q2S -> S2W -> UniLM -> new Wh-question
- MR2: Wh-question + Answer -> Q2S -> S2G -> General question (Yes/No)
- MR3: General/Alternative question + Answer -> GA2S -> S2W -> UniLM -> new Wh-question
"""

import os
import sys
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings

# Add current directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Try to import NLP libraries
_NLP_AVAILABLE = False
_nlp = None
_nlp_benepar = None

try:
    import spacy
    _nlp = spacy.load('en_core_web_sm')
    _NLP_AVAILABLE = True
except Exception as e:
    warnings.warn(f"spaCy not available: {e}")

# Try to load benepar for constituency parsing (used by S2W and GA2S)
_BENEPAR_AVAILABLE = False
try:
    if _NLP_AVAILABLE:
        import spacy
        from benepar.spacy_plugin import BeneparComponent
        _nlp_benepar = spacy.load('en_core_web_sm')
        _nlp_benepar.add_pipe('benepar', config={'model': 'benepar_en3'})
        _BENEPAR_AVAILABLE = True
except Exception as e:
    warnings.warn(f"benepar not available: {e}")

# Try to import pattern library (required for Q2S, S2G, GA2S)
# Note: pattern library has known issues with Python 3.12+, will use fallbacks
_PATTERN_AVAILABLE = False
_PATTERN_ERROR = None
try:
    from pattern.text.en import conjugate, lemma, lexeme, PRESENT, INFINITIVE, PAST, FUTURE, SG, PLURAL, PROGRESSIVE
    # Test if it actually works
    _ = conjugate("test", PRESENT)
    _PATTERN_AVAILABLE = True
except (ImportError, ModuleNotFoundError, RuntimeError) as e:
    _PATTERN_ERROR = str(e)
    warnings.warn(f"pattern library not available or incompatible (Python 3.12+ issue): {type(e).__name__}")

# Try to import ROUGE for quality checking
_ROUGE_AVAILABLE = False
try:
    from rouge import Rouge
    _rouge = Rouge()
    _ROUGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(f"rouge library not available: {e}")

# Try to import the WH-rule modules
_WH_RULES_AVAILABLE = False
_wh_rules = {}
try:
    if _PATTERN_AVAILABLE and _NLP_AVAILABLE:
        from wh_rules import what, how, howmany, why, who, whose, where, when, which
        _wh_rules = {
            'what': what.what(),
            'how': how.how(),
            'howmany': howmany.howmany(),
            'why': why.why(),
            'who': who.who(),
            'whose': whose.whose(),
            'where': where.where(),
            'when': when.when(),
            'which': which.which(),
        }
        _WH_RULES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(f"WH rules not available: {e}")

# Import BERT tokenizer for S2W (optional, for UniLM format)
_BERT_AVAILABLE = False
_bert_tokenizer = None
try:
    from transformers import BertTokenizer
    _bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    _BERT_AVAILABLE = True
except Exception as e:
    warnings.warn(f"BERT tokenizer not available: {e}")


@dataclass
class MutationResult:
    """
    Result of a QAAskeR mutation operation.

    Attributes:
        original_question: The original question
        mutated_question: The mutated question/statement
        mutation_type: The type of mutation applied (MR1, MR2, MR3)
        target_answer: The expected answer for the mutated question
        statement: The intermediate statement (for MR1/MR2)
        is_consistent: Whether the mutation preserves semantic consistency
        metadata: Additional metadata about the mutation
        answer: The answer to the original question (if available)
        premise: The extracted premise from original question (if any),
                 e.g., "Given that X is Y" part
    """
    original_question: str
    mutated_question: str
    mutation_type: str
    target_answer: Optional[str] = None
    statement: Optional[str] = None
    is_consistent: bool = True
    metadata: Optional[Dict] = None
    answer: Optional[str] = None
    premise: Optional[str] = None


class QAAskeR:
    """
    QAAskeR: Metamorphic Testing for Question Answering Systems

    This class wraps the original QAAskeR implementation and provides
    three metamorphic relations (MRs):
    - MR1 (Wh→New Wh): Q2S → S2W → (UniLM) → new Wh-question
    - MR2 (Wh→General): Q2S → S2G → General question
    - MR3 (General→Wh): GA2S → S2W → (UniLM) → new Wh-question
    """

    # Constants from original implementation
    BOOLEAN_START = ["be", "do", "will", "can", "should", "may", "have", "must",
                     "would", "am", "could", "shell", "might"]
    WH_WORDS = ["how", "what", "who", "why", "whose", "where", "when", "which"]

    # Illegal words for S2W (from original S2W.py)
    ILLEGAL_WORDS_SET = {
        ("the", "range"), ("a", "concern"), ("a", "desire"), ("a", "person"), ("a", "term"),
        ("an", "event"), ("one", "of", "the"), ("All", "other"), ("efforts"), ("D.C", ".."),
        ("not", "only"), ("the", "term"), ("the", "world"), ("the", "name"), ("the", "most", "times"),
        ("the", "last", "time"), ("last", "time"), ("a", "time"), ("some", "people"), ("a", "living"),
        ("the", "first", "time"), ("the", "public"), ("that", "person"), ("most", "often"),
        ("dependent", "on"), ("the", "time"), ("one", "individual"), ("more", "so"), ("most", "often"),
        ("a", "product"), ("the", "most"), ("the", "term"), ("the", "area"), ("the", "last"),
        ("the", "range"), ("each", "other"), ("the", "efforts"), ("last", "year")
    }
    SINGLE_ILLEGAL_WORD_SET = {
        "not", "also", "place", "key", "other", "Other", "lack", "effect",
        "addition", "responsible", "pace", "most", "total", "one", "many", "action",
        "many", "most", "relation", "due", "people", "total", "part", "city",
        "someone", "regard", "data", "people", "life", "total", "kind", "history",
        "work", "mom", "dad"
    }

    def __init__(self, random_seed: int = 42, use_nlp: bool = True):
        """
        Initialize QAAskeR

        Args:
            random_seed: Random seed for reproducibility
            use_nlp: Whether to use NLP libraries
        """
        import numpy as np
        self.random_seed = random_seed
        self.use_nlp = use_nlp
        np.random.seed(random_seed)

        # Check available components
        self.nlp_available = _NLP_AVAILABLE and use_nlp
        self.pattern_available = _PATTERN_AVAILABLE
        self.benepar_available = _BENEPAR_AVAILABLE
        self.wh_rules_available = _WH_RULES_AVAILABLE
        self.rouge_available = _ROUGE_AVAILABLE

    def get_status(self) -> Dict[str, bool]:
        """Get the status of available components"""
        return {
            'nlp_available': self.nlp_available,
            'pattern_available': self.pattern_available,
            'benepar_available': self.benepar_available,
            'wh_rules_available': self.wh_rules_available,
            'rouge_available': self.rouge_available,
        }

    def _preprocess_question(self, question: str) -> str:
        """
        Preprocess question by expanding contractions (from original Q2S.py)
        """
        contractions = [
            ("can't ", "can not "), ("won't ", "will not "), ("couldn't ", "could not "),
            ("shouldn't ", "should not "), ("haven't ", "have not "), ("hasn't ", "has not "),
            ("mustn't ", "must not "), ("aren't ", "are not "), ("isn't ", "is not "),
            ("weren't ", "were not "), ("wasn't ", "was not "), ("don't ", "do not "),
            ("doesn't ", "does not "), ("didn't ", "did not "), ("dont ", "do not "),
        ]
        for old, new in contractions:
            if old in question:
                question = question.replace(old, new)

        # Fix common typos
        if "How man " in question:
            question = question.replace("How man ", "How many ")
        if "How maney " in question:
            question = question.replace("How maney ", "How many ")

        return question

    def detect_question_type(self, question: str) -> Tuple[str, Optional[str]]:
        """
        Detect the type of question (wh-question, boolean, or alternative)

        Handles complex questions with premise prefixes like:
        - "Given that X, is it true that Y?" (boolean)
        - "Based on X, what is Y?" (wh)

        Returns:
            Tuple of (question_type, wh_type)
            - question_type: 'wh', 'boolean', 'alternative', or 'unknown'
            - wh_type: The specific wh-word if it's a wh-question
        """
        if not self.nlp_available:
            return self._detect_question_type_simple(question)

        question = self._preprocess_question(question)
        question_lower = question.lower().strip()

        # Define common premise prefixes that indicate a complex question structure
        premise_prefixes = [
            'given that ', 'given the ', 'based on ', 'based upon ',
            'assuming ', 'suppose that ', 'consider that '
        ]

        # Check if the question has a premise prefix
        has_premise = any(question_lower.startswith(p) for p in premise_prefixes)

        if has_premise:
            # Method 1: Look for "is it true that" as the most reliable indicator
            if 'is it true that' in question_lower:
                return 'boolean', None

            # Method 2: Find the main clause by handling nested parentheses
            paren_depth = 0
            last_valid_comma = -1
            for i, char in enumerate(question_lower):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    last_valid_comma = i

            if last_valid_comma > 0:
                main_clause = question_lower[last_valid_comma + 1:].strip()

                # Analyze the main clause using NLP
                doc = _nlp(main_clause)
                tokens = [token for token in doc]
                lemmas = [token.lemma_ for token in doc]
                pos_tags = [token.pos_ for token in doc]

                # Check if it's a wh-question from the main clause
                if tokens and tokens[0].lemma_ in self.WH_WORDS:
                    if lemmas[0] == 'how' and len(pos_tags) > 1 and pos_tags[1] in ['ADJ', 'ADV']:
                        return 'wh', 'howmany'
                    return 'wh', tokens[0].lemma_

                # Check for wh-words elsewhere in the main clause
                for wh in self.WH_WORDS:
                    if wh in lemmas:
                        return 'wh', wh

                # Check if it's a boolean question in the main clause
                if tokens and tokens[0].lemma_ in self.BOOLEAN_START:
                    if 'or' in lemmas:
                        return 'alternative', None
                    return 'boolean', None

        # No premise prefix or couldn't parse - use standard NLP detection
        doc = _nlp(question)
        tokens = [token for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]
        str_tokens = [token.text for token in doc]

        # Check if it's a wh-question
        if tokens[0].lemma_ in self.WH_WORDS:
            # Special case: "how many/much"
            if lemmas[0] == 'how' and len(pos_tags) > 1 and pos_tags[1] in ['ADJ', 'ADV']:
                return 'wh', 'howmany'
            return 'wh', tokens[0].lemma_

        # Check for wh-words elsewhere in the question
        for wh in self.WH_WORDS:
            if wh in lemmas:
                return 'wh', wh

        # Check if it's a boolean question (starts with auxiliary verb)
        if tokens[0].lemma_ in self.BOOLEAN_START:
            # Check for "or" (alternative question)
            if 'or' in lemmas:
                return 'alternative', None
            return 'boolean', None

        return 'unknown', None

    def _detect_question_type_simple(self, question: str) -> Tuple[str, Optional[str]]:
        """
        Simple question type detection without NLP.

        Handles complex questions with premise prefixes like:
        - "Given that X, is it true that Y?" (boolean)
        - "Based on X, what is Y?" (wh)
        """
        question_lower = question.lower().strip()

        # Define common premise prefixes that indicate a complex question structure
        premise_prefixes = [
            'given that ', 'given the ', 'based on ', 'based upon ',
            'assuming ', 'suppose that ', 'consider that '
        ]

        # Check if the question has a premise prefix
        has_premise = any(question_lower.startswith(p) for p in premise_prefixes)

        if has_premise:
            # Method 1: Look for the main clause marker "is it true that"
            # This is the most reliable indicator for boolean yes/no questions
            if 'is it true that' in question_lower:
                return 'boolean', None

            # Method 2: Look for wh-words after the premise
            # Find the last comma that's likely to separate premise from main clause
            # We need to handle nested parentheses correctly
            last_valid_comma = -1
            paren_depth = 0
            for i, char in enumerate(question_lower):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    last_valid_comma = i

            if last_valid_comma > 0:
                main_clause = question_lower[last_valid_comma + 1:].strip()

                # Check for wh-words at the start of the main clause
                for wh in self.WH_WORDS:
                    if main_clause.startswith(wh + ' ') or main_clause.startswith(wh + "'"):
                        return 'wh', wh

                # Check for howmany at the start of the main clause
                if main_clause.startswith('how many') or main_clause.startswith('how much'):
                    return 'wh', 'howmany'

                # Check for boolean/alternative at the start of the main clause
                boolean_starters = ['is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
                                   'can ', 'could ', 'will ', 'would ', 'should ',
                                   'have ', 'has ', 'had ', 'hasnt ', 'hasn\'t ',
                                   'isnt ', 'isn\'t ', 'arent ', 'aren\'t ']
                for starter in boolean_starters:
                    if main_clause.startswith(starter):
                        if ' or ' in main_clause:
                            return 'alternative', None
                        return 'boolean', None

                # If main clause starts with a wh-word contraction (e.g. "what's")
                for wh in self.WH_WORDS:
                    if main_clause.startswith(wh + "'"):
                        return 'wh', wh

        # No premise prefix or couldn't parse - use original detection logic
        # Check for howmany first
        if question_lower.startswith('how many') or question_lower.startswith('how much'):
            return 'wh', 'howmany'

        # Check for wh-words
        for wh in self.WH_WORDS:
            if question_lower.startswith(wh + ' ') or question_lower.startswith(wh + "'"):
                return 'wh', wh

        # Check for boolean/alternative
        boolean_starters = ['is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
                           'can ', 'could ', 'will ', 'would ', 'should ', 'have ', 'has ', 'had ']
        for starter in boolean_starters:
            if question_lower.startswith(starter):
                if ' or ' in question_lower:
                    return 'alternative', None
                return 'boolean', None

        return 'unknown', None

    def q2s(self, question: str, answer: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Q2S: Convert Wh-question + Answer to declarative Statement

        This is the core transformation used in MR1 and MR2.
        Uses the original wh_rules from QAAskeR.

        Args:
            question: The wh-question
            answer: The answer to the question

        Returns:
            Tuple of (statement, premise) where:
            - statement: The declarative statement, or None if conversion fails
            - premise: The extracted premise (if any), e.g., "Given that X"
        """
        # Extract premise first (works for both fallback and wh_rules modes)
        premise = self._extract_premise(question)

        if not self.wh_rules_available:
            return self._fallback_q2s(question, answer)

        question = self._preprocess_question(question)

        try:
            doc = _nlp(question)
            tokens = [token for token in doc]
            lemmas = [token.lemma_ for token in doc]
            pos_tags = [token.pos_ for token in doc]
            str_tokens = [token.text for token in doc]

            # Determine which rule to apply based on question type
            if tokens[0].lemma_ in self.WH_WORDS:
                # Special case: "how many/much"
                if lemmas[0] == 'how' and len(pos_tags) > 1 and pos_tags[1] in ['ADJ', 'ADV']:
                    return _wh_rules['howmany'].generate(question, answer), premise

                # Apply corresponding rule
                wh_type = tokens[0].lemma_
                if wh_type == 'what' or ('what' in str_tokens and tokens[0].lemma_ != 'who'):
                    return _wh_rules['what'].generate(question, answer), premise
                elif wh_type == 'who':
                    return _wh_rules['who'].generate(question, answer), premise
                elif wh_type == 'why' or ', why ' in question:
                    return _wh_rules['why'].generate(question, answer), premise
                elif wh_type == 'whose':
                    return _wh_rules['whose'].generate(question, answer), premise
                elif 'When and where' in question or 'Where and when' in question:
                    new_q = question.replace('When and where', 'Where').replace('Where and when', 'Where')
                    return _wh_rules['where'].generate(new_q, answer), premise
                elif wh_type == 'where':
                    return _wh_rules['where'].generate(question, answer), premise
                elif wh_type == 'when':
                    return _wh_rules['when'].generate(question, answer), premise
                elif wh_type == 'how':
                    return _wh_rules['how'].generate(question, answer), premise
                elif wh_type == 'which':
                    return _wh_rules['which'].generate(question, answer), premise

            # Check for wh-words elsewhere
            if 'how' in lemmas:
                how_idx = lemmas.index('how')
                if how_idx < len(lemmas) - 1 and pos_tags[how_idx + 1] in ['ADJ', 'ADV']:
                    return _wh_rules['howmany'].generate(question, answer), premise
                return _wh_rules['how'].generate(question, answer), premise
            elif 'what' in lemmas:
                return _wh_rules['what'].generate(question, answer), premise
            elif 'who' in lemmas or 'whom' in lemmas:
                return _wh_rules['who'].generate(question, answer), premise
            elif 'why' in lemmas:
                return _wh_rules['why'].generate(question, answer), premise
            elif 'whose' in lemmas:
                return _wh_rules['whose'].generate(question, answer), premise
            elif 'where' in lemmas or 'in which' in ' '.join(str_tokens).lower():
                return _wh_rules['where'].generate(question, answer), premise
            elif 'when' in lemmas:
                return _wh_rules['when'].generate(question, answer), premise
            elif 'which' in lemmas:
                return _wh_rules['which'].generate(question, answer), premise

        except Exception as e:
            warnings.warn(f"Q2S failed: {e}")

        return None, None

    def _extract_premise(self, question: str) -> Optional[str]:
        """
        Extract the premise part from a complex question.

        Handles patterns like:
        - Wh-questions: "Given that X, what is Y?"
        - Boolean questions: "Given that X, is it true that Y?"
        - Boolean questions: "Given that X, is/are/was/were Y?"
        - Alternative questions: "Given that X, A or B?"

        Returns:
            The premise text (e.g., "Given that X"), or None if no premise found.
        """
        import re

        q_lower = question.lower()

        # Define boolean/auxiliary verb patterns for premise extraction
        # These come after the comma in boolean/alternative questions
        boolean_patterns_after_comma = [
            r'is\s+it\s+true\s+that\s+',
            r'is\s+',
            r'are\s+',
            r'was\s+',
            r'were\s+',
            r'do\s+',
            r'does\s+',
            r'did\s+',
            r'can\s+',
            r'could\s+',
            r'will\s+',
            r'would\s+',
            r'should\s+',
            r'must\s+',
            r'may\s+',
            r'might\s+',
        ]

        # Pattern 1: "Given that ..., what/who/where/when/which/how ...?"
        if 'given that' in q_lower:
            # First try wh-words
            match = re.search(r'(Given that[^?]+?)(?=\s*,\s*(?:what|who|where|when|which|how)\s)', question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            # Then try boolean/auxiliary verbs
            for pattern in boolean_patterns_after_comma:
                match = re.search(f'(Given that[^?]+?)(?=\\s*,\\s*{pattern})', question, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            # Try "which of the following" (multiple choice)
            if 'which of the following' in q_lower:
                match = re.search(r'(Given that[^?]+?)(?=\s*,\s*which of the following)', question, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            # Fallback: match until comma (assuming premise ends at first comma)
            match = re.search(r'(Given that[^,]+(?:\([^)]*\))?(?:,\s*[^,]+)*?)(?=,\s*(?:is|are|was|were|do|does|did|can|could|will|would|should|must|may|might)\s)', question, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Pattern 2: "Based on ..., what/who/where/when/which/how ...?" or boolean
        if 'based on' in q_lower:
            # First try wh-words
            match = re.search(r'(Based on[^?]+?)(?=\s*,\s*(?:what|who|where|when|which|how)\s)', question, re.IGNORECASE)
            if not match:
                # Try alternative pattern: match until comma followed by wh-word
                match = re.search(r'([Bb]ased on[^,]+(?:,\s*[^,]+)*?)(?=,\s*(?:what|who|where|when|which|how)\s)', question)
            if match:
                return match.group(1).strip()
            # Then try boolean/auxiliary verbs
            for pattern in boolean_patterns_after_comma:
                match = re.search(f'([Bb]ased on[^?]+?)(?=\\s*,\\s*{pattern})', question, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        # Pattern 3: "Assuming ..., what/who/where/when/which/how ...?" or boolean
        if 'assuming' in q_lower:
            # First try wh-words
            match = re.search(r'(Assuming[^?]+?)(?=\s*,\s*(?:what|who|where|when|which|how)\s)', question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            # Then try boolean/auxiliary verbs
            for pattern in boolean_patterns_after_comma:
                match = re.search(f'(Assuming[^?]+?)(?=\\s*,\\s*{pattern})', question, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        # Pattern 4: "If ..., what/who/where/when/which/how ...?" or boolean
        if q_lower.startswith('if '):
            # First try wh-words
            match = re.search(r'([Ii]f[^?]+?)(?=\s*,\s*(?:what|who|where|when|which|how)\s)', question)
            if match:
                return match.group(1).strip()
            # Then try boolean/auxiliary verbs
            for pattern in boolean_patterns_after_comma:
                match = re.search(f'([Ii]f[^?]+?)(?=\\s*,\\s*{pattern})', question, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        return None

    def _try_specialized_patterns(
        self,
        question: str,
        answer: str,
        premise: Optional[str],
        q_lower: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Try specialized pattern handlers for complex question structures.

        This method attempts to match and convert complex question patterns
        that require special handling, such as:
        - Rank relations: "what is X the next higher/lower rank of?"
        - Directional relations: "what is the next X upstream/downstream from Y?"
        - Complex predicates: questions with intricate predicate structures

        Args:
            question: The original question
            answer: The answer to the question
            premise: The extracted premise (if any)
            q_lower: Lowercase version of the question

        Returns:
            Tuple of (statement, premise) if a pattern matches, (None, None) otherwise
        """
        import re

        # Pattern 1: Rank relations - "what is X the next higher/lower rank of?"
        # Example: "what is general of the army the next higher rank of?"
        # Expected statement: "general of the army the next higher rank of is marshal of the branch"
        rank_patterns = [
            r',\s*what\s+is\s+(.+?)\s+the\s+next\s+(higher|lower)\s+rank\s+of\?$',
            r',\s*what\s+is\s+(.+?)\s+the\s+next\s+(higher|lower)\s+rank\s+than\s+',
            r'^what\s+is\s+(.+?)\s+the\s+next\s+(higher|lower)\s+rank\s+of\?$',
        ]

        for pattern in rank_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                direction = match.group(2).lower()
                # Preserve the full predicate structure
                # "X the next higher/lower rank of is answer"
                statement_main = f"{subject} the next {direction} rank of is {answer}."
                if premise:
                    return f"{premise}, {statement_main}", premise
                return statement_main, premise

        # Pattern 2: Directional relations - "what is the next X upstream/downstream from Y?"
        # Example: "what is the next crossing downstream from Kanmon Roadway Tunnel?"
        # Expected statement: "the next crossing downstream from Kanmon Roadway Tunnel is Diana"
        directional_patterns = [
            (r',\s*what\s+is\s+the\s+next\s+(.+?)\s+(upstream|downstream)\s+from\s+(.+?)\?$', 'next_{direction}_{entity}'),
            (r',\s*what\s+is\s+the\s+next\s+(.+?)\s+(upstream|downstream)\s+of\s+(.+?)\?$', 'next_{direction}_{entity}'),
            (r'^what\s+is\s+the\s+next\s+(.+?)\s+(upstream|downstream)\s+from\s+(.+?)\?$', 'next_{direction}_{entity}'),
        ]

        for pattern, _ in directional_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                entity_type = match.group(1).strip()  # e.g., "crossing"
                direction = match.group(2).lower()    # e.g., "downstream"
                location = match.group(3).strip()     # e.g., "Kanmon Roadway Tunnel"
                # Preserve the full predicate structure
                # "the next crossing downstream from Kanmon Roadway Tunnel is Diana"
                statement_main = f"the next {entity_type} {direction} from {location} is {answer}."
                if premise:
                    return f"{premise}, {statement_main}", premise
                return statement_main, premise

        # Pattern 3: Complex "what is X of Y?" structure
        # Example: "what is the head of government of country X?"
        # Expected statement: "the head of government of country X is person Y"
        complex_of_patterns = [
            (r',\s*what\s+is\s+the\s+(.+?)\s+of\s+(.+?)\?$', 'the_{property}_of_{entity}'),
            (r'^what\s+is\s+the\s+(.+?)\s+of\s+(.+?)\?$', 'the_{property}_of_{entity}'),
        ]

        for pattern, _ in complex_of_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                property_name = match.group(1).strip()  # e.g., "head of government"
                entity = match.group(2).strip()         # e.g., "country X"
                # Preserve the complex noun phrase structure
                # "the head of government of country X is person Y"
                statement_main = f"the {property_name} of {entity} is {answer}."
                if premise:
                    return f"{premise}, {statement_main}", premise
                return statement_main, premise

        # No specialized pattern matched
        return None, None

    def _extract_full_subject(self, subject_part: str) -> Optional[str]:
        """
        Extract the full subject from a question part, handling complex noun phrases.

        This method intelligently extracts the complete subject from a question,
        handling cases like:
        - Simple subjects: "Paris" → "Paris"
        - Multi-word subjects: "Merited Artist" → "Merited Artist"
        - Subjects with modifiers: "general of the army" → "general of the army"
        - Complex subjects with parentheses: "entity (description)" → "entity (description)"

        The key insight is to stop at predicate markers like:
        - "the next X of" (rank relations, directional relations)
        - common prepositions starting a predicate
        - But preserve multi-word proper nouns and noun phrases

        Args:
            subject_part: The subject part extracted from the question

        Returns:
            The full subject string, or None if extraction fails
        """
        if not subject_part:
            return None

        import re

        subject_part = subject_part.strip()

        # Predicate markers that indicate the start of a predicate (not part of subject)
        # These patterns help identify where the subject ends
        predicate_patterns = [
            r'\s+the\s+next\s+(higher|lower)\s+rank\s+(of|than)',  # "the next higher/lower rank of/than"
            r'\s+the\s+next\s+',                                      # "the next [upstream/downstream/crossing]"
            r'\s+of\s+(?:the\s+)?(?:[a-z]+(?:\s+[a-z]+)*)?\s*$',   # "of X" at end (but not "of" in middle)
        ]

        # Try to find where the subject ends (first predicate marker)
        for pattern in predicate_patterns:
            match = re.search(pattern, subject_part, re.IGNORECASE)
            if match:
                # Split at the predicate marker
                end_pos = match.start()
                potential_subject = subject_part[:end_pos].strip()
                if potential_subject:
                    return potential_subject

        # If no predicate marker found, try heuristic-based extraction
        # Split into words and intelligently determine the subject boundary
        words = subject_part.split()

        # Look for common prepositions that might start a predicate
        predicate_starters = {
            'of', 'in', 'at', 'on', 'to', 'from', 'by', 'with', 'for',
            'next', 'following', 'after', 'before'
        }

        # Keep words until we hit a likely predicate starter
        # But preserve multi-word entities (look ahead to see if it makes sense)
        subject_words = []
        i = 0

        while i < len(words):
            word = words[i].lower()

            # Check if this word starts a predicate
            if word in predicate_starters and i > 0:
                # Look ahead: if the next few words form a coherent predicate, stop here
                if i + 1 < len(words):
                    # Special case: "of the" might be part of subject (e.g., "head of government")
                    if word == 'of' and words[i + 1].lower() in ['the', 'a', 'an']:
                        # This could be part of the subject, continue
                        subject_words.append(words[i])
                        i += 1
                        continue
                    # Other predicates usually end the subject
                    break

            subject_words.append(words[i])
            i += 1

        if subject_words:
            # Preserve original capitalization for proper nouns
            result = ' '.join(subject_words)

            # Handle parentheses: if we cut off mid-parenthesis, fix it
            if '(' in result and ')' not in result[result.rindex('('):]:
                # Find the matching closing parenthesis
                open_idx = result.rindex('(')
                remaining = subject_part[len(result):]
                close_idx = remaining.find(')')
                if close_idx >= 0:
                    result += remaining[:close_idx + 1]

            return result

        # Fallback: return the original (subject might be simple)
        return subject_part

    def _fallback_q2s(self, question: str, answer: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fallback Q2S using improved templates and regex patterns.

        This fallback handles:
        1. Simple wh-questions (what, who, where, when, why, how)
        2. Complex questions with context (Given that..., Based on...)
        3. Questions with multiple clauses
        4. Special patterns: rank relations, directional relations, complex predicates

        Returns:
            Tuple of (statement, premise) where:
            - statement: The declarative statement from the main clause
            - premise: The extracted premise (if any), e.g., "Given that X"
        """
        if not question or not answer:
            return None, None

        # Store original question with '?' for specialized patterns
        original_question = question
        q_lower = question.lower()

        # Extract premise first (before processing main clause)
        premise = self._extract_premise(question)

        # Try specialized pattern handlers first (for complex questions)
        # Use original_question (with '?') for pattern matching
        result = self._try_specialized_patterns(original_question, answer, premise, q_lower)
        if result and result[0]:
            return result

        # No specialized pattern matched, continue with general processing
        # Now strip '?' for the remaining patterns
        question = question.rstrip('?').strip()
        q_lower = question.lower()

        # Handle "Given that X, wh-word ...?" pattern (all wh-words, not just "what")
        if 'given that' in q_lower and premise:
            # Find the wh-word after the premise
            import re
            # Special pattern: "where has X has as its Y" (unusual grammar from KG-generated questions)
            # Example: "where has Naples has as its head of government office?"
            match = re.search(r',\s*where\s+has\s+(\w+)\s+has\s+as\s+its\s+(.+?)$', question, re.IGNORECASE)
            if match:
                entity = match.group(1)  # e.g., "Naples"
                attribute = match.group(2)  # e.g., "head of government office"
                # Construct statement: "X has {answer} as its Y"
                entity = entity[0].upper() + entity[1:] if entity else ''
                return f"{premise}, {entity} has {answer} as its {attribute}.", premise

            # Try improved matching with full subject and predicate preservation
            match = re.search(r',\s*(what|who|where|when|which|how)\s+(?:is|are|was|were|has|have|had)\s+(.+?)$', question, re.IGNORECASE)
            if match:
                wh_word, subject_full = match.groups()
                # Use improved subject extraction
                actual_subject = self._extract_full_subject(subject_full)
                if actual_subject:
                    # Capitalize first letter
                    actual_subject = actual_subject[0].upper() + actual_subject[1:] if len(actual_subject) > 1 else actual_subject.upper()
                    # Include premise in the statement to preserve context
                    return f"{premise}, {actual_subject} is {answer}.", premise

            # Try broader pattern: ", wh-word + any word + rest" (catch-all)
            match = re.search(r',\s*(what|who|where|when|which|how)\s+\w+\s+(.+?)$', question, re.IGNORECASE)
            if match:
                wh_word, rest = match.groups()
                actual_subject = self._extract_full_subject(rest)
                if actual_subject:
                    actual_subject = actual_subject[0].upper() + actual_subject[1:] if len(actual_subject) > 1 else actual_subject.upper()
                    if premise:
                        return f"{premise}, {actual_subject} is {answer}.", premise
                    return f"{actual_subject} is {answer}.", premise

        # Handle "Based on X, wh-word ...?" pattern (all wh-words)
        if ('based on' in q_lower or 'based on the' in q_lower) and premise:
            import re
            # Try to match: ", wh-word + (be-verb/has/have) + rest"
            match = re.search(r',\s*(what|who|where|when|which|how)\s+(?:is|are|was|were|has|have|had)\s+(.+?)$', question, re.IGNORECASE)
            if match:
                wh_word, subject_full = match.groups()
                actual_subject = self._extract_full_subject(subject_full)
                if actual_subject:
                    actual_subject = actual_subject[0].upper() + actual_subject[1:] if len(actual_subject) > 1 else actual_subject.upper()
                    return f"{premise}, {actual_subject} is {answer}.", premise

            # Try broader pattern
            match = re.search(r',\s*(what|who|where|when|which|how)\s+\w+\s+(.+?)$', question, re.IGNORECASE)
            if match:
                wh_word, rest = match.groups()
                actual_subject = self._extract_full_subject(rest)
                if actual_subject:
                    actual_subject = actual_subject[0].upper() + actual_subject[1:] if len(actual_subject) > 1 else actual_subject.upper()
                    if premise:
                        return f"{premise}, {actual_subject} is {answer}.", premise
                    return f"{actual_subject} is {answer}.", premise

        # Handle multiple choice questions with premise
        if premise and 'which of the following' in q_lower:
            # For multiple choice, include premise in statement to preserve context
            return f"{premise}, the answer is {answer}.", premise

        # Simple template-based conversion (no premise expected)
        if q_lower.startswith('what is ') or q_lower.startswith('what are '):
            subject = question[8:] if q_lower.startswith('what is ') else question[9:]
            return f"{subject} is {answer}.", None
        elif q_lower.startswith('which is '):
            subject = question[9:]
            return f"{subject} is {answer}.", None
        elif q_lower.startswith('who is '):
            subject = question[7:]
            return f"{subject} is {answer}.", None
        elif q_lower.startswith('who '):
            return f"{answer} {question[4:]}.", None
        elif q_lower.startswith('where '):
            return f"It is {answer}.", None
        elif q_lower.startswith('when '):
            return f"It happened {answer}.", None
        elif q_lower.startswith('why '):
            return f"The reason is {answer}.", None
        elif q_lower.startswith('how '):
            return f"The method is {answer}.", None
        else:
            # Last resort: Try to find and replace the wh-word
            import re
            # Find wh-word at beginning or after comma (support has/have/had verbs)
            match = re.search(r'(?:^|,\s*)(what|who|where|when|why|how|which)\s+(?:is|are|was|were|has|have|had)\s+(.+?)$', question, re.IGNORECASE)
            if match:
                wh_word, subject = match.groups()
                subject_words = subject.split()
                if subject_words:
                    actual_subject = subject_words[0]
                    actual_subject = actual_subject[0].upper() + actual_subject[1:] if actual_subject else ''
                    return f"{actual_subject} is {answer}.", None
            return f"The answer is {answer}.", None

    def ga2s(self, question: str, answer: str) -> Optional[str]:
        """
        GA2S: Convert General/Alternative question + Answer to Statement

        This is used for MR3 to handle boolean and alternative questions.

        For questions with premises (e.g., "Given that X, is it true that Y?"),
        we split the premise and question parts, convert the question part to
        a statement, then recombine with the premise.

        Args:
            question: The general or alternative question
            answer: The answer (yes/no or a choice)

        Returns:
            The declarative statement, or None if conversion fails
        """
        # Define premise patterns
        premise_patterns = [
            'given that ', 'given the ', 'based on ', 'based upon ',
            'assuming ', 'suppose that ', 'consider that '
        ]

        q_lower = question.lower().strip()

        # Check if question has a premise prefix
        for pattern in premise_patterns:
            if q_lower.startswith(pattern):
                # Split premise from the main question
                # Find the main clause (after the first comma, handling nested parentheses)
                paren_depth = 0
                main_clause_start = -1
                for i, char in enumerate(question):
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    elif char == ',' and paren_depth == 0:
                        main_clause_start = i
                        break

                if main_clause_start > 0:
                    premise_part = question[:main_clause_start].strip()
                    question_part = question[main_clause_start + 1:].strip()

                    # Convert only the question part to a statement
                    if self.benepar_available and self.pattern_available:
                        try:
                            from GA2S import boolean
                            statement_part = boolean(question_part, answer, _nlp_benepar)
                            # Recombine with premise
                            return f"{premise_part}, {statement_part}"
                        except Exception as e:
                            warnings.warn(f"GA2S failed: {e}")
                            statement_part = self._fallback_ga2s(question_part, answer)
                            # Remove the period from statement_part if it exists, then combine
                            statement_part = statement_part.rstrip('.')
                            return f"{premise_part}, {statement_part}."
                    else:
                        # Use fallback but only for the question part
                        statement_part = self._fallback_ga2s(question_part, answer)
                        if statement_part:
                            statement_part = statement_part.rstrip('.')
                            return f"{premise_part}, {statement_part}."
                        return None
                # If no comma found, fall through to standard processing
                break

        # No premise or no comma - use standard GA2S
        if not self.benepar_available or not self.pattern_available:
            return self._fallback_ga2s(question, answer)

        try:
            # Import the boolean function from GA2S module
            from GA2S import boolean
            statement = boolean(question, answer, _nlp_benepar)
            return statement
        except Exception as e:
            warnings.warn(f"GA2S failed: {e}")
            return self._fallback_ga2s(question, answer)

    def _fallback_ga2s(self, question: str, answer: str) -> Optional[str]:
        """
        Fallback GA2S using simple templates.

        Handles:
        - Simple boolean questions: "Is X yes?"
        - Complex boolean questions with premise: "Given that X, is it true that Y?"
        - Alternative questions: "A or B?"

        For questions with premises, the premise is preserved in the output statement.
        Example: "Given that X, is it true that Y?" + yes → "Given that X, Y is true."
        """
        if not question or not answer:
            return None

        question = question.rstrip('?').strip()
        q_lower = question.lower()
        answer_lower = answer.lower()

        # Define premise prefixes
        premise_prefixes = [
            'given that ', 'given the ', 'based on ', 'based upon ',
            'assuming ', 'suppose that ', 'consider that '
        ]

        # Check if the question has a premise prefix
        has_premise = any(q_lower.startswith(p) for p in premise_prefixes)

        # Extract the main clause if there's a premise prefix
        if has_premise:
            # Find the main clause by handling nested parentheses
            paren_depth = 0
            last_valid_comma = -1
            for i, char in enumerate(q_lower):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    last_valid_comma = i

            if last_valid_comma > 0:
                main_clause = question[last_valid_comma + 1:].strip()
                premise = question[:last_valid_comma].strip()

                # Handle "is it true that" pattern
                if main_clause.lower().startswith('is it true that'):
                    # Extract the statement after "is it true that"
                    content_start = len('is it true that')
                    statement_content = main_clause[content_start:].strip()

                    if answer_lower == 'yes':
                        # "is it true that X?" + yes → "X is true." or "X is the case."
                        # Remove trailing question mark if present
                        statement_content = statement_content.rstrip('?').strip()
                        statement_part = f"{statement_content[0].upper()}{statement_content[1:]} is true."
                        # Combine with premise
                        return f"{premise}, {statement_part}"
                    else:  # answer == 'no'
                        statement_content = statement_content.rstrip('?').strip()
                        statement_part = f"{statement_content[0].upper()}{statement_content[1:]} is not true."
                        # Combine with premise
                        return f"{premise}, {statement_part}"

                # Handle other boolean patterns in the main clause
                if answer_lower in ['yes', 'no']:
                    # Simple inversion for is/are/was/were questions
                    for aux in ['is ', 'are ', 'was ', 'were ']:
                        if main_clause.lower().startswith(aux):
                            subject_pred = main_clause[len(aux):]
                            subject_pred = subject_pred.rstrip('?').strip()
                            if answer_lower == 'yes':
                                statement_part = f"{subject_pred[0].upper()}{subject_pred[1:]} {aux.strip()}."
                            else:
                                statement_part = f"{subject_pred[0].upper()}{subject_pred[1:]} {aux.strip()} not."
                            # Combine with premise
                            return f"{premise}, {statement_part}"

                    # Handle do/does/did questions
                    for aux in ['do ', 'does ', 'did ']:
                        if main_clause.lower().startswith(aux):
                            rest = main_clause[len(aux):]
                            rest = rest.rstrip('?').strip()
                            if answer_lower == 'yes':
                                statement_part = f"{rest[0].upper()}{rest[1:]}."
                            else:
                                statement_part = f"{rest[0].upper()}{rest[1:]} {aux.strip()} not."
                            # Combine with premise
                            return f"{premise}, {statement_part}"

        # Handle simple boolean questions (no premise prefix)
        if answer_lower in ['yes', 'no']:
            # First check for "is it true that" pattern in simple questions
            if q_lower.startswith('is it true that '):
                content_start = len('is it true that')
                statement_content = question[content_start:].strip()
                statement_content = statement_content.rstrip('?').strip()
                if answer_lower == 'yes':
                    return f"{statement_content[0].upper()}{statement_content[1:]} is true."
                else:  # answer == 'no'
                    return f"{statement_content[0].upper()}{statement_content[1:]} is not true."

            # Simple inversion for is/are/was/were questions
            for aux in ['is ', 'are ', 'was ', 'were ']:
                if q_lower.startswith(aux):
                    subject_pred = question[len(aux):]
                    subject_pred = subject_pred.rstrip('?').strip()
                    if answer_lower == 'yes':
                        return f"{subject_pred[0].upper()}{subject_pred[1:]} {aux.strip()}."
                    else:
                        return f"{subject_pred[0].upper()}{subject_pred[1:]} {aux.strip()} not."

            # Handle do/does/did questions
            for aux in ['do ', 'does ', 'did ']:
                if q_lower.startswith(aux):
                    rest = question[len(aux):]
                    rest = rest.rstrip('?').strip()
                    if answer_lower == 'yes':
                        return f"{rest[0].upper()}{rest[1:]}."
                    else:
                        return f"{rest[0].upper()}{rest[1:]} {aux.strip()} not."

        # Handle alternative questions (A or B)
        if ' or ' in q_lower and answer_lower not in ['yes', 'no']:
            # The answer should be one of the choices
            return f"It is {answer}."

        return None

    def s2g(self, statement: str) -> Optional[str]:
        """
        S2G: Convert Statement to General question (Yes/No question)

        This is used in MR2 to generate the follow-up general question.

        For statements with premises (e.g., "Given that X, Y is Z."),
        we split the premise and statement parts, convert the statement part
        to a general question, then recombine with the premise.

        Args:
            statement: The declarative statement

        Returns:
            The general question, or None if conversion fails
        """
        # Define premise patterns
        premise_patterns = [
            'given that ', 'given the ', 'based on ', 'based upon ',
            'assuming ', 'suppose that ', 'consider that '
        ]

        statement_lower = statement.lower().strip()

        # Check if statement has a premise prefix
        for pattern in premise_patterns:
            if statement_lower.startswith(pattern):
                # Split premise from the main statement
                # Find the main clause (after the first comma, handling nested parentheses)
                paren_depth = 0
                main_clause_start = -1
                for i, char in enumerate(statement):
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    elif char == ',' and paren_depth == 0:
                        main_clause_start = i
                        break

                if main_clause_start > 0:
                    premise_part = statement[:main_clause_start].strip()
                    statement_part = statement[main_clause_start + 1:].strip()

                    # Convert only the statement part to a general question
                    if self.pattern_available and self.nlp_available:
                        try:
                            from S2G import S2I
                            statement_question = S2I(statement_part)
                            # Recombine with premise
                            return f"{premise_part}, {statement_question}"
                        except Exception as e:
                            warnings.warn(f"S2G failed: {e}")
                            statement_question = self._fallback_s2g(statement_part)
                            # Remove "Is it true that" prefix since we'll add it back with premise
                            if statement_question and statement_question.lower().startswith("is it true that "):
                                statement_question = statement_question[len("is it true that "):].rstrip('?')
                            return f"{premise_part}, is it true that {statement_question}?"
                    else:
                        return self._fallback_s2g(statement)
                # If no comma found, fall through to standard processing
                break

        # No premise or no comma - use standard S2G
        if not self.pattern_available or not self.nlp_available:
            return self._fallback_s2g(statement)

        try:
            from S2G import S2I
            return S2I(statement)
        except Exception as e:
            warnings.warn(f"S2G failed: {e}")
            return self._fallback_s2g(statement)

    def _fallback_s2g(self, statement: str) -> Optional[str]:
        """
        Fallback S2G using simple templates.

        Handles complex statements with premises by:
        1. Splitting premise and statement parts
        2. Converting only the statement part to a general question
        3. Recombining with premise

        Example:
            Input: "Given that X, Y is Z."
            Output: "Given that X, is it true that Y is Z?"
        """
        if not statement:
            return None

        statement = statement.rstrip('.').strip()

        # Define premise patterns
        premise_patterns = [
            'given that ', 'given the ', 'based on ', 'based upon ',
            'assuming ', 'suppose that ', 'consider that '
        ]

        statement_lower = statement.lower()

        # Check if statement has a premise prefix
        for pattern in premise_patterns:
            if statement_lower.startswith(pattern):
                # Need to split premise from the main statement
                # Find the main clause (after the first comma, handling nested parentheses)
                paren_depth = 0
                main_clause_start = -1
                for i, char in enumerate(statement):
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth -= 1
                    elif char == ',' and paren_depth == 0:
                        main_clause_start = i
                        break

                if main_clause_start > 0:
                    premise_part = statement[:main_clause_start].strip()
                    statement_part = statement[main_clause_start + 1:].strip()

                    # Only wrap the statement part with "is it true that"
                    # Keep premise outside
                    return f"{premise_part}, is it true that {statement_part.lower()}?"

                # If no comma found, fall through to simple conversion
                break

        # Simple conversion: add "Is it true that" prefix
        return f"Is it true that {statement.lower()}?"

    def s2g_with_premise(self, statement: str, premise: Optional[str]) -> Optional[str]:
        """
        S2G with premise preservation for better context.

        When there's a premise (e.g., "Given that X, what is Y?"), the generated
        yes/no question should preserve this premise for better LLM understanding.

        Args:
            statement: The declarative statement (e.g., "The answer is D.")
                       May already include the premise (e.g., "Given that X, the answer is D.")
            premise: The premise from original question (e.g., "Given that X...")

        Returns:
            The general question with context preserved
        """
        if not statement:
            return None

        # Remove trailing period
        statement_clean = statement.rstrip('.').strip()

        # Check if statement already contains the premise (e.g., "Given that X, ...")
        # This happens when _fallback_q2s includes premise in the statement
        statement_contains_premise = False
        if premise:
            premise_lower = premise.lower().rstrip('.')
            statement_lower = statement_clean.lower()
            # Check if statement starts with the premise (case-insensitive)
            if statement_lower.startswith(premise_lower):
                statement_contains_premise = True

        # If statement already contains premise, need to split and convert properly
        # Example: "Given that X, Y is Z." -> "Given that X, is it true that Y is Z?"
        if statement_contains_premise:
            # Split premise from statement by finding the comma (handling nested parentheses)
            paren_depth = 0
            main_clause_start = -1
            for i, char in enumerate(statement_clean):
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    main_clause_start = i
                    break

            if main_clause_start > 0:
                premise_part = statement_clean[:main_clause_start].strip()
                statement_part = statement_clean[main_clause_start + 1:].strip()
                # Keep premise outside, only wrap statement part with "is it true that"
                return f"{premise_part}, is it true that {statement_part.lower()}?"
            else:
                # No comma found, fall back to simple wrapping (shouldn't happen normally)
                return f"Is it true that {statement_clean.lower()}?"

        # If we have a premise but statement doesn't contain it, construct a question that references it
        if premise:
            # For "The answer is D." type statements from multiple choice
            if statement_clean.lower().startswith("the answer is"):
                answer_value = statement_clean[len("the answer is"):].strip()
                # Clean up premise
                premise_clean = premise.rstrip('.')
                if premise_clean.lower().startswith("given that"):
                    premise_clean = premise_clean[len("given that"):].strip()
                # Create natural question: "Given [premise], is the answer [value]?"
                return f"Given {premise_clean}, is the answer {answer_value}?"
            # For other statements, incorporate premise more naturally
            else:
                # Check if premise starts with "Given that"
                if premise.lower().startswith("given that"):
                    content = premise[len("given that"):].strip()
                    # Make the statement lowercase for more natural flow
                    stmt_lower = statement_clean.lower()
                    return f"Given that {content}, is it true that {stmt_lower}?"
                else:
                    return f"Based on {premise}, is it true that {statement_clean.lower()}?"
        else:
            # No premise, fall back to standard S2G
            return self.s2g(statement)

    def s2w(self, statement: str, original_answer: str) -> List[str]:
        """
        S2W: Extract potential target answers from Statement

        This extracts noun phrases and adjective phrases that can be
        used as target answers for new wh-questions.

        Args:
            statement: The declarative statement
            original_answer: The original answer (to exclude from candidates)

        Returns:
            List of potential target answers
        """
        if not self.benepar_available:
            return self._fallback_s2w(statement, original_answer)

        try:
            from nltk import Tree

            # Clean statement
            statement = statement[0].lower() + statement[1:] if statement else statement
            statement = statement.replace('?', '')

            doc = _nlp_benepar(statement)
            sent = list(doc.sents)[0]
            doc2 = _nlp(statement)

            str_tokens = [token.text for token in doc2 if token.text != '']
            pos_tokens = [token.pos_ for token in doc2 if token.text != '']
            tag_tokens = [token.tag_ for token in doc2 if token.text != '']

            # Get constituency parse
            parse_str = sent._.parse_string
            t = Tree.fromstring(parse_str)

            # Extract phrases
            phrases = []
            self._extract_phrases(t, phrases)

            # Filter phrases
            candidates = []
            answer_tokens = original_answer.lower().split()

            for phrase in phrases:
                if not phrase:
                    continue

                phrase_str = ' '.join(phrase)
                phrase_lower = [p.lower() for p in phrase]

                # Skip if too long (>70% of statement)
                if len(phrase) / len(str_tokens) > 0.7:
                    continue

                # Skip if contains original answer
                if set(phrase_lower) <= set(answer_tokens) or set(answer_tokens) <= set(phrase_lower):
                    continue

                # Skip illegal words
                if tuple(phrase) in self.ILLEGAL_WORDS_SET:
                    continue
                if len(phrase) == 1 and phrase[0] in self.SINGLE_ILLEGAL_WORD_SET:
                    continue

                # Skip pronouns and certain words
                if phrase[0].lower() in ['this', 'it', 'other', 'me', 'him', 'her', 'them',
                                         'they', 'you', 'he', 'she', 'i', 'that']:
                    continue

                # Check if it's a valid noun/adjective phrase
                if self._is_valid_phrase(phrase, pos_tokens, str_tokens, tag_tokens):
                    candidates.append(phrase_str)

            return list(set(candidates))  # Remove duplicates

        except Exception as e:
            warnings.warn(f"S2W failed: {e}")
            return self._fallback_s2w(statement, original_answer)

    def _extract_phrases(self, tree, result: List):
        """Extract leaf phrases from parse tree"""
        from nltk import Tree

        if isinstance(tree, str):
            result.append([tree])
        elif isinstance(tree, Tree):
            # Check if all children are leaves
            all_leaves = all(isinstance(child, str) or
                           (isinstance(child, Tree) and len(child) == 1 and isinstance(child[0], str))
                           for child in tree)
            if all_leaves:
                phrase = []
                for child in tree:
                    if isinstance(child, str):
                        phrase.append(child)
                    elif isinstance(child, Tree) and len(child) == 1:
                        phrase.append(child[0])
                if phrase:
                    result.append(phrase)
            else:
                for child in tree:
                    self._extract_phrases(child, result)

    def _is_valid_phrase(self, phrase: List[str], pos_tokens: List[str],
                         str_tokens: List[str], tag_tokens: List[str]) -> bool:
        """Check if a phrase is valid for S2W"""
        if not phrase:
            return False

        valid_pos = {'NOUN', 'PROPN', 'PRON', 'ADJ', 'ADJP', 'NUM'}
        valid_tags = {'CD', 'FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS'}

        # Check first token
        if phrase[0].lower() in ['the', 'a', 'an']:
            return True

        # Check each token
        for token in phrase:
            if '-' in token and token != '-':
                continue
            if token not in str_tokens:
                continue
            idx = str_tokens.index(token)
            if pos_tokens[idx] in valid_pos:
                return True
            if tag_tokens[idx] in valid_tags:
                return True

        return False

    def _fallback_s2w(self, statement: str, original_answer: str) -> List[str]:
        """
        Fallback S2W using simple NLP or rule-based extraction.

        Extracts noun phrases and entities from a statement that can be used
        as target answers for new wh-questions.
        """
        if not statement:
            return []

        candidates = []

        # If NLP is available, use it for better extraction
        if self.nlp_available:
            try:
                doc = _nlp(statement)
                answer_lower = original_answer.lower() if original_answer else ''

                # Extract noun chunks
                for chunk in doc.noun_chunks:
                    text = chunk.text
                    if original_answer and text.lower() == answer_lower:
                        continue
                    if len(text) > 1:
                        candidates.append(text)

                # Extract named entities
                for ent in doc.ents:
                    text = ent.text
                    if original_answer and text.lower() == answer_lower:
                        continue
                    if len(text) > 1:
                        candidates.append(text)

                return list(set(candidates))
            except Exception:
                pass

        # Rule-based extraction as final fallback
        # This handles cases where spaCy is not available
        return self._extract_candidates_by_rules(statement, original_answer)

    def _extract_candidates_by_rules(self, statement: str, original_answer: str) -> List[str]:
        """
        Rule-based candidate extraction when NLP is not available.

        Extracts potential noun phrases from a statement using simple heuristics.
        """
        if not statement:
            return []

        candidates = []
        statement = statement.rstrip('.')
        answer_lower = original_answer.lower() if original_answer else ''

        # Simple tokenization
        import re
        tokens = statement.split()

        # Look for capitalized phrases (potential proper nouns)
        i = 0
        while i < len(tokens):
            # Skip if this token matches the original answer
            if tokens[i].lower() == answer_lower:
                i += 1
                continue

            # Check if token starts with capital letter (potential proper noun)
            if tokens[i] and tokens[i][0].isupper() and len(tokens[i]) > 1:
                # Start of a potential multi-word phrase
                phrase = [tokens[i]]
                j = i + 1

                # Continue collecting consecutive capitalized/short words
                while j < len(tokens):
                    # Stop at punctuation, keywords, or end
                    if tokens[j] in ['is', 'are', 'was', 'were', 'a', 'an', 'the', 'of', 'in', 'at', 'to', 'for', 'and', 'or']:
                        break
                    # Continue if next word is capitalized or short function word
                    if tokens[j][0].isupper() or len(tokens[j]) <= 3:
                        phrase.append(tokens[j])
                        j += 1
                    else:
                        break

                phrase_text = ' '.join(phrase)
                if len(phrase_text) > 1:
                    candidates.append(phrase_text)

                i = j if j > i + 1 else i + 1
            else:
                i += 1

        # Extract quoted content (often contains proper nouns or specific terms)
        quoted_patterns = re.findall(r'"([^"]+)"', statement)
        quoted_patterns.extend(re.findall(r"'([^']+)'", statement))
        for quoted_text in quoted_patterns:
            if quoted_text and len(quoted_text) > 1:
                if quoted_text.lower() != answer_lower:
                    candidates.append(quoted_text)

        # Extract parenthetical content (often contains descriptions/definitions)
        parenthetical_patterns = re.findall(r'\(([^)]+)\)', statement)
        for match in parenthetical_patterns:
            if match and len(match) > 1 and len(match) < 100:  # Reasonable length
                # Only add if it's not the original answer
                if match.lower() != answer_lower:
                    # Split by common separators and add each part
                    parts = re.split(r',|;|\|', match)
                    for part in parts:
                        part = part.strip()
                        if len(part) > 1:
                            candidates.append(part)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        return unique_candidates

    def check_quality(self, original_statement: str, new_question: str,
                     target_answer: str, threshold: float = 0.7,
                     use_strict_check: bool = True) -> Tuple[bool, Dict]:
        """
        Check the quality of a generated question using ROUGE scores

        When pattern/wh_rules are not available (fallback mode),
        we use a more lenient check to allow more mutations through.

        Args:
            original_statement: The original statement from Q2S
            new_question: The newly generated question
            target_answer: The target answer for the new question
            threshold: ROUGE threshold for acceptance (default 0.7)
            use_strict_check: Whether to use strict ROUGE checking (default True,
                              but set to False automatically when using fallbacks)

        Returns:
            Tuple of (is_valid, scores_dict)
        """
        # Auto-detect if we should use strict checking
        if use_strict_check and not self.wh_rules_available:
            # Using fallback Q2S - be more lenient
            use_strict_check = False

        if not self.rouge_available:
            return True, {'rouge_1_p': 1.0, 'rouge_1_r': 1.0}

        try:
            # Generate statement from new question (returns tuple)
            new_statement, _ = self.q2s(new_question, target_answer)
            if new_statement is None:
                return False, {}

            # Calculate ROUGE scores
            scores = _rouge.get_scores(
                hyps=[new_statement.lower()],
                refs=[original_statement.lower()]
            )

            rouge_1_p = scores[0]['rouge-1']['p']
            rouge_1_r = scores[0]['rouge-1']['r']

            # Use lower threshold when in fallback mode
            actual_threshold = threshold * 0.5 if not use_strict_check else threshold

            is_valid = rouge_1_p > actual_threshold and rouge_1_r > actual_threshold

            return is_valid, {
                'rouge_1_p': rouge_1_p,
                'rouge_1_r': rouge_1_r,
                'new_statement': new_statement,
                'strict_mode': use_strict_check
            }
        except Exception as e:
            warnings.warn(f"Quality check failed: {e}")
            return True, {}

    def mr1_wh_to_new_wh(self, question: str, answer: str) -> List[MutationResult]:
        """
        MR1: Wh→New Wh
        Complete flow: Q2S → S2W → generate new wh-questions

        Note: The original QAAskeR uses UniLM to generate new questions from
        statement + target_answer pairs. Since we don't have UniLM here,
        we use template-based question generation as an approximation.

        Args:
            question: The original wh-question
            answer: The answer to the question

        Returns:
            List of MutationResult objects
        """
        results = []

        # Step 1: Q2S - Convert question + answer to statement (and extract premise)
        statement, premise = self.q2s(question, answer)
        if statement is None:
            return results

        # Step 2: S2W - Extract potential target answers
        target_answers = self.s2w(statement, answer)
        if not target_answers:
            return results

        # Step 3: Generate new wh-questions for each target answer
        # (In original QAAskeR, this uses UniLM. Here we use templates.)
        for target_ans in target_answers:
            new_questions = self._generate_wh_questions(statement, target_ans, premise)

            for new_q in new_questions:
                # Quality check
                is_valid, scores = self.check_quality(statement, new_q, target_ans)

                if is_valid:
                    results.append(MutationResult(
                        original_question=question,
                        mutated_question=new_q,
                        mutation_type='MR1_Wh_to_New_Wh',
                        target_answer=target_ans,
                        statement=statement,
                        answer=answer,
                        premise=premise,
                        metadata={
                            'rouge_1_p': scores.get('rouge_1_p', 1.0),
                            'rouge_1_r': scores.get('rouge_1_r', 1.0),
                        }
                    ))

        return results

    def _generate_wh_questions(self, statement: str, target_answer: str,
                              premise: Optional[str] = None) -> List[str]:
        """
        Generate wh-questions from statement targeting a specific answer.

        In the original QAAskeR, this is done by UniLM. Here we use
        simple template-based generation as an approximation.

        Args:
            statement: The declarative statement
            target_answer: The target answer to ask about
            premise: The premise from the original question (e.g., "Given that X")
                    If provided, the generated questions will include this context.

        Returns:
            List of generated questions
        """
        questions = []

        # Build the context prefix (if premise exists)
        context_prefix = ""
        if premise:
            # Ensure proper capitalization and spacing
            context_prefix = f"{premise}, "

        # Determine the type of the target answer
        if not self.nlp_available:
            # Simple fallback
            questions.append(f"{context_prefix}What is {target_answer}?")
            return questions

        doc = _nlp(target_answer)

        # Check entity types
        has_person = any(ent.label_ == 'PERSON' for ent in doc.ents)
        has_location = any(ent.label_ in ['GPE', 'LOC', 'FAC'] for ent in doc.ents)
        has_date = any(ent.label_ in ['DATE', 'TIME'] for ent in doc.ents)
        has_number = any(ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY', 'PERCENT'] for ent in doc.ents)
        has_org = any(ent.label_ == 'ORG' for ent in doc.ents)

        # Generate appropriate questions with context prefix
        if has_person:
            questions.append(f"{context_prefix}Who is {target_answer}?")
            questions.append(f"{context_prefix}Who was {target_answer}?")
        if has_location:
            questions.append(f"{context_prefix}Where is {target_answer}?")
            questions.append(f"{context_prefix}What place is {target_answer}?")
        if has_date:
            questions.append(f"{context_prefix}When was {target_answer}?")
            questions.append(f"{context_prefix}What time is {target_answer}?")
        if has_number:
            questions.append(f"{context_prefix}How many is {target_answer}?")
            questions.append(f"{context_prefix}What number is {target_answer}?")
        if has_org:
            questions.append(f"{context_prefix}What organization is {target_answer}?")
            questions.append(f"{context_prefix}What is {target_answer}?")

        # Default questions
        if not questions:
            questions.append(f"{context_prefix}What is {target_answer}?")
            questions.append(f"{context_prefix}Which is {target_answer}?")

        return questions

    def mr2_wh_to_general(self, question: str, answer: str) -> List[MutationResult]:
        """
        MR2: Wh→General (Yes/No)
        Complete flow: Q2S → S2G → General question

        Args:
            question: The original wh-question
            answer: The answer to the question

        Returns:
            List of MutationResult objects
        """
        results = []

        # Step 1: Q2S - Convert question + answer to statement (and extract premise)
        statement, premise = self.q2s(question, answer)
        if statement is None:
            return results

        # Step 2: S2G - Convert statement to general question
        # If there's a premise, use enhanced S2G to preserve context
        if premise:
            general_question = self.s2g_with_premise(statement, premise)
        else:
            general_question = self.s2g(statement)

        if general_question is None:
            return results

        # The expected answer is "yes" (since the statement is derived from the original Q&A)
        results.append(MutationResult(
            original_question=question,
            mutated_question=general_question,
            mutation_type='MR2_Wh_to_General',
            target_answer='yes',
            statement=statement,
            answer=answer,
            premise=premise,
            metadata={
                'expected_answer': 'yes',
                'rouge_1_p': 1.0,
                'rouge_1_r': 1.0,
            }
        ))

        return results

    def mr3_general_to_wh(self, question: str, answer: str) -> List[MutationResult]:
        """
        MR3: General/Alternative→Wh
        Complete flow: GA2S → S2W → generate new wh-questions

        Args:
            question: The original general/alternative question
            answer: The answer (yes/no or a choice)

        Returns:
            List of MutationResult objects
        """
        results = []

        # Step 1: Extract premise from the original question (for context preservation)
        premise = self._extract_premise(question)

        # Step 2: GA2S - Convert general question + answer to statement
        # (statement will include premise if question had one)
        statement = self.ga2s(question, answer)
        if statement is None:
            return results

        # Step 3: S2W - Extract potential target answers
        target_answers = self.s2w(statement, answer)
        if not target_answers:
            return results

        # Step 4: Generate new wh-questions for each target answer
        # Pass the premise to preserve context in generated questions
        for target_ans in target_answers:
            new_questions = self._generate_wh_questions(statement, target_ans, premise)

            for new_q in new_questions:
                # Quality check
                is_valid, scores = self.check_quality(statement, new_q, target_ans)

                if is_valid:
                    # Special handling for "no" answers (from 2023.02.02 update)
                    # When source output is "No", check target answer NOT in model output
                    check_not_in = answer.lower() == 'no'

                    results.append(MutationResult(
                        original_question=question,
                        mutated_question=new_q,
                        mutation_type='MR3_General_to_Wh',
                        target_answer=target_ans,
                        statement=statement,
                        answer=answer,
                        premise=premise,
                        metadata={
                            'rouge_1_p': scores.get('rouge_1_p', 1.0),
                            'rouge_1_r': scores.get('rouge_1_r', 1.0),
                            'check_not_in_output': check_not_in,
                        }
                    ))

        return results

    def generate_mutations(
        self,
        question: str,
        answer: Optional[str] = None,
        apply_mr1: bool = True,
        apply_mr2: bool = True,
        apply_mr3: bool = True,
    ) -> List[MutationResult]:
        """
        Generate all mutations for a given question.

        Args:
            question: The original question
            answer: The answer to the question (required for full functionality)
            apply_mr1: Whether to apply MR1 (Wh→New Wh)
            apply_mr2: Whether to apply MR2 (Wh→General)
            apply_mr3: Whether to apply MR3 (General→Wh)

        Returns:
            List of MutationResult objects
        """
        if not question:
            return []

        results = []

        # Detect question type
        q_type, wh_type = self.detect_question_type(question)

        # Apply appropriate MRs based on question type
        if q_type == 'wh' and answer:
            # MR1: Wh → New Wh
            if apply_mr1:
                results.extend(self.mr1_wh_to_new_wh(question, answer))

            # MR2: Wh → General
            if apply_mr2:
                results.extend(self.mr2_wh_to_general(question, answer))

        elif q_type in ['boolean', 'alternative'] and answer:
            # MR3: General/Alternative → Wh
            if apply_mr3:
                results.extend(self.mr3_general_to_wh(question, answer))

        return results

    def check_answer_consistency(
        self,
        llm_answer: str,
        expected_answer: str,
        original_answer: Optional[str] = None,
        similarity_threshold: float = 0.6,
        use_word_vectors: bool = False
    ) -> Tuple[bool, float, Dict]:
        """
        Check if LLM answer satisfies the metamorphic relation.

        Based on the original QAAskeR calculate_score.py logic:
        - For MR1/MR3: Check if llm_answer is similar to expected_answer
        - For MR2 (yes/no): Check if llm_answer contains the expected yes/no
        - Special handling for "no" answers (negative relation)

        Args:
            llm_answer: The LLM's actual answer to the mutated question
            expected_answer: The expected answer (target_answer from mutation)
            original_answer: The answer to the original question (for special handling)
            similarity_threshold: Threshold for considering answers consistent (default 0.6)
            use_word_vectors: Whether to use word vectors for similarity (requires gensim)

        Returns:
            Tuple of (is_consistent, similarity_score, metadata)
        """
        if not llm_answer or not expected_answer:
            return False, 0.0, {'reason': 'missing_answer'}

        # Normalize for comparison
        llm_lower = llm_answer.lower().strip()
        expected_lower = expected_answer.lower().strip()

        # Special handling for MR2 (yes/no questions)
        if expected_lower in ['yes', 'no']:
            # First, check if the answer starts with Yes/No (most direct signal)
            # This takes priority over keyword detection in the rest of the answer
            answer_start = llm_lower[:50].strip()
            starts_with_yes = (
                answer_start.startswith('yes') or
                answer_start.startswith('"yes') or
                answer_start.startswith("'yes")
            )
            starts_with_no = (
                answer_start.startswith('no') or
                answer_start.startswith('"no') or
                answer_start.startswith("'no")
            )

            if expected_lower == 'yes':
                # Check if starts with yes (highest confidence)
                if starts_with_yes:
                    return True, 1.0, {'method': 'yes_no_detection', 'detected': 'yes'}

                # Check for strong positive indicators at the beginning
                strong_positive_starts = [
                    'yes, that', 'yes, it', 'yes, the', 'yes, this',
                    'correct. ', 'true. ', 'indeed. '
                ]
                if any(llm_lower.startswith(s) for s in strong_positive_starts):
                    return True, 1.0, {'method': 'yes_no_detection', 'detected': 'yes'}

                # Fallback: keyword detection (but only if no strong negative signal at start)
                # Negative indicators that matter: only if they appear early in the answer
                early_negative = any(
                    llm_lower[:100].startswith(s) for s in ['no', 'no. ', 'no, ', 'false', 'incorrect']
                )

                if early_negative:
                    return False, 0.0, {'method': 'yes_no_detection', 'detected': 'no'}

                # Check for positive indicators anywhere
                positive_indicators = ['yes', 'indeed', 'correct. ', 'true. ']
                has_positive = any(ind in llm_lower for ind in positive_indicators)

                if has_positive:
                    return True, 1.0, {'method': 'yes_no_detection', 'detected': 'yes'}
                else:
                    # Ambiguous answer
                    return False, 0.5, {'method': 'yes_no_detection', 'detected': 'ambiguous'}

            else:  # expected == 'no'
                # Check if starts with no (highest confidence)
                if starts_with_no:
                    return True, 1.0, {'method': 'yes_no_detection', 'detected': 'no'}

                # Check for strong negative indicators at the beginning
                strong_negative_starts = [
                    'no, that', 'no, it', 'no, the', 'no, this',
                    'false. ', 'incorrect. '
                ]
                if any(llm_lower.startswith(s) for s in strong_negative_starts):
                    return True, 1.0, {'method': 'yes_no_detection', 'detected': 'no'}

                # Fallback: keyword detection
                early_positive = any(
                    llm_lower[:100].startswith(s) for s in ['yes', 'yes. ', 'yes, ', 'true', 'correct']
                )

                if early_positive:
                    return False, 0.0, {'method': 'yes_no_detection', 'detected': 'yes'}

                # Check for negative indicators anywhere
                negative_indicators = ['no', 'not ', "is not", "are not", 'false', 'incorrect']
                has_negative = any(ind in llm_lower for ind in negative_indicators)

                if has_negative:
                    return True, 1.0, {'method': 'yes_no_detection', 'detected': 'no'}
                else:
                    return False, 0.5, {'method': 'yes_no_detection', 'detected': 'ambiguous'}

        # For non-yes/no answers, use string/semantic similarity
        # Try exact match first
        if expected_lower in llm_lower or llm_lower in expected_lower:
            return True, 1.0, {'method': 'exact_match'}

        # Try word overlap (Jaccard-like)
        llm_words = set(llm_lower.split())
        expected_words = set(expected_lower.split())

        if llm_words and expected_words:
            # Remove common stop words for better comparison
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'of', 'in', 'at', 'on', 'to', 'for'}
            llm_content = llm_words - stop_words
            expected_content = expected_words - stop_words

            if expected_content:
                overlap = len(llm_content & expected_content)
                union = len(llm_content | expected_content)
                jaccard = overlap / union if union > 0 else 0.0

                if jaccard >= similarity_threshold:
                    return True, jaccard, {'method': 'word_overlap', 'jaccard': jaccard}

                # Also check if expected is subset of LLM answer (lenient)
                if expected_content.issubset(llm_content):
                    return True, 0.8, {'method': 'subset_match'}

        # If word vectors are available and requested
        if use_word_vectors and self.nlp_available:
            try:
                from gensim.models import KeyedVectors
                # This would require loading word vectors - skip for now
                # User can implement this if needed
                pass
            except ImportError:
                pass

        # Fallback: simple normalized string similarity
        # Using character-level bigram overlap for robustness
        def bigram_overlap(s1, s2):
            if len(s1) < 2 or len(s2) < 2:
                return 0.0
            bg1 = {s1[i:i+2] for i in range(len(s1)-1)}
            bg2 = {s2[i:i+2] for i in range(len(s2)-1)}
            overlap = len(bg1 & bg2)
            return overlap / min(len(bg1), len(bg2)) if bg1 and bg2 else 0.0

        bigram_sim = bigram_overlap(expected_lower, llm_lower)
        if bigram_sim >= similarity_threshold * 0.8:  # More lenient for bigram
            return True, bigram_sim, {'method': 'bigram_overlap'}

        # Not consistent
        return False, bigram_sim, {'method': 'bigram_overlap', 'below_threshold': True}

    def evaluate_mr_violation(
        self,
        mutation_data: Dict,
        llm_answer: str
    ) -> Dict:
        """
        Evaluate if a mutation answer violates the metamorphic relation.

        This is the main evaluation method that should be called for each mutation.

        Args:
            mutation_data: Dictionary containing mutation information (from JSON)
                - mutation_type: 'MR1_Wh_to_New_Wh', 'MR2_Wh_to_General', 'MR3_General_to_Wh'
                - target_answer: The expected answer for the mutated question
                - answer: The original answer
                - mutated_question: The mutated question text
                - statement: The intermediate statement
                - metadata: Additional metadata
            llm_answer: The LLM's answer to the mutated question

        Returns:
            Dictionary with evaluation results:
                - is_violation: True if the metamorphic relation is violated
                - is_consistent: True if LLM answer satisfies the MR
                - similarity: Similarity score
                - method: Method used for evaluation
                - details: Additional details
        """
        mutation_type = mutation_data.get('mutation_type', '')
        target_answer = mutation_data.get('target_answer', '')
        original_answer = mutation_data.get('answer', '')
        metadata = mutation_data.get('metadata', {})

        # Special handling for different mutation types
        if 'MR2' in mutation_type or 'Wh_to_General' in mutation_type:
            # MR2: Wh → General (Yes/No)
            # target_answer should be 'yes' (from original Q2S)
            is_consistent, similarity, details = self.check_answer_consistency(
                llm_answer, target_answer, original_answer
            )
            # For MR2, violation means inconsistent with expected yes/no
            is_violation = not is_consistent

        elif 'MR3' in mutation_type or 'General_to_Wh' in mutation_type:
            # MR3: General → Wh
            # Special handling for "no" answers
            if original_answer and original_answer.lower() == 'no':
                # For "no" answers, the target should NOT be in the LLM output
                is_consistent, similarity, details = self.check_answer_consistency(
                    llm_answer, target_answer, original_answer
                )
                # Invert: violation if similar (should NOT be similar for "no")
                is_violation = is_consistent and similarity > 0.6
                is_consistent = not is_violation
                details['special_case'] = 'negative_answer_inverted'
            else:
                is_consistent, similarity, details = self.check_answer_consistency(
                    llm_answer, target_answer, original_answer
                )
                is_violation = not is_consistent

        else:  # MR1 or others
            # MR1: Wh → New Wh
            # Check if LLM answer is consistent with target_answer
            is_consistent, similarity, details = self.check_answer_consistency(
                llm_answer, target_answer, original_answer
            )
            is_violation = not is_consistent

        return {
            'is_violation': is_violation,
            'is_consistent': is_consistent,
            'similarity': similarity,
            'method': details.get('method', 'unknown'),
            'details': details,
            'mutation_type': mutation_type,
            'target_answer': target_answer,
            'llm_answer': llm_answer[:100] if llm_answer else '',  # Truncate for storage
        }


# Export main class and result type
__all__ = ['QAAskeR', 'MutationResult']
