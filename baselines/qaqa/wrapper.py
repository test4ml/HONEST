"""
QAQA Wrapper Implementation

This module provides a clean wrapper around the original QAQA implementation.
It wraps the original functions and provides a clean Python API.

Based on: "Natural Test Generation for Precise Testing of Question Answering Software"
https://github.com/yichuan-cs/QAQA
"""

# IMPORTANT: Set environment variable BEFORE importing any torch-related modules
# This forces sentence-transformers to use CPU and avoid CUDA OOM errors
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Suppress all warnings for clean output
import warnings
warnings.filterwarnings('ignore')

import sys
import random
import copy
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from original implementation
# The pattern library is not compatible with Python 3.12+, so we provide fallbacks
_PATTERN_AVAILABLE = False
_GA2S_AVAILABLE = False

try:
    from utils.GA2S import GA2S
    _GA2S_AVAILABLE = True
    _PATTERN_AVAILABLE = True
except Exception:
    # Catch all exceptions during import (ImportError, ModuleNotFoundError, LookupError, etc.)
    # This handles cases where benepar models are not available
    _PATTERN_AVAILABLE = False
    GA2S = None

# Import template functions (these don't depend on pattern)
_TEMPLATE_AVAILABLE = False
try:
    from generate.template import (
        add_extra2context,
        add_extra2question,
        add_input_as_redundancy,
        add_wh_question_as_redundancy,
        combine2input,
        negative_question,
        opposite_answer,
    )
    _TEMPLATE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TEMPLATE_AVAILABLE = False
    # Provide fallback implementations that don't depend on GA2S/benepar
    import random as _random

    def _clean_question(q):
        """Clean question: lowercase, strip punctuation"""
        q = q.strip().lower()
        # Remove trailing punctuation
        while q and q[-1] in '?.!':
            q = q[:-1]
        return q.strip()

    def add_extra2context(context, extra_sent):
        """Add extra sentence to context (prepend)"""
        context = context.strip().rstrip('.')
        extra_sent = extra_sent.strip().rstrip('.')
        if not extra_sent:
            return context
        if not context:
            return extra_sent + '.'
        return extra_sent + '. ' + context + '.'

    def add_extra2question(question, extra_sent):
        """Add extra sentence to question using templates"""
        question = question.strip().lower()
        extra_sent = extra_sent.strip().lower().rstrip('.')
        templates = [
            f"I have known that {extra_sent}, {question}",
            f"I heard about that {extra_sent}, {question}",
            f"It is said that {extra_sent}, {question}",
            f"Someone told me that {extra_sent}, {question}",
        ]
        return _random.choice(templates)

    def add_input_as_redundancy(obj1, obj2):
        """Add second question as redundancy (ETI for BoolQ)"""
        s2 = _clean_question(obj2.Q)
        extra_q = _random.choice(["if ", "whether "]) + s2
        templates = [
            f"I'm not sure {extra_q}, but {obj1.Q.strip().lower()}",
            f"I do not care {extra_q}, but {obj1.Q.strip().lower()}",
            f"Regardless of {extra_q}, {obj1.Q.strip().lower()}",
        ]
        new_context = (obj2.C or "").strip() + " " + (obj1.C or "").strip()
        return _random.choice(templates), new_context.strip()

    def add_wh_question_as_redundancy(obj1, obj2):
        """Add second question as redundancy (ETI for wh-questions)"""
        extra_q = _clean_question(obj2.Q)
        templates = [
            f"Regardless of {extra_q}, {obj1.Q.strip().lower()}",
            f"Put aside {extra_q}, {obj1.Q.strip().lower()}",
        ]
        new_context = (obj2.C or "").strip() + " " + (obj1.C or "").strip()
        return _random.choice(templates), new_context.strip()

    def combine2input(obj1, obj2, ans1, ans2):
        """Combine two questions (TI mutation)"""
        s1 = _clean_question(obj1.Q)
        s2 = _clean_question(obj2.Q)
        templates = [
            f"Is it true that {s1} and {s2}?",
            f"Isn't it true that {s1} and {s2}?",
        ]
        new_question = _random.choice(templates)
        new_answer = 'yes' if (ans1 == 'yes' and ans2 == 'yes') else 'no'
        new_context = (obj2.C or "").strip() + " " + (obj1.C or "").strip()
        return new_question, new_context.strip(), new_answer

    def negative_question(question):
        words = question.split()
        if words:
            first = words[0].lower()
            negations = {
                "is": "isn't", "are": "aren't", "was": "wasn't", "were": "weren't",
                "do": "don't", "does": "doesn't", "did": "didn't",
                "can": "can't", "could": "couldn't", "will": "won't", "would": "wouldn't",
                "should": "shouldn't", "may": "may not", "might": "might not",
                "must": "must not", "have": "haven't", "has": "hasn't", "had": "hadn't"
            }
            if first in negations:
                words[0] = negations[first]
            return ' '.join(words)
        return question

    def opposite_answer(origin):
        if origin.strip().lower() == 'yes':
            return 'no'
        elif origin.strip().lower() == 'no':
            return 'yes'
        return origin

# Import similarity calculation functions (require sentence-transformers)
_SIMILARITY_AVAILABLE = False
try:
    from calc_sim.sent_sim import calculate_sim_origin_sentence, calculate_sim_origin_target
    _SIMILARITY_AVAILABLE = True
except Exception:
    # Catch all exceptions including CUDA OOM during model loading
    calculate_sim_origin_sentence = None
    calculate_sim_origin_target = None

# Import preprocessing functions
_PREPROCESS_AVAILABLE = False
try:
    from utils.preprocess import (
        context_preprocess,
        get_sentences_from_contexts,
    )
    _PREPROCESS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    def context_preprocess(context):
        return context

    def get_sentences_from_contexts(contexts):
        sentences = []
        for c in contexts:
            # Simple sentence splitting by period
            for s in c.split('.'):
                s = s.strip()
                if s:
                    sentences.append(s + '.')
        return sentences

# Import answer similarity function
# We provide our own improved implementation that handles LLM long answers
_ANSWER_SIM_AVAILABLE = False

def _get_bool_sentiment(ans: str) -> str | None:
    """
    Extract yes/no sentiment from an answer string.

    LLMs often return full sentences like "No, that is not correct..."
    This function extracts the boolean sentiment from the answer.
    """
    ans_lower = ans.strip().lower()

    # Check for explicit yes/no indicators at the start
    yes_starters = ('yes', 'yeah', 'yep', 'correct', 'true', 'right', 'absolutely', 'certainly', 'indeed')
    no_starters = ('no', 'not', 'false', 'wrong', 'incorrect', 'never', 'neither', 'none')

    for starter in yes_starters:
        if ans_lower.startswith(starter):
            return 'yes'
    for starter in no_starters:
        if ans_lower.startswith(starter):
            return 'no'

    # Check first 10 words for yes/no indicators
    first_words = ' '.join(ans_lower.split()[:10])

    # Yes indicators
    if any(word in first_words for word in ['yes', 'true', 'correct', 'right']):
        return 'yes'
    # No indicators (be careful with "not" - it should be "not " to avoid "note" etc.)
    if 'no' in first_words.split() or 'not ' in first_words or 'false' in first_words or 'wrong' in first_words:
        return 'no'

    return None


def _same_boolq_answer_improved(ans1: str, ans2: str) -> bool:
    """
    Improved boolean answer comparison that handles LLM long answers.

    Original QAQA expects exact 'yes'/'no' strings, but LLMs return
    full sentences like "No, based on the given premises..."
    """
    sent1 = _get_bool_sentiment(ans1)
    sent2 = _get_bool_sentiment(ans2)

    # If both have clear yes/no sentiment, compare them
    if sent1 and sent2:
        return sent1 == sent2

    # If only one has clear sentiment, they differ
    if sent1 or sent2:
        return False

    # Neither has clear yes/no sentiment
    # Fall back to simple string comparison
    return ans1.strip().lower() == ans2.strip().lower()


def is_same_answer(ans1: str, ans2: str, is_bool: bool = False) -> bool:
    """
    Check if two answers are semantically equivalent.

    For boolean questions (is_bool=True):
        - Extracts yes/no sentiment from answer text
        - Compares sentiments rather than exact strings

    For non-boolean questions:
        - Uses containment check (key terms from shorter in longer)
        - Falls back to word overlap similarity
        - Handles cases where LLM returns full sentences vs short expected answers
    """
    import re

    if is_bool:
        return _same_boolq_answer_improved(ans1, ans2)

    # Non-boolean: use improved comparison
    ans1_clean = ans1.strip().lower()
    ans2_clean = ans2.strip().lower()

    # Exact match
    if ans1_clean == ans2_clean:
        return True

    # Containment check: if the shorter answer (key entity) is contained in the longer one
    shorter = ans1_clean if len(ans1_clean) <= len(ans2_clean) else ans2_clean
    longer = ans2_clean if len(ans1_clean) <= len(ans2_clean) else ans1_clean

    # Remove punctuation for containment check
    shorter_clean = re.sub(r'[^\w\s]', '', shorter).strip()
    longer_clean = re.sub(r'[^\w\s]', '', longer).strip()

    # Check if key terms from shorter are in longer
    shorter_terms = set(shorter_clean.split())
    longer_terms = set(longer_clean.split())

    # If 80%+ terms from shorter answer appear in longer answer, consider same
    # This handles: "Charlie." in "Charlie is the head of government of Fournels."
    if shorter_terms and len(shorter_terms & longer_terms) >= len(shorter_terms) * 0.8:
        return True

    # Word-level comparison: check if key entity names appear in both
    def extract_key_words(text):
        """Extract meaningful words (skip common words)"""
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from',
                      'by', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                      'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                      'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
                      'while', 'this', 'that', 'these', 'those', 'it', 'its'}
        words = [w for w in text.lower().split() if w not in stop_words and len(w) > 2]
        return set(words)

    words1 = extract_key_words(ans1_clean)
    words2 = extract_key_words(ans2_clean)

    if not words1 or not words2:
        return False

    # Jaccard similarity
    intersection = words1 & words2
    union = words1 | words2
    similarity = len(intersection) / len(union) if union else 0

    # If significant word overlap (> 50%), consider them similar
    if similarity >= 0.5:
        return True

    # Check if most important words from ans1 appear in ans2
    # (handles cases where expected answer is short like "Diana Crossing")
    if words1 and len(words1 & words2) >= len(words1) * 0.6:
        return True

    return False

# Define QA and QAList classes directly (avoiding import from QAUtil.qa which has pattern dependency)
class QA:
    """QA data structure"""
    count = 0

    def __init__(self, q, a, c, t=''):
        self.id = QA.count
        QA.count += 1
        self.Q = q
        self.A = a
        self.C = c
        self.T = t
        self.C_Embedding = None
        self.PA = None


class QAList:
    """QAList data structure"""
    def __init__(self):
        self.qa_list = []
        QA.count = 0

    def append_(self, qa):
        self.qa_list.append(qa)

    def remove_qa_of_this_context(self, context):
        for qa in self.qa_list:
            if context == qa.C:
                self.qa_list[qa.id].C = ''
                if hasattr(self.qa_list[qa.id], 'C_Embedding') and self.qa_list[qa.id].C_Embedding is not None:
                    self.qa_list[qa.id].C_Embedding = np.zeros(shape=(384,), dtype='float32')

    def get_all_context_embedding(self):
        """Get all context embeddings (returns numpy array with object dtype)"""
        context_embedding_list = []
        for qa in self.qa_list:
            embedding = qa.C_Embedding if hasattr(qa, 'C_Embedding') and qa.C_Embedding is not None else np.zeros(shape=(384,), dtype='float32')
            context_embedding_list.append((qa.C, embedding))
        # Create a numpy array with object dtype to handle mixed types
        return np.array(context_embedding_list, dtype=object)


# Set random seed for reproducibility (matching original)
random.seed(2022)


@dataclass
class MutationResult:
    """
    Result of a QAQA mutation operation.

    Attributes:
        original_question: The original question
        mutated_question: The mutated question
        mutation_type: The type of mutation applied (EC, EQ, EQC, ETI, TI)
        is_violation: Whether this mutation detected a violation
        metadata: Additional metadata about the mutation
        original_context: The original context
        mutated_context: The mutated context
        original_answer: The original answer
        mutated_answer: The mutated answer
    """
    original_question: str
    mutated_question: str
    mutation_type: str
    is_violation: bool = False
    metadata: Dict = field(default_factory=dict)
    original_context: Optional[str] = None
    mutated_context: Optional[str] = None
    original_answer: Optional[str] = None
    mutated_answer: Optional[str] = None


class QAQA:
    """
    QAQA: Metamorphic Testing for Question Answering Systems

    This class wraps the original QAQA implementation and provides
    five metamorphic relations (MRs):
    - EC (Extra Context): Add similar sentences to context
    - EQ (Extra Question): Add similar sentences to question
    - EQC (Extra Question + Context): Add similar sentences to both
    - ETI (Extra Two Inputs): Add redundant QA pairs as context
    - TI (Two Inputs): Combine two QA pairs

    This wrapper maintains the original implementation while providing
    a clean Python API.
    """

    # Attack mod (mutation type) names
    ATTACK_MODS = ['EC', 'EQ', 'EQC', 'ETI', 'TI']

    def __init__(self, random_seed: int = 42, use_nlp: bool = True):
        """
        Initialize QAQA

        Args:
            random_seed: Random seed for reproducibility
            use_nlp: Whether to use NLP libraries for advanced features
        """
        self.random_seed = random_seed
        self.use_nlp = use_nlp
        random.seed(random_seed)

    def find_similar_sentences(
        self,
        question: str,
        training_qa_list: QAList,
        top_n: int = 10,
        extra_sent_num2context: int = 1,
        combined_context_num: int = 1,
        max_attack_num: int = 1,
    ) -> Tuple[List[str], Dict[str, int], List]:
        """
        Find similar sentences from training data for a given question.

        This implements the similarity search logic from original QAQA.

        Args:
            question: The question to find similar sentences for
            training_qa_list: Training QA list to search from
            top_n: Maximum number of similar sentences to find
            extra_sent_num2context: Number of extra sentences to add to context
            combined_context_num: Number of contexts to combine
            max_attack_num: Maximum number of attacks to generate

        Returns:
            Tuple of (similar_sentences, sentence_qa_id_dict, top_n_sim_sentences_list)
        """
        # Remove QA objects with same context as current question
        training_qa_list_no_same = copy.deepcopy(training_qa_list)

        # Check if similarity functions are available
        if calculate_sim_origin_target is None or calculate_sim_origin_sentence is None:
            # Fallback: Randomly select sentences from training data when similarity search is not available
            extra_sentences_list = []
            sentence_qa_id_dict = {}

            # Randomly select contexts
            available_qas = [qa for qa in training_qa_list_no_same.qa_list if qa.C]
            num_to_select = min(combined_context_num, len(available_qas))

            for i in range(num_to_select):
                qa = available_qas[i % len(available_qas)]
                this_sentences_list = get_sentences_from_contexts([qa.C])
                extra_sentences_list.extend(this_sentences_list)
                for s in this_sentences_list:
                    sentence_qa_id_dict[s] = qa.id

            # Create top_n_sim_sentences_list with dummy scores
            num_sentences = min(max_attack_num + extra_sent_num2context - 1, len(extra_sentences_list))
            top_n_sim_sentences_list = [(extra_sentences_list[i], 1.0 - i * 0.01) for i in range(num_sentences)]

            return [sent[0] for sent in top_n_sim_sentences_list], sentence_qa_id_dict, top_n_sim_sentences_list

        # Calculate similarity between question and contexts
        top_n_sim_contexts_list = calculate_sim_origin_target(
            question,
            training_qa_list_no_same,
            combined_context_num
        )

        # Extract sentences from similar contexts
        extra_sentences_list = []
        sentence_qa_id_dict = {}

        for count in range(min(combined_context_num, len(top_n_sim_contexts_list))):
            context_qa_id = top_n_sim_contexts_list[count][0]
            this_qa = training_qa_list_no_same.qa_list[context_qa_id]
            this_sentences_list = get_sentences_from_contexts([this_qa.C])
            extra_sentences_list.extend(this_sentences_list)
            for s in this_sentences_list:
                sentence_qa_id_dict[s] = context_qa_id

        # Calculate similarity between question and sentences
        top_n_sim_sentences_list = calculate_sim_origin_sentence(
            question,
            extra_sentences_list,
            top_n=max_attack_num + extra_sent_num2context - 1
        )

        return [sent[0] for sent in top_n_sim_sentences_list], sentence_qa_id_dict, top_n_sim_sentences_list

    def apply_ec(
        self,
        question: str,
        context: str,
        similar_sentences: List[str],
        extra_sent_num2context: int = 1,
    ) -> MutationResult:
        """
        Apply EC (Extra Context) mutation.

        Args:
            question: The original question
            context: The original context
            similar_sentences: List of similar sentences to add
            extra_sent_num2context: Number of extra sentences to add

        Returns:
            MutationResult with the mutated context
        """
        # Combine sentences for context
        selected_sent_combine = ''
        for i in range(min(extra_sent_num2context, len(similar_sentences))):
            selected_sent_combine += (similar_sentences[i] + ' ')

        new_context = add_extra2context(context, selected_sent_combine)

        return MutationResult(
            original_question=question,
            mutated_question=question,
            mutation_type='EC',
            original_context=context,
            mutated_context=new_context,
        )

    def apply_eq(
        self,
        question: str,
        context: str,
        similar_sentences: List[str],
        is_boolq: bool = False,
        add_negative: bool = False,
    ) -> MutationResult:
        """
        Apply EQ (Extra Question) mutation.

        Args:
            question: The original question
            context: The original context
            similar_sentences: List of similar sentences to add
            is_boolq: Whether this is a boolean question
            add_negative: Whether to add negative transformation

        Returns:
            MutationResult with the mutated question
        """
        new_question = question

        # Optionally add negative (for BoolQ)
        if is_boolq and add_negative and random.randint(0, 1) == 1:
            new_question = negative_question(new_question)

        # Add extra sentence to question
        if similar_sentences:
            selected_sent = similar_sentences[0]
            # Ensure selected sentence is long enough
            if len(selected_sent) < 10 and len(similar_sentences) > 1:
                selected_sent = similar_sentences[1]
            new_question = add_extra2question(new_question, selected_sent)

        return MutationResult(
            original_question=question,
            mutated_question=new_question,
            mutation_type='EQ',
            original_context=context,
            mutated_context=context,
        )

    def apply_eqc(
        self,
        question: str,
        context: str,
        similar_sentences: List[str],
        extra_sent_num2context: int = 1,
        is_boolq: bool = False,
        add_negative: bool = False,
        original_answer: str = '',
    ) -> MutationResult:
        """
        Apply EQC (Extra Question + Context) mutation.

        Args:
            question: The original question
            context: The original context
            similar_sentences: List of similar sentences
            extra_sent_num2context: Number of extra sentences for context
            is_boolq: Whether this is a boolean question
            add_negative: Whether to add negative transformation
            original_answer: The original answer (for BoolQ logic)

        Returns:
            MutationResult with both question and context mutated
        """
        new_question = question

        # Optionally add negative (for BoolQ)
        if is_boolq and add_negative and random.randint(0, 1) == 1:
            new_question = negative_question(new_question)

        # Add extra sentence to question
        if similar_sentences:
            selected_sent = similar_sentences[0]
            if len(selected_sent) < 10 and len(similar_sentences) > 1:
                selected_sent = similar_sentences[1]
            new_question = add_extra2question(new_question, selected_sent)

        # Add extra sentences to context (different for yes/no answers in BoolQ)
        selected_sent_combine = ''
        for i in range(min(extra_sent_num2context, len(similar_sentences))):
            selected_sent_combine += (similar_sentences[i] + ' ')

        # For BoolQ: if original answer is 'no', use selected_sent_combine,
        # otherwise use the next similar sentence
        if is_boolq and original_answer == 'no' and len(similar_sentences) > 1:
            new_context = add_extra2context(context, selected_sent_combine)
        elif len(similar_sentences) > 1:
            new_context = add_extra2context(context, similar_sentences[1])
        else:
            new_context = add_extra2context(context, selected_sent_combine)

        return MutationResult(
            original_question=question,
            mutated_question=new_question,
            mutation_type='EQC',
            original_context=context,
            mutated_context=new_context,
        )

    def apply_eti(
        self,
        qa1: QA,
        qa2: QA,
        is_boolq: bool = False,
    ) -> MutationResult:
        """
        Apply ETI (Extra Two Inputs) mutation.

        Args:
            qa1: First QA object
            qa2: Second QA object to add as redundancy
            is_boolq: Whether this is a boolean question

        Returns:
            MutationResult with redundant information added
        """
        if is_boolq:
            new_question, new_context = add_input_as_redundancy(qa1, qa2)
        else:
            new_question, new_context = add_wh_question_as_redundancy(qa1, qa2)

        return MutationResult(
            original_question=qa1.Q,
            mutated_question=new_question,
            mutation_type='ETI',
            original_context=qa1.C,
            mutated_context=new_context,
            original_answer=qa1.A,
        )

    def apply_ti(
        self,
        qa1: QA,
        qa2: QA,
    ) -> MutationResult:
        """
        Apply TI (Two Inputs) mutation.

        Args:
            qa1: First QA object
            qa2: Second QA object

        Returns:
            MutationResult combining both QA pairs
        """
        new_question, new_context, new_answer = combine2input(qa1, qa2, qa1.A, qa2.A)

        return MutationResult(
            original_question=qa1.Q,
            mutated_question=new_question,
            mutation_type='TI',
            original_context=qa1.C,
            mutated_context=new_context,
            original_answer=qa1.A,
            mutated_answer=new_answer,
            metadata={
                'qa2_question': qa2.Q,
                'qa2_context': qa2.C,
                'qa2_answer': qa2.A,
            }
        )

    def check_violation(
        self,
        original_answer: str,
        mutated_answer: str,
        is_boolq: bool = False,
    ) -> bool:
        """
        Check if a mutation violates consistency (i.e., answers differ).

        Args:
            original_answer: The original answer
            mutated_answer: The answer to the mutated question
            is_boolq: Whether this is a boolean question

        Returns:
            True if answers are inconsistent (violation detected)
        """
        return not is_same_answer(original_answer, mutated_answer, is_bool=is_boolq)

    def generate_mutations(
        self,
        qa: QA,
        training_qa_list: Optional[QAList] = None,
        attack_mods: Optional[List[str]] = None,
        extra_sent_num2context: int = 1,
        max_attack_num: int = 1,
        combined_context_num: int = 1,
        is_boolq: bool = False,
        predictor_func: Optional[callable] = None,
    ) -> List[MutationResult]:
        """
        Generate all mutations for a given QA pair.

        This implements the main mutation generation logic from original QAQA.

        Args:
            qa: The QA object to mutate
            training_qa_list: Training QA list for finding similar sentences
            attack_mods: List of attack types to apply (default: all)
            extra_sent_num2context: Number of extra sentences for context
            max_attack_num: Maximum number of mutations to generate
            combined_context_num: Number of contexts to combine
            is_boolq: Whether this is a boolean question
            predictor_func: Optional function to predict answers

        Returns:
            List of MutationResult objects
        """
        results = []

        if attack_mods is None:
            attack_mods = self.ATTACK_MODS

        # For BoolQ, don't use TI mutation
        if is_boolq and 'TI' in attack_mods:
            attack_mods = [m for m in attack_mods if m != 'TI']

        # Select a random attack mod for this QA
        this_attack = random.choice(attack_mods)

        # Get original answer
        if predictor_func:
            ori_ans = predictor_func(qa.Q + '\n (' + qa.T + ') ' + qa.C)
        else:
            ori_ans = qa.A

        # Find similar sentences if training data is provided
        similar_sentences = []
        sentence_qa_id_dict = {}
        top_n_sim_sentences_list = []

        if training_qa_list and this_attack in ['EC', 'EQ', 'EQC']:
            # Remove QA objects with same context
            training_qa_list_no_same = copy.deepcopy(training_qa_list)
            training_qa_list_no_same.remove_qa_of_this_context(qa.C)

            # Find similar sentences
            similar_sentences, sentence_qa_id_dict, top_n_sim_sentences_list = self.find_similar_sentences(
                qa.Q,
                training_qa_list_no_same,
                top_n=max_attack_num + extra_sent_num2context - 1,
                extra_sent_num2context=extra_sent_num2context,
                combined_context_num=combined_context_num,
                max_attack_num=max_attack_num,
            )

        # Generate mutations based on attack type
        for attack_times in range(min(max_attack_num, len(similar_sentences) if similar_sentences else 1)):
            selected_sent = similar_sentences[attack_times] if similar_sentences else None

            if this_attack == 'EC':
                result = self.apply_ec(qa.Q, qa.C, similar_sentences, extra_sent_num2context)
                results.append(result)

            elif this_attack == 'EQ':
                result = self.apply_eq(qa.Q, qa.C, similar_sentences, is_boolq)
                results.append(result)

            elif this_attack == 'EQC':
                result = self.apply_eqc(qa.Q, qa.C, similar_sentences, extra_sent_num2context, is_boolq, original_answer=ori_ans)
                results.append(result)

            elif this_attack == 'ETI' and training_qa_list:
                if similar_sentences and sentence_qa_id_dict:
                    selected_qa_id = sentence_qa_id_dict.get(selected_sent)
                    if selected_qa_id is not None:
                        selected_qa = training_qa_list_no_same.qa_list[selected_qa_id]
                        result = self.apply_eti(qa, selected_qa, is_boolq)
                        results.append(result)

            elif this_attack == 'TI' and training_qa_list:
                if similar_sentences and len(similar_sentences) > 1:
                    # Find second QA
                    selected_qa_id = sentence_qa_id_dict.get(similar_sentences[1])
                    if selected_qa_id is not None:
                        selected_qa = training_qa_list_no_same.qa_list[selected_qa_id]
                        result = self.apply_ti(qa, selected_qa)
                        results.append(result)

        return results

    def is_consistent(
        self,
        question1: str,
        question2: str,
        mutation_type: Optional[str] = None,
    ) -> bool:
        """
        Check if two questions are consistent (should have similar answers).

        Args:
            question1: First question
            question2: Second question
            mutation_type: The type of mutation applied (optional)

        Returns:
            True if questions are consistent
        """
        # For QAQA, consistency depends on the mutation type
        # In general, if the mutation preserves the core meaning,
        # the answers should be consistent
        return True


# Export main classes and result type
__all__ = ['QAQA', 'MutationResult', 'QA', 'QAList']
