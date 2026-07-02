# IMPORTANT: Set environment variable BEFORE importing any torch-related modules
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
import torch
from torch import nn

model_path = 'whaleloops/phrase-bert'

# Try to load the model, handle CUDA OOM gracefully
# Force CPU to avoid CUDA memory issues
phrase_bert_model = None
try:
    phrase_bert_model = SentenceTransformer(model_path, device='cpu')
except Exception as e:
    phrase_bert_model = None  # Silent failure

cos_sim = nn.CosineSimilarity(dim=0)


def same_boolq_answer(ans1, ans2):
    if 'no' not in [ans1, ans2] and 'yes' not in [ans1, ans2]:
        return False
    if 'no' in [ans1, ans2]:
        return ans1 == ans2
    elif 'yes' in [ans1, ans2]:
        if 'no' in [ans1, ans2]:
            return False
        elif ans1 == ans2:
            return True
        else:
            sim = pbert_ans_similarity(ans1, ans2)
            if sim >= 0.76:
                return True
            else:
                return False


def pbert_ans_similarity(ans1, ans2):
    if phrase_bert_model is None:
        # Fallback: simple string matching
        print(f"Warning: phrase-bert model not available, using simple string matching")
        return 1.0 if ans1.lower() == ans2.lower() else 0.0

    try:
        phrase_embs = phrase_bert_model.encode([ans1, ans2], show_progress_bar=False)
        [p1, p2] = phrase_embs
        similarity = float(cos_sim(torch.tensor(p1), torch.tensor(p2)))
        # Silence verbose output
        return similarity
    except Exception as e:
        print(f"Warning: Failed to compute phrase-bert similarity: {e}")
        return 1.0 if ans1.lower() == ans2.lower() else 0.0


def is_same_answer(ans1, ans2, is_bool=False, threshold=0.65):
    """
    Check if two answers are semantically the same.

    Uses multiple strategies:
    1. For boolean questions: special boolean logic
    2. Containment check: if one answer contains the other (key entities)
    3. Phrase-BERT semantic similarity

    Args:
        ans1: First answer (expected answer)
        ans2: Second answer (LLM answer)
        is_bool: Whether this is a boolean question
        threshold: Similarity threshold (default 0.65)

    Returns:
        bool: True if answers are considered the same
    """
    import re

    ans1 = ans1.strip().lower()
    ans2 = ans2.strip().lower()

    # Exact match
    if ans1 == ans2:
        return True

    if is_bool:
        return same_boolq_answer(ans1, ans2)

    # Containment check: if the shorter answer (key entity) is contained in the longer one
    # This handles cases like: "Charlie." vs "Charlie is the head of government of Fournels."
    shorter = ans1 if len(ans1) <= len(ans2) else ans2
    longer = ans2 if len(ans1) <= len(ans2) else ans1

    # Remove punctuation for containment check
    shorter_clean = re.sub(r'[^\w\s]', '', shorter).strip()
    longer_clean = re.sub(r'[^\w\s]', '', longer).strip()

    # Check if key terms from shorter are in longer
    shorter_terms = set(shorter_clean.split())
    longer_terms = set(longer_clean.split())

    # If most terms from shorter answer appear in longer answer (80% threshold)
    if shorter_terms and len(shorter_terms & longer_terms) >= len(shorter_terms) * 0.8:
        return True

    # Semantic similarity check as fallback
    similarity = pbert_ans_similarity(ans1, ans2)
    if similarity >= threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    res = is_same_answer("Jack", "jack wisen")
    print(res)
    res = is_same_answer("yes", "No", is_bool=True)
    print(res)
    res = is_same_answer("No answer", "no", is_bool=True)
    print(res)
