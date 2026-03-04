"""
ablation.py — Run ablation studies on your RAG model.

Usage:
    python ablation.py --questions data/questions.txt --answers data/reference_answers.json

This script runs multiple RAG configurations and prints a comparison table.
Tweak the ABLATIONS list to match the configs you want to compare.
"""

import json
import re
import string
import argparse
from collections import Counter
from typing import List, Tuple


# ──────────────────────────────────────────────
# Evaluation helpers (mirror official script)
# ──────────────────────────────────────────────

def normalize(s: str) -> str:
    """Matches the official SQuAD v1.1 evaluation script exactly:
    lowercase → remove punctuation → remove articles (a/an/the) → collapse whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, golds: List[str]) -> int:
    return int(any(normalize(pred) == normalize(g) for g in golds))


def token_f1(pred: str, golds: List[str]) -> float:
    pred_tokens = normalize(pred).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        p = n_common / len(pred_tokens)
        r = n_common / len(gold_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


def evaluate(predictions: List[str], references: List[List[str]]) -> Tuple[float, float]:
    """Returns (avg_EM, avg_F1)."""
    ems, f1s = [], []
    for pred, golds in zip(predictions, references):
        ems.append(exact_match(pred, golds))
        f1s.append(token_f1(pred, golds))
    return sum(ems) / len(ems) * 100, sum(f1s) / len(f1s) * 100


# ──────────────────────────────────────────────
# Ablation configurations
# Modify these to test different design choices.
# ──────────────────────────────────────────────

ABLATIONS = [
    # (label, bm25_weight, dense_weight, top_k, chunk_size)
    ("BM25 only",           1.0, 0.0, 5, 200),
    ("Dense only",          0.0, 1.0, 5, 200),
    ("Hybrid 30/70 k=5",    0.3, 0.7, 5, 200),  # default
    ("Hybrid 30/70 k=10",   0.3, 0.7, 10, 200),
    ("Hybrid 50/50 k=5",    0.5, 0.5, 5, 200),
    ("Chunk=100",           0.3, 0.7, 5, 100),
    ("Chunk=400",           0.3, 0.7, 5, 400),
]


def run_ablation(questions, references, bm25_w, dense_w, top_k, chunk_size):
    """
    Import rag.py with patched constants and run predictions.
    To avoid reloading the index every time, we monkey-patch the module constants.
    """
    import rag
    rag.BM25_WEIGHT = bm25_w
    rag.DENSE_WEIGHT = dense_w
    rag.TOP_K_RETRIEVE = top_k
    rag.CHUNK_SIZE = chunk_size

    model = rag.RAGModel()
    preds = model.predict(questions)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="Path to questions.txt")
    parser.add_argument("--answers", required=True,
                        help="Path to reference_answers.json (list of lists or list of strings)")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [l.strip() for l in f if l.strip()]

    with open(args.answers) as f:
        raw = json.load(f)
    # Support both list-of-strings and list-of-lists (multiple valid answers)
    references = [r if isinstance(r, list) else [r] for r in raw]

    assert len(questions) == len(references), "Question/answer count mismatch."

    print(f"\n{'Config':<25} {'EM':>6} {'F1':>6}")
    print("-" * 40)
    for label, bm25_w, dense_w, top_k, chunk_size in ABLATIONS:
        preds = run_ablation(questions, references, bm25_w, dense_w, top_k, chunk_size)
        em, f1 = evaluate(preds, references)
        print(f"{label:<25} {em:>5.1f}% {f1:>5.1f}%")


if __name__ == "__main__":
    main()
