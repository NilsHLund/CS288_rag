"""
Inter-Annotator Agreement (IAA) computation for QA dataset.

Usage:
    python compute_iaa.py --a1 annotator1.json --a2 annotator2.json

Each JSON file should be a list of dicts like:
    [{"question": "...", "answer": "..."}, ...]
"""

import json
import argparse
import string
import re
from collections import Counter


def normalize(text: str) -> str:
    """Lowercase and remove punctuation (mirrors official eval)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def compute_iaa(file1: str, file2: str):
    with open(file1) as f:
        ann1 = json.load(f)  # list of {"question": ..., "answer": ...}
    with open(file2) as f:
        ann2 = json.load(f)

    assert len(ann1) == len(ann2), "Annotation files must have the same number of entries."

    em_scores, f1_scores = [], []
    for a, b in zip(ann1, ann2):
        em_scores.append(int(exact_match(a["answer"], b["answer"])))
        f1_scores.append(token_f1(a["answer"], b["answer"]))

    print(f"Number of shared questions: {len(ann1)}")
    print(f"Exact Match Agreement:       {sum(em_scores)/len(em_scores)*100:.1f}%")
    print(f"Average Token F1 Agreement:  {sum(f1_scores)/len(f1_scores)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True, help="Path to annotator 1 JSON")
    parser.add_argument("--a2", required=True, help="Path to annotator 2 JSON")
    args = parser.parse_args()
    compute_iaa(args.a1, args.a2)
