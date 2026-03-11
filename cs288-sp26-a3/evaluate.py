"""
evaluate.py — Evaluate RAG predictions against reference.jsonl

Scoring is identical to the official SQuAD v1.1 evaluation script:
  - normalize: lowercase, remove punctuation, remove articles (a/an/the), collapse whitespace
  - Exact Match: 1 if normalized prediction == any normalized gold answer
  - Token F1: best token-overlap F1 across all gold answers

reference.jsonl format (one JSON object per line):
    {"question": "...", "answer": "CS 161", "url": "..."}
    {"question": "...", "answer": "Alice|Bob", "url": "..."}   <- pipe-separated multiple answers

Usage:
    # predictions.txt has one answer per line, in same order as reference.jsonl
    python evaluate.py reference.jsonl predictions.txt

    # Or evaluate against a plain questions.txt + answers.txt pair
    python evaluate.py --questions data/questions.txt \\
                       --gold data/reference_answers.txt \\
                       --pred data/predictions.txt
"""

from pathlib import Path
import re
import string
import json
import argparse
import sys
from collections import Counter
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Normalization — exactly mirrors SQuAD v1.1 evaluate-v1.1.py
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    print(f"Pred: {prediction_tokens}, GT: {ground_truth_tokens}")
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_reference_jsonl(path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Returns (questions, list_of_gold_answer_lists, urls).
    Pipe-separated answers are split into multiple valid answers.
    """
    questions, golds, urls = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj["question"])
            # Split pipe-separated answers; strip whitespace around each
            answers = [a.strip() for a in obj["answer"].split("|")]
            golds.append(answers)
            urls.append(obj.get("url", ""))
    return questions, golds, urls


def load_predictions(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_answers(path: str) -> List[List[str]]:
    with open(path, encoding="utf-8") as f:
        return [[a.strip() for a in line.split("|")] for line in f]

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(predictions: List[str], golds: List[List[str]]) -> dict:
    assert len(predictions) == len(golds), (
        f"Prediction count ({len(predictions)}) != reference count ({len(golds)})"
    )
    total_em = 0.0
    total_f1 = 0.0
    n = len(predictions)

    per_question = []
    for pred, gold_list in zip(predictions, golds):
        em = metric_max_over_ground_truths(exact_match_score, pred, gold_list)
        f1 = metric_max_over_ground_truths(f1_score, pred, gold_list)
        total_em += em
        total_f1 += f1
        per_question.append({"pred": pred, "golds": gold_list, "em": em, "f1": f1})

    return {
        "exact_match": 100.0 * total_em / n,
        "f1": 100.0 * total_f1 / n,
        "n": n,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG predictions (SQuAD v1.1 scoring)")
    parser.add_argument("reference", help="Path to reference.jsonl or answers.txt")
    parser.add_argument("predictions", help="Path to predictions.txt (one answer per line)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question results")
    args = parser.parse_args()

    if Path(args.reference).suffix == ".jsonl":
        questions, golds, urls = load_reference_jsonl(args.reference)
    elif Path(args.reference).suffix == ".txt":
        golds = load_answers(args.reference)
    else:
        raise ValueError("Reference should be .txt or .jsonl")
    
    predictions = load_predictions(args.predictions)

    results = evaluate(predictions, golds)

    print(f"\n{'='*50}")
    print(f"  Exact Match: {results['exact_match']:.2f}%")
    print(f"  F1 Score:    {results['f1']:.2f}%")
    print(f"  Questions:   {results['n']}")
    print(f"{'='*50}\n")

    if args.verbose:
        print(f"\n{'Q':<5} {'EM':>4} {'F1':>5}  Prediction  |  Gold Answers")
        print("-" * 80)
        for i, item in enumerate(results["per_question"]):
            gold_str = " | ".join(item["golds"])
            marker = "✓" if item["em"] else ("~" if item["f1"] > 0 else "✗")
            print(f"{i+1:<5} {marker} {item['f1']:>4.2f}  {item['pred']!r:<30}  {gold_str}")

    # Print per-question F1=0 cases for error analysis
    zero_f1 = [r for r in results["per_question"] if r["f1"] == 0.0]
    if zero_f1:
        print(f"\n{len(zero_f1)} questions with F1=0.0 (for error analysis):")
        for i, item in enumerate(zero_f1[:10]):  # show first 10
            print(f"  [{i+1}] pred={item['pred']!r}  golds={item['golds']}")
        if len(zero_f1) > 10:
            print(f"  ... and {len(zero_f1)-10} more.")


if __name__ == "__main__":
    main()
