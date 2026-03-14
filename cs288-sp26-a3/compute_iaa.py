"""
Inter-Annotator Agreement (IAA) computation for QA annotations.

Supports two formats:

1) Legacy format (JSON list):
   [{"question": "...", "answer": "..."}, ...]
   -> reports answer-to-answer EM and token F1.

2) Annotation format (JSONL or JSON list), each item containing:
   {
     "id": "q001",
     "question": "...",
     "gold_answer": "...",
     "url": "...",
     "validity_label": "valid|invalid|1|0|yes|no",
     "annotated_answer": "...",
     "notes": "..."
   }
   -> reports label agreement, Cohen's kappa, and answer EM/F1 on jointly-valid items.
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def _load_json_or_jsonl(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def _parse_validity(value) -> Optional[int]:
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "y", "valid", "correct", "supported"}:
        return 1
    if txt in {"0", "false", "no", "n", "invalid", "incorrect", "unsupported"}:
        return 0
    return None


def _cohen_kappa(labels1: List[int], labels2: List[int]) -> float:
    n = len(labels1)
    if n == 0:
        return float("nan")

    agree = sum(int(a == b) for a, b in zip(labels1, labels2)) / n
    p1_pos = sum(labels1) / n
    p1_neg = 1.0 - p1_pos
    p2_pos = sum(labels2) / n
    p2_neg = 1.0 - p2_pos
    pe = p1_pos * p2_pos + p1_neg * p2_neg

    if abs(1.0 - pe) < 1e-12:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - pe) / (1.0 - pe)


def _looks_like_legacy(ann) -> bool:
    if not isinstance(ann, list) or not ann:
        return False
    return isinstance(ann[0], dict) and "answer" in ann[0] and "validity_label" not in ann[0]


def _compute_legacy(ann1: List[dict], ann2: List[dict]) -> None:
    if len(ann1) != len(ann2):
        raise ValueError("Legacy annotation files must have the same number of entries.")

    em_scores, f1_scores = [], []
    for a, b in zip(ann1, ann2):
        em_scores.append(int(exact_match(a.get("answer", ""), b.get("answer", ""))))
        f1_scores.append(token_f1(a.get("answer", ""), b.get("answer", "")))

    print(f"Mode: legacy answer comparison")
    print(f"Number of shared questions: {len(ann1)}")
    print(f"Exact Match Agreement:       {sum(em_scores)/len(em_scores)*100:.1f}%")
    print(f"Average Token F1 Agreement:  {sum(f1_scores)/len(f1_scores)*100:.1f}%")


def _index_annotations(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for i, row in enumerate(rows):
        key = str(row.get("id") or row.get("question") or f"idx_{i}")
        out[key] = row
    return out


def _get_annotated_answer(row: dict) -> str:
    # Accept a few possible field names.
    return (
        row.get("annotated_answer")
        or row.get("answer")
        or row.get("corrected_answer")
        or ""
    )


def _compute_annotation_mode(ann1: List[dict], ann2: List[dict]) -> None:
    idx1 = _index_annotations(ann1)
    idx2 = _index_annotations(ann2)
    shared = sorted(set(idx1.keys()) & set(idx2.keys()))
    if not shared:
        raise ValueError("No shared IDs/questions found between annotation files.")

    labels1, labels2 = [], []
    jointly_valid_a1, jointly_valid_a2 = [], []
    missing_label = 0

    for k in shared:
        r1, r2 = idx1[k], idx2[k]
        v1 = _parse_validity(r1.get("validity_label"))
        v2 = _parse_validity(r2.get("validity_label"))
        if v1 is None or v2 is None:
            missing_label += 1
            continue

        labels1.append(v1)
        labels2.append(v2)

        if v1 == 1 and v2 == 1:
            jointly_valid_a1.append(_get_annotated_answer(r1))
            jointly_valid_a2.append(_get_annotated_answer(r2))

    if not labels1:
        raise ValueError("No comparable labeled items found. Check validity_label values.")

    agreement = sum(int(a == b) for a, b in zip(labels1, labels2)) / len(labels1)
    kappa = _cohen_kappa(labels1, labels2)

    print("Mode: annotation agreement")
    print(f"Shared items:               {len(shared)}")
    print(f"Comparable labeled items:   {len(labels1)}")
    if missing_label:
        print(f"Skipped (missing labels):   {missing_label}")
    print(f"Label agreement:            {agreement*100:.1f}%")
    print(f"Cohen's kappa:              {kappa:.3f}")

    if jointly_valid_a1:
        em_scores = [int(exact_match(a, b)) for a, b in zip(jointly_valid_a1, jointly_valid_a2)]
        f1_scores = [token_f1(a, b) for a, b in zip(jointly_valid_a1, jointly_valid_a2)]
        print(f"Jointly valid items:        {len(jointly_valid_a1)}")
        print(f"Answer EM (joint-valid):    {sum(em_scores)/len(em_scores)*100:.1f}%")
        print(f"Answer F1 (joint-valid):    {sum(f1_scores)/len(f1_scores)*100:.1f}%")
    else:
        print("Jointly valid items:        0")
        print("Answer EM/F1 skipped because there are no jointly valid items.")


def compute_iaa(file1: str, file2: str) -> None:
    ann1 = _load_json_or_jsonl(file1)
    ann2 = _load_json_or_jsonl(file2)

    if _looks_like_legacy(ann1) and _looks_like_legacy(ann2):
        _compute_legacy(ann1, ann2)
        return

    if not isinstance(ann1, list) or not isinstance(ann2, list):
        raise ValueError("Both inputs must be JSON arrays or JSONL arrays of objects.")
    _compute_annotation_mode(ann1, ann2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True, help="Path to annotator 1 JSON/JSONL")
    parser.add_argument("--a2", required=True, help="Path to annotator 2 JSON/JSONL")
    args = parser.parse_args()
    compute_iaa(args.a1, args.a2)
