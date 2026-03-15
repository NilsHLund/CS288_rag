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
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


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


def _krippendorff_alpha_nominal_two_annotators(labels1: List[int], labels2: List[int]) -> float:
    """
    Krippendorff's alpha (nominal) for paired annotations.
    This implementation is equivalent to alpha on items with two coders each.
    """
    if len(labels1) != len(labels2):
        raise ValueError("Label lists must have equal length for alpha.")
    n_items = len(labels1)
    if n_items == 0:
        return float("nan")

    # Observed disagreement Do: nominal distance is 0(same)/1(different).
    do = sum(int(a != b) for a, b in zip(labels1, labels2)) / n_items

    # Expected disagreement De from overall label marginals.
    counts = Counter(labels1 + labels2)
    total = sum(counts.values())
    if total <= 1:
        return float("nan")

    # For nominal alpha, De = 1 - sum_c p(c)^2
    sum_sq = 0.0
    for cnt in counts.values():
        p = cnt / total
        sum_sq += p * p
    de = 1.0 - sum_sq

    if abs(de) < 1e-12:
        return 1.0 if abs(do) < 1e-12 else 0.0
    return 1.0 - (do / de)


def _reliability_band(score: float) -> str:
    """
    Krippendorff-style reliability interpretation commonly used in NLP:
    >= 0.800: reliable
    0.667-0.799: tentative
    < 0.667: low
    """
    if score != score:  # NaN check
        return "n/a"
    if score >= 0.800:
        return "reliable"
    if score >= 0.667:
        return "tentative"
    return "low"


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


def _short(text: str, n: int = 140) -> str:
    txt = " ".join(str(text or "").split())
    if len(txt) <= n:
        return txt
    return txt[: n - 3] + "..."


def _compute_annotation_mode(ann1: List[dict], ann2: List[dict], show_mismatches: bool, max_mismatches: int) -> None:
    idx1 = _index_annotations(ann1)
    idx2 = _index_annotations(ann2)
    shared = sorted(set(idx1.keys()) & set(idx2.keys()))
    if not shared:
        raise ValueError("No shared IDs/questions found between annotation files.")

    labels1, labels2 = [], []
    jointly_valid_a1, jointly_valid_a2 = [], []
    jointly_valid_meta = []
    missing_label = 0
    label_mismatches = []

    for k in shared:
        r1, r2 = idx1[k], idx2[k]
        v1 = _parse_validity(r1.get("validity_label"))
        v2 = _parse_validity(r2.get("validity_label"))
        if v1 is None or v2 is None:
            missing_label += 1
            continue

        labels1.append(v1)
        labels2.append(v2)

        if v1 != v2:
            label_mismatches.append(
                {
                    "id": k,
                    "question": r1.get("question") or r2.get("question") or "",
                    "label1": r1.get("validity_label"),
                    "label2": r2.get("validity_label"),
                    "answer1": _get_annotated_answer(r1),
                    "answer2": _get_annotated_answer(r2),
                }
            )

        if v1 == 1 and v2 == 1:
            jointly_valid_a1.append(_get_annotated_answer(r1))
            jointly_valid_a2.append(_get_annotated_answer(r2))
            jointly_valid_meta.append(
                {
                    "id": k,
                    "question": r1.get("question") or r2.get("question") or "",
                    "answer1": _get_annotated_answer(r1),
                    "answer2": _get_annotated_answer(r2),
                }
            )

    if not labels1:
        raise ValueError("No comparable labeled items found. Check validity_label values.")

    agreement = sum(int(a == b) for a, b in zip(labels1, labels2)) / len(labels1)
    kappa = _cohen_kappa(labels1, labels2)
    alpha = _krippendorff_alpha_nominal_two_annotators(labels1, labels2)

    print("Mode: annotation agreement")
    print(f"Shared items:               {len(shared)}")
    print(f"Comparable labeled items:   {len(labels1)}")
    if missing_label:
        print(f"Skipped (missing labels):   {missing_label}")
    print(f"Label agreement:            {agreement*100:.1f}%")
    print(f"Cohen's kappa:              {kappa:.3f}")
    print(f"Kappa band:                 {_reliability_band(kappa)}")
    print(f"Krippendorff alpha:         {alpha:.3f}")
    print(f"Alpha band:                 {_reliability_band(alpha)}")

    if jointly_valid_a1:
        em_scores = [int(exact_match(a, b)) for a, b in zip(jointly_valid_a1, jointly_valid_a2)]
        f1_scores = [token_f1(a, b) for a, b in zip(jointly_valid_a1, jointly_valid_a2)]
        print(f"Jointly valid items:        {len(jointly_valid_a1)}")
        print(f"Answer EM (joint-valid):    {sum(em_scores)/len(em_scores)*100:.1f}%")
        print(f"Answer F1 (joint-valid):    {sum(f1_scores)/len(f1_scores)*100:.1f}%")

        answer_mismatches = []
        for i, (a1, a2) in enumerate(zip(jointly_valid_a1, jointly_valid_a2)):
            if not exact_match(a1, a2):
                answer_mismatches.append(
                    {
                        "id": jointly_valid_meta[i]["id"],
                        "question": jointly_valid_meta[i]["question"],
                        "answer1": a1,
                        "answer2": a2,
                        "f1": token_f1(a1, a2),
                    }
                )

        print(f"Non-matching answers:       {len(answer_mismatches)}")

        if show_mismatches:
            if label_mismatches:
                print("\nLabel mismatches:")
                for m in label_mismatches[:max_mismatches]:
                    print(f"- [{m['id']}] {_short(m['question'])}")
                    print(f"  a1 label={m['label1']!r} answer={m['answer1']!r}")
                    print(f"  a2 label={m['label2']!r} answer={m['answer2']!r}")
                if len(label_mismatches) > max_mismatches:
                    print(f"... {len(label_mismatches) - max_mismatches} more label mismatches omitted.")

            if answer_mismatches:
                print("\nAnswer mismatches (among jointly valid items):")
                for m in answer_mismatches[:max_mismatches]:
                    print(f"- [{m['id']}] {_short(m['question'])}")
                    print(f"  a1 answer={m['answer1']!r}")
                    print(f"  a2 answer={m['answer2']!r}")
                    print(f"  token_f1={m['f1']:.3f}")
                if len(answer_mismatches) > max_mismatches:
                    print(f"... {len(answer_mismatches) - max_mismatches} more answer mismatches omitted.")
    else:
        print("Jointly valid items:        0")
        print("Answer EM/F1 skipped because there are no jointly valid items.")


def _compute_answer_only_mode(ann1: List[dict], ann2: List[dict], show_mismatches: bool, max_mismatches: int) -> None:
    idx1 = _index_annotations(ann1)
    idx2 = _index_annotations(ann2)
    shared = sorted(set(idx1.keys()) & set(idx2.keys()))
    if not shared:
        raise ValueError("No shared IDs/questions found between annotation files.")

    comparable = []
    skipped_missing_answer = 0
    for k in shared:
        r1, r2 = idx1[k], idx2[k]
        a1 = str(_get_annotated_answer(r1)).strip()
        a2 = str(_get_annotated_answer(r2)).strip()
        if not a1 or not a2:
            skipped_missing_answer += 1
            continue
        comparable.append(
            {
                "id": k,
                "question": r1.get("question") or r2.get("question") or "",
                "answer1": a1,
                "answer2": a2,
            }
        )

    if not comparable:
        raise ValueError("No comparable answers found. Check annotated_answer fields.")

    em_scores = [int(exact_match(x["answer1"], x["answer2"])) for x in comparable]
    f1_scores = [token_f1(x["answer1"], x["answer2"]) for x in comparable]
    mismatches = []
    for i, row in enumerate(comparable):
        if not em_scores[i]:
            mismatches.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer1": row["answer1"],
                    "answer2": row["answer2"],
                    "f1": f1_scores[i],
                }
            )

    print("Mode: answer-only agreement (ignoring validity labels)")
    print(f"Shared items:               {len(shared)}")
    print(f"Comparable answer items:    {len(comparable)}")
    if skipped_missing_answer:
        print(f"Skipped (missing answer):   {skipped_missing_answer}")
    print(f"Answer EM:                  {sum(em_scores)/len(em_scores)*100:.1f}%")
    print(f"Answer F1:                  {sum(f1_scores)/len(f1_scores)*100:.1f}%")
    print(f"Non-matching answers:       {len(mismatches)}")

    if show_mismatches and mismatches:
        print("\nAnswer mismatches:")
        for m in mismatches[:max_mismatches]:
            print(f"- [{m['id']}] {_short(m['question'])}")
            print(f"  a1 answer={m['answer1']!r}")
            print(f"  a2 answer={m['answer2']!r}")
            print(f"  token_f1={m['f1']:.3f}")
        if len(mismatches) > max_mismatches:
            print(f"... {len(mismatches) - max_mismatches} more answer mismatches omitted.")


def compute_iaa(
    file1: str,
    file2: str,
    show_mismatches: bool,
    max_mismatches: int,
    answer_only: bool,
) -> None:
    ann1 = _load_json_or_jsonl(file1)
    ann2 = _load_json_or_jsonl(file2)

    if _looks_like_legacy(ann1) and _looks_like_legacy(ann2):
        _compute_legacy(ann1, ann2)
        return

    if not isinstance(ann1, list) or not isinstance(ann2, list):
        raise ValueError("Both inputs must be JSON arrays or JSONL arrays of objects.")
    if answer_only:
        _compute_answer_only_mode(ann1, ann2, show_mismatches=show_mismatches, max_mismatches=max_mismatches)
        return
    _compute_annotation_mode(ann1, ann2, show_mismatches=show_mismatches, max_mismatches=max_mismatches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", required=True, help="Path to annotator 1 JSON/JSONL")
    parser.add_argument("--a2", required=True, help="Path to annotator 2 JSON/JSONL")
    parser.add_argument(
        "--show-mismatches",
        action="store_true",
        help="Print label and answer mismatches for manual inspection.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=200,
        help="Maximum number of mismatches to print per section.",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Ignore validity labels and compute agreement directly on annotated answers.",
    )
    args = parser.parse_args()
    compute_iaa(
        args.a1,
        args.a2,
        show_mismatches=args.show_mismatches,
        max_mismatches=args.max_mismatches,
        answer_only=args.answer_only,
    )
