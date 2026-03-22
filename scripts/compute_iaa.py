"""
Inter-annotator agreement on annotated_answer: normalize (lowercase, strip punctuation),
then exact string match. Prints agreement % and disagreement list.

Usage (from repo root):
  python scripts/compute_iaa.py
  python scripts/compute_iaa.py --a1 annotations/foo.jsonl --a2 annotations/bar.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def normalize_answer(text: str) -> str:
    """Lowercase; remove all non-alphanumeric characters; collapse whitespace."""
    if not text:
        return ""
    lowered = text.lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return " ".join(cleaned.split())


def load_by_id(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["id"]
            if qid in by_id:
                raise ValueError(f"Duplicate id {qid!r} in {path}")
            by_id[qid] = obj
    return by_id


def main() -> None:
    p = argparse.ArgumentParser(description="Compute IAA on annotated_answer fields")
    p.add_argument(
        "--a1",
        type=Path,
        default=REPO_ROOT / "annotations/iaa_annotation_sample_60.jsonl",
    )
    p.add_argument(
        "--a2",
        type=Path,
        default=REPO_ROOT / "annotations/iaa_annotation_sample_60_annotator2.jsonl",
    )
    args = p.parse_args()

    a1 = load_by_id(args.a1)
    a2 = load_by_id(args.a2)

    ids1, ids2 = set(a1), set(a2)
    if ids1 != ids2:
        only1 = sorted(ids1 - ids2)
        only2 = sorted(ids2 - ids1)
        raise SystemExit(
            f"id mismatch: only in annotator1: {only1[:10]}{'...' if len(only1) > 10 else ''} "
            f"only in annotator2: {only2[:10]}{'...' if len(only2) > 10 else ''}"
        )

    ordered_ids = sorted(ids1, key=lambda x: (len(x), x))

    matches = 0
    disagreements: list[dict] = []

    for qid in ordered_ids:
        ans1 = a1[qid].get("annotated_answer") or ""
        ans2 = a2[qid].get("annotated_answer") or ""
        n1, n2 = normalize_answer(ans1), normalize_answer(ans2)
        if n1 == n2:
            matches += 1
        else:
            disagreements.append(
                {
                    "id": qid,
                    "question": a1[qid].get("question", ""),
                    "annotator1_answer": ans1,
                    "annotator2_answer": ans2,
                    "normalized_1": n1,
                    "normalized_2": n2,
                    "validity_1": a1[qid].get("validity_label", ""),
                    "validity_2": a2[qid].get("validity_label", ""),
                }
            )

    n = len(ordered_ids)
    pct = 100.0 * matches / n if n else 0.0

    print(f"Compared {n} items (annotated_answer, normalized exact match)")
    print(f"IAA (exact match after normalization): {matches}/{n} = {pct:.2f}%")
    print()

    if disagreements:
        print(f"Disagreements ({len(disagreements)}):")
        for d in disagreements:
            print(f"  [{d['id']}] {d['question']}")
            print(f"      A1: {d['annotator1_answer']!r}")
            print(f"      A2: {d['annotator2_answer']!r}")
            print(f"      norm A1: {d['normalized_1']!r}")
            print(f"      norm A2: {d['normalized_2']!r}")
            if d["validity_1"] != d["validity_2"]:
                print(
                    f"      validity: {d['validity_1']!r} vs {d['validity_2']!r}"
                )
    else:
        print("No disagreements on normalized annotated_answer.")


if __name__ == "__main__":
    main()
