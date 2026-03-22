"""
List question IDs where both annotators agreed an item was invalid.

A row matches if ANY of these holds (all require agreement on both sides):
  - both validity_label are "invalid" (case-insensitive), OR
  - both annotated_answer are empty (after strip), OR
  - both annotated_answer are "UNKNOWN" (case-insensitive), OR
  - both annotated_answer contain the substring "invalid" (case-insensitive)

Usage:
  python scripts/list_agreed_invalid.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_A1 = REPO_ROOT / "annotations/iaa_annotation_sample_60.jsonl"
PATH_A2 = REPO_ROOT / "annotations/iaa_annotation_sample_60_annotator2.jsonl"


def load_by_id(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out[o["id"]] = o
    return out


def agreed_invalid(a1: dict, a2: dict) -> tuple[bool, list[str]]:
    """Return (match, list of reason tags)."""
    v1 = (a1.get("validity_label") or "").strip().lower()
    v2 = (a2.get("validity_label") or "").strip().lower()
    s1 = (a1.get("annotated_answer") or "").strip()
    s2 = (a2.get("annotated_answer") or "").strip()

    reasons: list[str] = []
    if v1 == "invalid" and v2 == "invalid":
        reasons.append("both_validity_invalid")
    if s1 == "" and s2 == "":
        reasons.append("both_answer_empty")
    if s1.lower() == "unknown" and s2.lower() == "unknown":
        reasons.append("both_unknown")
    if "invalid" in s1.lower() and "invalid" in s2.lower():
        reasons.append("both_answer_contain_invalid")

    return (len(reasons) > 0, reasons)


def main() -> None:
    m1 = load_by_id(PATH_A1)
    m2 = load_by_id(PATH_A2)
    ids = sorted(set(m1) & set(m2), key=lambda x: int(x[1:]))

    hits: list[tuple[str, str, str, str, list[str]]] = []
    for qid in ids:
        ok, reasons = agreed_invalid(m1[qid], m2[qid])
        if ok:
            a1 = (m1[qid].get("annotated_answer") or "").strip()
            a2 = (m2[qid].get("annotated_answer") or "").strip()
            hits.append((qid, m1[qid].get("question", ""), a1, a2, reasons))

    print(f"Files: {PATH_A1.name} + {PATH_A2.name}")
    print(f"Questions where both annotators agreed invalid ({len(hits)}):\n")
    for qid, qtext, ans1, ans2, reasons in hits:
        print(f"  {qid}  [{', '.join(reasons)}]")
        print(f"    {qtext}")
        print(f"    annotator1: {ans1!r}")
        print(f"    annotator2: {ans2!r}\n")


if __name__ == "__main__":
    main()
