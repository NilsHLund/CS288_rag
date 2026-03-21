"""
Stratified sample from cleaned_qa_500.jsonl -> final_qa_200.jsonl,
then IAA annotation JSONLs for annotate_app.py (two parallel files per annotator).

Does not read or write any existing annotation templates; only creates new outputs.

Usage (from repo root):
  python scripts/sample_final_qa.py
  python scripts/sample_final_qa.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCES: dict[str, Path] = {
    "generated_qa_hard": REPO_ROOT / "data/qa/generated_qa_hard.jsonl",
    "generated_qa_dense_02": REPO_ROOT / "data/qa/generated_qa_dense_02.jsonl",
    "generated_qa_dense": REPO_ROOT / "data/qa/generated_qa_dense.jsonl",
    "generated_qa_num": REPO_ROOT / "data/qa/generated_qa_num.jsonl",
    "generated_qa_yesorno": REPO_ROOT / "data/qa/generated_qa_yesorno.jsonl",
}

# Exact counts for 200 rows: 54%, 21%, 17%, 4%, 4%
TARGET_COUNTS_200: dict[str, int] = {
    "generated_qa_hard": 108,
    "generated_qa_dense_02": 42,
    "generated_qa_dense": 34,
    "generated_qa_num": 8,
    "generated_qa_yesorno": 8,
}

FINAL_N = 200
IAA_N = 60


def _norm_answer(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


def _load_source_index(sources: dict[str, Path]) -> dict[tuple[str, str, str], str]:
    """Map (question, answer, url) -> source key."""
    index: dict[tuple[str, str, str], str] = {}
    for src_name, path in sources.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                q, a, u = o["question"], _norm_answer(o["answer"]), o["url"].strip()
                index[(q, a, u)] = src_name
    return index


def _load_question_url_index(sources: dict[str, Path]) -> dict[tuple[str, str], str]:
    """Fallback when answer text differs (encoding edits in cleaned file)."""
    index: dict[tuple[str, str], str] = {}
    for src_name, path in sources.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                q, u = o["question"], o["url"].strip()
                index[(q, u)] = src_name
    return index


def classify_row(
    obj: dict,
    triple_index: dict[tuple[str, str, str], str],
    qu_index: dict[tuple[str, str], str],
) -> str:
    q = obj["question"]
    a = _norm_answer(obj["answer"])
    u = obj["url"].strip()
    k3 = (q, a, u)
    if k3 in triple_index:
        return triple_index[k3]
    k2 = (q, u)
    if k2 in qu_index:
        return qu_index[k2]
    raise ValueError(f"Unclassified row: question={q[:80]!r}...")


def main() -> None:
    p = argparse.ArgumentParser(description="Sample final_qa_200 and IAA annotation JSONLs")
    p.add_argument(
        "--cleaned",
        type=Path,
        default=REPO_ROOT / "data/qa/cleaned_qa_500.jsonl",
        help="Input cleaned JSONL",
    )
    p.add_argument(
        "--out-final",
        type=Path,
        default=REPO_ROOT / "data/qa/final_qa_200.jsonl",
        help="Output 200-question JSONL",
    )
    p.add_argument(
        "--out-iaa1",
        type=Path,
        default=REPO_ROOT / "annotations/iaa_annotation_sample_60.jsonl",
        help="IAA file for annotator 1",
    )
    p.add_argument(
        "--out-iaa2",
        type=Path,
        default=REPO_ROOT / "annotations/iaa_annotation_sample_60_annotator2.jsonl",
        help="IAA file for annotator 2 (same items as annotator 1)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    triple_index = _load_source_index(DEFAULT_SOURCES)
    qu_index = _load_question_url_index(DEFAULT_SOURCES)

    by_source: dict[str, list[dict]] = defaultdict(list)
    with open(args.cleaned, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = classify_row(obj, triple_index, qu_index)
            by_source[src].append(obj)

    want = dict(TARGET_COUNTS_200)
    if sum(want.values()) != FINAL_N:
        raise SystemExit(f"TARGET_COUNTS_200 must sum to {FINAL_N}")
    rng = random.Random(args.seed)

    final_rows: list[dict] = []
    for src, need in want.items():
        pool = by_source[src]
        if len(pool) < need:
            raise SystemExit(
                f"Not enough items in {src}: need {need}, have {len(pool)}"
            )
        picked = rng.sample(pool, need)
        final_rows.extend(picked)

    rng.shuffle(final_rows)

    args.out_final.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_final, "w", encoding="utf-8") as out:
        for row in final_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if IAA_N > len(final_rows):
        raise SystemExit(f"IAA sample size {IAA_N} exceeds final set {len(final_rows)}")
    iaa_indices = sorted(rng.sample(range(len(final_rows)), IAA_N))
    iaa_rows = [final_rows[i] for i in iaa_indices]

    def write_iaa(path: Path, annotator: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as out:
            for i, row in enumerate(iaa_rows, start=1):
                rec = {
                    "id": f"q{i:03d}",
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "url": row["url"],
                    "validity_label": "",
                    "annotated_answer": "",
                    "notes": "",
                    "annotator": annotator,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_iaa(args.out_iaa1, "annotator1")
    write_iaa(args.out_iaa2, "annotator2")

    # Summary
    from collections import Counter

    c = Counter()
    for row in final_rows:
        c[classify_row(row, triple_index, qu_index)] += 1
    print(f"Wrote {args.out_final} ({len(final_rows)} rows)")
    print("  per-source counts:", dict(c))
    print(f"Wrote {args.out_iaa1}")
    print(f"Wrote {args.out_iaa2} (same items, annotator2)")


if __name__ == "__main__":
    main()
