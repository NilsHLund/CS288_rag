#!/usr/bin/env python3
"""
Extract questions and gold answers from a QA JSONL into .txt files (one line per question/answer).
Then run evaluate_rag_model.py and evaluate.py.

Example:
  python scripts/qa_to_eval.py data/qa/generated_qa_100.jsonl data/qa/qa_100
  # Creates data/qa/qa_100_questions.txt and data/qa/qa_100_gold.txt
"""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/qa_to_eval.py <in.jsonl> <out-prefix>")
        print("Example: python scripts/qa_to_eval.py data/qa/generated_qa_100.jsonl data/qa/qa_100")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_prefix = Path(sys.argv[2])
    in_path.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    questions = []
    gold = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj["question"])
            gold.append(obj.get("answer", ""))
    q_path = out_prefix.parent / f"{out_prefix.name}_questions.txt"
    g_path = out_prefix.parent / f"{out_prefix.name}_gold.txt"
    with open(q_path, "w", encoding="utf-8") as f:
        f.write("\n".join(questions) + "\n")
    with open(g_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gold) + "\n")
    print(f"Wrote {len(questions)} questions → {q_path}")
    print(f"Wrote {len(gold)} gold answers → {g_path}")
    print()
    print("Next run:")
    pred_path = out_prefix.parent / f"{out_prefix.name}_predictions.txt"
    print(f"  python scripts/evaluate_rag_model.py {q_path} {pred_path}")
    print(f"  python scripts/evaluate.py {g_path} {pred_path}")

if __name__ == "__main__":
    main()
