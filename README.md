See **STRUCTURE.md** for layout.

**Index cache:** `rag.py` stores a fingerprint of `corpus/pages_all.json` under `cache/…/corpus_fingerprint.txt`. If you change the corpus or chunk settings, the index rebuilds automatically. Old cache folders (e.g. `sent_100_20`, `sent_100_28`) can be deleted to save disk space.

**Latency:** The RAG model uses 3 self-consistency samples per question (better accuracy, ~1.5× more LLM calls than 2 samples).

**Evaluate (dense):** Predictions go to `data/answers/` (created automatically). Run from repo root (or any cwd — `evaluate_rag_model.py` switches to the project root).
```bash
python3 scripts/evaluate_rag_model.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
python3 scripts/evaluate.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
```

**Docker (3GB RAM, 2 CPU):** Only answer generation is time-constrained; scoring is local.
```bash
docker build -t cs288-rag .
docker run --rm --cpus="2" --memory="3g" -v .:/app -w /app cs288-rag python3 scripts/evaluate_rag_model.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
# Check score (EM + F1):
python3 scripts/evaluate.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
```
