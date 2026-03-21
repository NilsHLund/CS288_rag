See **STRUCTURE.md** for layout.

**Latency / Gradescope:** The model uses **1** OpenRouter call per question by default (`CS288_RAG_SAMPLES` unset). `CS288_RAG_FAST=1` (default in `run.sh` / Docker) also uses **sequential** `predict` and smaller retrieve/rerank windows for **2 CPU / 4GB RAM** (no GPU). For heavier local runs: `CS288_RAG_FAST=0` (larger `top_k`) and optionally `CS288_RAG_SAMPLES=3` (3-sample vote, ~3× more LLM calls).

**Timing logs:** `scripts/rag.py` prints `[RAGModel timing]` for `__init__` (cache vs embedder vs cross-encoder) and a **`predict` summary** (wall clock + avg retrieve / rerank / generate per question). Silence with `CS288_RAG_NO_PROFILE=1`. Per-question lines: `CS288_RAG_PROFILE_PER_QUESTION=1`.

**Evaluate (dense):** Predictions go to `data/answers/` (created automatically). Run from repo root (or any cwd — `evaluate_rag_model.py` switches to the project root).
```bash
python3 scripts/evaluate_rag_model.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
python3 scripts/evaluate.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
```

**Docker (4GB RAM, 2 CPU, no GPU):** Only answer generation is time-constrained; scoring is local.
```bash
docker build -t cs288-rag .
docker run --rm --cpus="2" --memory="4g" -v .:/app -w /app cs288-rag python3 scripts/evaluate_rag_model.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
# Check score (EM + F1):
python3 scripts/evaluate.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
```
