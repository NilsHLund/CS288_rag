See **STRUCTURE.md** for layout.

**Evaluate (dense):** Predictions go to `data/answers/` (created automatically).
```bash
python scripts/evaluate_rag_model.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
python scripts/evaluate.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
```

**Docker (3GB RAM, 2 CPU):** Only answer generation is time-constrained; scoring is local.
```bash
docker build -t cs288-rag .
docker run --rm --cpus="2" --memory="3g" -v .:/app -w /app cs288-rag python3 scripts/evaluate_rag_model.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
# Check score (EM + F1):
python3 scripts/evaluate.py data/qa/generated_qa_dense.jsonl data/answers/generated_qa_dense_predictions.txt
```
