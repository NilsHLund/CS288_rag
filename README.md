See **STRUCTURE.md** for layout and “How it fits together”.

**Evaluate (100 questions):** Predictions go to `data/answers/` (created automatically).
```bash
python3 scripts/evaluate_rag_model.py data/qa/generated_qa_100.jsonl data/answers/generated_qa_100_predictions.txt

python3 scripts/evaluate.py data/qa/generated_qa_100.jsonl data/answers/generated_qa_100_predictions.txt
```

**Docker (3GB RAM, 2 CPU):**
```bash
docker build -t cs288-rag .

docker run --rm --cpus="2" --memory="3g" -v .:/app -w /app cs288-rag python3 scripts/evaluate_rag_model.py data/qa/generated_qa_100.jsonl data/answers/generated_qa_100_predictions.txt
```
