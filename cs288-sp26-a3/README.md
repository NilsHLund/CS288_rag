## How to run locally

```bash
python evaluate_rag_model.py data_500/questions.txt data_500/answers.txt
python evaluate.py data_500/answers_gold.txt data_500/answers.txt > results.txt
```

## How to mimic test environment (with 3GB RAM, 2 CPUs)
```bash
# Run once
docker build -t cs288-rag .

docker run --rm --cpus="2" --memory="3g" -v .:/app -w /app cs288-rag python3 evaluate_rag_model.py data_500/questions_100.txt data_500/answers_100.txt
```
