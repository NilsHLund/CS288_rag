# Project structure

## Overview

```
cs288-sp26-a3/
├── README.md
├── STRUCTURE.md
├── Dockerfile
├── requirements.txt
├── run.sh                 # Autograder: run.sh <questions.txt> <predictions.txt>
├── .env                    # API keys (GEMINI_API_KEY, etc.)
├── .gitignore
│
├── scripts/                # All Python scripts — run from project root
│   ├── generate_qa_dataset.py   # corpus → data/qa/generated_qa.jsonl (500)
│   ├── test_generate.py         # corpus → data/qa/test_qa_10.jsonl (10)
│   ├── qa_to_eval.py             # jsonl → questions.txt + answers.txt
│   ├── covert_qa.py               # jsonl → questions.txt + answers.txt
│   ├── evaluate_rag_model.py      # RAG: questions → predictions
│   ├── evaluate.py                # Compare predictions to reference
│   ├── rag.py                     # RAG model (used by evaluate_rag_model)
│   ├── llm.py                     # LLM calls (used by rag.py)
│   ├── compute_iaa.py             # IAA between annotator 1 and 2
│   ├── annotate_app.py            # Flask app for annotation
│   ├── ablation.py                # Ablation experiments
│   ├── crawl.py                   # Simple crawler
│   └── crawl_multi.py             # Parallel crawler
│
├── data/
│   ├── qa/                 # Q&A source (JSONL)
│   │   ├── generated_qa.jsonl
│   │   ├── generated_qa_100.jsonl
│   │   ├── generated_qa_30.jsonl
│   │   └── test_qa_10.jsonl
│   └── answers/            # Model predictions (.txt, one per line)
│
├── corpus/                 # Crawled web content
│   ├── pages_all.json
│   └── ...
│
└── annotations/            # IAA (Inter-Annotator Agreement)
    ├── generated_qa_30_annotation.jsonl
    ├── iaa_annotator1_template.jsonl
    ├── iaa_annotator2_template.jsonl
    ├── HAND_ANNOTATION_GUIDELINES.md
    └── README.md
```

## Run from project root

All `scripts/*.py` expect to be run with the current directory set to **cs288-sp26-a3/** (so that `corpus/`, `data/`, `annotations/` resolve correctly).

**Examples:**
```bash
python scripts/generate_qa_dataset.py
python scripts/test_generate.py
python scripts/annotate_app.py annotations/iaa_annotator1_template.jsonl
python scripts/compute_iaa.py --a1 annotations/iaa_annotator1_template.jsonl --a2 annotations/iaa_annotator2_template.jsonl
```

Gold answers for data/qa are in the `answer` field of each JSONL line.
