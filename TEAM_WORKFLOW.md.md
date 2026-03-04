# RAG Assignment 3 – Team Workflow

Team Members:

- AIdan
- Hanyoon
- Nils

Goal:
Build a Retrieval-Augmented Generation (RAG) system that answers questions about <https://eecs.berkeley.edu>.

System requirements from assignment:

- CPU only
- ≤3GB RAM
- <0.6 seconds per question
- 100 questions must finish <1 minute

---

# System Pipeline

crawl website  
↓  
clean HTML  
↓  
chunk documents  
↓  
build retrieval index (BM25 + FAISS)  
↓  
retrieve passages  
↓  
LLM generation  
↓  
answer output

---

# Repository Structure

repo/

data/

- questions.txt
- answers.txt
- corpus.json
- clean_corpus.json

indices/

- passages.pkl
- bm25.pkl
- faiss.index

scripts/

- crawl.py
- clean_html.py
- chunk_docs.py
- build_index.py

core files

- rag.py
- llm.py
- evaluate_rag_model.py
- run.sh
- requirements.txt
- Dockerfile
- TEAM_WORKFLOW.md

---

## Team Responsibilities

## AIdan – Data Pipeline

Branch:
feature-data-pipeline

Files:
scripts/crawl.py  
scripts/clean_html.py  
scripts/chunk_docs.py  
data/

Responsibilities:

- Crawl EECS website
- Extract text from HTML
- Remove navigation, scripts and boilerplate
- Clean HTML content
- Chunk documents into passages

Outputs:
data/corpus.json  
data/clean_corpus.json  
indices/passages.pkl

---

## Nils – Retrieval System

Branch:
feature-retrieval

Files:
scripts/build_index.py  
rag.py  
indices/

Responsibilities:

- Build BM25 index
- Build FAISS embedding index
- Implement hybrid retrieval
- Optimize retrieval speed
- Return top-k passages

Outputs:
indices/bm25.pkl  
indices/faiss.index

Retrieval architecture goal:

BM25 top 20  
+  
FAISS top 20  
→ union  
→ rerank  
→ top 5 passages

Retrieval quality determines most of the F1 score.

---

## Hanyoon – Generation & Evaluation

Branch:
feature-generation

Files:
rag.py  
run.sh  
evaluate_rag_model.py  

Responsibilities:

- Implement LLM prompting
- Format context for LLM
- Postprocess answers
- Ensure output format is correct
- Optimize runtime (<0.6 sec per question)

Example prompt:

Answer the question using ONLY the context.  
Return a short answer (<10 words).  
If the answer is not present return UNKNOWN.

---

# GitHub Workflow

Main branch:
main

Rules:

- main must always contain working code
- no direct commits to main
- use feature branches
- merge using Pull Requests

---

# Feature Branches

feature-data-pipeline  
feature-retrieval  
feature-generation  

Each developer works only on their branch.

---

# Development Workflow

1 Pull latest code

git checkout main  
git pull  

---

2 Create feature branch

Example:

git checkout -b feature-retrieval  

---

3 Work and commit changes

Example commit messages:

add website crawler  
implement html cleaner  
add BM25 retrieval  
add FAISS index  
improve prompt formatting  
optimize retrieval speed  

---

4 Push branch

git push origin feature-retrieval  

---

5 Create Pull Request

Merge into main only after testing.

---

# Testing the System

Run evaluation script:

python evaluate_rag_model.py data/questions.txt data/answers.txt

Input format:
one question per line

Output format:
one answer per line  
same order as questions  
no newline characters

---

# Performance Constraints

Environment:

Python 3.10  
2 CPUs  
3GB RAM  
No GPU  

Runtime requirement:

100 questions < 1 minute  
Average < 0.6 seconds per question  

Test locally with:

ulimit -v $((3 *1024* 1024))  
CUDA_VISIBLE_DEVICES=""  
taskset -c 0,1 python evaluate_rag_model.py data/questions.txt data/answers.txt  

---

# Collaboration Strategy

Correct workflow:

baseline system  
↓  
each person experiments on their module  
↓  
measure F1  
↓  
merge best improvements  

Do NOT build three completely separate RAG systems.

Work on one shared pipeline.

---

# Experiments

Optional folder:

experiments/

Example experiments:

experiments/test_chunk_sizes.py  
experiments/test_bm25_vs_faiss.py  
experiments/test_prompts.py  

These experiments will be used for ablations in the report.

---

# RAG Improvements

Possible improvements:

Hybrid retrieval

BM25  
+  
FAISS embeddings

---

Reranking

score = bm25_score + cosine_similarity

---

Chunk tuning

Test chunk sizes:

100 tokens  
200 tokens  
300 tokens  

Often ~200 tokens performs best.

---

Prompt tuning

Answer the question using ONLY the context.  
Return a short answer (<10 words).  
If the answer is not in the context return UNKNOWN.

---

# Deliverables

Final submission must include:

- code
- retrieval datastore
- QA dataset
- run.sh

Autograder runs:

bash run.sh <questions_txt_path> <predictions_out_path>

Your script must produce:

one answer per question  
same order as input  
no newline characters  

---

# Final Notes

Key factors for high performance:

- high quality corpus
- good chunking
- strong retrieval
- fast inference

Retrieval quality usually determines the majority of the final F1 score.
