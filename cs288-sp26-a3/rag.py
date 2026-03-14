"""
rag.py — RAG model for CS288 Assignment 3.
"""

import json
import os
import pickle
import string
from pathlib import Path
from typing import List

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CORPUS_PATH = "corpus/pages_all.json"
CACHE_DIR = "cache"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 40

TOP_K_RETRIEVE = 5

BM25_WEIGHT = 0.7
DENSE_WEIGHT = 0.3

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about UC Berkeley EECS. "
    "Answer using ONLY the provided context. "
    "Give a SHORT answer (under 10 words). "
    "If the answer is not clearly in the context reply UNKNOWN."
    "If the answer asks for Yes/No, reply only with Yes or No."
    "If there are multiple possible answers, only reply with one most probable one."
)


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def normalize(text: str) -> str:
    import re

    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks


def build_corpus_chunks(pages):
    chunks = []

    for page in pages:

        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")

        full_text = f"{title}\n{text}" if title else text

        for i, chunk in enumerate(chunk_text(full_text)):
            chunks.append(
                {
                    "url": url,
                    "title": title,
                    "chunk_id": i,
                    "text": chunk,
                }
            )

    return chunks


def load_questions_from_jsonl(path):
    questions = []

    with open(path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])

    return questions


# ──────────────────────────────────────────────
# RAGModel
# ──────────────────────────────────────────────

class RAGModel:
    def __init__(self):

        os.makedirs(CACHE_DIR, exist_ok=True)

        self.llm = call_llm

        chunks_cache = Path(CACHE_DIR) / "chunks.pkl"
        bm25_cache = Path(CACHE_DIR) / "bm25.pkl"
        faiss_cache = Path(CACHE_DIR) / "faiss.index"
        embeddings_cache = Path(CACHE_DIR) / "embeddings.npy"

        if (
            chunks_cache.exists()
            and bm25_cache.exists()
            and faiss_cache.exists()
        ):

            print("[RAGModel] Loading cached index...")

            with open(chunks_cache, "rb") as f:
                self.chunks = pickle.load(f)

            with open(bm25_cache, "rb") as f:
                self.bm25 = pickle.load(f)

            self.index = faiss.read_index(str(faiss_cache))

            self.embeddings = np.load(str(embeddings_cache))

        else:

            print("[RAGModel] Building index...")

            with open(CORPUS_PATH) as f:
                pages = json.load(f)

            self.chunks = build_corpus_chunks(pages)

            tokenized = [normalize(c["text"]).split() for c in self.chunks]

            self.bm25 = BM25Okapi(tokenized)

            embedder = SentenceTransformer(EMBED_MODEL)

            texts = [c["text"] for c in self.chunks]

            self.embeddings = embedder.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

            dim = self.embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dim)

            self.index.add(self.embeddings)

            with open(chunks_cache, "wb") as f:
                pickle.dump(self.chunks, f)

            with open(bm25_cache, "wb") as f:
                pickle.dump(self.bm25, f)

            faiss.write_index(self.index, str(faiss_cache))

            np.save(str(embeddings_cache), self.embeddings)

        self.embedder = SentenceTransformer(EMBED_MODEL)

    # ──────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────

    def _retrieve(self, question, top_k=TOP_K_RETRIEVE):

        n = len(self.chunks)

        fetch_k = min(top_k * 10, n)

        bm25_scores = np.array(
            self.bm25.get_scores(normalize(question).split())
        )

        if bm25_scores.max() > 0:
            bm25_scores /= bm25_scores.max()

        q_emb = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        dense_scores_raw, dense_indices = self.index.search(q_emb, fetch_k)

        dense_scores = np.zeros(n)

        for idx, score in zip(dense_indices[0], dense_scores_raw[0]):
            dense_scores[idx] = score

        hybrid = BM25_WEIGHT * bm25_scores + DENSE_WEIGHT * dense_scores

        top_indices = np.argsort(hybrid)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices]

    # ──────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────

    def _generate(self, question, chunks):

        context = "\n\n---\n\n".join(
            f"[Source: {c['url']}]\n{c['text']}" for c in chunks
        )

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Short answer:"
        )

        try:

            response = self.llm(
                system_prompt=SYSTEM_PROMPT,
                query=prompt,
                model="meta-llama/llama-3.1-8b-instruct",
                max_tokens=16,
                temperature=0.0,
                timeout=10,
            )

            answer = response.strip().splitlines()[0].strip()

            return answer[:80]

        except Exception as e:

            print(e)

            return "UNKNOWN"

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def predict(self, questions: list[str]) -> list[str]:
        answers = ["UNKNOWN"] * len(questions)

        def process(i, q):
            try:
                chunks = self._retrieve(q)
                return i, self._generate(q, chunks)
            except Exception as e:
                print(f"Exception during inference, {e}")
                return i, "UNKNOWN"

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(process, i, q): i for i, q in enumerate(questions)}
            for future in as_completed(futures):
                i, answer = future.result()
                answers[i] = answer

        return answers


# ──────────────────────────────────────────────
# Run on generated QA dataset
# ──────────────────────────────────────────────

if __name__ == "__main__":

    model = RAGModel()

    questions = load_questions_from_jsonl("generated_qa.jsonl")

    answers = model.predict(questions[:20])

    for q, a in zip(questions, answers):

        print("Q:", q)
        print("A:", a)
        print()