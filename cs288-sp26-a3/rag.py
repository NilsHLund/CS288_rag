"""
rag.py — RAG model for CS288 Assignment 3.

Architecture:
  1. Chunking:    Pages are split into overlapping 200-word chunks at load time.
  2. Retrieval:   Hybrid BM25 (sparse) + MiniLM sentence embeddings (dense).
                  Scores are combined and top-k chunks are selected.
  3. Generation:  Top chunks + question sent to LLM via llm.py for a short answer.

Constraints respected:
  - No GPU (CPU-only FAISS + sentence-transformers)
  - ≤ 3 GB RAM  → small embedding model (MiniLM, ~90MB), BM25 is RAM-light
  - ≤ 0.6s per question after loading
"""

import json
import os
import pickle
import re
import string
from pathlib import Path
from typing import List
import numpy as np

import re
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from llm import call_llm  # provided by course staff — do NOT modify llm.py

# ──────────────────────────────────────────────
# Configuration — tweak these for ablations
# ──────────────────────────────────────────────
CORPUS_PATH   = "corpus/pages.json"    # output of crawl.py
CACHE_DIR     = "cache"                # where we store pre-built index files
CHUNK_SIZE    = 200                    # words per chunk
CHUNK_OVERLAP = 50                     # word overlap between consecutive chunks
TOP_K_RETRIEVE = 5                     # number of chunks passed to the LLM
BM25_WEIGHT   = 0.3                    # weight for BM25 score in hybrid ranking
DENSE_WEIGHT  = 0.7                    # weight for dense score in hybrid ranking
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # ~90 MB, fast on CPU

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about UC Berkeley EECS. "
    "Answer using ONLY the provided context. "
    "Give a SHORT answer (under 10 words). "
    "If the answer is not in the context, reply with UNKNOWN."
    "If there are multiple answers, answer with any single correct answer."
    "If the question asks for Yes/No, only reply with exactly Yes or No. Do not give additional explanations."
)


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def normalize(text: str) -> str:
    """Matches official SQuAD v1.1 eval exactly:
    lowercase -> remove punctuation -> remove articles (a/an/the) -> collapse whitespace."""
    import re as _re
    def remove_articles(t):
        return _re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t):
        return ' '.join(t.split())
    def remove_punc(t):
        return ''.join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks."""
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


def build_corpus_chunks(pages: List[dict]) -> List[dict]:
    """Convert raw page dicts into a flat list of chunk dicts."""
    chunks = []
    for page in pages:
        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")
        # Prepend title to each chunk for better relevance signal
        full_text = f"{title}\n{text}" if title else text
        for i, chunk in enumerate(chunk_text(full_text)):
            chunks.append({
                "url": url,
                "title": title,
                "chunk_id": i,
                "text": chunk,
            })
    return chunks


# ──────────────────────────────────────────────
# RAGModel
# ──────────────────────────────────────────────

class RAGModel:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.llm = call_llm

        chunks_cache    = Path(CACHE_DIR) / "chunks.pkl"
        bm25_cache      = Path(CACHE_DIR) / "bm25.pkl"
        faiss_cache     = Path(CACHE_DIR) / "faiss.index"
        embeddings_cache = Path(CACHE_DIR) / "embeddings.npy"

        # ── Load or build index ──
        if chunks_cache.exists() and bm25_cache.exists() and faiss_cache.exists():
            print("[RAGModel] Loading cached index...")
            with open(chunks_cache, "rb") as f:
                self.chunks = pickle.load(f)
            with open(bm25_cache, "rb") as f:
                self.bm25 = pickle.load(f)
            self.index = faiss.read_index(str(faiss_cache))
            self.embeddings = np.load(str(embeddings_cache))
        else:
            print("[RAGModel] Building index from scratch...")
            with open(CORPUS_PATH, "r", encoding="utf-8") as f:
                pages = json.load(f)

            self.chunks = build_corpus_chunks(pages)
            print(f"  Total chunks: {len(self.chunks)}")

            # BM25
            tokenized = [normalize(c["text"]).split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)

            # Dense embeddings
            print("  Building dense index (CPU) ...")
            embedder = SentenceTransformer(EMBED_MODEL)
            texts = [c["text"] for c in self.chunks]
            self.embeddings = embedder.encode(
                texts, batch_size=64, show_progress_bar=True,
                normalize_embeddings=True, convert_to_numpy=True
            ).astype("float32")

            # FAISS flat inner-product index (cosine since embeddings are normalized)
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

            # Cache everything
            with open(chunks_cache, "wb") as f:
                pickle.dump(self.chunks, f)
            with open(bm25_cache, "wb") as f:
                pickle.dump(self.bm25, f)
            faiss.write_index(self.index, str(faiss_cache))
            np.save(str(embeddings_cache), self.embeddings)
            print("  Index cached.")

        # Load embedding model for query-time use
        self.embedder = SentenceTransformer(EMBED_MODEL)

    # ── Retrieval ──

    def _retrieve(self, question: str, top_k: int = TOP_K_RETRIEVE) -> List[dict]:
        n = len(self.chunks)
        fetch_k = min(top_k * 10, n)  # fetch more, then re-rank

        # BM25 scores
        bm25_scores = np.array(
            self.bm25.get_scores(normalize(question).split()), dtype="float32"
        )
        # Normalize BM25 to [0, 1]
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_scores /= bm25_max

        # Dense scores (cosine similarity via FAISS)
        q_emb = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        dense_scores_raw, dense_indices = self.index.search(q_emb, fetch_k)
        dense_scores_full = np.zeros(n, dtype="float32")
        for idx, score in zip(dense_indices[0], dense_scores_raw[0]):
            if idx >= 0:
                dense_scores_full[idx] = score

        # Hybrid combination
        hybrid_scores = BM25_WEIGHT * bm25_scores + DENSE_WEIGHT * dense_scores_full
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    # ── Generation ──

    def _generate(self, question: str, chunks: List[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c['url']}]\n{c['text']}" for c in chunks
        )
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Short answer (under 10 words):"
        )
        try:
            response = self.llm(
                system_prompt=SYSTEM_PROMPT,
                query=prompt,
                model="meta-llama/llama-3.1-8b-instruct",
            )
            # Strip to first line, truncate aggressively
            answer = response.strip().splitlines()[0].strip()
            return answer
        except Exception as e:
            print(f"Exception: {e}")
            return "UNKNOWN"

    # ── Public API ──

    def predict(self, questions: List[str]) -> List[str]:
        answers = []
        for q in questions:
            try:
                chunks = self._retrieve(q)
                answer = self._generate(q, chunks)
            except Exception:
                answer = "UNKNOWN"
            answers.append(answer)
        return answers
