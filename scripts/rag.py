"""
rag.py — RAG model for CS288 Assignment 3.
"""

import gc
import json
import os
import pickle
import re
import string
import time
from pathlib import Path
from typing import List

import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm

torch.set_num_threads(2)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CORPUS_PATH = "corpus/pages_all.json"
CACHE_DIR = "cache"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 40

TOP_K_RETRIEVE = 30

BM25_WEIGHT = 0.5
DENSE_WEIGHT = 1.0

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 33M params, 384d (between MiniLM-L6 and BGE-base)

ENABLE_RERANKER = os.environ.get("RAG_ENABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "y"}

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about UC Berkeley EECS. "
    "Answer using ONLY the provided context. "
    "Extract the EXACT answer phrase from the context; do not paraphrase or give surrounding text. "
    "Give a SHORT answer (under 10 words). "
    "Only reply UNKNOWN if the answer is clearly absent from the context. "
    "If the question asks for Yes/No, reply only with Yes or No. "
    "If the question asks for an acronym or abbreviation (e.g. HKN, AUWICSEE), use that form. "
    "If the question asks for a specific identifier (course number, person name, organization), extract that exact one—not a related or parent concept. "
    "If there are multiple possible answers, pick the one that most directly answers the question."
)


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def normalize(text: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


_TAG_RE = re.compile(r"<[^>]+>")
_ZW_CHARS_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_chunk_text(text: str) -> str:
    """Best-effort prompt compressor for crawled text."""
    if not text:
        return ""
    t = _TAG_RE.sub(" ", text)
    t = _ZW_CHARS_RE.sub("", t)
    t = t.replace("\xa0", " ")
    t = _WS_RE.sub(" ", t)
    t = _NEWLINE_RE.sub("\n\n", t)
    return t.strip()


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
        self._profile_llm = os.environ.get("RAG_PROFILE_LLM", "").strip().lower() in {"1", "true", "yes", "y"}
        if ENABLE_RERANKER:
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                device="cpu",
                max_length=256,
                default_activation_function=torch.nn.Sigmoid(),
            )
        else:
            self.reranker = None
        self._rerank_keep_k = 3

        chunks_cache = Path(CACHE_DIR) / "chunks.pkl"
        bm25_cache = Path(CACHE_DIR) / "bm25.pkl"
        faiss_cache = Path(CACHE_DIR) / "faiss.index"
        embeddings_cache = Path(CACHE_DIR) / "embeddings.npy"

        if (
            chunks_cache.exists()
            and bm25_cache.exists()
            and faiss_cache.exists()
            and embeddings_cache.exists()
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

        fetch_k = min(top_k * 15, n)

        bm25_scores = np.array(
            self.bm25.get_scores(normalize(question).split())
        )

        if bm25_scores.max() > 0:
            bm25_scores /= bm25_scores.max()

        q_emb = self.embedder.encode(
            ["Represent this sentence for searching relevant passages: " + question],
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

    def _rerank(self, question: str, chunks: list[dict], keep_k: int | None = None) -> list[dict]:
        if not chunks or not self.reranker:
            return chunks
        keep_k = keep_k or self._rerank_keep_k
        pairs = [(question, c.get("text", "")) for c in chunks]
        with torch.no_grad():
            scores = self.reranker.predict(
                pairs,
                batch_size=10,
                show_progress_bar=False,
            )
        order = np.argsort(scores)[::-1]
        top_idx = order[:keep_k]
        reranked = [chunks[i] for i in top_idx]
        top_scores = scores[top_idx]

        del pairs, scores, order, top_idx, top_scores
        gc.collect()

        return reranked

    # ──────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────

    def _generate(self, question, chunks):

        context = "\n\n---\n\n".join(
            f"[Source: {c['url']}]\n{clean_chunk_text(c.get('text', ''))}" for c in chunks
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
                max_tokens=24,
                temperature=0.0,
                timeout=120,
            )
            if self._profile_llm:
                print(f"[RAG_PROFILE_LLM] chunks={len(chunks)} prompt_chars={len(prompt)}")

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
        for i, q in enumerate(questions):
            try:
                chunks = self._retrieve(q, top_k=TOP_K_RETRIEVE)
                chunks = self._rerank(q, chunks, keep_k=3)
                t0 = time.time() if self._profile_llm else None
                ans = self._generate(q, chunks)
                if t0 is not None:
                    print(f"[RAG_PROFILE_LLM] openrouter_s={time.time() - t0:.3f}")
                answers[i] = ans
            except Exception as e:
                print(f"Exception during inference for question {i}: {e}")
                answers[i] = "UNKNOWN"
        return answers


# ──────────────────────────────────────────────
# Run on generated QA dataset
# ──────────────────────────────────────────────

if __name__ == "__main__":

    model = RAGModel()

    questions = load_questions_from_jsonl("data/qa/generated_qa.jsonl")

    answers = model.predict(questions[:20])

    for q, a in zip(questions, answers):

        print("Q:", q)
        print("A:", a)
        print()