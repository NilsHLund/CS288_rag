"""
rag.py — RAG model for CS288 Assignment 3.

Retrieval pipeline:
  1. Small child chunks (100 words) for precise hybrid BM25+dense retrieval
  2. Cross-encoder re-ranking (ms-marco-MiniLM-L-12-v2)
  3. URL deduplication (max 2 chunks per source URL)
  4. Parent-document expansion (300-word window) passed to LLM
  5. Lost-in-the-middle reordering (best chunks at context edges)
"""

import json
import os
import pickle
import re
import string
from pathlib import Path
from typing import List

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import call_llm


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CORPUS_PATH = "corpus/pages_all.json"
CACHE_DIR = "cache"

CHUNK_SIZE = 100            # child chunk words (small for retrieval precision)
CHUNK_OVERLAP = 20          # overlap between child chunks
PARENT_WINDOW = 300         # wider context window expanded for LLM generation

TOP_K_RETRIEVE = 8          # final chunks after re-ranking + dedup
RERANK_FETCH_K = 60         # candidates fetched before re-ranking
MAX_CHUNKS_PER_URL = 2      # URL dedup cap after re-ranking

BM25_WEIGHT = 0.6
DENSE_WEIGHT = 0.4

EMBED_MODEL = "Snowflake/snowflake-arctic-embed-s"  # 384d, under ~400MB; rebuild cache after model change
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # 33M params, stronger cross-encoder

SYSTEM_PROMPT = (
    "You are a precise answer extractor for UC Berkeley EECS questions. "
    "Rules (follow strictly):\n"
    "1. Extract the EXACT answer phrase from the context — copy it verbatim, do not paraphrase.\n"
    "2. Answer must be UNDER 10 words. Never give full sentences.\n"
    "3. For Yes/No questions: reply only 'Yes' or 'No'.\n"
    "4. For numbers: always use digits ('3' not 'three', '4' not 'four').\n"
    "5. For acronyms or abbreviations: use the short form (e.g. 'HKN', 'BAIR', 'NSF').\n"
    "6. For names, courses, organizations: extract the exact identifier, nothing more.\n"
    "7. Never start with 'The answer is', 'According to', or any preamble — give only the answer.\n"
    "8. Reply UNKNOWN only if the answer is completely absent from the context."
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


_PREAMBLE_RE = re.compile(
    r'^(?:the answer is|answer:|according to (?:the )?context,?|based on (?:the )?context,?'
    r'|short answer:|yes[,.]?\s+|no[,.]?\s+(?=\w))\s*',
    re.IGNORECASE,
)
_TRAILING_PUNC_RE = re.compile(r'[.\s]+$')


def _clean_answer(text: str) -> str:
    """Strip common LLM preambles and trailing punctuation from answers."""
    text = _PREAMBLE_RE.sub('', text.strip())
    lower = text.lower()
    if lower.startswith('yes'):
        return 'Yes'
    if lower.startswith('no'):
        return 'No'
    return _TRAILING_PUNC_RE.sub('', text)


def build_corpus_chunks(pages, chunk_size=None, overlap=None):
    """
    Build child chunks (small) from pages, tracking word positions for
    parent-document expansion at generation time.

    Returns (chunks, page_word_lists).
    """
    cs = chunk_size if chunk_size is not None else CHUNK_SIZE
    co = overlap if overlap is not None else CHUNK_OVERLAP

    chunks = []
    page_word_lists = []

    for page_idx, page in enumerate(pages):
        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")
        full_text = f"{title}\n{text}" if title else text
        words = full_text.split()

        page_word_lists.append({"url": url, "words": words})

        start = 0
        chunk_id = 0
        while start < len(words):
            end = min(start + cs, len(words))
            chunks.append({
                "url": url,
                "title": title,
                "chunk_id": chunk_id,
                "text": " ".join(words[start:end]),
                "page_idx": page_idx,
                "word_start": start,
                "word_end": end,
            })
            chunk_id += 1
            if end == len(words):
                break
            start += cs - co

    return chunks, page_word_lists


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
        pages_cache = Path(CACHE_DIR) / "pages.pkl"
        bm25_cache = Path(CACHE_DIR) / "bm25.pkl"
        faiss_cache = Path(CACHE_DIR) / "faiss.index"
        embeddings_cache = Path(CACHE_DIR) / "embeddings.npy"

        if (
            chunks_cache.exists()
            and pages_cache.exists()
            and bm25_cache.exists()
            and faiss_cache.exists()
            and embeddings_cache.exists()
        ):
            print("[RAGModel] Loading cached index...")

            with open(chunks_cache, "rb") as f:
                self.chunks = pickle.load(f)

            with open(pages_cache, "rb") as f:
                self.page_word_lists = pickle.load(f)

            with open(bm25_cache, "rb") as f:
                self.bm25 = pickle.load(f)

            self.index = faiss.read_index(str(faiss_cache))
            self.embeddings = np.load(str(embeddings_cache))

        else:
            print("[RAGModel] Building index...")

            with open(CORPUS_PATH) as f:
                pages = json.load(f)

            self.chunks, self.page_word_lists = build_corpus_chunks(pages)

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
            with open(pages_cache, "wb") as f:
                pickle.dump(self.page_word_lists, f)
            with open(bm25_cache, "wb") as f:
                pickle.dump(self.bm25, f)
            faiss.write_index(self.index, str(faiss_cache))
            np.save(str(embeddings_cache), self.embeddings)

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)

    # ──────────────────────────────────────────────
    # Parent-document expansion
    # ──────────────────────────────────────────────

    def _get_parent_text(self, chunk: dict) -> str:
        """Expand a child chunk to a larger parent context window."""
        page = self.page_word_lists[chunk["page_idx"]]
        words = page["words"]
        center = (chunk["word_start"] + chunk["word_end"]) // 2
        half = PARENT_WINDOW // 2
        start = max(0, center - half)
        end = min(len(words), start + PARENT_WINDOW)
        return " ".join(words[start:end])

    # ──────────────────────────────────────────────
    # Lost-in-the-middle mitigation
    # ──────────────────────────────────────────────

    @staticmethod
    def _reorder_lost_in_middle(items: list) -> list:
        """
        Place highest-ranked chunks at the edges of the context window,
        where LLMs attend most reliably (counteracts lost-in-the-middle).
        """
        if len(items) <= 2:
            return items
        result: list = [None] * len(items)
        left, right = 0, len(items) - 1
        for i, item in enumerate(items):
            if i % 2 == 0:
                result[left] = item
                left += 1
            else:
                result[right] = item
                right -= 1
        return result

    # ──────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────

    def _retrieve(self, question, top_k=TOP_K_RETRIEVE, fetch_k=RERANK_FETCH_K):

        n = len(self.chunks)
        fetch_k = min(fetch_k, n)

        # Stage 1: Hybrid BM25 + dense retrieval
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
        candidate_indices = np.argsort(hybrid)[::-1][:fetch_k]
        candidates = [self.chunks[i] for i in candidate_indices]

        # Stage 2: Cross-encoder re-ranking
        try:
            pairs = [[question, c["text"]] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
        except Exception as e:
            print(f"[reranker fallback] {type(e).__name__}: {e}")
            ranked = [(0, c) for c in candidates]

        # Stage 3: URL deduplication — cap chunks per source
        url_counts: dict = {}
        deduped = []
        for _, chunk in ranked:
            url = chunk["url"]
            if url_counts.get(url, 0) < MAX_CHUNKS_PER_URL:
                deduped.append(chunk)
                url_counts[url] = url_counts.get(url, 0) + 1
            if len(deduped) >= top_k:
                break

        return deduped

    # ──────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────

    def _generate(self, question, chunks):

        # Expand each child chunk to its wider parent context window
        expanded = [
            {"url": c["url"], "text": self._get_parent_text(c)}
            for c in chunks
        ]

        # Reorder so best chunks appear at edges (counteracts lost-in-the-middle)
        expanded = self._reorder_lost_in_middle(expanded)

        context = "\n\n---\n\n".join(
            f"[Source: {c['url']}]\n{c['text']}" for c in expanded
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
                max_tokens=32,
                temperature=0.0,
                timeout=120,
            )
            answer = response.strip().splitlines()[0].strip()
            answer = _clean_answer(answer)
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
                print(f"Exception during inference [{type(e).__name__}]: {e}")
                return i, "UNKNOWN"

        with ThreadPoolExecutor(max_workers=2) as executor:
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

    questions = load_questions_from_jsonl("data/qa/generated_qa.jsonl")

    answers = model.predict(questions[:20])

    for q, a in zip(questions, answers):
        print("Q:", q)
        print("A:", a)
        print()
