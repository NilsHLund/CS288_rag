"""
rag.py — RAG model for CS288 Assignment 3.

Features:
- Sentence-boundary aware chunking (child chunks, small for retrieval precision)
- Parent-document retrieval: expand to larger context window for LLM generation
- Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking (ms-marco-MiniLM-L-12-v2)
- URL-based chunk deduplication (max N chunks per URL after reranking)
- Query expansion via LLM
- HyDE (Hypothetical Document Embeddings)
- Lost-in-the-middle mitigation (best chunks placed at context edges)
- Answer post-processing to strip common LLM prefixes
"""

import json
import os
import pickle
import re
import string
from pathlib import Path
from typing import List
from urllib.parse import urlparse

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

PATH_PREFIX_EXCLUDE = []

CHILD_CHUNK_SIZE = 100       # words — small chunks for precise retrieval
CHILD_CHUNK_OVERLAP = 20     # word overlap between child chunks
PARENT_WINDOW = 350          # words — wider context window passed to LLM

CHUNK_SIZE = CHILD_CHUNK_SIZE  # alias kept for ablation.py compatibility

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GENERATION_MODEL = "meta-llama/llama-3.1-8b-instruct"

_cache_suffix = f"sent_{CHILD_CHUNK_SIZE}_{CHILD_CHUNK_OVERLAP}"
_embed_tag = EMBED_MODEL.split("/")[-1].replace(".", "_")  # e.g. bge-small-en-v1_5
_filter_tag = "_filtered" if PATH_PREFIX_EXCLUDE else ""
CACHE_DIR = f"cache/{_embed_tag}{_filter_tag}_{_cache_suffix}"

TOP_K_RETRIEVE = 30
TOP_K_RERANK = 8
MAX_CHUNKS_PER_URL = 2       # deduplication cap per URL after reranking

BM25_WEIGHT = 1.0
DENSE_WEIGHT = 1.0
RRF_K = 60                   # RRF smoothing constant (standard: 60)

ENABLE_RERANKER = True
ENABLE_QUERY_EXPANSION = False  # Disabled: often times out, adds 2 extra LLM calls
ENABLE_HYDE = False            # Disabled: often times out, adds latency

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

# Regex to strip common LLM answer prefixes before scoring
_ANSWER_PREFIX_RE = re.compile(
    r"^(short answer[:\s]+|answer[:\s]+|the answer is[:\s]+|based on the context[,\s]+)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────

def get_path_prefix(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.split("/")[0] if path else ""


def filter_pages_by_path(pages: list, exclude_prefixes: list) -> list:
    exclude = set(p.strip().lower() for p in exclude_prefixes if p.strip())
    if not exclude:
        return pages
    return [p for p in pages if get_path_prefix(p.get("url", "")).lower() not in exclude]


def normalize(text: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with simple regex heuristics."""
    # Split on . ! ? followed by whitespace + capital/digit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"])", text)
    sentences = []
    for part in parts:
        # Also split on newlines (common in web-crawled content)
        for line in part.split("\n"):
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def make_sentence_chunks(text: str, target_words: int, overlap_words: int) -> list[str]:
    """Build word-count-bounded chunks that respect sentence boundaries."""
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_wc = 0

    for sent in sentences:
        sent_wc = len(sent.split())
        if current_wc + sent_wc > target_words and current:
            chunks.append(" ".join(current))
            # Overlap: keep trailing sentences up to overlap_words
            overlap: list[str] = []
            overlap_wc = 0
            for s in reversed(current):
                sw = len(s.split())
                if overlap_wc + sw <= overlap_words:
                    overlap.insert(0, s)
                    overlap_wc += sw
                else:
                    break
            current = overlap
            current_wc = overlap_wc
        current.append(sent)
        current_wc += sent_wc

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_corpus_chunks(
    pages: list,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> tuple[list[dict], list[dict]]:
    """
    Build child chunks (small, for retrieval) and store page word lists
    (for parent-document context expansion at generation time).

    Returns:
        chunks: list of chunk dicts with page_idx, word_start, word_end
        page_word_lists: list of {"url", "title", "words"} per page
    """
    chunks: list[dict] = []
    page_word_lists: list[dict] = []

    for page_idx, page in enumerate(pages):
        url = page.get("url") or ""
        title = page.get("title") or ""
        text = page.get("text") or ""
        full_text = f"{title}\n{text}" if title else (text or "")
        words = full_text.split()
        page_word_lists.append({"url": url, "title": title, "words": words})

        child_texts = make_sentence_chunks(full_text, child_size, child_overlap)
        word_cursor = 0
        for chunk_text in child_texts:
            chunk_words = chunk_text.split()
            chunks.append({
                "url": url,
                "title": title,
                "text": chunk_text,
                "page_idx": page_idx,
                "word_start": word_cursor,
                "word_end": word_cursor + len(chunk_words),
            })
            # Advance cursor accounting for overlap
            word_cursor += max(1, len(chunk_words) - child_overlap)

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
        bm25_cache = Path(CACHE_DIR) / "bm25.pkl"
        faiss_cache = Path(CACHE_DIR) / "faiss.index"
        embeddings_cache = Path(CACHE_DIR) / "embeddings.npy"
        page_lists_cache = Path(CACHE_DIR) / "page_word_lists.pkl"

        all_cached = all(p.exists() for p in [
            chunks_cache, bm25_cache, faiss_cache, embeddings_cache, page_lists_cache
        ])

        if all_cached:
            print("[RAGModel] Loading cached index...")
            with open(chunks_cache, "rb") as f:
                self.chunks = pickle.load(f)
            with open(bm25_cache, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(page_lists_cache, "rb") as f:
                self.page_word_lists = pickle.load(f)
            self.index = faiss.read_index(str(faiss_cache))
            self.embeddings = np.load(str(embeddings_cache))
        else:
            print("[RAGModel] Building index...")
            with open(CORPUS_PATH) as f:
                pages = json.load(f)
            pages = filter_pages_by_path(pages, PATH_PREFIX_EXCLUDE)
            print(f"[RAGModel] Using {len(pages)} pages (excluded: {PATH_PREFIX_EXCLUDE})")

            self.chunks, self.page_word_lists = build_corpus_chunks(pages)
            print(f"[RAGModel] Built {len(self.chunks)} child chunks")

            tokenized = [normalize(c["text"]).split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)

            embedder = SentenceTransformer(EMBED_MODEL)
            texts = [c["text"] for c in self.chunks]
            self.embeddings = embedder.encode(
                texts, batch_size=64, show_progress_bar=True,
                normalize_embeddings=True, convert_to_numpy=True,
            ).astype("float32")

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)

            with open(chunks_cache, "wb") as f:
                pickle.dump(self.chunks, f)
            with open(bm25_cache, "wb") as f:
                pickle.dump(self.bm25, f)
            with open(page_lists_cache, "wb") as f:
                pickle.dump(self.page_word_lists, f)
            faiss.write_index(self.index, str(faiss_cache))
            np.save(str(embeddings_cache), self.embeddings)

        self.embedder = SentenceTransformer(EMBED_MODEL)

        if ENABLE_RERANKER:
            self.reranker = CrossEncoder(RERANKER_MODEL)
        else:
            self.reranker = None

    # ──────────────────────────────────────────────
    # Query Expansion
    # ──────────────────────────────────────────────

    def _expand_query(self, question: str) -> list[str]:
        """Generate 2 alternative phrasings of the question via LLM."""
        if not ENABLE_QUERY_EXPANSION:
            return [question]
        try:
            response = self.llm(
                system_prompt=(
                    "Rewrite the following question in 2 different ways. "
                    "Keep the meaning identical. Output one rewrite per line, nothing else."
                ),
                query=question,
                model="meta-llama/llama-3.1-8b-instruct",
                max_tokens=80,
                temperature=0.3,
                timeout=15,
            )
            response = (response or "").strip()
            variants = [
                line.strip().lstrip("0123456789.-) ")
                for line in response.splitlines()
                if line and line.strip()
            ]
            return [question] + variants[:2]
        except Exception as e:
            print(f"[QueryExpansion] Failed: {e}")
            return [question]

    # ──────────────────────────────────────────────
    # HyDE
    # ──────────────────────────────────────────────

    def _generate_hypothetical_doc(self, question: str) -> str:
        """Generate a short hypothetical answer passage for embedding-based retrieval."""
        if not ENABLE_HYDE:
            return ""
        try:
            response = self.llm(
                system_prompt=(
                    "Write a short factual paragraph (2-3 sentences) that would answer "
                    "the following question about UC Berkeley EECS. "
                    "Write as if it were an excerpt from the EECS website."
                ),
                query=question,
                model="meta-llama/llama-3.1-8b-instruct",
                max_tokens=80,
                temperature=0.0,
                timeout=15,
            )
            return (response or "").strip()
        except Exception as e:
            print(f"[HyDE] Failed: {e}")
            return ""

    # ──────────────────────────────────────────────
    # Parent-document retrieval
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
    # Lost-in-the-middle context ordering
    # ──────────────────────────────────────────────

    @staticmethod
    def _reorder_lost_in_middle(items: list) -> list:
        """
        Interleave items so highest-ranked (most relevant) appear at the
        edges of the context window, where LLMs attend most reliably.
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
    # Retrieval (Reciprocal Rank Fusion)
    # ──────────────────────────────────────────────

    def _retrieve(self, question: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        n = len(self.chunks)
        fetch_k = min(top_k * 15, n)

        # Run query expansion and HyDE concurrently — both are independent LLM calls
        with ThreadPoolExecutor(max_workers=2) as pool:
            expand_future = pool.submit(self._expand_query, question)
            hyde_future = pool.submit(self._generate_hypothetical_doc, question)
            queries = expand_future.result()
            hyde_doc = hyde_future.result()

        # BM25 — RRF scores (take best rank across query variants)
        bm25_rrf = np.zeros(n)
        for q in queries:
            scores = np.array(self.bm25.get_scores(normalize(q).split()))
            for rank, idx in enumerate(np.argsort(scores)[::-1][:fetch_k]):
                rrf = 1.0 / (RRF_K + rank + 1)
                if rrf > bm25_rrf[idx]:
                    bm25_rrf[idx] = rrf

        # Dense — RRF scores (take best rank across query + HyDE embeddings)
        embed_texts = [
            "Represent this sentence for searching relevant passages: " + q
            for q in queries
        ]
        if hyde_doc:
            embed_texts.append(
                "Represent this sentence for searching relevant passages: " + hyde_doc
            )

        q_embs = self.embedder.encode(
            embed_texts, normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")

        dense_rrf = np.zeros(n)
        for q_emb in q_embs:
            _, indices = self.index.search(q_emb.reshape(1, -1), fetch_k)
            for rank, idx in enumerate(indices[0]):
                if idx >= 0:
                    rrf = 1.0 / (RRF_K + rank + 1)
                    if rrf > dense_rrf[idx]:
                        dense_rrf[idx] = rrf

        hybrid = BM25_WEIGHT * bm25_rrf + DENSE_WEIGHT * dense_rrf
        top_indices = np.argsort(hybrid)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]

    # ──────────────────────────────────────────────
    # Reranking + URL deduplication
    # ──────────────────────────────────────────────

    def _rerank(self, question: str, chunks: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
        if not self.reranker or not chunks:
            return chunks[:top_k]

        pairs = [[question, c["text"]] for c in chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        # Keep at most MAX_CHUNKS_PER_URL chunks per source URL
        url_counts: dict[str, int] = {}
        deduped: list[dict] = []
        for chunk, _ in ranked:
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

    def _generate(self, question: str, chunks: list[dict]) -> str:
        # Expand each child chunk to its parent context window
        contexts = [
            {"url": c["url"], "text": self._get_parent_text(c)}
            for c in chunks
        ]

        # Reorder to counteract lost-in-the-middle degradation
        contexts = self._reorder_lost_in_middle(contexts)

        context_str = "\n\n---\n\n".join(
            f"[Source: {c['url']}]\n{c['text']}" for c in contexts
        )

        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Short answer:"
        )

        for attempt in range(3):
            try:
                response = self.llm(
                    system_prompt=SYSTEM_PROMPT,
                    query=prompt,
                    model=GENERATION_MODEL,
                    max_tokens=24,
                    temperature=0.0,
                    timeout=120,
                )
                response = (response or "").strip()
                answer = response.splitlines()[0].strip() if response else "UNKNOWN"
                answer = _ANSWER_PREFIX_RE.sub("", answer).strip()
                return answer[:80]
            except Exception as e:
                if attempt < 2:
                    continue
                return "UNKNOWN"

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def predict(self, questions: list[str]) -> list[str]:
        answers = ["UNKNOWN"] * len(questions)

        def process(i, q):
            try:
                chunks = self._retrieve(q)
                chunks = self._rerank(q, chunks)
                return i, self._generate(q, chunks)
            except Exception as e:
                print(f"Exception during inference: {e}")
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
