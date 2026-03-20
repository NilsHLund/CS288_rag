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
from urllib.parse import unquote, urlparse

import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from llm import call_llm

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CORPUS_PATH = "corpus/pages_all.json"
CACHE_DIR = "cache"
CACHE_VERSION = "v2_table_signal"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 40

TOP_K_RETRIEVE = 15

BM25_WEIGHT = 0.5
DENSE_WEIGHT = 1.0

EMBED_MODEL = "./models/all-MiniLM-L6-v2"
RERANK_MODEL = "./models/ms-marco-TinyBERT-L-2-v2"

ENABLE_RERANKER = os.environ.get("RAG_ENABLE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "y"}
ENABLE_PROGRESS_LOGS = os.environ.get("RAG_PROGRESS_LOGS", "0").strip().lower() in {"1", "true", "yes", "y"}
RERANKER_BACKEND = os.environ.get("RAG_RERANKER_BACKEND", "safe").strip().lower()
FORCE_BM25_ONLY = os.environ.get("RAG_FORCE_BM25_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"}
LLM_RETRIES = max(1, int(os.environ.get("RAG_LLM_RETRIES", "1")))
LLM_TIMEOUT_SECONDS = max(5, int(os.environ.get("RAG_LLM_TIMEOUT", "25")))
LLM_RETRY_SLEEP_SECONDS = max(0.0, float(os.environ.get("RAG_LLM_RETRY_SLEEP", "0.5")))

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


def get_path_prefix(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.split("/")[0] if path else ""


TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|\s*$")
ACRONYM_RE = re.compile(r"\b([A-Za-z][A-Za-z/&\-\s]{3,80}?)\s+\(([A-Z][A-Z0-9&]{1,10})\)")
COURSE_RE = re.compile(r"\b([A-Z]{2,5})\s*[-]?\s*(\d{1,3}[A-Z]?)\b")


def is_markdown_table_start(lines: list[str], idx: int) -> bool:
    if idx + 1 >= len(lines):
        return False
    return bool(TABLE_ROW_RE.match(lines[idx])) and bool(TABLE_SEP_RE.match(lines[idx + 1]))


def split_content_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    i = 0

    while i < len(lines):
        if is_markdown_table_start(lines, i):
            start = i
            i += 2
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                i += 1
            table_block = "\n".join(l.strip() for l in lines[start:i] if l.strip())
            if table_block:
                blocks.append(("table", table_block))
            continue

        start = i
        i += 1
        while i < len(lines) and not is_markdown_table_start(lines, i):
            i += 1
        text_block = "\n".join(lines[start:i]).strip()
        if text_block:
            blocks.append(("text", text_block))

    return blocks


def split_md_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def parse_markdown_table(table_block: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in table_block.splitlines() if line.strip()]
    if len(lines) < 2:
        return [], []

    headers = split_md_row(lines[0])
    data_start = 2 if TABLE_SEP_RE.match(lines[1]) else 1
    rows = [split_md_row(line) for line in lines[data_start:] if TABLE_ROW_RE.match(line)]
    if not headers or not rows:
        return headers, rows

    width = max(len(headers), max(len(r) for r in rows))
    headers = headers + [""] * (width - len(headers))
    rows = [r + [""] * (width - len(r)) for r in rows]
    return headers, rows


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def path_terms(url: str, max_terms: int = 10) -> list[str]:
    raw_path = unquote(urlparse(url).path)
    tokens = re.findall(r"[A-Za-z0-9]+", raw_path.lower())
    return tokens[:max_terms]


def build_corpus_paraphrase(
    text: str,
    title: str,
    url: str,
    chunk_type: str,
    table_columns: list[str] | None = None,
) -> str:
    signals = []

    if title:
        signals.append(f"topic {title}")

    for phrase, acronym in ACRONYM_RE.findall(f"{title} {text}"):
        phrase = " ".join(phrase.split())
        if len(phrase.split()) <= 12:
            signals.append(f"{acronym} {phrase}")
        if len(signals) >= 12:
            break

    for dept, code in COURSE_RE.findall(text):
        signals.append(f"{dept} {code}")
        signals.append(f"{dept}{code}")
        if len(signals) >= 18:
            break

    tokens = path_terms(url)
    if tokens:
        signals.append("path " + " ".join(tokens))

    if chunk_type == "table_row" and table_columns:
        cols = [c for c in table_columns if c][:6]
        if cols:
            signals.append("columns " + " ".join(cols))

    deduped = []
    seen = set()
    for s in signals:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(s.strip())
        if len(deduped) >= 16:
            break

    return " ; ".join(deduped)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []

    if not words:
        return chunks

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
        path_prefix = get_path_prefix(url)

        full_text = f"{title}\n{text}" if title else text
        blocks = split_content_blocks(full_text)
        chunk_id = 0

        for block_type, block in blocks:
            if block_type == "table":
                columns, rows = parse_markdown_table(block)
                if rows:
                    for row in rows:
                        pairs = []
                        for col, val in zip(columns, row):
                            if col and val:
                                pairs.append(f"{col}: {val}")
                        row_text = "; ".join(pairs) if pairs else " | ".join(row)
                        row_text = truncate_words(row_text, 130)
                        text_out = (
                            f"Table columns: {' | '.join(c for c in columns if c)}\n"
                            f"Table row: {row_text}"
                        ).strip()
                        paraphrase = build_corpus_paraphrase(
                            text_out, title, url, "table_row", table_columns=columns
                        )
                        retrieval_text = f"{text_out}\n{paraphrase}" if paraphrase else text_out
                        chunks.append(
                            {
                                "url": url,
                                "title": title,
                                "path_prefix": path_prefix,
                                "chunk_id": chunk_id,
                                "chunk_type": "table_row",
                                "text": text_out,
                                "retrieval_text": retrieval_text,
                            }
                        )
                        chunk_id += 1
                    continue

            for text_chunk in chunk_text(block):
                paraphrase = build_corpus_paraphrase(text_chunk, title, url, "text")
                retrieval_text = f"{text_chunk}\n{paraphrase}" if paraphrase else text_chunk
                chunks.append(
                    {
                        "url": url,
                        "title": title,
                        "path_prefix": path_prefix,
                        "chunk_id": chunk_id,
                        "chunk_type": "text",
                        "text": text_chunk,
                        "retrieval_text": retrieval_text,
                    }
                )
                chunk_id += 1

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
        self._progress_logs = ENABLE_PROGRESS_LOGS
        self._dense_enabled = not FORCE_BM25_ONLY
        if self._progress_logs:
            print(
                f"[RAG_PROGRESS] init start | reranker_enabled={ENABLE_RERANKER} "
                f"force_bm25_only={FORCE_BM25_ONLY}"
            )
        if self._dense_enabled and ENABLE_RERANKER and RERANKER_BACKEND == "crossencoder":
            self.reranker = CrossEncoder(
                RERANK_MODEL,
                device="cpu",
                max_length=256,
                activation_fn=torch.nn.Sigmoid(),
            )
        else:
            self.reranker = None
        self._rerank_keep_k = 3

        cache_tag = f"{CACHE_VERSION}_cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}"
        tagged_chunks_cache = Path(CACHE_DIR) / f"chunks_{cache_tag}.pkl"
        tagged_bm25_cache = Path(CACHE_DIR) / f"bm25_{cache_tag}.pkl"
        tagged_faiss_cache = Path(CACHE_DIR) / f"faiss_{cache_tag}.index"
        tagged_embeddings_cache = Path(CACHE_DIR) / f"embeddings_{cache_tag}.npy"

        legacy_chunks_cache = Path(CACHE_DIR) / "chunks.pkl"
        legacy_bm25_cache = Path(CACHE_DIR) / "bm25.pkl"
        legacy_faiss_cache = Path(CACHE_DIR) / "faiss.index"
        legacy_embeddings_cache = Path(CACHE_DIR) / "embeddings.npy"

        if (
            tagged_chunks_cache.exists()
            and tagged_bm25_cache.exists()
            and tagged_faiss_cache.exists()
            and tagged_embeddings_cache.exists()
        ):
            chunks_cache = tagged_chunks_cache
            bm25_cache = tagged_bm25_cache
            faiss_cache = tagged_faiss_cache
            embeddings_cache = tagged_embeddings_cache
        else:
            # Autograder safety: prefer pre-existing legacy cache files if the
            # new tagged cache artifacts were not committed.
            chunks_cache = legacy_chunks_cache
            bm25_cache = legacy_bm25_cache
            faiss_cache = legacy_faiss_cache
            embeddings_cache = legacy_embeddings_cache

        cache_exists = chunks_cache.exists() and bm25_cache.exists() and (
            (self._dense_enabled and faiss_cache.exists() and embeddings_cache.exists())
            or (not self._dense_enabled)
        )
        cache_loaded = False

        if cache_exists:

            print("[RAGModel] Loading cached index...")
            if self._progress_logs:
                print(
                    "[RAG_PROGRESS] cache load from "
                    f"{chunks_cache.name}, {bm25_cache.name}, {faiss_cache.name}, {embeddings_cache.name}"
                )

            try:
                with open(chunks_cache, "rb") as f:
                    self.chunks = pickle.load(f)

                with open(bm25_cache, "rb") as f:
                    self.bm25 = pickle.load(f)

                if self._dense_enabled:
                    self.index = faiss.read_index(str(faiss_cache))
                    self.embeddings = np.load(str(embeddings_cache))
                else:
                    self.index = None
                    self.embeddings = None
                cache_loaded = True

                if self._progress_logs:
                    print(
                        f"[RAG_PROGRESS] cache loaded | chunks={len(self.chunks)} "
                        f"embeddings_shape={getattr(self.embeddings, 'shape', None)}"
                    )
            except Exception as e:
                print(f"[RAGModel] Warning: cache load failed, rebuilding index. Error: {e}")

        if not cache_loaded:

            print("[RAGModel] Building index...")
            if self._progress_logs:
                print("[RAG_PROGRESS] cache miss, building index from corpus")

            with open(CORPUS_PATH) as f:
                pages = json.load(f)

            self.chunks = build_corpus_chunks(pages)

            tokenized = [normalize(c["retrieval_text"]).split() for c in self.chunks]

            self.bm25 = BM25Okapi(tokenized)

            self.index = None
            self.embeddings = None
            if self._dense_enabled:
                try:
                    embedder = SentenceTransformer(EMBED_MODEL)
                    texts = [c["retrieval_text"] for c in self.chunks]
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
                except Exception as e:
                    self._dense_enabled = False
                    self.reranker = None
                    self.index = None
                    self.embeddings = None
                    print(f"[RAGModel] Warning: dense index build failed, using BM25-only retrieval. Error: {e}")

            with open(chunks_cache, "wb") as f:
                pickle.dump(self.chunks, f)

            with open(bm25_cache, "wb") as f:
                pickle.dump(self.bm25, f)

            if self._dense_enabled and self.index is not None and self.embeddings is not None:
                faiss.write_index(self.index, str(faiss_cache))
                np.save(str(embeddings_cache), self.embeddings)
            if self._progress_logs:
                print(
                    f"[RAG_PROGRESS] index build complete | chunks={len(self.chunks)} "
                    f"embeddings_shape={getattr(self.embeddings, 'shape', None)}"
                )

        if not self._dense_enabled:
            self.embedder = None
            self.reranker = None
            if self._progress_logs:
                print("[RAG_PROGRESS] dense retrieval disabled; BM25-only mode")
        else:
            try:
                self.embedder = SentenceTransformer(EMBED_MODEL)
                if self._progress_logs:
                    print("[RAG_PROGRESS] embedder ready")
            except Exception as e:
                self.embedder = None
                self._dense_enabled = False
                self.reranker = None
                print(f"[RAGModel] Warning: embedder unavailable, using BM25-only retrieval. Error: {e}")

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

        if not self._dense_enabled or self.embedder is None:
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            return [self.chunks[i] for i in top_indices]

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
        if not chunks:
            return chunks
        keep_k = keep_k or self._rerank_keep_k
        # Default "safe" backend: no CrossEncoder.predict call (avoids segfault on some macOS stacks).
        if ENABLE_RERANKER and RERANKER_BACKEND != "crossencoder":
            if self.embedder is None:
                return chunks[:keep_k]
            q_emb = self.embedder.encode(
                ["Represent this sentence for searching relevant passages: " + question],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")[0]
            chunk_texts = [c.get("text", "") for c in chunks]
            c_emb = self.embedder.encode(
                chunk_texts,
                batch_size=16,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            dense_scores = np.dot(c_emb, q_emb)
            q_tok = set(normalize(question).split())
            overlap_scores = []
            for c in chunks:
                t_tok = set(normalize(c.get("text", "")).split())
                overlap_scores.append(len(q_tok & t_tok) / max(1, len(q_tok)))
            overlap_scores = np.array(overlap_scores, dtype="float32")
            scores = 0.85 * dense_scores + 0.15 * overlap_scores
            order = np.argsort(scores)[::-1]
            top_idx = order[:keep_k]
            reranked = [chunks[i] for i in top_idx]
            del q_emb, chunk_texts, c_emb, dense_scores, q_tok, overlap_scores, scores, order, top_idx
            gc.collect()
            return reranked

        if not self.reranker:
            return chunks

        pairs = [(question, c.get("text", "")) for c in chunks]
        with torch.no_grad():
            scores = self.reranker.predict(
                pairs,
                batch_size=1,
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

        for attempt in range(LLM_RETRIES):
            try:
                response = self.llm(
                    system_prompt=SYSTEM_PROMPT,
                    query=prompt,
                    model="meta-llama/llama-3.1-8b-instruct",
                    max_tokens=24,
                    temperature=0.0,
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                if self._profile_llm:
                    print(f"[RAG_PROFILE_LLM] chunks={len(chunks)} prompt_chars={len(prompt)}")

                answer = response.strip().splitlines()[0].strip()
                return answer[:80]
            except Exception as e:
                if attempt < LLM_RETRIES - 1:
                    print(f"[RAGModel] generation attempt {attempt + 1} failed: {e}")
                    time.sleep(LLM_RETRY_SLEEP_SECONDS)
                    continue
                print(e)
                return "UNKNOWN"

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def predict(self, questions: list[str]) -> list[str]:
        answers = ["UNKNOWN"] * len(questions)
        if self._progress_logs:
            print(f"[RAG_PROGRESS] predict start | total_questions={len(questions)}")
        for i, q in enumerate(questions):
            try:
                if self._progress_logs:
                    print(f"[RAG_PROGRESS] q{i + 1}/{len(questions)} retrieve start")
                chunks = self._retrieve(q, top_k=TOP_K_RETRIEVE)
                if self._progress_logs:
                    print(f"[RAG_PROGRESS] q{i + 1}/{len(questions)} retrieved_chunks={len(chunks)}")
                chunks = self._rerank(q, chunks, keep_k=3)
                if self._progress_logs:
                    print(f"[RAG_PROGRESS] q{i + 1}/{len(questions)} post_rerank_chunks={len(chunks)}")
                t0 = time.time() if self._profile_llm else None
                ans = self._generate(q, chunks)
                if t0 is not None:
                    print(f"[RAG_PROFILE_LLM] openrouter_s={time.time() - t0:.3f}")
                answers[i] = ans
                if self._progress_logs:
                    print(f"[RAG_PROGRESS] q{i + 1}/{len(questions)} done")
            except Exception as e:
                print(f"Exception during inference for question {i}: {e}")
                answers[i] = "UNKNOWN"
        if self._progress_logs:
            print("[RAG_PROGRESS] predict complete")
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