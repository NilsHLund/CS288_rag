"""
rag.py — RAG model for CS288 Assignment 3.

Changes from v3:
  - FIX: Dense embeddings now use retrieval_text (same as BM25), not raw text
  - FIX: minmax_normalize replaced with stable_softmax for dense scores —
         preserves absolute similarity signal instead of amplifying noise
  - FIX: MAX_CHUNKS_PER_URL raised 2 → 5 for list-heavy pages
  - FIX: max_tokens raised 24 → 48 to avoid truncating long answers
  - FIX: Page summaries rebuilt from title + section headers + first sentences
         instead of a raw word-truncation of the page text
  - NEW: Query-time HyDE — LLM generates a hypothetical answer passage before
         retrieval; its embedding is averaged with the real query embedding so
         dense retrieval is pulled toward the answer space. One LLM call per
         question at inference time (same budget as generation).
"""

import json
import os
import pickle
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import List
from urllib.parse import unquote, urlparse

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

# Exclude low-value path prefixes to trim corpus (empty = use full corpus)
PATH_PREFIX_EXCLUDE = ["Pubs", "news"]  # [] for full corpus; ["Pubs","news"] ~7k pages removed

# Separate cache per config
CACHE_DIR = "cache/bge_base_filtered" if PATH_PREFIX_EXCLUDE else "cache"
CACHE_VERSION = "v4_fixes"  # bumped — forces index rebuild with all fixes

CHUNK_SIZE = 170
CHUNK_OVERLAP = 60

TOP_K_RETRIEVE = 10
PAGE_TOP_K = 25
MAX_CHUNKS_PER_URL = 5          # FIX: was 2; raised for list-heavy pages

BM25_WEIGHT = 0.7
DENSE_WEIGHT = 1.0
PAGE_BM25_WEIGHT = 0.6
PAGE_DENSE_WEIGHT = 1.0
PAGE_PRIOR_WEIGHT = 0.25

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# HyDE: weight of the hypothetical passage embedding blended with the real query embedding.
# 0.0 = pure query embedding (HyDE disabled); 0.5 = equal blend; 1.0 = hypothetical only.
HYDE_WEIGHT = 0.5

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about UC Berkeley EECS. "
    "Answer using ONLY the provided context. "
    "Extract the EXACT answer phrase from the context; do not paraphrase or give surrounding text. "
    "Give a SHORT answer (under 10 words). "
    "Only reply UNKNOWN if the answer is clearly absent from the context. "
    "If the question asks for Yes/No, reply only with Yes or No. "
    "If the question asks for an acronym or abbreviation (e.g. HKN, AUWICSEE), use that form. "
    "If the question asks for a specific identifier (course number, person name, organization), extract that exact one—not a related or parent concept. "
    "When context is from table rows, return the exact table cell value needed. "
    "If there are multiple possible answers, pick the one that most directly answers the question."
)


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def get_path_prefix(url: str) -> str:
    """Extract first path segment from URL (e.g. 'academics' from /academics/graduate/...)."""
    path = urlparse(url).path.strip("/")
    return path.split("/")[0] if path else ""


def filter_pages_by_path(pages: list, exclude_prefixes: list[str]) -> list:
    """Keep only pages whose URL path prefix is not in exclude_prefixes."""
    exclude = set(p.strip().lower() for p in exclude_prefixes if p.strip())
    if not exclude:
        return pages
    kept = []
    for p in pages:
        prefix = get_path_prefix(p.get("url", "")).lower()
        if prefix not in exclude:
            kept.append(p)
    return kept


def normalize(text: str) -> str:
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


# ──────────────────────────────────────────────
# FIX: Smarter page summary
# ──────────────────────────────────────────────

def build_page_summary(text: str, title: str, max_words: int = 220) -> str:
    """
    Build a page-level summary from title + section headings + first sentence
    of each paragraph, rather than a raw word-truncation of the full text.
    This gives the page-level ranker signal from across the whole page,
    not just the first N words.
    """
    lines = text.splitlines()
    parts = []
    if title:
        parts.append(title)

    word_count = len(title.split()) if title else 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Heuristic: heading if short, no period, possibly title-cased or ALL CAPS
        is_heading = (
            len(stripped.split()) <= 8
            and not stripped.endswith(".")
            and (stripped.istitle() or stripped.isupper() or stripped[0].isupper())
        )

        if is_heading:
            candidate = stripped
        else:
            # Take just the first sentence of the paragraph
            first_sentence = re.split(r"(?<=[.!?])\s", stripped)[0]
            candidate = first_sentence

        candidate_words = candidate.split()
        if word_count + len(candidate_words) > max_words:
            remaining = max_words - word_count
            if remaining > 3:
                parts.append(" ".join(candidate_words[:remaining]))
            break

        parts.append(candidate)
        word_count += len(candidate_words)

    return "\n".join(parts)


# ──────────────────────────────────────────────
# Keyword paraphrase signals (kept but no longer sole source)
# ──────────────────────────────────────────────

def build_keyword_signals(
    text: str,
    title: str,
    url: str,
    chunk_type: str,
    table_columns: list[str] | None = None,
) -> str:
    """
    Regex-derived keyword signals: acronyms, course codes, URL path tokens.
    These complement the LLM-generated hypothetical questions.
    """
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


# ──────────────────────────────────────────────
# NEW: Query-time HyDE
# ──────────────────────────────────────────────

def generate_hypothetical_passage(question: str) -> str:
    """
    HyDE (Hypothetical Document Embeddings): ask the LLM to draft a short passage
    that would answer the question, without access to the real corpus.
    Its embedding is blended with the real query embedding so dense retrieval is
    pulled toward the answer space. One LLM call per question at inference time.
    Returns empty string on any failure (non-fatal — falls back to query-only).
    """
    prompt = (
        f"Write a short factual passage (2-3 sentences) that directly answers the "
        f"following question about UC Berkeley EECS. Do not say you don't know — "
        f"write a plausible answer as if you were an expert.\n\nQuestion: {question}"
    )
    try:
        return call_llm(
            query=prompt,
            system_prompt="You are a knowledgeable assistant about UC Berkeley EECS.",
            max_tokens=80,
            temperature=0.0,
        )
    except Exception:
        return ""


def build_retrieval_text(
    text: str,
    title: str,
    url: str,
    chunk_type: str,
    table_columns: list[str] | None = None,
) -> str:
    """
    Compose the full retrieval_text used for BOTH BM25 and dense embedding.
    Order: chunk text → keyword signals.
    """
    keyword_signals = build_keyword_signals(text, title, url, chunk_type, table_columns)
    parts = [text]
    if keyword_signals:
        parts.append(keyword_signals)
    return "\n".join(parts)


# ──────────────────────────────────────────────
# Page and chunk builders
# ──────────────────────────────────────────────

def build_page_records(pages: list[dict]) -> list[dict]:
    records = []
    for page_id, page in enumerate(pages):
        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")
        meta_description = page.get("meta_description", "")
        prefix = get_path_prefix(url)

        # FIX: use structured summary instead of raw word-truncation
        summary = build_page_summary(text, title="", max_words=220)
        terms = " ".join(path_terms(url))
        page_doc = "\n".join(
            part
            for part in [
                title,
                meta_description,
                f"section {prefix}" if prefix else "",
                terms,
                summary,
            ]
            if part
        )

        records.append(
            {
                "page_id": page_id,
                "url": url,
                "title": title,
                "path_prefix": prefix,
                "page_doc": page_doc,
            }
        )

    return records


def build_corpus_chunks(pages: list[dict]):
    """
    Build chunks from pages. retrieval_text = chunk text + keyword signals.
    HyDE is applied at query time in _encode_query, not here.
    """
    # Collect raw chunks
    chunks = []

    for page_id, page in enumerate(pages):
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
                        chunks.append(
                            {
                                "url": url,
                                "title": title,
                                "path_prefix": path_prefix,
                                "page_id": page_id,
                                "chunk_id": chunk_id,
                                "chunk_type": "table_row",
                                "table_columns": columns,
                                "text": text_out,
                            }
                        )
                        chunk_id += 1
                    continue

            for text_chunk in chunk_text(block):
                chunks.append(
                    {
                        "url": url,
                        "title": title,
                        "path_prefix": path_prefix,
                        "page_id": page_id,
                        "chunk_id": chunk_id,
                        "chunk_type": "text",
                        "text": text_chunk,
                        }
                )
                chunk_id += 1

    # Finalize chunks with retrieval_text (keyword signals only; HyDE is query-time)
    for chunk in chunks:
        chunk["retrieval_text"] = build_retrieval_text(
            text=chunk["text"],
            title=chunk["title"],
            url=chunk["url"],
            chunk_type=chunk["chunk_type"],
            table_columns=chunk.get("table_columns"),
        )

    return chunks


# ──────────────────────────────────────────────
# FIX: Score normalization
# ──────────────────────────────────────────────

def max_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 0:
        return arr / max_val
    return arr


def stable_softmax(arr: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """
    FIX: Replace minmax_normalize for dense scores.

    minmax_normalize makes the worst candidate score 0 and best score 1,
    regardless of absolute similarity — it amplifies noise and flattens
    meaningful score gaps. softmax over cosine similarities preserves the
    relative ordering while keeping the score distribution meaningful.

    temperature: lower = sharper (more winner-takes-all); 0.1 works well
    for cosine sims in [0, 1].
    """
    arr = arr.astype("float32")
    if not arr.size:
        return arr
    shifted = (arr - arr.max()) / temperature   # numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum()


# ──────────────────────────────────────────────
# RAGModel
# ──────────────────────────────────────────────

class RAGModel:
    def __init__(self):

        os.makedirs(CACHE_DIR, exist_ok=True)

        self.llm = call_llm
        self.embedder = SentenceTransformer(EMBED_MODEL)

        cache_tag = f"{CACHE_VERSION}_cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}"
        chunks_cache = Path(CACHE_DIR) / f"chunks_{cache_tag}.pkl"
        pages_cache = Path(CACHE_DIR) / f"pages_{cache_tag}.pkl"
        bm25_cache = Path(CACHE_DIR) / f"bm25_{cache_tag}.pkl"
        page_bm25_cache = Path(CACHE_DIR) / f"page_bm25_{cache_tag}.pkl"
        faiss_cache = Path(CACHE_DIR) / f"faiss_{cache_tag}.index"
        page_faiss_cache = Path(CACHE_DIR) / f"page_faiss_{cache_tag}.index"
        embeddings_cache = Path(CACHE_DIR) / f"embeddings_{cache_tag}.npy"
        page_embeddings_cache = Path(CACHE_DIR) / f"page_embeddings_{cache_tag}.npy"

        if (
            chunks_cache.exists()
            and pages_cache.exists()
            and bm25_cache.exists()
            and page_bm25_cache.exists()
            and faiss_cache.exists()
            and page_faiss_cache.exists()
            and embeddings_cache.exists()
            and page_embeddings_cache.exists()
        ):

            print("[RAGModel] Loading cached index...")

            with open(chunks_cache, "rb") as f:
                self.chunks = pickle.load(f)

            with open(pages_cache, "rb") as f:
                self.pages = pickle.load(f)

            with open(bm25_cache, "rb") as f:
                self.bm25 = pickle.load(f)

            with open(page_bm25_cache, "rb") as f:
                self.page_bm25 = pickle.load(f)

            self.index = faiss.read_index(str(faiss_cache))
            self.page_index = faiss.read_index(str(page_faiss_cache))

            self.embeddings = np.load(str(embeddings_cache))
            self.page_embeddings = np.load(str(page_embeddings_cache))

        else:

            print("[RAGModel] Building index...")

            with open(CORPUS_PATH) as f:
                pages = json.load(f)

            pages = filter_pages_by_path(pages, PATH_PREFIX_EXCLUDE)
            print(f"[RAGModel] Using {len(pages)} pages (excluded: {PATH_PREFIX_EXCLUDE})")

            self.pages = build_page_records(pages)
            self.chunks = build_corpus_chunks(pages)

            tokenized = [normalize(c["retrieval_text"]).split() for c in self.chunks]
            page_tokenized = [normalize(p["page_doc"]).split() for p in self.pages]

            self.bm25 = BM25Okapi(tokenized)
            self.page_bm25 = BM25Okapi(page_tokenized)

            # FIX: encode retrieval_text for dense embeddings too (was: c["text"])
            retrieval_texts = [c["retrieval_text"] for c in self.chunks]
            page_texts = [p["page_doc"] for p in self.pages]

            self.embeddings = self.embedder.encode(
                retrieval_texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            self.page_embeddings = self.embedder.encode(
                page_texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

            dim = self.embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dim)
            self.page_index = faiss.IndexFlatIP(dim)

            self.index.add(self.embeddings)
            self.page_index.add(self.page_embeddings)

            with open(chunks_cache, "wb") as f:
                pickle.dump(self.chunks, f)

            with open(pages_cache, "wb") as f:
                pickle.dump(self.pages, f)

            with open(bm25_cache, "wb") as f:
                pickle.dump(self.bm25, f)

            with open(page_bm25_cache, "wb") as f:
                pickle.dump(self.page_bm25, f)

            faiss.write_index(self.index, str(faiss_cache))
            faiss.write_index(self.page_index, str(page_faiss_cache))

            np.save(str(embeddings_cache), self.embeddings)
            np.save(str(page_embeddings_cache), self.page_embeddings)

        self.page_to_chunk_ids = defaultdict(list)
        for idx, chunk in enumerate(self.chunks):
            self.page_to_chunk_ids[int(chunk.get("page_id", 0))].append(idx)

    # ──────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────

    def _encode_query(self, question: str) -> np.ndarray:
        """
        HyDE at query time: generate a hypothetical answer passage, embed it,
        and blend with the real query embedding. HYDE_WEIGHT controls the mix
        (0.0 = query only, 0.5 = equal blend, 1.0 = hypothetical only).
        Falls back to query-only if the LLM call fails.
        """
        query_emb = self.embedder.encode(
            ["Represent this sentence for searching relevant passages: " + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        if HYDE_WEIGHT <= 0.0:
            return query_emb

        hyp_passage = generate_hypothetical_passage(question)
        if not hyp_passage:
            return query_emb

        hyp_emb = self.embedder.encode(
            [hyp_passage],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        blended = (1.0 - HYDE_WEIGHT) * query_emb + HYDE_WEIGHT * hyp_emb
        # Re-normalize after blending so dot-product scores stay meaningful
        norm = np.linalg.norm(blended, axis=1, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        return (blended / norm).astype("float32")

    def _retrieve(self, question, top_k=TOP_K_RETRIEVE):

        n_chunks = len(self.chunks)
        n_pages = len(self.pages)
        if n_chunks == 0 or n_pages == 0:
            return []

        query_tokens = normalize(question).split()
        chunk_bm25_scores = np.array(self.bm25.get_scores(query_tokens), dtype="float32")
        page_bm25_scores = np.array(self.page_bm25.get_scores(query_tokens), dtype="float32")
        page_bm25_scores = max_normalize(page_bm25_scores)

        q_emb = self._encode_query(question)
        q_vec = q_emb[0]

        page_fetch_k = min(max(PAGE_TOP_K * 4, top_k * 4), n_pages)
        page_dense_raw, page_dense_indices = self.page_index.search(q_emb, page_fetch_k)
        page_dense_scores = np.zeros(n_pages, dtype="float32")
        for idx, score in zip(page_dense_indices[0], page_dense_raw[0]):
            page_dense_scores[idx] = score
        # FIX: use stable_softmax instead of minmax_normalize for dense scores
        page_dense_scores = stable_softmax(page_dense_scores)

        page_hybrid = (
            PAGE_BM25_WEIGHT * page_bm25_scores
            + PAGE_DENSE_WEIGHT * page_dense_scores
        )
        page_top_ids = np.argsort(page_hybrid)[::-1][: min(PAGE_TOP_K, n_pages)]

        candidate_indices: list[int] = []
        for page_id in page_top_ids:
            candidate_indices.extend(self.page_to_chunk_ids.get(int(page_id), []))

        if not candidate_indices:
            candidate_indices = list(range(n_chunks))

        candidate_indices = list(dict.fromkeys(candidate_indices))
        candidate_arr = np.array(candidate_indices, dtype=np.int32)

        candidate_bm25 = max_normalize(chunk_bm25_scores[candidate_arr])
        candidate_dense = np.dot(self.embeddings[candidate_arr], q_vec)
        # FIX: use stable_softmax instead of minmax_normalize for chunk dense scores
        candidate_dense = stable_softmax(candidate_dense)

        page_prior = np.array(
            [page_hybrid[int(self.chunks[idx].get("page_id", 0))] for idx in candidate_arr],
            dtype="float32",
        )
        page_prior = max_normalize(page_prior)

        candidate_hybrid = (
            BM25_WEIGHT * candidate_bm25
            + DENSE_WEIGHT * candidate_dense
            + PAGE_PRIOR_WEIGHT * page_prior
        )
        ranked_pos = np.argsort(candidate_hybrid)[::-1]
        ranked_chunk_ids = candidate_arr[ranked_pos].tolist()

        selected_ids = []
        selected_set = set()
        per_url_count = defaultdict(int)

        for idx in ranked_chunk_ids:
            url = self.chunks[idx]["url"]
            # FIX: MAX_CHUNKS_PER_URL is now 5 (was 2)
            if per_url_count[url] >= MAX_CHUNKS_PER_URL:
                continue
            selected_ids.append(idx)
            selected_set.add(idx)
            per_url_count[url] += 1
            if len(selected_ids) >= top_k:
                break

        if len(selected_ids) < top_k:
            for idx in ranked_chunk_ids:
                if idx in selected_set:
                    continue
                selected_ids.append(idx)
                if len(selected_ids) >= top_k:
                    break

        return [self.chunks[i] for i in selected_ids]

    # ──────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────

    def _format_chunk_for_prompt(self, chunk: dict) -> str:
        source = f"[Source: {chunk['url']}]"
        if chunk.get("chunk_type") == "table_row":
            return f"{source}\n[Type: table]\n{chunk['text']}"
        return f"{source}\n{chunk['text']}"

    def _generate(self, question, chunks):

        context = "\n\n---\n\n".join(
            self._format_chunk_for_prompt(c) for c in chunks
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
                max_tokens=48,      # FIX: was 24; raised to avoid truncating long answers
                temperature=0.0,
                timeout=120,
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

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process, i, q): i for i, q in enumerate(questions)}
            for future in as_completed(futures):
                i, answer = future.result()
                answers[i] = answer

        return answers


# ──────────────────────────────────────────────
# Run on generated QA dataset
# ──────────────────────────────────────────────

def load_questions_from_jsonl(path: str) -> list[str]:
    questions = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    return questions


if __name__ == "__main__":

    model = RAGModel()

    questions = load_questions_from_jsonl("data/qa/generated_qa.jsonl")

    answers = model.predict(questions[:20])

    for q, a in zip(questions, answers):

        print("Q:", q)
        print("A:", a)
        print()