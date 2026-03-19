#!/usr/bin/env python3
"""
Download required local models for offline autograder runs.
"""

from pathlib import Path

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: sentence-transformers.\n"
        "Install project dependencies first:\n"
        "  python3 -m pip install -r requirements.txt"
    ) from exc


def main() -> None:
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    embed_src = "BAAI/bge-small-en-v1.5"
    embed_dst = models_dir / "bge-small-en-v1.5"

    rerank_src = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    rerank_dst = models_dir / "ms-marco-TinyBERT-L-2-v2"

    print(f"Downloading embedder: {embed_src} -> {embed_dst}")
    embedder = SentenceTransformer(embed_src)
    embedder.save(str(embed_dst))

    print(f"Downloading reranker: {rerank_src} -> {rerank_dst}")
    reranker = CrossEncoder(rerank_src)
    reranker.save(str(rerank_dst))

    print("Done. Models are saved under ./models/")


if __name__ == "__main__":
    main()
