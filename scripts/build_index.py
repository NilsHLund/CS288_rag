import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

print("Loading passages...")

with open("indices/passages.pkl", "rb") as f:
    passages = pickle.load(f)

print("Building BM25...")

tokenized = [p.lower().split() for p in passages]

bm25 = BM25Okapi(tokenized)

with open("indices/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("Building embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(passages, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print("Building FAISS index...")

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)

faiss.normalize_L2(embeddings)

index.add(embeddings)

faiss.write_index(index, "indices/faiss.index")

print("Index built successfully")