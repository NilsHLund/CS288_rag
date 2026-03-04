import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from llm import call_llm


class RAGModel:

    def __init__(self):

        print("Loading indices...")

        with open("indices/passages.pkl", "rb") as f:
            self.passages = pickle.load(f)

        with open("indices/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        self.index = faiss.read_index("indices/faiss.index")

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, question, k=5):

        tokens = question.lower().split()

        bm25_scores = self.bm25.get_scores(tokens)

        bm25_top = np.argsort(bm25_scores)[-10:]

        q_emb = self.embedder.encode([question]).astype("float32")

        faiss.normalize_L2(q_emb)

        _, faiss_ids = self.index.search(q_emb, 10)

        candidates = set(bm25_top) | set(faiss_ids[0])

        ranked = sorted(
            candidates,
            key=lambda i: bm25_scores[i],
            reverse=True
        )

        return [self.passages[i] for i in ranked[:k]]

    def generate(self, question, contexts):

        context = "\n\n".join(contexts)

        prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{question}

Return a short answer (<10 words).
If not found return UNKNOWN.
"""

        try:
            answer = call_llm(prompt)
        except:
            answer = "UNKNOWN"

        return answer.strip()

    def predict(self, questions):

        answers = []

        for q in questions:

            contexts = self.retrieve(q)

            ans = self.generate(q, contexts)

            answers.append(ans)

        return answers