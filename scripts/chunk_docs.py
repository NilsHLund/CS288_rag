import json
import pickle

CHUNK_SIZE = 300
OVERLAP = 50


def chunk_text(text):

    words = text.split()

    chunks = []

    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):

        chunk = words[i:i + CHUNK_SIZE]

        if len(chunk) < 50:
            continue

        chunks.append(" ".join(chunk))

    return chunks


with open("data/clean_corpus.json") as f:
    pages = json.load(f)

passages = []

for p in pages:

    passages.extend(chunk_text(p["text"]))

print("Total passages:", len(passages))

with open("indices/passages.pkl", "wb") as f:
    pickle.dump(passages, f)