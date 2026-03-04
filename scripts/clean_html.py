import json

def clean_text(text):

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text


with open("data/corpus.json") as f:
    pages = json.load(f)

cleaned = []

for p in pages:

    cleaned.append({
        "url": p["url"],
        "text": clean_text(p["text"])
    })

with open("data/clean_corpus.json", "w") as f:
    json.dump(cleaned, f)

print("Cleaned corpus saved")