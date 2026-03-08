import json
import requests
import random
import time

INPUT_FILE = "corpus/pages_all.json"
OUTPUT_FILE = "generated_qa.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def extract_json(text):
    """Extract JSON block from LLM output."""
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end])
    except:
        return None


def ask_llm(context):

    prompt = f"""
Generate ONE factoid question-answer pair from this EECS webpage text.

Rules:
- Answer must be under 10 words
- Answer must appear exactly in the text
- Question must be about Berkeley EECS
- Output JSON only

Example:
{{"question": "...", "answer": "..."}}

TEXT:
{context}
"""

    for attempt in range(3):

        try:

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            text = response.json()["response"]

            qa = extract_json(text)

            if qa:
                return qa

            print("Bad JSON:", text[:150])

        except Exception as e:

            print("LLM error:", e)

        print("Retrying...", attempt + 1)
        time.sleep(1)

    return None


def main():

    with open(INPUT_FILE) as f:
        pages = json.load(f)

    dataset = []

    random.shuffle(pages)

    for i, page in enumerate(pages):

        print(f"\nProcessing page {i+1}/{len(pages)}")
        print(page["url"])

        context = page.get("text", "")[:2000]

        if len(context) < 50:
            print("Too short")
            continue

        qa = ask_llm(context)

        if qa is None:
            print("Skipped")
            continue

        if "question" not in qa or "answer" not in qa:
            print("Bad format:", qa)
            continue

        dataset.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "url": page["url"]
        })

        print("Generated:", qa)

        if len(dataset) >= 500:
            break

    with open(OUTPUT_FILE, "w") as f:

        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print("\nGenerated", len(dataset), "QA pairs")


if __name__ == "__main__":
    main()