import json
import os
import requests
import random
import time
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

INPUT_FILE = "corpus/pages_all.json"
OUTPUT_FILE = "data/qa/generated_qa.jsonl"

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"

CHUNK_WORDS = 300


def extract_json(text):
    """Extract the first JSON object from LLM output."""
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def get_chunk(text):
    """Select a random 300-word window from the text."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return text
    start = random.randint(0, len(words) - CHUNK_WORDS)
    return " ".join(words[start:start + CHUNK_WORDS])


def ask_llm(context):
    prompt = f"""Generate ONE factoid question-answer pair from this EECS webpage text.

Rules:
- The answer MUST be an exact substring extracted directly from the text.
- Answer must be under 10 words.
- Ask about specific, niche details: course numbers, dates, faculty names, research awards, or specific locations.
- DO NOT ask general questions like "Where is the department located?" or "What is this department?"
- The question MUST be completely self-contained. It must include the specific names, titles, or entities it is asking about so it can be searched in a database.
- NEVER use pronouns like "he", "she", "it", or "this" in the question.
- BAD questions: "Who is the advisor?" or "What is the report number?"
- GOOD questions: "Who advised John Doe's dissertation?" or "What is the report number for the paper on Optimal Controls?"
- Output valid JSON only, nothing else.

Example:
{{"question": "Who won the 2022 Guggenheim Fellowship?", "answer": "Venkatesan Guruswami"}}

TEXT:
{context}
"""

    for attempt in range(3):
        try:
            response = requests.post(
                GEMINI_URL,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30,
            )
            data = response.json()
            if "error" in data:
                print(f"API error: {data['error'].get('message', data['error'])}")
                time.sleep(2)
                continue
            text = data["candidates"][0]["content"]["parts"][0]["text"]
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
    seen_questions = set()

    random.shuffle(pages)

    for i, page in enumerate(pages):
        print(f"\nProcessing page {i+1}/{len(pages)}")
        print(page["url"])

        full_text = page.get("text", "")
        if len(full_text.split()) < 30:
            print("Too short")
            continue

        context = get_chunk(full_text)
        qa = ask_llm(context)

        if qa is None:
            print("Skipped (no response)")
            continue

        if "question" not in qa or "answer" not in qa:
            print("Rejected (missing keys):", qa)
            continue

        question = qa["question"].strip()
        answer = qa["answer"].strip()

        if len(answer.split()) >= 10:
            print(f"Rejected (answer too long — {len(answer.split())} words): {answer}")
            continue

        if answer.lower() not in context.lower():
            print(f"Rejected (answer not in context): {answer}")
            continue

        q_key = question.lower()
        if q_key in seen_questions:
            print(f"Rejected (duplicate question): {question}")
            continue
        seen_questions.add(q_key)

        dataset.append({
            "question": question,
            "answer": answer,
            "url": page["url"],
        })

        print(f"Accepted [{len(dataset)}]: {question} → {answer}")

        if len(dataset) >= 500:
            break

    with open(OUTPUT_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\nGenerated {len(dataset)} QA pairs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
