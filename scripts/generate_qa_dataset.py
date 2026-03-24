import argparse
import json
import os
import random
import time
from collections import defaultdict
from urllib.parse import urlparse, unquote
from urllib.request import urlopen, Request

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# --- Config ---
INPUT_FILE = "corpus/pages_all.json"
OUTPUT_FILE = "data/qa/generated_qa_yesorno.jsonl"
DEFAULT_LIMIT = int(os.environ.get("QA_LIMIT", "500"))  # Max QA pairs to generate
CHUNK_WORDS = 800
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"

def get_path_prefix(url):
    # gets the prefix of a URL
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path).strip("/")
        if not path:
            return ""
        return path.split("/")[0]
    except Exception:
        return ""

def get_chunk(text, chunk_size=CHUNK_WORDS):
    words = text.split()
    if len(words) <= chunk_size:
        return text
    else:
        highest_start_idx = len(words) - chunk_size
        start = random.randint(0, highest_start_idx)
        return " ".join(words[start:start + chunk_size]) #this will remove newlines but that's probably fine

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        json_text = json.loads(text[start:end])
        if "question" not in json_text or "answer" not in json_text:
            return None
        return {"question": json_text["question"], "answer": json_text["answer"]}
    except Exception:
        return None

def _build_prompt(context):
    return f"""Generate ONE factoid question-answer pair from this EECS webpage text.
    
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

def ask_llm(context):
    prompt = _build_prompt(context)
    for attempt in range(3):
        try:
            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
            req = Request(GEMINI_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            if "error" in data:
                raise RuntimeError(data["error"].get("message", data["error"]))
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed_qa = extract_json(raw_text)
            if parsed_qa:
                return parsed_qa
        except Exception as e:
            print("Error:", e)


            print("Retrying...", attempt + 1)
        time.sleep(1)

    return None

def main():
    with open(INPUT_FILE, "r") as f:
        pages = json.load(f)

    dataset = []
    seen_questions = set()

    random.shuffle(pages)

    with open(OUTPUT_FILE, "a") as f:
        for i, page in enumerate(pages):
            if len(dataset) >= DEFAULT_LIMIT:
                break
            full_text = page.get("text", "")
            if len(full_text.split()) < 30:
                print("Too short")
                continue
            context = get_chunk(full_text)
            qa = ask_llm(context)
            if qa is None:
                print("No response")
                continue
            question = qa["question"].strip()
            answer = qa["answer"].strip()
            # check various filters
            if len(answer.split()) >= 10:
                print(f"Rejected (answer too long — {len(answer.split())} words): {answer}")
                continue
            if answer.lower() not in context.lower():
                print(f"Rejected (answer not in context): {answer}")
                continue
            question_lower = question.lower()
            if question_lower in seen_questions:
                print(f"Rejected (duplicate question): {question}")
                continue
            # add to seen questions
            seen_questions.add(question_lower)
            # add url to qa
            qa["url"] = page.get("url", "")

            #passed through all filters, add to dataset and write to file
            dataset.append(qa)
            f.write(json.dumps(qa) + "\n")
            print(f"Saved: {question}")

    

if __name__ == "__main__":
    main()
