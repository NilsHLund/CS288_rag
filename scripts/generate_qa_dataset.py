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
        return json_text["question"], json_text["answer"]
    except Exception:
        return None

def _build_prompt(context):
    return f"""Generate ONE yes/no question-answer pair from this UC Berkeley EECS webpage text.

Goal:
- Create a question that is answerable from this webpage alone.
- The answer must be exactly "Yes" or "No".
- Prefer realistic yes/no questions that a student, applicant, researcher, or visitor might naturally ask.
- Only generate a question if the page contains clear evidence for a definite yes or no answer.

Rules:
- The question must be fully self-contained and understandable on its own.
- NEVER use pronouns like "he", "she", "it", "this", or "they" in the question.
- The answer must be exactly one of: "Yes" or "No".
- The answer must be clearly supported by the webpage text.
- Do not generate a question if the page does not provide a definite answer.
- Prefer questions about:
  - requirements or eligibility
  - whether a course, program, or option has a certain feature
  - whether a rule, policy, or restriction applies
  - whether a specific program is intended for a certain group
- Avoid questions that are trivial, vague, ambiguous, or based on weak inference.
- Avoid questions that require outside knowledge.
- Avoid questions about office numbers, phone numbers, room numbers, email addresses, report numbers, or publication dates.
- Avoid questions whose answer depends on a highly time-sensitive detail unless the page clearly states it.

Good examples:
- "Is pass/fail grading allowed for this course?"
- "Is the program intended for transfer students?"
- "Does the course have a final exam?"
- "Can students from outside the United States apply for this fellowship?"

Bad examples:
- "Is UC Berkeley in California?"
- "Is the professor important?"
- "Does this page mention a course?"
- "Is the phone number listed on the page?"

Process:
1. Find a fact on the page that clearly supports a yes/no question.
2. Make sure the question is realistic and self-contained.
3. Make sure the answer is definitely Yes or definitely No from the page alone.
4. If no good yes/no question exists, output {{"question": "", "answer": ""}}.

Webpage text:
{context}

Output valid JSON only, with keys "question" and "answer".
"""




if __name__ == "__main__":
    main()
