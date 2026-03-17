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

# LLM provider: "gemini" or "chatgpt"
LLM_PROVIDER = os.environ.get("QA_LLM_PROVIDER", "gemini")

# Path prefix filter: comma-separated allowlist, or empty for all
PATH_PREFIX_ALLOWLIST = [
    p.strip() for p in os.environ.get("QA_PATH_PREFIXES", "").split(",") if p.strip()
]
# Path prefix exclude: comma-separated blocklist
PATH_PREFIX_EXCLUDE = [
    p.strip() for p in os.environ.get("QA_PATH_EXCLUDE", "").split(",") if p.strip()
]
# Default path weights: prioritize Faculty/Courses/resources/etc, moderate news, exclude Pubs/category
DEFAULT_PATH_WEIGHTS = {
    "Faculty": 0.05,
    "Courses": 0.12,
    "resources": 0.18,
    "research": 0.12,
    "Research": 0.04,
    "academics": 0.18,
    "people": 0.05,
    "about": 0.06,
    "connect": 0.03,
    "news": 0.12,
    "Pubs": 0.01,
}
# Override with QA_PATH_WEIGHTS env (JSON) if set
PATH_PREFIX_WEIGHTS = {}
_w = os.environ.get("QA_PATH_WEIGHTS", "")
if _w:
    try:
        PATH_PREFIX_WEIGHTS = json.loads(_w)
    except json.JSONDecodeError:
        PATH_PREFIX_WEIGHTS = dict(DEFAULT_PATH_WEIGHTS)
else:
    PATH_PREFIX_WEIGHTS = dict(DEFAULT_PATH_WEIGHTS)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.environ.get("QA_OPENAI_MODEL", "gpt-5.4")


def get_path_prefix(url):
    """Extract top-level path segment (first path segment after domain)."""
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path).strip("/")
        if not path:
            return "/"
        return path.split("/")[0]
    except Exception:
        return ""


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


def _ask_gemini(prompt):
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = Request(GEMINI_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode())
    if "error" in data:
        raise RuntimeError(data["error"].get("message", data["error"]))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _ask_chatgpt(prompt):
    payload = json.dumps({
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = Request(
        OPENAI_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode())
    if "error" in data:
        raise RuntimeError(data["error"].get("message", data["error"]))
    return data["choices"][0]["message"]["content"]


def ask_llm(context):
    prompt = _build_prompt(context)
    for attempt in range(3):
        try:
            if LLM_PROVIDER == "chatgpt":
                if not OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY required for ChatGPT")
                text = _ask_chatgpt(prompt)
            else:
                if not GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY required for Gemini")
                text = _ask_gemini(prompt)
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
    global LLM_PROVIDER
    parser = argparse.ArgumentParser(description="Generate QA dataset from corpus")
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max QA pairs to generate (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "chatgpt"],
        default=LLM_PROVIDER,
        help="LLM provider: gemini or chatgpt (default: from QA_LLM_PROVIDER env)",
    )
    parser.add_argument(
        "--path-prefixes",
        type=str,
        default="",
        help="Comma-separated allowlist of path prefixes (e.g. news,Faculty,Courses). Only process these.",
    )
    parser.add_argument(
        "--path-exclude",
        type=str,
        default="",
        help="Comma-separated blocklist of path prefixes (e.g. Pubs,category). Skip these.",
    )
    parser.add_argument(
        "--path-weights",
        type=str,
        default="",
        help='JSON weights per prefix for sampling. Overrides default priority weights.',
    )
    parser.add_argument(
        "--all-paths",
        action="store_true",
        help="Disable path filtering; process all pages (ignores default priority weights).",
    )
    args = parser.parse_args()
    limit = args.limit
    provider = args.provider
    LLM_PROVIDER = provider

    # Path filter overrides from CLI
    path_allowlist = [p.strip() for p in args.path_prefixes.split(",") if p.strip()]
    path_exclude = [p.strip() for p in args.path_exclude.split(",") if p.strip()]
    if args.all_paths:
        path_weights = {}
    elif args.path_weights:
        try:
            path_weights = json.loads(args.path_weights)
        except json.JSONDecodeError:
            print("Warning: invalid --path-weights JSON, using defaults")
            path_weights = dict(PATH_PREFIX_WEIGHTS)
    else:
        path_weights = dict(PATH_PREFIX_WEIGHTS)

    if not path_allowlist:
        path_allowlist = PATH_PREFIX_ALLOWLIST
    if not path_exclude:
        path_exclude = PATH_PREFIX_EXCLUDE

    print(f"Config: provider={LLM_PROVIDER}, limit={limit}, output={OUTPUT_FILE}")
    if path_allowlist:
        print(f"  path allowlist: {path_allowlist}")
    if path_exclude:
        print(f"  path exclude: {path_exclude}")
    if path_weights:
        print(f"  path weights: {path_weights}")

    with open(INPUT_FILE) as f:
        all_pages = json.load(f)

    # Apply path prefix filter / weighted sampling
    if path_weights:
        # Weighted sampling: build list proportionally by prefix
        by_prefix = defaultdict(list)
        for p in all_pages:
            prefix = get_path_prefix(p.get("url", ""))
            if prefix in path_weights:
                by_prefix[prefix].append(p)
        total_weight = sum(path_weights.values())
        if total_weight <= 0:
            pages = all_pages
        else:
            pages = []
            n_per_prefix = {k: max(1, int(limit * (v / total_weight) * 2)) for k, v in path_weights.items()}
            for prefix, weight in path_weights.items():
                pool = by_prefix.get(prefix, [])
                n = n_per_prefix.get(prefix, limit)
                pages.extend(random.sample(pool, min(n, len(pool))) if pool else [])
            random.shuffle(pages)
    else:
        # Filter by allowlist / blocklist
        pages = []
        for p in all_pages:
            prefix = get_path_prefix(p.get("url", ""))
            if path_allowlist and prefix not in path_allowlist:
                continue
            if path_exclude and prefix in path_exclude:
                continue
            pages.append(p)
        random.shuffle(pages)

    print(f"Pages to process: {len(pages)} (from {len(all_pages)} total)")

    dataset = []
    seen_questions = set()

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

        # Allow answers not in context (LLM may paraphrase or extract from full page)
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

        if len(dataset) >= limit:
            break

    with open(OUTPUT_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\nGenerated {len(dataset)} QA pairs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
