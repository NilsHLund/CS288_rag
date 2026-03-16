import argparse
import json
import os
import random
import time
from collections import defaultdict
from urllib.parse import urlparse, unquote

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- Config ---
INPUT_FILE = "corpus/pages_all.json"
OUTPUT_FILE = "data/qa/generated_qa_dense.jsonl"
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
    "Faculty": 0.10,
    "Courses": 0.10,
    "resources": 0.10,
    "research": 0.09,
    "Research": 0.03,   # www2 uses capital R
    "academics": 0.10,
    "people": 0.10,
    "about": 0.10,
    "connect": 0.10,
    "news": 0.13,      
    "Pubs": 0.05,      
    # category excluded
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
    return f"""Generate ONE factoid question-answer pair from this UC Berkeley EECS webpage text.

Goal:
- Create a question that is answerable from this webpage alone.
- The question should favor dense retrieval / semantic retrieval, meaning it should be easier to retrieve by semantic similarity than by exact keyword overlap alone.

Rules:
- The question must be answerable from this webpage alone.
- The answer MUST be an exact substring extracted directly from the text.
- The answer must be under 10 words. If the natural answer would be longer, choose a different question.
- Prefer extractive questions whose answer appears explicitly on the page.
- The question must be fully self-contained and understandable on its own.
- NEVER use pronouns like "he", "she", "it", "this", or "they" in the question.
- Prefer realistic questions a student, applicant, researcher, or visitor might naturally ask about UC Berkeley EECS.
- Favor questions about:
  - program purpose or structure
  - requirements or eligibility
  - roles or responsibilities
  - relationships between people, programs, courses, awards, or research topics
  - research themes, publication venues, academic distinctions, or substantive policies
- Prefer questions whose wording is a natural semantic reformulation of the source text.
- Prefer questions that require matching by meaning, role, purpose, attribute, or relationship, rather than by exact phrase overlap.
- The answer must still be a short exact span from the page, but the question should not simply copy distinctive words or titles from the source unless necessary for clarity.
- Prefer questions where an embedding-based retriever would have an advantage over exact lexical matching.
- Avoid questions that can be answered mainly by copying one rare phrase from the question into a search box.
- Avoid questions whose solution mainly depends on exact string matching of:
  - office numbers
  - room numbers
  - office locations
  - email addresses
  - phone numbers
  - exact event titles
  - exact award names
  - exact course codes alone
  - full report names
  - technical report numbers
  - version numbers
  - patent numbers
  - grant numbers
  - article publication dates
- Avoid repetitive metadata lookup questions such as:
  - publication year
  - article publication date
  - version number
  - report number
  - technical report number
  - patent number
  - grant number
  - advisor names
  - paper authors
  - phone numbers
  - room numbers
  - email addresses
- Avoid directory-style, contact-book, or identifier-lookup questions unless the fact is unusually important and central to the page.
- DO NOT ask general questions like "Where is the department located?" or "What is this department?"
- Avoid overly generic questions.
- The question and answer must refer to the same clearly identified person, role, event, policy, program, course, or entity on the page.
- Do not generate a question if the entity-to-answer mapping is ambiguous.
- Avoid long or messy list answers unless the list is short, explicit, and central to the page.
- Avoid questions that refer to "this page", "this report", "this paper", "the article", or similar non-self-contained phrases.

Process:
1. Identify a concrete fact on the page with a short extractive answer.
2. Rewrite the question so that it asks for the fact by meaning, role, purpose, attribute, or relationship.
3. Reduce exact lexical overlap between the question and the answer-bearing sentence whenever possible without making the question unnatural.
4. If the question mostly works because of exact title matching, identifier matching, or contact lookup, choose a different fact.

Bad questions:
- "What is Dan Klein's office number?"
- "Which email address should students use to submit the EECS Minor Declaration Form?"
- "What room hosted the colloquium on October 25, 2023?"
- "What is the phone number for the Computer Science Division office?"
- "Who won the 2021-22 Google-CMD-IT LEAP Fellowship Award?"
- "Which speaker is scheduled to present 'Reading Alan Turing'?"
- "What is the technical report number for this report?"

Good questions:
- "Which degree option is intended as a professional master’s program?"
- "What minimum GPA is expected from applicants using a 4.0 grading scale?"
- "Which program supports graduate students preparing for faculty careers?"
- "What journal published the research on the cockroach-inspired robot?"
- "Which Berkeley EECS option is designed for accelerated study?"
- "What teaching requirement must doctoral students complete in GSI hours?"

Output valid JSON only, with keys "question" and "answer".


TEXT:
{context}
"""


def _ask_gemini(prompt):
    response = requests.post(
        GEMINI_URL,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30,
    )
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", data["error"]))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _ask_chatgpt(prompt):
    response = requests.post(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    data = response.json()
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

        if len(dataset) >= limit:
            break

    with open(OUTPUT_FILE, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\nGenerated {len(dataset)} QA pairs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
