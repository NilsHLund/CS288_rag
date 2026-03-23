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
        return json.loads(text[start:end])
    except Exception:
        return None


if __name__ == "__main__":
    main()
