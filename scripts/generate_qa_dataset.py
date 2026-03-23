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


if __name__ == "__main__":
    main()
