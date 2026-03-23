import json
import time
import re
import argparse
import os
import logging
import threading
from collections import deque
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm

seeds = [
    "https://eecs.berkeley.edu",
    "https://www2.eecs.berkeley.edu",
]
ALLOWED_DOMAIN_RE = re.compile(r"https?:\/\/(?:www\d*\.)?eecs\.berkeley\.edu(?:\/[^\s]*)?")
SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4", ".svg", ".ico"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded EECS website crawler")
    parser.add_argument("--seed",
                        default="https://eecs.berkeley.edu,https://www2.eecs.berkeley.edu",
                        help="Comma-separated seed URLs (default: %(default)s)")
    parser.add_argument("--output", default="corpus/pages.json",
                        help="Output JSON file path")
    parser.add_argument("--max_pages", type=int, default=None,
                        help="Max pages to crawl (omit or set to 0 for unlimited)")
    parser.add_argument("--threads", type=int, default=10,
                        help="Number of concurrent threads (default: 10)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Per-thread delay between requests in seconds (default: 0.1)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save corpus to disk every N pages (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous checkpoint instead of starting fresh")
    parser.add_argument("--log", default=None,
                        help="Path to a log file for warnings (default: console only)")
    args = parser.parse_args()

    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else None
    