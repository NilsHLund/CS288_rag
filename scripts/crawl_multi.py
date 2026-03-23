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
HEADERS = {"User-Agent": "CS288 Assignment 3 Crawler"}
BACKOFF_SECONDS = 5
UNWANTED_TAGS = ["script", "style", "noscript", "nav", "footer", "aside"]


def is_allowed_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if any(parsed.path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    return bool(ALLOWED_DOMAIN_RE.match(url.split("#")[0].split("?")[0]))

def process_link(link: str):
    if is_allowed_url(link):
        if link.startswith("http://"):
            link = link.replace("http://", "https://", 1)
        return link
    return None

def fetch_page(url: str):
    backoff = BACKOFF_SECONDS

    url = process_link(url)
    if url is None:
        return None

    for attempt in range(3):
        try: 
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=10) as resp:
                status = resp.getcode()
                if "text/html" not in resp.headers.get("Content-Type", ""):
                    return None
                if status == 200:
                    # check type
                    html = resp.read().decode("utf-8", errors="replace")
                elif status in (429, 503):
                    raise Exception("Rate limited with status " + str(status) + " for " + url)
                else:
                    return None
            
        except Exception as e:
            logging.warning("Exception occured. " + url + ", " + str(e) + ", " + str(attempt+1) + "/3")
            time.sleep(backoff)
            backoff *= 2
            continue

        soup = BeautifulSoup(html, "html.parser") # contains webpage content

        # extract links
        links = set()
        for url_tag in soup.find_all("a", href=True):
            link = urljoin(url, url_tag["href"]).split("#")[0].split("?")[0]
            link = process_link(link)
            if link is not None:
                links.add(link)

        # filter data
        title, text, meta_description = extract_text_from_soup(soup)
        # TODO: come back after extract text is implemented

def extract_text_from_soup(soup):
    title = soup.title.text.strip() if soup.title and soup.title.text else ""
    meta_description = soup.find("meta", attrs={"name": "description"}).get("content", "").strip()
    
    for tag in soup(UNWANTED_TAGS):
        tag.decompose()

    # get main text
    raw_text = (soup.find("main") or soup.find("article") or soup.find("div", {"id": "content"}) or soup.body or soup).get_text(separator="\n").strip()
    
    # clean text
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if not line: 
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r" {2,}", " ", text)
    return title, text, meta_description

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
