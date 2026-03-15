"""
EECS Website Crawler
Crawls all pages under eecs.berkeley.edu, extracts clean text, and saves as JSON.

Usage:
    python crawl.py --output corpus/pages.json --max_pages 2000

Output JSON format:
    [{"url": "...", "title": "...", "text": "..."}, ...]

Requirements (add to your env):
    pip install requests beautifulsoup4 tqdm
"""

import json
import time
import re
import argparse
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

SEED_URL = "https://eecs.berkeley.edu"
ALLOWED_DOMAIN_RE = re.compile(r"https?:\/\/(?:www\d*\.)?eecs\.berkeley\.edu(?:\/[^\s]*)?")
SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4", ".svg", ".ico"}

def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    # Skip non-http and fragment-only links
    if parsed.scheme not in ("http", "https"):
        return False
    # Skip file extensions we don't want
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    return bool(ALLOWED_DOMAIN_RE.match(url.split("#")[0].split("?")[0]))


def extract_text(soup: BeautifulSoup) -> str:
    """Extract clean text from a BeautifulSoup object."""
    # Remove script, style, nav, footer noise
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Get title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Get main content (prefer <main> or <article> if available)
    main = soup.find("main") or soup.find("article") or soup.find("div", {"id": "content"}) or soup.body
    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    # Clean whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return title, text


def crawl(seed_url: str, max_pages: int, output_path: str, delay: float = 0.3):
    visited = set()
    queue = deque([seed_url])
    corpus = []

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (CS288 RAG Assignment)"})

    pbar = tqdm(total=max_pages, desc="Crawling")

    while queue and len(corpus) < max_pages:
        url = queue.popleft()
        url = url.split("#")[0]  # strip fragments
        url = "https://www2.eecs.berkeley.edu/Directories/directory-nostudents.html"

        if url in visited or not is_allowed_url(url):
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title, text = extract_text(soup)

            if len(text) < 100:  # skip nearly empty pages
                continue

            corpus.append({"url": url, "title": title, "text": text})
            pbar.update(1)

            # Discover new links
            for a_tag in soup.find_all("a", href=True):
                link = urljoin(url, a_tag["href"]).split("#")[0]
                if link not in visited and is_allowed_url(link):
                    queue.append(link)

            time.sleep(delay)  # be polite

        except Exception as e:
            print(f"  [skip] {url}: {e}")

    pbar.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\nCrawled {len(corpus)} pages → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="corpus/pages.json")
    parser.add_argument("--max_pages", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between requests")
    args = parser.parse_args()

    import os; os.makedirs("corpus", exist_ok=True)
    crawl(SEED_URL, args.max_pages, args.output, args.delay)
