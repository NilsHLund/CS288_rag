"""
EECS Website Crawler (multi-threaded)
Crawls all pages under eecs.berkeley.edu using a thread pool.

Usage:
    python crawl.py --output corpus/pages.json                  # unlimited pages
    python crawl.py --output corpus/pages.json --max_pages 500  # cap at 500

Output JSON format:
    [{"url": "...", "title": "...", "text": "..."}, ...]

Requirements:
    pip install requests beautifulsoup4 tqdm
"""

import json
import time
import re
import argparse
import os
import threading
from collections import deque
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

SEED_URL = "https://eecs.berkeley.edu"
ALLOWED_DOMAIN_RE = re.compile(r"https?:\/\/(?:www\d*\.)?eecs\.berkeley\.edu(?:\/[^\s]*)?")
SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4", ".svg", ".ico"}


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    return bool(ALLOWED_DOMAIN_RE.match(url.split("#")[0].split("?")[0]))


def extract_text(soup: BeautifulSoup):
    """Extract title and all visible text from a BeautifulSoup object."""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "content"})
        or soup.body
        or soup
    )
    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return title, text


def fetch_page(url: str, session: requests.Session, delay: float):
    """
    Fetch a single page and return a result dict, or None on failure.
    """
    time.sleep(delay)
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        title, text = extract_text(soup)

        # Collect all outgoing links from this page
        links = set()
        for a_tag in soup.find_all("a", href=True):
            link = urljoin(url, a_tag["href"]).split("#")[0].split("?")[0]
            if is_allowed_url(link):
                links.add(link)

        return {"url": url, "title": title, "text": text, "links": links}

    except Exception:
        return None


def _save(corpus, output_path, lock):
    """Atomically write corpus to disk."""
    with lock:
        tmp = output_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        os.replace(tmp, output_path)


def crawl(
    seed_url: str,
    output_path: str,
    max_pages: Optional[int],
    num_threads: int,
    delay: float,
    save_every: int,
):
    visited = set()
    visited_lock = threading.Lock()

    corpus = []
    corpus_lock = threading.Lock()

    frontier = deque([seed_url])
    frontier_lock = threading.Lock()

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (CS288 RAG Assignment)"})

    pbar = tqdm(desc="Crawling", unit="pages")
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            # Pull a URL off the frontier
            with frontier_lock:
                if not frontier:
                    return
                url = frontier.popleft()

            result = fetch_page(url, session, delay)

            if result is None:
                continue

            # Enqueue newly discovered links
            with visited_lock:
                new_links = result["links"] - visited
                visited.update(new_links)
            with frontier_lock:
                frontier.extend(new_links)

            # Save page — no minimum text length check, keep everything
            page = {"url": result["url"], "title": result["title"], "text": result["text"]}
            with corpus_lock:
                corpus.append(page)
                count = len(corpus)

            pbar.update(1)
            pbar.set_postfix(frontier=len(frontier), visited=len(visited))

            # Periodic save so progress survives a crash
            if count % save_every == 0:
                _save(corpus, output_path, corpus_lock)

            # Honour optional page cap
            if max_pages is not None and count >= max_pages:
                stop_event.set()
                return

    # Mark seed as visited before spawning workers
    seed_clean = seed_url.split("#")[0].split("?")[0]
    visited.add(seed_clean)

    # Keep the pool saturated: submit a new worker whenever one finishes,
    # as long as there is still frontier work and we haven't hit the cap.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker) for _ in range(num_threads)}

        while futures:
            still_running = set()
            for f in list(futures):
                if f.done():
                    # Re-submit if there's still work to do
                    with frontier_lock:
                        has_work = bool(frontier)
                    if has_work and not stop_event.is_set():
                        still_running.add(executor.submit(worker))
                else:
                    still_running.add(f)
            futures = still_running
            time.sleep(0.05)

    pbar.close()
    _save(corpus, output_path, corpus_lock)
    print(f"\nFinished. Crawled {len(corpus)} pages → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded EECS website crawler")
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
    args = parser.parse_args()

    # Treat --max_pages 0 as unlimited
    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else None

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    crawl(
        seed_url=SEED_URL,
        output_path=args.output,
        max_pages=max_pages,
        num_threads=args.threads,
        delay=args.delay,
        save_every=args.save_every,
    )