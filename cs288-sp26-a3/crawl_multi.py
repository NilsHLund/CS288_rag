"""
EECS Website Crawler (multi-threaded)
Crawls all pages under eecs.berkeley.edu using a thread pool.

Usage:
    python crawl_multi.py --output corpus/pages.json                  # unlimited pages
    python crawl_multi.py --output corpus/pages.json --max_pages 500  # cap at 500
    python crawl_multi.py --output corpus/pages.json --resume         # resume from checkpoint

Checkpoint: a companion file (e.g. corpus/pages.checkpoint.json) is saved
alongside the output and stores the frontier queue + visited set so the crawl
can pick up where it left off with --resume.

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
from bs4 import BeautifulSoup, NavigableString
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


def _html_table_to_markdown(table_tag) -> str:
    """Convert a single <table> element into a Markdown table string."""
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    grid = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        grid.append([c.get_text(separator=" ").strip() for c in cells])

    if not grid:
        return ""

    n_cols = max(len(r) for r in grid)
    for r in grid:
        r.extend([""] * (n_cols - len(r)))

    lines = []
    for i, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in row) + " |")

    return "\n".join(lines)


def extract_text(soup: BeautifulSoup):
    """Extract title and visible text, converting HTML tables to Markdown
    and aggressively stripping boilerplate."""
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    for table in soup.find_all("table"):
        md = _html_table_to_markdown(table)
        if md:
            table.replace_with(NavigableString("\n" + md + "\n"))
        else:
            table.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "content"})
        or soup.body
        or soup
    )

    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    text = "\n".join(cleaned).strip()
    text = re.sub(r" {2,}", " ", text)
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


def _checkpoint_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return base + ".checkpoint" + ext


def _save(corpus, output_path, lock, frontier=None, visited=None):
    """Atomically write corpus and checkpoint to disk."""
    with lock:
        tmp = output_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        os.replace(tmp, output_path)

        if frontier is not None and visited is not None:
            ckpt = {
                "frontier": list(frontier),
                "visited": list(visited),
            }
            ckpt_path = _checkpoint_path(output_path)
            tmp_ckpt = ckpt_path + ".tmp"
            with open(tmp_ckpt, "w", encoding="utf-8") as f:
                json.dump(ckpt, f, ensure_ascii=False)
            os.replace(tmp_ckpt, ckpt_path)


def _load_checkpoint(output_path: str):
    """Load previous corpus and checkpoint state for resume.
    Returns (corpus, visited, frontier) or ([], set(), deque()) if nothing exists."""
    corpus = []
    visited = set()
    frontier = deque()

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        print(f"Resumed: loaded {len(corpus)} pages from {output_path}")

    ckpt_path = _checkpoint_path(output_path)
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        visited = set(ckpt.get("visited", []))
        frontier = deque(ckpt.get("frontier", []))
        print(f"Resumed: {len(visited)} visited URLs, {len(frontier)} frontier URLs from checkpoint")
    elif corpus:
        visited = {page["url"] for page in corpus}
        print("No checkpoint file found — rebuilt visited set from corpus URLs only")

    return corpus, visited, frontier


def crawl(
    seed_url: str,
    output_path: str,
    max_pages: Optional[int],
    num_threads: int,
    delay: float,
    save_every: int,
    resume: bool = False,
):
    if resume:
        corpus, visited, frontier = _load_checkpoint(output_path)
        if not frontier and not corpus:
            frontier = deque([seed_url])
        elif not frontier:
            print("Checkpoint has empty frontier — crawl already finished or frontier was exhausted.")
            print(f"Corpus has {len(corpus)} pages. Nothing to resume.")
            return
    else:
        corpus = []
        visited = set()
        frontier = deque([seed_url])

    visited_lock = threading.Lock()
    corpus_lock = threading.Lock()
    frontier_lock = threading.Lock()

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (CS288 RAG Assignment)"})

    already = len(corpus)
    pbar = tqdm(desc="Crawling", unit="pages", initial=already)
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            with frontier_lock:
                if not frontier:
                    return
                url = frontier.popleft()

            result = fetch_page(url, session, delay)

            if result is None:
                continue

            with visited_lock:
                new_links = result["links"] - visited
                visited.update(new_links)
            with frontier_lock:
                frontier.extend(new_links)

            page = {"url": result["url"], "title": result["title"], "text": result["text"]}
            with corpus_lock:
                corpus.append(page)
                count = len(corpus)

            pbar.update(1)
            pbar.set_postfix(frontier=len(frontier), visited=len(visited))

            if count % save_every == 0:
                with visited_lock:
                    vis_snap = set(visited)
                with frontier_lock:
                    front_snap = list(frontier)
                _save(corpus, output_path, corpus_lock,
                      frontier=front_snap, visited=vis_snap)

            if max_pages is not None and count >= max_pages:
                stop_event.set()
                return

    seed_clean = seed_url.split("#")[0].split("?")[0]
    visited.add(seed_clean)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker) for _ in range(num_threads)}

        while futures:
            still_running = set()
            for f in list(futures):
                if f.done():
                    with frontier_lock:
                        has_work = bool(frontier)
                    if has_work and not stop_event.is_set():
                        still_running.add(executor.submit(worker))
                else:
                    still_running.add(f)
            futures = still_running
            time.sleep(0.05)

    pbar.close()

    with visited_lock:
        vis_snap = set(visited)
    with frontier_lock:
        front_snap = list(frontier)
    _save(corpus, output_path, corpus_lock,
          frontier=front_snap, visited=vis_snap)

    new_pages = len(corpus) - already
    print(f"\nFinished. Crawled {new_pages} new pages ({len(corpus)} total) → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded EECS website crawler")
    parser.add_argument("--seed", default=SEED_URL,
                        help="Seed URL to start crawling from (default: %(default)s)")
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
    args = parser.parse_args()

    max_pages = args.max_pages if args.max_pages and args.max_pages > 0 else None

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    crawl(
        seed_url=args.seed,
        output_path=args.output,
        max_pages=max_pages,
        num_threads=args.threads,
        delay=args.delay,
        save_every=args.save_every,
        resume=args.resume,
    )