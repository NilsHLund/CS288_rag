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
import logging
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

    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_tag["content"].strip() if meta_tag and meta_tag.get("content") else ""

    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside"]):
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
    return title, text, meta_description


FAILED_URLS_PATH = "corpus/failed_urls.txt"
_failed_urls_lock = threading.Lock()


def fetch_page(url: str, session: requests.Session, delay: float,
               is_retry: bool = False):
    """
    Fetch a single page and return a result dict, or None on failure.
    Retries with exponential backoff on 429/503 and connection errors.
    """
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)
    time.sleep(delay)
    backoff = 5

    for attempt in range(3):
        try:
            resp = session.get(url, timeout=10)
        except Exception as e:
            logging.warning("Request exception for %s: %s (attempt %d/3)",
                            url, e, attempt + 1)
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code in (429, 503):
            logging.warning("Rate limited (%d) for %s — retrying in %ds (attempt %d/3)",
                            resp.status_code, url, backoff, attempt + 1)
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code != 200:
            logging.warning("HTTP %d for %s", resp.status_code, url)
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        links = set()
        for a_tag in soup.find_all("a", href=True):
            link = urljoin(url, a_tag["href"]).split("#")[0].split("?")[0]
            if link.startswith("http://"):
                link = link.replace("http://", "https://", 1)
            if is_allowed_url(link):
                links.add(link)

        title, text, meta_description = extract_text(soup)

        raw_len = len(resp.text)
        if raw_len > 0:
            ratio = len(text) / raw_len
            if ratio < 0.005:
                logging.warning("Low text ratio (%.1f%%) for %s", ratio * 100, url)

        return {"url": url, "title": title, "text": text,
                "meta_description": meta_description, "links": links}

    logging.warning("Failed after 3 retries for %s", url)
    if not is_retry:
        with _failed_urls_lock:
            with open(FAILED_URLS_PATH, "a", encoding="utf-8") as f:
                f.write(url + "\n")
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
    seed_urls: list[str],
    output_path: str,
    max_pages: Optional[int],
    num_threads: int,
    delay: float,
    save_every: int,
    resume: bool = False,
):
    clean_seeds = [u.split("#")[0].split("?")[0] for u in seed_urls]

    if resume:
        corpus, visited, frontier = _load_checkpoint(output_path)
        if not frontier and not corpus:
            frontier = deque(clean_seeds)
        elif not frontier:
            print("Checkpoint has empty frontier — crawl already finished or frontier was exhausted.")
            print(f"Corpus has {len(corpus)} pages. Nothing to resume.")
            return
    else:
        corpus = []
        visited = set()
        frontier = deque(clean_seeds)

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

            page = {"url": result["url"], "title": result["title"],
                    "text": result["text"], "meta_description": result["meta_description"]}
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

    visited.update(clean_seeds)
    interrupted = False

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        try:
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
        except KeyboardInterrupt:
            print("\nInterrupted — stopping threads...")
            stop_event.set()
            interrupted = True
        finally:
            pbar.close()
            with visited_lock:
                vis_snap = set(visited)
            with frontier_lock:
                front_snap = list(frontier)
            _save(corpus, output_path, corpus_lock,
                  frontier=front_snap, visited=vis_snap)
            new_pages = len(corpus) - already
            print(f"\nSaved. Crawled {new_pages} new pages ({len(corpus)} total) → {output_path}")

    if interrupted:
        return

    # --- Slow retry phase for failed URLs ---
    if not os.path.exists(FAILED_URLS_PATH):
        return

    with open(FAILED_URLS_PATH, "r", encoding="utf-8") as f:
        failed = list(dict.fromkeys(line.strip() for line in f if line.strip()))

    if not failed:
        os.remove(FAILED_URLS_PATH)
        return

    print(f"\nRetrying {len(failed)} failed URLs at a slow rate...")
    permanently_failed_path = "corpus/permanently_failed.txt"

    for i, url in enumerate(failed, 1):
        print(f"  Retry {i}/{len(failed)}: {url}")
        result = fetch_page(url, session, delay=3.0, is_retry=True)

        # Remove this URL from the failed list regardless of outcome
        remaining = failed[i:]
        if remaining:
            with open(FAILED_URLS_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(remaining) + "\n")
        elif os.path.exists(FAILED_URLS_PATH):
            os.remove(FAILED_URLS_PATH)

        if result is None:
            with open(permanently_failed_path, "a", encoding="utf-8") as f:
                f.write(url + "\n")
            continue

        page = {"url": result["url"], "title": result["title"],
                "text": result["text"], "meta_description": result["meta_description"]}
        corpus.append(page)
        _save(corpus, output_path, corpus_lock,
              frontier=list(frontier), visited=visited)
        print(f"    Recovered: {result['title'][:80]}")

    print(f"Retry phase complete. Corpus now has {len(corpus)} pages.")


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

    handlers = [logging.StreamHandler()]
    if args.log:
        os.makedirs(os.path.dirname(args.log) if os.path.dirname(args.log) else ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    seed_urls = [s.strip() for s in args.seed.split(",") if s.strip()]

    crawl(
        seed_urls=seed_urls,
        output_path=args.output,
        max_pages=max_pages,
        num_threads=args.threads,
        delay=args.delay,
        save_every=args.save_every,
        resume=args.resume,
    )