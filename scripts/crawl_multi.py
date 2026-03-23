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
FAILED_URLS_PATH = "corpus/failed_urls.txt"
FAILED_LOCK = threading.Lock()


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

def fetch_page(url: str, is_retry: bool = False, failed_urls_path: str = FAILED_URLS_PATH):
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

        raw_len = len(html)
        if raw_len > 0:
            ratio = len(text) / raw_len
            if ratio < 0.005:
                logging.warning("Low text ratio " + str(ratio * 100) + "% for " + url)
        
        return {
            "url": url,
            "title": title,
            "text": text,
            "meta_description": meta_description,
            "links": links,
        }
    
    logging.warning("Failed after 3 retries for " + url)
    if not is_retry:
        with FAILED_LOCK:
            with open(failed_urls_path, "a") as f:
                f.write(url + "\n")
    return None

def extract_text_from_soup(soup):
    title = soup.title.text.strip() if soup.title and soup.title.text else ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_tag["content"].strip() if meta_tag and meta_tag.get("content") else ""
    
    for tag in soup(UNWANTED_TAGS):
        tag.decompose()

    # handle tables
    for table in soup.find_all("table"):
        lines = []
        grid = []
        rows = table.find_all("tr")
        if not rows:
            table.decompose()
            continue
        for row in rows:
            cells = row.find_all(["th", "td"])
            grid.append([c.get_text(separator=" ").strip() for c in cells])

        if not grid:
            table.decompose()
            continue

        # add padding in case of uneven number of columns
        num_cols = max(len(r) for r in grid)
        for r in grid: 
            r.extend([""] * (num_cols - len(r)))

        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                lines.append("| " + " | ".join("---" for _ in row) + " |")



        table.replace_with(NavigableString("\n\n" + "\n".join(lines) + "\n\n"))

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

def crawl(
    seed_urls: list[str],
    output_path: str,
    max_pages: Optional[int],
    num_threads: int,
    delay: float,
    save_every: int,
    resume: bool = False,
):
    corpus, frontier, visited = _load_checkpoint(output_path, resume)

    if not frontier:
        frontier = deque(seed_urls)   
        visited = set(seed_urls)

    visited_lock = threading.Lock()
    corpus_lock = threading.Lock()
    frontier_lock = threading.Lock()

    pbar = tqdm(desc="Crawling", unit="pages")
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            # Pull a URL off the frontier
            with frontier_lock:
                if not frontier:
                    return
                url = frontier.popleft()

            time.sleep(delay)

            result = fetch_page(url)

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
                with visited_lock:
                    visited_snap = list(visited)
                with frontier_lock:
                    frontier_snap = list(frontier)
                _save(corpus, output_path, corpus_lock, frontier=frontier_snap, visited=visited_snap)

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

def _save(corpus, output_path, lock, frontier=None, visited=None):
    """Atomically write corpus to disk."""
    with lock:
        tmp = output_path + ".tmp"
        
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        
        checkpoint_data = {
            "frontier": list(frontier) if frontier is not None else [],
            "visited": list(visited) if visited is not None else []
        }
        with open(_checkpoint_path(output_path), "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False)
        os.replace(tmp, output_path)

def _checkpoint_path(output_path):
    base, ext = os.path.splitext(output_path)
    return base + ".checkpoint" + ext

def _load_checkpoint(output_path, resume):
    check_path = _checkpoint_path(output_path)
    if resume and os.path.exists(check_path):
        with open(check_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            frontier = deque(data.get("frontier", []))
            visited = set(data.get("visited", []))
        with open(output_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        return corpus, frontier, visited
    return [], deque(), set()

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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    crawl(
        seed_url=args.seed.split(","),
        output_path=args.output,
        max_pages=max_pages,
        num_threads=args.threads,
        delay=args.delay,
        save_every=args.save_every
    )