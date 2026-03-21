"""
patch_corpus.py — Fetch missing QA URLs and merge all corpus sources into pages_all.json

1. Reads all QA URLs from generated_qa_dense.jsonl
2. Identifies which are missing/empty in pages_all.json
3. Fetches them from the web
4. Merges pages.json (1000 quality pages) into pages_all.json
5. Writes updated pages_all.json (JSONL format)
"""

import json
import os
import re
import time
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup, NavigableString


def norm_url(u):
    u = u.rstrip("/")
    u = u.replace("http://", "https://", 1)
    return u


def extract_text(soup):
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript", "nav", "footer", "aside"]):
        tag.decompose()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if rows:
            grid = []
            for row in rows:
                cells = row.find_all(["th", "td"])
                grid.append([c.get_text(separator=" ").strip() for c in cells])
            n_cols = max(len(r) for r in grid) if grid else 0
            for r in grid:
                r.extend([""] * (n_cols - len(r)))
            lines = []
            for i, row in enumerate(grid):
                lines.append("| " + " | ".join(row) + " |")
                if i == 0:
                    lines.append("| " + " | ".join("---" for _ in row) + " |")
            md = "\n".join(lines)
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


def fetch_url(url, delay=0.5):
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)
    headers = {"User-Agent": "Mozilla/5.0 (CS288 RAG Assignment)"}
    time.sleep(delay)
    for attempt in range(3):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                status = resp.getcode()
                ct = resp.headers.get("Content-Type", "")
                if status != 200 or "text/html" not in ct:
                    return None
                html = resp.read().decode(errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            title, text = extract_text(soup)
            return {"url": url, "title": title, "text": text}
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def main():
    corpus_path = "corpus/pages_all.json"
    qa_path = "data/qa/generated_qa_dense.jsonl"
    pages_json_path = "corpus/pages.json"

    # Load existing corpus (JSONL)
    existing = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing.append(json.loads(line))
    print(f"Existing corpus: {len(existing)} pages")

    # Build URL→page index (normalized)
    url_index = {}
    for i, p in enumerate(existing):
        nu = norm_url(p["url"])
        url_index[nu] = i

    # Load pages.json (JSON array with 1000 quality pages)
    with open(pages_json_path, encoding="utf-8") as f:
        pages_json = json.load(f)
    print(f"pages.json: {len(pages_json)} pages")

    merged_count = 0
    for p in pages_json:
        nu = norm_url(p["url"])
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if nu in url_index:
            idx = url_index[nu]
            old_text = (existing[idx].get("text") or "").strip()
            if not old_text or len(text) > len(old_text) * 1.2:
                existing[idx]["text"] = text
                existing[idx]["title"] = p.get("title", existing[idx].get("title", ""))
                merged_count += 1
        else:
            existing.append({"url": p["url"], "title": p.get("title", ""), "text": text})
            url_index[nu] = len(existing) - 1
            merged_count += 1
    print(f"Merged {merged_count} pages from pages.json")

    # Load QA URLs
    qa_urls = set()
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qa_urls.add(item["url"])

    # Find missing/empty QA URLs
    missing_urls = []
    for url in qa_urls:
        nu = norm_url(url)
        if nu not in url_index:
            missing_urls.append(url)
        else:
            idx = url_index[nu]
            if not (existing[idx].get("text") or "").strip():
                missing_urls.append(url)

    print(f"\nMissing/empty QA URLs to fetch: {len(missing_urls)}")

    # Fetch missing URLs
    fetched = 0
    failed = 0
    for i, url in enumerate(missing_urls):
        print(f"  [{i+1}/{len(missing_urls)}] Fetching: {url}")
        result = fetch_url(url)
        if result and result["text"].strip():
            nu = norm_url(url)
            if nu in url_index:
                idx = url_index[nu]
                existing[idx]["text"] = result["text"]
                existing[idx]["title"] = result["title"]
            else:
                existing.append(result)
                url_index[nu] = len(existing) - 1
            fetched += 1
            print(f"    OK ({len(result['text'])} chars)")
        else:
            failed += 1
            print(f"    FAILED")

    print(f"\nFetched: {fetched}, Failed: {failed}")

    # Remove pages with empty text to clean up the corpus
    non_empty = [p for p in existing if (p.get("text") or "").strip()]
    print(f"Final corpus: {len(non_empty)} pages (removed {len(existing) - len(non_empty)} empty)")

    # Write back as JSONL
    with open(corpus_path, "w", encoding="utf-8") as f:
        for p in non_empty:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved to {corpus_path}")


if __name__ == "__main__":
    main()
