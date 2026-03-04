import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from collections import deque

BASE = "https://eecs.berkeley.edu"

visited = set()
queue = deque([BASE])

pages = []

MAX_PAGES = 500   # begrens crawling

while queue and len(pages) < MAX_PAGES:

    url = queue.popleft()

    if url in visited:
        continue

    visited.add(url)

    try:
        r = requests.get(url, timeout=5)
    except:
        continue

    if r.status_code != 200:
        continue

    soup = BeautifulSoup(r.text, "html.parser")

    text = soup.get_text(separator=" ", strip=True)

    pages.append({
        "url": url,
        "text": text
    })

    for link in soup.find_all("a"):
        href = link.get("href")

        if not href:
            continue

        new_url = urljoin(BASE, href)

        if "eecs.berkeley.edu" in new_url and new_url not in visited:
            queue.append(new_url)


with open("data/corpus.json", "w") as f:
    json.dump(pages, f)

print("Saved", len(pages), "pages")