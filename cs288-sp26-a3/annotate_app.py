"""
Local annotation web app for IAA.
Run: python annotate_app.py [path/to/annotations.jsonl]
Open: http://localhost:5050
"""

import json
import os
import sys
import requests as req
from flask import Flask, render_template_string, request, redirect, url_for
from urllib.parse import quote, unquote

ANNOTATION_FILE = sys.argv[1] if len(sys.argv) > 1 else "annotations/iaa_annotator2_template.jsonl"
CORPUS_FILE = "corpus/pages_all.json"

app = Flask(__name__)

CORPUS_BY_URL = {}

def _load_corpus():
    global CORPUS_BY_URL
    if CORPUS_BY_URL:
        return
    try:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            pages = json.load(f)
        for p in pages:
            url = p.get("url", "")
            CORPUS_BY_URL[url] = p
            if url.startswith("https://"):
                CORPUS_BY_URL[url.replace("https://", "http://", 1)] = p
            elif url.startswith("http://"):
                CORPUS_BY_URL[url.replace("http://", "https://", 1)] = p
    except Exception as e:
        print(f"Warning: could not load corpus: {e}")


def load_annotations():
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_annotations(items):
    with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QA Annotation — {{ item.id }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }

  .layout { display: flex; height: 100vh; }

  .panel { width: 420px; min-width: 420px; padding: 20px; overflow-y: auto;
           background: #1e293b; border-right: 1px solid #334155; display: flex; flex-direction: column; gap: 16px; }

  .source-frame { flex: 1; display: flex; flex-direction: column; }
  .source-frame iframe { flex: 1; width: 100%; border: none; background: #fff; }

  .progress { font-size: 13px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; }
  .progress-bar { height: 6px; background: #334155; border-radius: 3px; flex: 1; margin: 0 12px; }
  .progress-fill { height: 100%; background: #22c55e; border-radius: 3px; transition: width 0.3s; }

  .nav-buttons { display: flex; gap: 8px; }
  .nav-buttons a { padding: 6px 14px; background: #334155; color: #e2e8f0; text-decoration: none;
                   border-radius: 6px; font-size: 13px; }
  .nav-buttons a:hover { background: #475569; }
  .nav-buttons a.disabled { opacity: 0.3; pointer-events: none; }

  .card { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
  .card h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 8px; }
  .question { font-size: 16px; font-weight: 600; color: #f8fafc; line-height: 1.4; }
  .gold { font-size: 14px; color: #fbbf24; font-family: 'SF Mono', monospace; background: #1a1a2e;
          padding: 8px 12px; border-radius: 6px; margin-top: 4px; }

  .guide { font-size: 12px; color: #94a3b8; line-height: 1.5; }
  .guide strong { color: #e2e8f0; }
  .guide ul { padding-left: 16px; margin-top: 4px; }

  .form-group { display: flex; flex-direction: column; gap: 6px; }
  .form-group label { font-size: 13px; font-weight: 600; color: #94a3b8; }

  .radio-group { display: flex; gap: 12px; }
  .radio-option { flex: 1; }
  .radio-option input { display: none; }
  .radio-option span { display: block; text-align: center; padding: 10px; border: 2px solid #334155;
                       border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.2s; }
  .radio-option input:checked + span { border-color: #3b82f6; background: #1e3a5f; color: #60a5fa; }
  .radio-option:first-child input:checked + span { border-color: #22c55e; background: #14532d; color: #4ade80; }
  .radio-option:last-child input:checked + span { border-color: #ef4444; background: #450a0a; color: #f87171; }
  .radio-option span:hover { border-color: #475569; }

  input[type="text"], textarea { width: 100%; padding: 10px 12px; background: #0f172a; border: 1px solid #334155;
                                  border-radius: 8px; color: #e2e8f0; font-size: 14px; font-family: inherit; }
  input[type="text"]:focus, textarea:focus { outline: none; border-color: #3b82f6; }
  textarea { resize: vertical; min-height: 60px; }

  .submit-btn { width: 100%; padding: 12px; background: #3b82f6; color: white; border: none;
                border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
  .submit-btn:hover { background: #2563eb; }

  .status-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
  .status-done { background: #14532d; color: #4ade80; }
  .status-todo { background: #1e293b; color: #64748b; }
</style>
</head>
<body>
<div class="layout">
  <div class="panel">
    <div class="progress">
      <span>{{ done }}/{{ total }} done</span>
      <div class="progress-bar"><div class="progress-fill" style="width:{{ (done/total*100)|round }}%"></div></div>
      <div class="nav-buttons">
        <a href="{{ url_for('annotate', idx=idx-1) }}" class="{{ 'disabled' if idx == 0 else '' }}">← Prev</a>
        <a href="{{ url_for('annotate', idx=idx+1) }}" class="{{ 'disabled' if idx == total-1 else '' }}">Next →</a>
      </div>
    </div>

    <div class="card">
      <h3>Question ({{ item.id }}) {% if item.validity_label %}<span class="status-badge status-done">DONE</span>{% else %}<span class="status-badge status-todo">TODO</span>{% endif %}</h3>
      <div class="question">{{ item.question }}</div>
    </div>

    <div class="card">
      <h3>Gold Answer (from LLM)</h3>
      <div class="gold">{{ item.gold_answer }}</div>
    </div>

    <div class="card guide">
      <h3>Annotation Guide</h3>
      <ul>
        <li><strong>Valid</strong> = The question is clear, self-contained, and the gold answer is correct and found on the source page.</li>
        <li><strong>Invalid</strong> = The question is vague, unanswerable, the answer is wrong, or not on the page.</li>
        <li>If valid, copy/paste the gold answer or write your own if you disagree.</li>
        <li>Answer must be an <strong>exact text span</strong> from the source page, <strong>under 10 words</strong>.</li>
      </ul>
    </div>

    <form method="POST" action="{{ url_for('save', idx=idx) }}">
      <div style="display:flex; flex-direction:column; gap:14px;">
        <div class="form-group">
          <label>Validity</label>
          <div class="radio-group">
            <label class="radio-option">
              <input type="radio" name="validity_label" value="valid" {{ 'checked' if item.validity_label == 'valid' }}>
              <span>✓ Valid</span>
            </label>
            <label class="radio-option">
              <input type="radio" name="validity_label" value="invalid" {{ 'checked' if item.validity_label == 'invalid' }}>
              <span>✗ Invalid</span>
            </label>
          </div>
        </div>

        <div class="form-group">
          <label>Your Answer</label>
          <input type="text" name="annotated_answer" value="{{ item.annotated_answer }}"
                 placeholder="Paste exact text span from source page">
        </div>

        <div class="form-group">
          <label>Notes (optional)</label>
          <textarea name="notes" placeholder="Why invalid? Any ambiguity?">{{ item.notes }}</textarea>
        </div>

        <button type="submit" class="submit-btn">Save & Next →</button>
      </div>
    </form>
  </div>
<script>
  const goldAnswer = {{ item.gold_answer | tojson }};
  const answerInput = document.querySelector('input[name="annotated_answer"]');
  document.querySelectorAll('input[name="validity_label"]').forEach(radio => {
    radio.addEventListener('change', () => {
      if (radio.value === 'valid' && !answerInput.value) {
        answerInput.value = goldAnswer;
      }
    });
  });

  let showingCrawled = false;
  function toggleView() {
    showingCrawled = !showingCrawled;
    document.getElementById('frame-live').style.display = showingCrawled ? 'none' : '';
    document.getElementById('frame-crawled').style.display = showingCrawled ? '' : 'none';
    document.getElementById('toggle-btn').textContent = showingCrawled ? 'Show Live Page' : 'Show Crawled Text';
    document.getElementById('toggle-btn').style.background = showingCrawled ? '#f59e0b' : '#3b82f6';
  }
</script>

  <div class="source-frame">
    <div style="background:#1e293b;padding:6px 16px;border-bottom:1px solid #334155;display:flex;align-items:center;gap:12px;font-size:13px;">
      <button id="toggle-btn" onclick="toggleView()" style="padding:5px 14px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;white-space:nowrap;">Show Crawled Text</button>
      <a href="{{ item.url }}" target="_blank" style="color:#60a5fa;text-decoration:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;">{{ item.url }}</a>
      <a href="{{ item.url }}" target="_blank" style="color:#94a3b8;text-decoration:none;white-space:nowrap;">Open in new tab ↗</a>
    </div>
    <iframe id="frame-live" src="{{ url_for('proxy', url=item.url) }}"></iframe>
    <iframe id="frame-crawled" src="{{ url_for('crawled', url=item.url, gold=item.gold_answer) }}" style="display:none;"></iframe>
  </div>
</div>
</body>
</html>
"""


@app.route("/crawled")
def crawled():
    """Serve the crawled text for a URL, with the gold answer highlighted."""
    _load_corpus()
    target = request.args.get("url", "")
    gold = request.args.get("gold", "")
    page = CORPUS_BY_URL.get(target)
    if not page:
        return (
            '<html><body style="font-family:sans-serif;padding:40px;color:#94a3b8;background:#0f172a;">'
            '<h2>Not found in corpus</h2>'
            f'<p>No crawled data for:<br><code>{target}</code></p>'
            '</body></html>'
        ), 404
    import html as html_mod
    text = html_mod.escape(page.get("text", ""))
    title = html_mod.escape(page.get("title", ""))
    if gold:
        escaped_gold = html_mod.escape(gold)
        text = text.replace(escaped_gold,
            f'<mark style="background:#fbbf24;color:#000;padding:2px 4px;border-radius:3px;">{escaped_gold}</mark>')
    paragraphs = "\n".join(f"<p>{p}</p>" for p in text.split("\n") if p.strip())
    return (
        f'<html><head><style>'
        f'body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 20px;'
        f'       line-height: 1.7; color: #1e293b; background: #fff; }}'
        f'h1 {{ font-size: 22px; margin-bottom: 8px; color: #0f172a; }}'
        f'.url {{ font-size: 12px; color: #64748b; margin-bottom: 24px; word-break: break-all; }}'
        f'p {{ margin-bottom: 12px; }}'
        f'mark {{ scroll-margin-top: 100px; }}'
        f'</style></head><body>'
        f'<h1>{title}</h1>'
        f'<div class="url">{html_mod.escape(target)}</div>'
        f'{paragraphs}'
        f'<script>document.querySelector("mark")?.scrollIntoView({{behavior:"smooth",block:"center"}})</script>'
        f'</body></html>'
    )


@app.route("/proxy")
def proxy():
    """Fetch and serve external page to bypass X-Frame-Options restrictions."""
    target = request.args.get("url", "")
    if not target:
        return "No URL provided", 400
    try:
        if target.startswith("http://"):
            target = target.replace("http://", "https://", 1)
        r = req.get(target, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        content = r.text
        base_tag = f'<base href="{target}" target="_blank">'
        content = content.replace("<head>", f"<head>{base_tag}", 1)
        return content, r.status_code, {"Content-Type": r.headers.get("Content-Type", "text/html")}
    except Exception as e:
        return f"<h2>Could not load page</h2><p>{e}</p><p><a href='{target}' target='_blank'>Open directly</a></p>", 502


@app.route("/")
def index():
    items = load_annotations()
    first_todo = next((i for i, it in enumerate(items) if not it.get("validity_label")), 0)
    return redirect(url_for("annotate", idx=first_todo))


@app.route("/annotate/<int:idx>")
def annotate(idx):
    items = load_annotations()
    idx = max(0, min(idx, len(items) - 1))
    done = sum(1 for it in items if it.get("validity_label"))
    return render_template_string(TEMPLATE, item=items[idx], idx=idx,
                                  total=len(items), done=done)


@app.route("/save/<int:idx>", methods=["POST"])
def save(idx):
    items = load_annotations()
    items[idx]["validity_label"] = request.form.get("validity_label", "")
    items[idx]["annotated_answer"] = request.form.get("annotated_answer", "")
    items[idx]["notes"] = request.form.get("notes", "")
    save_annotations(items)
    next_idx = min(idx + 1, len(items) - 1)
    return redirect(url_for("annotate", idx=next_idx))


if __name__ == "__main__":
    app.run(port=5050, debug=True)
