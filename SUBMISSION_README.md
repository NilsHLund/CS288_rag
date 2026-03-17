# CS288 RAG Submission

**Cache required** — do not delete `cache/`.

```bash
rm submission.zip
zip -r submission.zip . \
  -x "*.pyc" \
  -x "__pycache__/*" \
  -x ".git/*" \
  -x "corpus/pages.json" \
  -x "*.tmp" \
  -x ".env"
```
**Cache required** — do not delete `cache/`.

```bash
bash run.sh <questions_txt_path> <predictions_out_path>
```
