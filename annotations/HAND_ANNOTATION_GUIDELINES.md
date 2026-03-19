# Hand Annotation Guidelines (30-Question IAA Set)

## Goal
Two annotators independently judge whether each QA pair is supported by the source URL and provide a corrected answer when needed.

## Files
- `annotations/iaa_annotator1_template.jsonl`
- `annotations/iaa_annotator2_template.jsonl`

Each line has:
- `id`
- `question`
- `gold_answer`
- `url`
- `validity_label` (to fill)
- `annotated_answer` (to fill)
- `notes` (optional)

## What to Label
For each item:
1. Open the `url`.
2. Decide whether `gold_answer` correctly answers `question` using evidence from that page.
3. Fill:
   - `validity_label`: `valid` or `invalid`
   - `annotated_answer`:
     - if `valid`: copy the minimal correct answer span from the page
     - if `invalid`: write the correct answer from the page
4. Add brief reasoning in `notes` if the case is ambiguous.

## Decision Rules
1. `valid` if the answer is fully supported by the page and semantically correct.
2. `invalid` if the answer is wrong, incomplete, overly broad, or unsupported.
3. Use exact entities where possible (full name, exact number/date, official title).
4. Prefer concise spans over full sentences.
5. If multiple equivalent forms exist, use the most canonical page form.
6. If the page does not contain enough evidence, mark `invalid` and set `annotated_answer` to `UNKNOWN`.

## Normalization Conventions
1. Names: preserve capitalization and accents if shown.
2. Dates: keep page style when possible (e.g., `March 18, 2016`).
3. Numbers/codes: keep exact punctuation and hyphenation (e.g., `UCB/EECS-2012-96`).
4. Multi-part answers: separate with ` | ` when truly multiple valid answers are required.

## Quality Checklist
1. No discussion between annotators before submission.
2. Finish all 30 items.
3. Ensure `validity_label` is always either `valid` or `invalid`.
4. Ensure `annotated_answer` is non-empty for every item.

## Compute IAA
After both files are complete:

```bash
python compute_iaa.py --a1 annotations/iaa_annotator1_template.jsonl --a2 annotations/iaa_annotator2_template.jsonl
```

This reports:
- label agreement (%)
- Cohen's kappa (valid/invalid)
- answer EM/F1 on jointly valid items
