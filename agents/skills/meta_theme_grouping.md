---
name: meta-theme-grouping
description: Group cluster ids into 4–5 meta-themes and output strict JSON.
---

# Meta-Theme Grouping

## Instructions
Group all provided cluster labels into exactly 4 or 5 broad meta-themes.

Rules (always follow):
- Output **only** valid JSON.
- The JSON must have exactly one top-level key: `meta_themes`.
- `meta_themes` is a list of objects with keys: `name`, `cluster_ids`.
- Every provided cluster id must appear in exactly one group (no missing ids).
- Meta-theme names should be short (2–6 words).

Do not add any other keys.

## Examples
{"meta_themes":[{"name":"Performance and stability","cluster_ids":["0","3"]},{"name":"Usability and UX","cluster_ids":["1","2"]}]}

