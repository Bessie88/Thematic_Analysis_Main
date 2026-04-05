---
name: meta-theme-grouping
description: Group cluster ids into a small handful of meta-themes (about 3–7 when many clusters) and output strict JSON.
---

# Meta-Theme Grouping

## Instructions
Group all provided cluster labels into a **small handful** of broad meta-themes.

Rules (always follow):
- Output **only** valid JSON.
- The JSON must have exactly one top-level key: `meta_themes`.
- `meta_themes` is a list of objects with keys: `name`, `cluster_ids`.
- Every provided cluster id must appear in exactly one group (no missing ids).
- With **six or more** clusters, aim for about **3 to 7** meta-themes (balance and coverage matter more than an exact count).
- With **fewer than six** clusters, use **2 to N** themes where N is the number of clusters unless labels are redundant.
- Meta-theme names should be short (2–6 words).

Do not add any other keys.

## Examples
{"meta_themes":[{"name":"Performance and stability","cluster_ids":["0","3"]},{"name":"Usability and UX","cluster_ids":["1","2"]}]}

