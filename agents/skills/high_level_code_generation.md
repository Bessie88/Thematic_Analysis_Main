---
name: high-level-code-generation
description: Produce a single JSON object with label/confidence/rationale for one cluster.
---

# High-Level Code Generation

## Instructions
You are generating a single high-level label for one cluster from its bullet list of open codes.

Output rules (always follow):
- Output **only** one valid JSON object (no markdown, no surrounding text).
- The JSON must contain exactly these keys:
  - `label`
  - `confidence`
  - `rationale`
- `label`: short (2–6 words).
- `confidence`: integer 1–5.
- `rationale`: one sentence explaining why the cluster coheres.

## Examples
{"label":"Interface usability issues","confidence":4,"rationale":"Codes all describe usability and interaction problems across the cluster."}

