---
name: high-level-synthesis
description: Synthesize multiple candidate cluster labels into one final label/confidence/rationale.
---

# High-Level Label Synthesis

## Instructions
You are reconciling multiple candidate labels, each generated from a different random
sample of open codes within the same cluster.

Output rules (always follow):
- Output **only** one valid JSON object (no markdown, no surrounding text).
- The JSON must contain exactly these keys:
  - `label`
  - `confidence`
  - `rationale`
- `label`: short (2–6 words) final label.
- `confidence`: integer 1–5. High (4–5) when candidates agree; low (1–2) when they diverge.
- `rationale`: one sentence explaining the synthesis decision.

## Examples
{"label":"Payment and checkout friction","confidence":5,"rationale":"All five candidates described payment or checkout problems; wording varied but theme was identical."}
{"label":"App stability issues","confidence":2,"rationale":"Candidates split between crash reports and slow performance, suggesting a heterogeneous cluster."}
