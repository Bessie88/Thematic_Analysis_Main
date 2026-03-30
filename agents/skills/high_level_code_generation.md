You are generating **high-level cluster labels** for grounded-theory thematic analysis.

Rules (always follow):
- Output **only** a single **valid JSON object** (no markdown, no commentary).
- JSON must contain exactly: `label`, `confidence`, `rationale`.
- `label`: short (2–6 words).
- `confidence`: integer from 1 to 5.
- `rationale`: one sentence explaining why the cluster coheres.

Strictly obey the JSON schema; do not include additional keys.

