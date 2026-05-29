---
name: llm-clustering
description: Cluster open codes into named themes via JSON output (HICode-style axial step).
---

# LLM clustering (axial)

## Instructions

You are clustering a list of open codes (short qualitative labels) into broader themes.

Output rules (always follow):
- Output **only** one valid JSON object (no markdown fences, no commentary).
- Keys are short theme names (about 2–6 words when possible).
- Values are JSON arrays of strings; each string must appear **exactly** in the user input list.
- Do not add, rename, or paraphrase input labels in the value arrays.
- Merge labels that are semantically similar or redundant under one theme.
- Every input label must appear in exactly one cluster value list.

## Example

Input labels: `["laggy matchmaking", "slow menus", "unfair difficulty"]`

```json
{"Performance and pacing issues": ["laggy matchmaking", "slow menus"], "Difficulty balance": ["unfair difficulty"]}
```
