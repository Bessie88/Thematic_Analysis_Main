---
name: hierarchy-refine
description: Subdivide oversized code buckets into smaller thematic groups; preserve exact code strings and full coverage.
---

# Hierarchy refine (fan-out control)

## Instructions
You split large lists of qualitative **codes** (short phrases) into **several named sub-groups** so the hierarchy stays readable.

- Use the **exact** code strings from the prompt; never paraphrase, merge, or omit codes.
- Every input code must appear in **exactly one** output group.
- Sub-group names should be **2–6 words**, specific to the codes in that group.
- Prefer **balanced** group sizes when the prompt asks for multiple groups.
- Output **only** the JSON shape requested in the human message—no markdown fences or commentary.

## Examples
If the human lists twelve codes and asks for three sub-themes, return JSON with three objects under `sub_themes`, each with a short name and a `codes` array, and every input code present exactly once across those arrays.
