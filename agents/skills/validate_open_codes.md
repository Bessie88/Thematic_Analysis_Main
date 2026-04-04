---
name: validate-open-codes
description: Validate open-coding outputs for grounded theory; enforce PASS/FAIL format and groundedness.
---

# Validate Open Codes

## Instructions
Validate the generated codes against the review text.

Rules (always follow):
- Respond with exactly one token on the first line: `PASS` or `FAIL`.
- If the first line is `FAIL`, list the issues on subsequent lines.
- Codes must be grounded in the review (evidence supports the claim).
- Codes must not be duplicates or near-duplicates.
- Codes must be concise concept labels and must include evaluative direction/quality (not neutral topics only).
- codes must be relevant to the research question 

Never output JSON.

## Examples
PASS example:
PASS

FAIL example:
FAIL
Issue 1: A code has no supporting evidence in the review.
Issue 2: Two codes are near-duplicates.

