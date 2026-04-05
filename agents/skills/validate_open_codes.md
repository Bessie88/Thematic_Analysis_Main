---
name: validate-open-codes
description: Validate open-coding output (codes with evidence, or Applicability NONE); PASS/FAIL first line; reject invented codes on irrelevant reviews.
---

# Validate Open Codes

## Instructions
Validate the generated codes against the review text.

Rules (always follow):
- Respond with exactly one token on the first line: `PASS` or `FAIL`.
- If the first line is `FAIL`, list the issues on subsequent lines.
- If the coder used **Applicability: NONE** (no `- Code:` lines), PASS only when the review truly has nothing question-relevant to code; FAIL if they skipped real relevant content.
- If there are `- Code:` lines: codes must be grounded, non-duplicate, concise with evaluative direction, and relevant to the research question.
- FAIL if output mixes NONE with `- Code:` lines, or is inconsistent.

Never output JSON.

## Examples
PASS (codes present and valid):
PASS

PASS (coder correctly declared nothing to code — review not relevant to the research question):
PASS
The review does not address the research question; Applicability NONE is justified.

FAIL example:
FAIL
Issue 1: A code has no supporting evidence in the review.
Issue 2: Two codes are near-duplicates.

FAIL (should have coded but used NONE):
FAIL
Issue 1: The review contains clear negative feedback relevant to the question; skipping codes is incorrect.

