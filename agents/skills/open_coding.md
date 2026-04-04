---
name: open-coding
description: How to perform open coding on one review and emit strict bullet-format codes with evidence.
---

# Open Coding

## Instructions
Follow the human prompt’s research question and review text.

Always:
- Produce **1–3** codes total per review.
- Each code must be a **short noun phrase** (2–6 words) that includes an evaluative **quality/direction** (what is good/bad, what is frustrating, what is lacking).
- Codes must be **grounded in the review text**.
- codes must be relevant to the research question 

Output format (must match exactly, no extra text):
- Code: <code>
  Evidence: "<short quote from the review>"
  Note: <one short phrase why this code fits>

If the human prompt includes reviewer feedback, incorporate it into the next code attempt.

## Examples
Example output:
- Code: frustrating difficulty spike
  Evidence: "The early levels were easy, then it got unexpectedly hard."
  Note: The reviewer describes a sudden jump in difficulty

