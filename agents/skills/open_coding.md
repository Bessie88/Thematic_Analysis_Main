---
name: open-coding
description: Open-code one review (0–3 codes or Applicability NONE when the text does not answer the research question); strict bullet format with evidence.
---

# Open Coding

## Instructions
Follow the human prompt’s research question and review text.

Always:
- Produce **0–3** codes per review. If the review has **nothing** that answers the research question, output **no** `- Code:` lines — use the **Applicability: NONE** block from the human prompt instead of inventing codes.
- When you do code, each label must be a **short noun phrase** (2–6 words) with evaluative **quality/direction**.
- Codes must be **grounded in the review text** and **relevant to the research question**.

Output formats (must match the human prompt exactly, no extra text):
- When nothing applies: `- Applicability: NONE` with Reason and Evidence (no `- Code:` lines).
- When coding: for each code, `- Code: ...` with Evidence and Note.

If the human prompt includes reviewer feedback, incorporate it into the next attempt.

## Examples
No applicable content:
- Applicability: NONE
  Reason: The review only says the game is cool and does not mention negative feedback asked for by the question.
  Evidence: "the game is very cool"

Example with codes:
- Code: frustrating difficulty spike
  Evidence: "The early levels were easy, then it got unexpectedly hard."
  Note: The reviewer describes a sudden jump in difficulty

