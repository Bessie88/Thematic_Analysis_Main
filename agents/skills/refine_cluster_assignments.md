---
name: refine-cluster-assignments
description: Move codes across clusters using MOVE/NONE lines, or output NONE when no confident move exists.
---

# Refine Cluster Assignments

## Instructions
Given a cluster label and its assigned codes, you may output moves to up to five candidate target cluster labels.

Output rules (must match exactly):
- Output `NONE` (exactly) if no code can be moved with full confidence.
- Otherwise output one move line per moved code in this exact form:
  MOVE: "<code>" → "<target cluster label>"

Always:
- Move a code only if it unambiguously belongs in the target cluster.
- If the fit is borderline, or the target label could plausibly match multiple clusters, output `NONE` instead.
- Do not output JSON, explanations, or any text outside the MOVE/NONE format.

## Examples
Move example:
MOVE: "crashing during loading" → "stability and bugs"

No moves example:
NONE

