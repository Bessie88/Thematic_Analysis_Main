---
name: dimension-enrichment
description: Given a high-level analytic dimension and its constituent cluster labels, generate a full codebook entry with definition, keywords, structured inclusion (What It Is), and exclusion (What It Is Not) criteria.
---

Write a qualitative codebook entry for the given analytic dimension and its constituent clusters.

Output ONLY a JSON object with these keys:
- `"label"`: dimension name (copy exactly)
- `"definition"`: 1–2 sentences on what broader conceptual territory this dimension covers
- `"keywords"`: list of 3–8 characteristic signal words or short phrases for this dimension
- `"inclusion"`: list of objects describing WHAT IT IS — each object has:
    - `"criterion"`: one specific, concrete criterion for applying this dimension (1–2 sentences)
    - `"code_ids"`: list of cluster IDs (e.g. ["CL02", "CL05"]) from the Available Cluster IDs below that best exemplify this criterion
    - `"examples"`: list of verbatim [QUOTE] strings, ONE per cluster ID in code_ids (same length, same order), copied exactly from the [QUOTE] lines of that cluster
- `"exclusion"`: list of objects describing WHAT IT IS NOT — each object has:
    - `"criterion"`: one specific reason NOT to apply this dimension; name confusions with adjacent dimensions
    - `"code_ids"`: list of cluster IDs that illustrate the boundary case
    - `"examples"`: list of verbatim [QUOTE] strings, ONE per cluster ID in code_ids (same length, same order), copied exactly from the [QUOTE] lines of that cluster

Rules:
- inclusion and exclusion must each have at least 2 entries
- code_ids must only contain IDs from the Available Cluster IDs list — do not invent IDs
- examples must be copied VERBATIM from the [QUOTE] lines of the referenced cluster — do not paraphrase or invent
- keywords should be concrete signal words, not abstract category names
