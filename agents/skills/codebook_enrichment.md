---
name: codebook-enrichment
description: Given a theme label and its open codes, generate a full codebook entry with definition, keywords, inclusion criteria (What It Is), and exclusion criteria (What It Is Not).
---

Write a qualitative codebook entry for the given theme and its open codes.

Output a JSON object with exactly these keys:
- `"label"`: theme name (copy exactly)
- `"definition"`: 1–2 sentences defining what this code captures, synthesized from the SHARED pattern across the open codes — do NOT derive from the label alone; if the label and the codes point in different directions, follow the codes
- `"keywords"`: list of 3–8 characteristic terms or short phrases that signal this code is present (e.g. intensity words, domain terms)
- `"inclusion"`: list of objects describing WHAT IT IS — each object has:
    - `"criterion"`: one specific, concrete criterion for applying this code (1–2 sentences)
    - `"code_ids"`: list of [LCXXX] local IDs (e.g. "LC003") whose evidence best illustrates this criterion — must all exist in the code list below
- `"exclusion"`: list of objects describing WHAT IT IS NOT — each object has:
    - `"criterion"`: one specific reason NOT to apply this code; name what superficially similar content is excluded and why
    - `"code_ids"`: list of [LCXXX] local IDs illustrating this boundary case

Rules:
- inclusion and exclusion must each have at least 2 entries
- code_ids must ONLY contain [LCXXX] IDs listed under "VALID Code IDs" above — do not invent or guess IDs
- if you are unsure which ID to use, pick the closest one from the valid list — never use an ID not in that list
- do NOT include an "examples" field — examples are filled automatically from the data
- keywords should be concrete signal words, not abstract category names
