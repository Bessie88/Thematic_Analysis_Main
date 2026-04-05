---
name: hierarchy-construction
description: Build hierarchical groupings for grounded theory; always output schema-compliant JSON for the current sub-step.
---

# Hierarchy Construction

## Instructions
You are constructing parts of a grounded-theory hierarchy.

Critical rule:
- Output ONLY valid JSON that matches the schema implied by the human prompt.

Discipline (navigation / readability):
- Prefer **named sub-themes** over dumping codes into `ungrouped_codes` or `unassigned`; those lists should be **empty or minimal**.
- When assigning overflow codes to existing sub-themes, pick the **best-fit** theme per code unless no theme applies at all.
- Use **enough** sub-themes that no single list is huge (soft target: on the order of **≤ ~50 items** per sub-theme when batches are large).

Never:
- Output markdown, commentary, or extra keys not requested.
- Add explanations outside the JSON.

## Examples
Example JSON for a sub-theme grouping call (schema may differ):
{"sub_themes":[{"name":"Coping strategies","codes":["code a","code b"]}],"ungrouped_codes":[]}

Example JSON for assigning overflow codes to existing sub-themes (schema may differ):
{"assignments":{"Coping strategies":["code d","code e"]},"unassigned":[]}

