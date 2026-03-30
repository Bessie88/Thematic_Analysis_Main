---
name: hierarchy-construction
description: Build hierarchical groupings for grounded theory; always output schema-compliant JSON for the current sub-step.
---

# Hierarchy Construction

## Instructions
You are constructing parts of a grounded-theory hierarchy.

Critical rule:
- Output ONLY valid JSON that matches the schema implied by the human prompt.

Never:
- Output markdown, commentary, or extra keys not requested.
- Add explanations outside the JSON.

## Examples
Example JSON for a sub-theme grouping call (schema may differ):
{"sub_themes":[{"name":"Coping strategies","codes":["code a","code b"]}],"ungrouped_codes":["code c"]}

Example JSON for assigning overflow codes to existing sub-themes (schema may differ):
{"assignments":{"Coping strategies":["code d","code e"]},"unassigned":["code f"]}

