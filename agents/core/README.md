# `agents/core`

Python package that implements the grounded-theory pipeline: LangGraph wiring, per-step tools, shared paths, prompts, and small utilities used by `agents/cli.py` and scripts.

## Layout

| Module | Role |
|--------|------|
| `app.py` | Compiles the LangGraph (`agent` ↔ `tool` loop). |
| `state.py` | `GTState`, routing, and dispatch into named tools. |
| `tools.py` | LLM-backed pipeline steps (open/axial coding, hierarchy, meta-themes, tree, etc.). |
| `paths.py` | Canonical paths under `agents/outputs/` and related data files. |
| `prompts.py` | Prompt strings imported by the tools. |
| `skills.py` | Loads markdown skills from `agents/skills/`. |
| `utils.py` | Logging, JSON cleanup, small shared helpers. |
| `pipeline_helpers.py` | Embeddings, clustering, and non-LLM hierarchy/meta-theme helpers. |
| `hierarchy_refine.py` | Caps fan-out and regroups codes inside `gt_hierarchy.json`. |
| `report.py` | Research report generation from the global graph. |
| `cooccurrence.py` | Builds co-occurrence stats from clustered codes + global graph. |