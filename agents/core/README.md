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
| `pipeline_helpers.py` | Embeddings, embedding axial clustering, hierarchy helpers. |
| `llm_clustering.py` | HICode-style LLM axial clustering (`USE_LLM_CLUSTERING` toggle). |
| `hierarchy_refine.py` | Caps fan-out and regroups codes inside `gt_hierarchy.json`. |
| `report.py` | Research report generation from the global graph. |
| `cooccurrence.py` | Builds co-occurrence stats from clustered codes + global graph. |

## Axial clustering modes

Default: **embedding + K-means** (`deduplicate_codes` → `axial_embed_and_cluster`).

Optional: **LLM clustering** (HICode-style). Set `USE_LLM_CLUSTERING = True` at the top of [`llm_clustering.py`](llm_clustering.py) (single toggle for CLI and `launch_sgl.sh`).

When enabled, `axial_coding` calls `axial_llm_cluster` from that module, a faithful port of HICode's
`cluster_labels_gpt` + `metrics.create_mapping` ([repo](https://github.com/mianzg/HICode),
[paper](https://arxiv.org/pdf/2509.17946)): unique labels are batch-clustered, the resulting
cluster names are re-clustered for a few rounds, then each label is chained to its final theme.
Labels the model omits or that aren't carried forward are **dropped** (HICode's `abandoned_labels`
behavior — no rescue / no coverage repair). The clustering call uses its own client at
`max_tokens=8192, temperature=0` (matching the paper). Output adds `clustering_method: "llm"`
and `cluster_theme_names` on [`gt_clustered_codes.json`](../outputs/data/gt_clustered_codes.json);
high-level labeling copies those names into `codebook.json` without extra LLM calls.

Env knobs (defaults match HICode): `GT_LLM_CLUSTER_MAX_ITER` (default 3, HICode `max_n_iter`),
`GT_LLM_CLUSTER_BATCH_SIZE` (default 100, HICode batch size), `GT_LLM_CLUSTER_SAVE_ITER`
(default 1 → writes `outputs/data/llm_clustering/cluster_iter_*.json`).

Refine still runs (LLM MOVEs; cluster-label similarity uses the embed model on **CPU**).

### Manual smoke (LLM path)

```bash
# SGLang must be up for open coding, axial, high-level, refine
python -m agents.cli --open-coding-only --research-question "..." --data ...
# Set USE_LLM_CLUSTERING = True in llm_clustering.py first
python -m agents.cli --axial-only --research-question "..."
python -m agents.cli --high-level-only --research-question "..."
python -m agents.cli --refine-only --research-question "..."
```