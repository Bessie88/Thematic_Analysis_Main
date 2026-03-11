# Grounded Theory (LOGOS) Pipeline

This directory contains a Grounded Theory / LOGOS pipeline that turns a research question and a corpus into a structured codebook graph.

## Repo Layout

```text
agents/
├── gt_agents.py              # stable CLI entrypoint
├── run.sh                    # SLURM batch entrypoint
├── launch_sgl.sh             # orchestrates server lifecycle + phases
├── run_visualize.sh          # SLURM wrapper for graph visualization
├── pipeline/                 # implementation code
│   ├── app.py
│   ├── state.py
│   ├── tools.py
│   ├── utils.py
│   ├── paths.py
│   ├── embed_and_cluster.py
│   └── visualize_graph.py
├── outputs/
│   ├── data/                 # generated JSON / JSONL / markdown / html artifacts
│   └── logs/                 # pipeline logs
├── old_slurm.out/
└── weights/
```

The top-level entrypoints stay stable, while the implementation now lives under `pipeline/`.

## End-To-End Flow

```text
Research question + corpus
  -> Step 1: Open coding
  -> Step 2: Axial coding (embed + K-means)
  -> Step 2b: High-level label generation
  -> Step 4: Hierarchy construction
  -> Step 5: Per-cluster graph construction
  -> Step 6: Global graph construction
  -> Step 6 (clean-up): Codebook clean-up (optional post-process)
```

## Pipeline Steps

| Step | Name | Uses LLM | What it produces |
|------|------|----------|------------------|
| 1 | Open coding | Yes | Short grounded codes per review |
| 2 | Axial coding | No | Clusters over all extracted codes |
| 2b | High-level generation | Yes | One representative label per cluster |
| 4 | Hierarchy construction | Yes | Within-cluster equivalence and subsumption relations |
| 5 | Graph construction | No | Per-cluster transitive closure with deduction-first conflict handling |
| 6 | Global graph construction | Optional LLM | One global codebook graph with optional cross-cluster links |

## Execution Model

- `sbatch run.sh` schedules the full workflow on the cluster.
- `run.sh` launches the containerized environment and calls `launch_sgl.sh`.
- `launch_sgl.sh` handles phase ordering and starts/stops SGLang as needed.
- `gt_agents.py` is the stable CLI entrypoint for phase-specific runs.

Phase breakdown:

1. SGLang up: open coding
2. SGLang down: axial coding / embeddings
3. SGLang up again: high-level labels, hierarchy, graph, global graph

## Stable Commands

```bash
# Full pipeline via SLURM
sbatch run.sh

# Direct CLI
python gt_agents.py --research-question "..."

# Individual phases
python gt_agents.py --open-coding-only --research-question "..."
python gt_agents.py --axial-only --research-question "..."
python gt_agents.py --high-level-only --research-question "..."
python gt_agents.py --hierarchy-only --research-question "..." --sim-threshold 0.6
python gt_agents.py --graph-only --research-question "..."
python gt_agents.py --global-graph-only --research-question "..." --skip-cross-cluster
```

## Outputs

Generated artifacts now live under `outputs/data/`.

| Output | Purpose |
|--------|---------|
| `outputs/data/gt_codes_only.json` | All extracted open codes and codes-per-review |
| `outputs/data/gt_clustered_codes.json` | Axial clustering output |
| `outputs/data/codebook.json` | Cluster representative labels plus `cluster_to_codes` |
| `outputs/data/gt_hierarchy_edges.jsonl` | Non-orthogonal relation classifications |
| `outputs/data/gt_hierarchy.json` | Per-cluster hierarchy after merges + explicit directed edges |
| `outputs/data/gt_graph.json` | Per-cluster graph after transitive inference |
| `outputs/data/gt_cross_cluster_edges.jsonl` | Optional cross-cluster relation classifications |
| `outputs/data/gt_global_graph.json` | Final global codebook graph |
| `outputs/data/gt_global_graph_cleaned.json` | Cleaned codebook (LOGOS Step 6: merge equiv, collapse low-freq, remove orphans) |
| `outputs/data/gt_open_codes_all_reviews.md` | Human-readable open coding dump |
| `outputs/data/gt_graph.html` | Quick HTML visualization of the per-cluster graph |

Logs:

| Output | Purpose |
|--------|---------|
| `outputs/logs/gt_agent_trace.log` | Step-by-step pipeline trace |
| `server.log` | SGLang server stdout/stderr |

## Main Modules

| File | Responsibility |
|------|----------------|
| `pipeline/paths.py` | Centralizes artifact, log, and weights paths |
| `pipeline/utils.py` | Logging and parsing helpers |
| `pipeline/tools.py` | LLM tools, embedding logic, hierarchy/graph/global graph builders |
| `pipeline/state.py` | LangGraph state, router, and tool dispatch |
| `pipeline/app.py` | Compiled LangGraph application |
| `pipeline/embed_and_cluster.py` | Standalone embed + cluster utility |
| `pipeline/visualize_graph.py` | HTML visualization generator for `gt_graph.json` |

## Visualization

To generate a quick HTML view of the per-cluster graph:

```bash
sbatch run_visualize.sh
```

This writes `outputs/data/gt_graph.html`.

## Codebook clean-up (LOGOS Step 6)

After global graph construction, run codebook clean-up to reduce the graph to a compact codebook (~20–40 concepts): equivalence closure, representative selection by datapoint frequency and in-degree, merge and redirect, collapse low-frequency children (direct edges only), remove orphans, then rebuild and optionally recompute inferred edges.

```bash
./run_cleanup.sh
# or
python3 codebook_cleanup.py
```

Optional: `--graph`, `--clustered`, `--codes-only`, `--out`, `--min-freq`, `--w-freq`, `--w-indeg`, `--no-inferred`.

## Configuration

- Research question: pass `--research-question` or set `RESEARCH_QUESTION` in `launch_sgl.sh`
- Embed model: `GT_EMBED_MODEL` env var or `weights/Qwen3-Embedding-0.6B`
- LLM endpoint: `http://localhost:8000/v1`
- LLM weights: `weights/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit`
