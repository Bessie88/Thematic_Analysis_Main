# `agents/scripts`

Shell entrypoints for **bringing up inference servers** and **submitting Slurm jobs** against this repo’s paths and containers.

**Versioned in git:** only the four files below. Anything else in this folder on a machine is local (untracked, ignored, or personal).

| Script | Role |
|--------|------|
| `launch.sh` | Start a **vLLM** OpenAI-compatible API server (paths point at `agents/weights/…`). |
| `launch_sgl.sh` | Start **SGLang** (and related setup) for pipeline / report servers that expect that stack. |
| `run.sh` | **Slurm** batch script: run the GT pipeline inside the project Apptainer image (cluster-specific `#SBATCH` and paths inside the file). |
| `run_embed.sh` | **Slurm** job for **embedding + clustering** after open coding has produced `gt_codes_only.json` (encode + K-means; no long-lived API server). |

## LLM axial clustering (`launch_sgl.sh`)

Set `USE_LLM_CLUSTERING = True` in [`agents/core/llm_clustering.py`](../core/llm_clustering.py). `launch_sgl.sh` reads that flag automatically.

Effects when `True`:

- SGLang **stays up** after open coding (no GPU handoff for embedding-based axial).
- Axial uses LLM clustering via `agents.cli --axial-only` (no extra CLI flags).
- Embed weights are downloaded only before **refine** (CPU embed for similar cluster labels).
- High-level step copies `cluster_theme_names` from `gt_clustered_codes.json` (no per-cluster labeling LLM).

Default (`False`): unchanged behavior (stop SGLang → embedding axial → restart SGLang).
