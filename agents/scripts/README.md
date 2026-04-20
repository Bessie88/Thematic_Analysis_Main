# `agents/scripts`

Shell entrypoints for **bringing up inference servers** and **submitting Slurm jobs** against this repo’s paths and containers.

**Versioned in git:** only the four files below. Anything else in this folder on a machine is local (untracked, ignored, or personal).

| Script | Role |
|--------|------|
| `launch.sh` | Start a **vLLM** OpenAI-compatible API server (paths point at `agents/weights/…`). |
| `launch_sgl.sh` | Start **SGLang** (and related setup) for pipeline / report servers that expect that stack. |
| `run.sh` | **Slurm** batch script: run the GT pipeline inside the project Apptainer image (cluster-specific `#SBATCH` and paths inside the file). |
| `run_embed.sh` | **Slurm** job for **embedding + clustering** after open coding has produced `gt_codes_only.json` (encode + K-means; no long-lived API server). |
