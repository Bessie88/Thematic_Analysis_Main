# `agents/scripts`

Shell entrypoints for **bringing up inference servers** and **submitting Slurm jobs** against this repo’s paths and containers.

| Script | Role |
|--------|------|
| `launch.sh` | Start a **vLLM** OpenAI-compatible API server (paths point at `agents/weights/…`). |
| `launch_sgl.sh` | **SGLang** pipeline: start/stop server around axial embedding; Mistral for report. |
| `launch_lm.sh` | **LM Studio** via Apptainer (`lmstudio-llmster-preview.sif`): Qwen LLM + Qwen3-Embedding on port 1234; pipeline runs in `pytorch-langgraph-sgl.sif`. |
| `pull_lmstudio_sif.sh` | Pull `docker://lmstudio/llmster-preview:cpu` into `agents/lmstudio-llmster-preview.sif`. |
| `download_lm_models.sh` | Download GGUF weights for `launch_lm.sh` (`lms get` or `--hf-only` via `hf download`). |
| `run.sh` | **Slurm** batch script: Apptainer + launcher (`GT_LAUNCHER=sgl` default, or `GT_LAUNCHER=lm`). |
| `run_embed.sh` | **Slurm** job for embedding + clustering after open coding (`gt_codes_only.json`). |

## Launcher selection

```bash
# Default: SGLang (AWQ weights under agents/weights/)
sbatch run.sh

# LM Studio (Apptainer llmster-preview + GGUF weights on scratch)
bash agents/scripts/pull_lmstudio_sif.sh
bash agents/scripts/download_lm_models.sh --hf-only
GT_LAUNCHER=lm sbatch run.sh
```

### Alliance clusters (rorqual, fir)

No host `lms` install is required for `GT_LAUNCHER=lm`. Pull the LM Studio SIF and download GGUF weights on a login node:

```bash
module load apptainer
bash agents/scripts/pull_lmstudio_sif.sh

module load StdEnv/2023 python/3.11
bash agents/scripts/download_lm_models.sh --hf-only
```

`run.sh` runs `launch_lm.sh` on the **host** (not nested inside the PyTorch SIF). The launcher starts `lmstudio-llmster-preview.sif` for inference and uses `pytorch-langgraph-sgl.sif` for pipeline Python stages.

LM Studio path env vars (set in `launch_lm.sh` or override before `sbatch`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `GT_LAUNCHER` | `sgl` | `lm` selects `launch_lm.sh` on host |
| `LM_SIF_PATH` | `agents/lmstudio-llmster-preview.sif` | LM Studio Apptainer image |
| `PYTORCH_SIF_PATH` | same as `SIF_PATH` | Pipeline Python container |
| `LM_APPTAINER_HOME` | `/scratch/nimamot/lmstudio_apptainer_home` | Writable home for `lms` / models |
| `LM_PORT` | `1234` | OpenAI-compatible API port (container default) |
| `LM_LLM_LOAD` | `lmstudio-community/...-GGUF/<file>.gguf` | Chat path for `lms load` after import |
| `LM_EMBED_LOAD` | `Qwen/Qwen3-Embedding-0.6B-GGUF/<file>.gguf` | Embedding path for `lms load` |
| `GT_OPENAI_BASE` | `http://127.0.0.1:1234/v1` | Set by launcher for Python clients |
| `GT_EMBED_BACKEND` | `sentence_transformers` (sgl) / `lmstudio` (lm) | Embedding backend |
| `GT_LLM_MODEL` | `llm` | Chat model id for OpenAI API |

Python inference env (both launchers): `GT_OPENAI_BASE`, `GT_LLM_MODEL`, `GT_EMBED_BACKEND`, `GT_EMBED_MODEL`.
