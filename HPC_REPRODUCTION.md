# Reproducing the GT pipeline on an HPC (after `git clone`)

This document compares **what is tracked in git** with **what the Slurm / Apptainer job needs at runtime**, and gives a **concrete run recipe** (data, research question, weights, `sbatch`).

There is **no GitHub Actions (or similar) CI workflow** in this repository. The batch path is:

`agents/scripts/run.sh` (Slurm) → **Apptainer** → `agents/scripts/launch_sgl.sh` → staged `python -m agents.cli` calls.

---

## 1. How to run the pipeline (`sbatch run.sh`)

Do these **once per clone / machine**, then submit the job.

### 1.1 Clone and go to the scripts directory

```bash
git clone <your-repo-url> grounded-theory-pipeline   # example name
cd grounded-theory-pipeline/agents/scripts
```

You can call `sbatch` from any directory if you pass the **absolute path** to `run.sh`; the script resolves the repo from its own location.

### 1.2 Put your input CSV on disk

1. Create the repo-root **`data/`** directory if it does not exist (same level as `agents/`, `README.md`):

   ```text
   <repo_root>/
     agents/
     data/          ← your CSV lives here
     README.md
   ```

2. Save your file (for example `my_corpus.csv`). The pipeline expects **exactly one** of these column names:

   - **`text_review`** (preferred), or  
   - **`review_text`**

3. Open **`agents/scripts/launch_sgl.sh`** and set **`GT_DATA_CSV`** to that file. By default the script points at:

   ```bash
   GT_DATA_CSV="$REPO_ROOT/data/train.csv"
   ```

   So either **name your file `train.csv`** under `data/`, or change the line to match your filename, for example:

   ```bash
   GT_DATA_CSV="$REPO_ROOT/data/my_corpus.csv"
   ```

   `REPO_ROOT` is the parent of `agents/` (the repository root). **`data/` is not in git** (see `.gitignore`); you must supply the CSV yourself.

### 1.3 Set the research question

In **`agents/scripts/launch_sgl.sh`**, edit **`RESEARCH_QUESTION`** to match your study (keep it analytic and avoid “naming” expected themes in the question text—see comments in that file for guidance).

```bash
RESEARCH_QUESTION="Your question here?"
export RESEARCH_QUESTION
```

**Note:** `RESEARCH_QUESTION` and `GT_DATA_CSV` are assigned **inside `launch_sgl.sh`**. Exporting them in the shell before `sbatch` does **not** override those lines unless you change the script to use something like `${RESEARCH_QUESTION:-'default'}`.

### 1.4 Model weights and Apptainer image

Under **`agents/weights/`** (not in git), you need at least:

| Path (under `agents/weights/`) | Role |
|--------------------------------|------|
| `Qwen3-30B-A3B-Instruct-2507-AWQ-4bit` | Main SGLang model (`MODEL_PATH` in `launch_sgl.sh`). Job **fails** if missing. |
| `Mistral-7B-Instruct-v0.3` | Research report phase (`REPORT_MODEL_PATH`). Job **fails** if missing. |
| `Qwen3-Embedding-0.6B` | Axial embeddings (or allow **one-time** Hugging Face download inside the job; after that, axial uses offline mode). |

Place the **`.sif`** Apptainer image at **`agents/pytorch-langgraph-sgl.sif`** by default, or set **`SIF_PATH`** when submitting (see below).

### 1.5 Slurm and Apptainer in `run.sh`

Edit **`agents/scripts/run.sh`** for your allocation:

- `#SBATCH` lines: `--account`, `--mail-user`, GPU type (`--gpus=`), time, memory.
- **`apptainer exec`** bind flags (`-B /project -B /scratch`) if your site uses different mount points.

**Portable defaults inside `run.sh`:**

- **`SIF_PATH`** — defaults to `$AGENTS_ROOT/pytorch-langgraph-sgl.sif`; override:  
  `SIF_PATH=/path/to/image.sif sbatch run.sh`
- **`APPTAINER_HOME`** — defaults to `$REPO_ROOT/vllm_env_home` (created if missing); writable home inside the container. Override if you want that cache on scratch:  
  `APPTAINER_HOME=/scratch/$USER/vllm_env_home sbatch run.sh`
- **`HF_CACHE`** — defaults to `$REPO_ROOT/cache/huggingface` (Hugging Face cache env vars are exported in `run.sh`).

### 1.6 Optional: Supabase

If **`agents/scripts/.env.supabase`** exists (not in git), `run.sh` sources it. If both `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set, **`UPLOAD_TO_SUPABASE`** defaults to **`1`**. Set `UPLOAD_TO_SUPABASE=0` in that file (or in the environment) to skip upload.

### 1.7 Submit the job

From **`agents/scripts`**:

```bash
sbatch run.sh
```

Or with overrides (examples):

```bash
SIF_PATH=/project/shared/pytorch-langgraph-sgl.sif sbatch /absolute/path/to/repo/agents/scripts/run.sh
```

Monitor Slurm output (`slurm-<jobid>.out` in the submission directory, or your site’s default). Logs and artifacts also land under **`agents/outputs/`** and **`agents/server.log`** (see §5).

---

## 2. What git contains vs what stays local

| Item | In git? | Notes |
|------|--------|--------|
| Python package (`agents/`), skills (`agents/skills/`), `agents/core/paths.py`, CLI | Yes | Stages, prompts, default Python path constants. |
| `agents/scripts/run.sh`, `launch_sgl.sh`, `launch.sh` | Yes | Slurm / Apptainer / SGLang wiring; **edit `run.sh` and `launch_sgl.sh` for your site and study.** |
| `agents/requirements-pipeline.txt` | Yes | LangChain stack; `launch_sgl.sh` may `pip install --user` if imports are missing in the image. |
| `agents/weights/` | **No** | All large model directories. |
| `agents/*.sif` | **No** | Apptainer image. |
| Repo-root `data/` | **No** | Your CSV corpus. |
| `agents/outputs/` | **No** | Run outputs. |
| `agents/scripts/.env.supabase` | **No** | Optional upload secrets. |
| `vllm_env_home/`, `venv/`, SLURM `*.out` | **No** | Runtime / job logs. |
| Root `docs/`, `agents/docs/`, `notebooks/` | **No** (`.gitignore`) | Local only; upload errors may mention `agents/docs/SUPABASE_SETUP.md` as a hint—that path may not exist in a clone. |

**Takeaway:** A fresh clone has **code and shell wiring** only. **Image, weights, CSV, and optional secrets** are yours to provide.

---

## 3. End-to-end flow (what the job does)

`launch_sgl.sh`:

1. Ensures LangChain-related deps import; if not, installs from `agents/requirements-pipeline.txt` (`pip install --user`).
2. Starts **SGLang** with the **Qwen** path under `agents/weights/` for open coding and most LLM steps.
3. Runs **`python -m agents.cli --open-coding-only … --data "$GT_DATA_CSV"`**.
4. Stops SGLang, runs **axial** (embed + cluster) using **`GT_EMBED_MODEL`** (local `weights/` or first-time HF download).
5. Restarts SGLang: **high-level → refine → hierarchy → meta-themes → global graph**.
6. Switches to **Mistral** for **research report**, then **co-occurrence** (no LLM).
7. Optionally uploads if **`UPLOAD_TO_SUPABASE=1`** and credentials are set.

`run.sh` adds Slurm, `module load apptainer`, HF cache env vars, optional secrets, **`APPTAINER_HOME`**, and **`apptainer exec … launch_sgl.sh`**.

---

## 4. Detailed reference (weights, CSV schema, API)

### 4.1 Main generation model (Qwen)

- **Path in `launch_sgl.sh`:** `MODEL_PATH="$AGENTS_ROOT/weights/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"`
- Change **`MODEL_PATH`** if your weights live elsewhere.

### 4.2 Embedding model (axial)

- **Local:** `agents/weights/Qwen3-Embedding-0.6B`
- If missing, `launch_sgl.sh` tries a one-time download from **`Qwen/Qwen3-Embedding-0.6B`**, then sets **`HF_HUB_OFFLINE=1`** for the axial step—embed weights must exist on disk before that step succeeds.

### 4.3 Research report model (Mistral)

- **Path:** `agents/weights/Mistral-7B-Instruct-v0.3`
- If missing, the script may reference **`download_mistral_7b_v03.sh`**, which is **not** in this repository—download weights manually or adjust **`REPORT_MODEL_PATH`** in `launch_sgl.sh`.

### 4.4 Python / OpenAI-compatible server

- **`agents/core/tools.py`** uses `http://localhost:8000/v1` and model name **`llm`**, matching SGLang’s **`--served-model-name`** in `launch_sgl.sh`.

---

## 5. Outputs after a successful run

Under **`agents/outputs/`** (gitignored), especially **`agents/outputs/data/`**:

- `gt_codes_only.json`, `gt_clustered_codes.json`, `codebook.json`, hierarchy and meta-theme JSON, `gt_global_graph.json`, `research_report.md`, `gt_cooccurrence.json`, and related logs.

Optional upload reads those paths via **`agents/scripts/upload_pipeline_to_supabase.py`**.

---

## 6. Known rough edges

| Issue | Detail |
|-------|--------|
| **Mistral download script** | Error text may cite `download_mistral_7b_v03.sh`; not shipped in repo. |
| **Supabase setup doc** | Hints may mention `agents/docs/SUPABASE_SETUP.md`; that tree is often absent on a clean clone. |
| **`GT_DATA_CSV` / `RESEARCH_QUESTION`** | Set in **`launch_sgl.sh`**; edit there for each study unless you refactor to env-based defaults. |

---

## 7. `agents/core/paths.py` vs the batch job

`paths.py` defines **`DEFAULT_DATA_CSV`** and artifact paths under **`agents/outputs/`**. The **Slurm + `launch_sgl.sh`** path does **not** use Python’s `DEFAULT_DATA_CSV` for open coding; it uses **`GT_DATA_CSV`** in **`launch_sgl.sh`**. For manual CLI runs without `launch_sgl.sh`, use **`python -m agents.cli --data … --research-question …`** as documented in `agents/cli.py`.
