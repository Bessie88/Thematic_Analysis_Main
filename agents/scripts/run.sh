#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G
#SBATCH --time=0-12:00:00
#SBATCH --gpus=h100_40gb:1
#SBATCH --account=rrg-lingjzhu

# TODO: Change to relative paths.
# Use absolute paths so the script works when SLURM runs from the job spool dir (where only this file exists).
AGENTS_SCRIPTS="/scratch/nimamot/agents/scripts"
SIF_PATH="/scratch/nimamot/agents/pytorch-langgraph-sgl.sif"

module load apptainer

nvidia-smi

export HF_DATASETS_CACHE=/scratch/lingjzhu/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/lingjzhu/cache/huggingface
export HF_HOME=/scratch/lingjzhu/cache/huggingface

# TODO: Change to relative paths.
# Override path: SECRETS_FILE=/path/to/file sbatch run.sh
SECRETS_FILE="${SECRETS_FILE:-/scratch/nimamot/agents/scripts/.env.supabase}"
if [ -f "$SECRETS_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
else
  echo "Note: $SECRETS_FILE not found — set SUPABASE_* and UPLOAD_TO_SUPABASE there if you want uploads." >&2
fi

# TODO: upload should happen automatically
# If Supabase keys are set but UPLOAD_TO_SUPABASE was omitted, default to upload (set UPLOAD_TO_SUPABASE=0 to skip).
if [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ] && [ -n "${SUPABASE_URL:-}" ]; then
  export UPLOAD_TO_SUPABASE="${UPLOAD_TO_SUPABASE:-1}"
fi

# Default CSV is reddit_comment_text_10000.csv (launch_sgl.sh). Override examples:
# export GT_DATA_CSV=/scratch/nimamot/data/school_burnout_text_review.csv
# export GT_DATA_CSV=/scratch/nimamot/data/reddit_comment_text_1000.csv

apptainer exec -C --nv --home /scratch/nimamot/vllm_env_home -W /scratch/nimamot/vllm_env_home -B /project -B /scratch "$SIF_PATH" bash "$AGENTS_SCRIPTS/launch_sgl.sh"
