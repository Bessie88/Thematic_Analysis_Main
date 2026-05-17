#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --gpus=h100:1
#SBATCH --account=rrg-lingjzhu
#SBATCH --mail-user=nima.motieifard@gmail.com
#SBATCH --mail-type=END,FAIL
#
# Fir: same GPU/account and path assumptions as run.sh.

# Embedding + clustering job (no server; loads model once and runs encode + K-means).
# Run after agents.cli has produced gt_codes_only.json.
# Usage: sbatch run_embed.sh
# Use absolute paths so the script works when SLURM runs from the job spool dir.
AGENTS_SCRIPTS="/scratch/nimamot/agents/scripts"
SIF_PATH="/scratch/nimamot/agents/pytorch-langgraph-sgl.sif"

module load apptainer

nvidia-smi

export HF_DATASETS_CACHE=/scratch/nimamot/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/nimamot/cache/huggingface
export HF_HOME=/scratch/nimamot/cache/huggingface

apptainer exec -C --nv --home /scratch/nimamot/vllm_env_home -W /scratch/nimamot/vllm_env_home -B /project -B /scratch "$SIF_PATH" bash "$AGENTS_SCRIPTS/launch_embed.sh"
