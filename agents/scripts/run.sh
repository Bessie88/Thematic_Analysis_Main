#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G
#SBATCH --time=0-03:00:00
#SBATCH --gpus=h100_40gb:1
#SBATCH --account=rrg-lingjzhu

# Use absolute paths so the script works when SLURM runs from the job spool dir (where only this file exists).
AGENTS_SCRIPTS="/scratch/nimamot/agents/scripts"
SIF_PATH="/scratch/nimamot/agents/pytorch-langgraph-sgl.sif"

module load apptainer

nvidia-smi

export HF_DATASETS_CACHE=/scratch/lingjzhu/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/lingjzhu/cache/huggingface
export HF_HOME=/scratch/lingjzhu/cache/huggingface

apptainer exec -C --nv --home /scratch/nimamot/vllm_env_home -W /scratch/nimamot/vllm_env_home -B /project -B /scratch "$SIF_PATH" bash "$AGENTS_SCRIPTS/launch_sgl.sh"
