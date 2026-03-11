#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --gpus=h100_40gb:1
#SBATCH --account=rrg-lingjzhu

# Embedding + clustering job (no server; loads model once and runs encode + K-means).
# Run after gt_agents.py has produced gt_codes_only.json.
# Usage: sbatch run_embed.sh

module load apptainer

nvidia-smi

export HF_DATASETS_CACHE=/scratch/lingjzhu/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/lingjzhu/cache/huggingface
export HF_HOME=/scratch/lingjzhu/cache/huggingface

apptainer exec -C --nv --home /scratch/nimamot/vllm_env_home -W /scratch/nimamot/vllm_env_home -B /project -B /scratch pytorch-langgraph-sgl.sif bash /scratch/nimamot/agents/launch_embed.sh
