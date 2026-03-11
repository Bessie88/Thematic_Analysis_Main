#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=0-00:05:00

cd /scratch/nimamot/agents
python3 visualize_graph.py
