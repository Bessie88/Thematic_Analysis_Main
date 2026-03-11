#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1024M
#SBATCH --time=0-00:05:00

# LOGOS Step 6: Codebook clean-up. Run after global graph construction.
# Uses gt_global_graph.json in agents dir; writes outputs/data/gt_global_graph_cleaned.json.

cd /scratch/nimamot/agents
python3 codebook_cleanup.py --graph /scratch/nimamot/agents/gt_global_graph.json
exit $?
