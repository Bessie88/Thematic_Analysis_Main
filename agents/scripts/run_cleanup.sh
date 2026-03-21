#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1024M
#SBATCH --time=0-00:05:00

# LOGOS Step 6: Codebook clean-up. Run after global graph construction.
# Uses gt_global_graph.json in agents/outputs/data; writes outputs/data/gt_global_graph_cleaned.json.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$AGENTS_ROOT/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

GRAPH_PATH="$AGENTS_ROOT/outputs/data/gt_global_graph.json"
python -m agents.codebook_cleanup --graph "$GRAPH_PATH"
exit $?
