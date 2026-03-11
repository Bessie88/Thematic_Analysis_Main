#!/bin/bash
# Run embedding + clustering in the same job environment as gt_agents.
# Same pattern as the LLM: use weights/ if present (pre-loaded); otherwise download once and save there.

cd /scratch/nimamot/agents/

EMBED_WEIGHTS_DIR="weights/Qwen3-Embedding-0.6B"
EMBED_HF_ID="Qwen/Qwen3-Embedding-0.6B"

if [ -d "$EMBED_WEIGHTS_DIR" ]; then
    echo "Using pre-loaded embed model: $EMBED_WEIGHTS_DIR"
    EMBED_MODEL="$EMBED_WEIGHTS_DIR"
else
    echo "Embed model not in weights/; downloading once and saving to $EMBED_WEIGHTS_DIR ..."
    python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('$EMBED_HF_ID').save('$EMBED_WEIGHTS_DIR')
"
    echo "Using embed model: $EMBED_WEIGHTS_DIR"
    EMBED_MODEL="$EMBED_WEIGHTS_DIR"
fi

python embed_and_cluster.py --model "$EMBED_MODEL" --codes outputs/data/gt_codes_only.json --out-dir outputs/data
exit $?
