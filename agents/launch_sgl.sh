#!/bin/bash

cd /scratch/nimamot/agents/


# Model path: local dir under agents/ (pre-downloaded weights from Hugging Face)
MODEL_PATH="weights/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
PORT=8000

RESEARCH_QUESTION="What factors shape player experience and satisfaction in video games?"

# --- 2. Launch Server in Background ---
echo "Starting SGLang server..."


python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.90 \
  --context-length 8000 \
  > server.log 2>&1 &

# Capture the Process ID (PID) of the server so we can kill it later
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# --- 3. The 'Wait-For-Ready' Loop ---
echo "Waiting for server to become ready..."

MAX_RETRIES=1440  # wait up to 60 minutes (720*5s) for large model load
COUNTER=0

while true; do
    # Try to curl the models endpoint. 
    # If successful (HTTP 200), grep returns 0 and we break.
    if curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; then
        echo "Server is READY!"
        break
    fi

    # Check if timeout reached
    if [ $COUNTER -ge $MAX_RETRIES ]; then
        echo "Timeout waiting for server. Printing last 50 lines of log:"
        tail -n 50 server.log
        kill $SERVER_PID
        exit 1
    fi

    # Check if the process died prematurely
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Server process died unexpectedly. Check server.log."
        tail -n 50 server.log
        exit 1
    fi

    echo "Waiting for server... ($COUNTER/$MAX_RETRIES)"
    sleep 5
    ((COUNTER++))
done

# --- 4. Open coding only (then kill SGLang so axial can use GPU for embed model) ---
echo "Starting Open Coding (gt_agents.py --open-coding-only)..."
python gt_agents.py --open-coding-only --research-question "$RESEARCH_QUESTION"
OPEN_EXIT=$?
echo "Open coding finished with exit code $OPEN_EXIT. Killing SGLang server..."
kill $SERVER_PID
if [ $OPEN_EXIT -ne 0 ]; then
    exit $OPEN_EXIT
fi

# --- 5. Resolve embed model for axial step (same graph, second invocation) ---
EMBED_WEIGHTS_DIR="$(cd /scratch/nimamot/agents && pwd)/weights/Qwen3-Embedding-0.6B"
EMBED_WEIGHTS_REL="weights/Qwen3-Embedding-0.6B"
EMBED_HF_ID="Qwen/Qwen3-Embedding-0.6B"
if [ -d "$EMBED_WEIGHTS_DIR" ]; then
    export GT_EMBED_MODEL="$EMBED_WEIGHTS_DIR"
else
    echo "Embed model not in weights/; downloading once to $EMBED_WEIGHTS_REL ..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$EMBED_HF_ID" --local-dir "$EMBED_WEIGHTS_REL"
    else
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$EMBED_HF_ID', local_dir='$EMBED_WEIGHTS_REL', local_dir_use_symlinks=False)
"
    fi
    export GT_EMBED_MODEL="$EMBED_WEIGHTS_DIR"
fi

# --- 6. Axial step in the same graph (embed + K-means); offline so compute node does not hit HF ---
export HF_HUB_OFFLINE=1
echo "Starting Axial step (gt_agents.py --axial-only)..."
python gt_agents.py --axial-only --research-question "$RESEARCH_QUESTION"
AXIAL_EXIT=$?
if [ $AXIAL_EXIT -ne 0 ]; then
    exit $AXIAL_EXIT
fi

# --- 7. Restart SGLang for high-level code generation (Phase 3) ---
echo "Restarting SGLang for high-level code generation..."
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.90 \
  --context-length 8000 \
  > server.log 2>&1 &
SERVER_PID=$!
COUNTER=0
while true; do
    if curl -s http://localhost:$PORT/v1/models | grep -q '"object":"list"'; then
        echo "SGLang READY for high-level step."
        break
    fi
    if [ $COUNTER -ge $MAX_RETRIES ]; then
        echo "Timeout waiting for SGLang (phase 3)."
        tail -n 50 server.log
        kill $SERVER_PID
        exit 1
    fi
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "SGLang died (phase 3)."
        tail -n 50 server.log
        exit 1
    fi
    echo "Waiting for SGLang... ($COUNTER/$MAX_RETRIES)"
    sleep 5
    ((COUNTER++))
done

# --- 8. High-level code generation (one label per cluster) ---
echo "Starting high-level code generation (gt_agents.py --high-level-only)..."
python gt_agents.py --high-level-only --research-question "$RESEARCH_QUESTION"
HL_EXIT=$?
echo "High-level step finished with exit code $HL_EXIT."
if [ $HL_EXIT -ne 0 ]; then
    kill $SERVER_PID
    exit $HL_EXIT
fi

# --- 9. Refine cluster assignments (same SGLang session, before hierarchy) ---
echo "Starting refine cluster assignments (gt_agents.py --refine-only)..."
python gt_agents.py --refine-only --research-question "$RESEARCH_QUESTION"
REFINE_EXIT=$?
echo "Refine step finished with exit code $REFINE_EXIT."
if [ $REFINE_EXIT -ne 0 ]; then
    kill $SERVER_PID
    exit $REFINE_EXIT
fi

# --- 10. Hierarchy construction (relationship classification + edges), SGLang still up ---
echo "Starting hierarchy construction (gt_agents.py --hierarchy-only)..."
python gt_agents.py --hierarchy-only --research-question "$RESEARCH_QUESTION"
HIER_EXIT=$?
echo "Hierarchy step finished with exit code $HIER_EXIT."
if [ $HIER_EXIT -ne 0 ]; then
    kill $SERVER_PID
    exit $HIER_EXIT
fi

# --- 11. Graph construction (transitivity inference), no LLM needed ---
echo "Starting graph construction (gt_agents.py --graph-only)..."
python gt_agents.py --graph-only --research-question "$RESEARCH_QUESTION"
GRAPH_EXIT=$?
if [ $GRAPH_EXIT -ne 0 ]; then
    kill $SERVER_PID
    exit $GRAPH_EXIT
fi

# --- 12. Global graph construction (merge clusters, optional cross-cluster linking), SGLang still up ---
echo "Starting global graph construction (gt_agents.py --global-graph-only)..."
python gt_agents.py --global-graph-only --research-question "$RESEARCH_QUESTION"
GLOBAL_EXIT=$?
echo "Global graph step finished with exit code $GLOBAL_EXIT. Killing SGLang..."
kill $SERVER_PID
exit $GLOBAL_EXIT