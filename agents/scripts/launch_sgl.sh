#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$AGENTS_ROOT/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

MODEL_PATH="$AGENTS_ROOT/weights/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
SERVER_LOG="$AGENTS_ROOT/server.log"
PORT=8000
RESEARCH_QUESTION="How do commenters frame the severity of climate change and the possibility or impossibility of meaningful response?"
# RESEARCH_QUESTION="What do players dislike about the game?"

# Stop SGLang reliably so GPU VRAM is freed before axial (embedding) and other steps.
# Previously: ( cd ... && python ... ) &  made $! the subshell PID, so kill left the Python tree alive.
stop_sglang_server() {
    local pid="${1:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping SGLang main process (PID $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        local i=0
        while [ "$i" -lt 45 ] && kill -0 "$pid" 2>/dev/null; do
            sleep 1
            i=$((i + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "SGLang PID $pid still running; sending SIGKILL..."
            kill -KILL "$pid" 2>/dev/null || true
            sleep 2
        fi
    fi
    # Catch worker/orphan processes that still hold VRAM (same user only).
    local u="${USER:-$(whoami)}"
    if pkill -TERM -u "$u" -f "sglang.launch_server" 2>/dev/null; then
        sleep 3
    fi
    pkill -KILL -u "$u" -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
    echo "nvidia-smi (after SGLang stop):"
    nvidia-smi 2>/dev/null || true
}

# Wait until OpenAI-compatible /v1/models responds (shared by Qwen and Mistral phases).
# Args: port, log_file, pid, label
wait_for_openai_ready() {
    local port="$1"
    local log_file="$2"
    local pid="$3"
    local phase_label="${4:-SGLang}"
    local counter=0
    echo "Waiting for $phase_label to become ready on port $port..."
    while true; do
        if curl -s "http://localhost:${port}/v1/models" | grep -q '"object":"list"'; then
            echo "$phase_label is READY!"
            return 0
        fi
        if [ $counter -ge $MAX_RETRIES ]; then
            echo "Timeout waiting for $phase_label. Last 50 lines of $log_file:"
            tail -n 50 "$log_file"
            stop_sglang_server "$pid"
            return 1
        fi
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "$phase_label process died unexpectedly. Check $log_file."
            tail -n 50 "$log_file"
            return 1
        fi
        echo "Waiting for $phase_label... ($counter/$MAX_RETRIES)"
        sleep 5
        ((counter++))
    done
}

# --- 2. Launch Server in Background ---
echo "Starting SGLang server..."
# Run python directly (no subshell) so $! is the real server PID.
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.90 \
  --context-length 8000 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

MAX_RETRIES=2400
# --- 3. The 'Wait-For-Ready' Loop ---
if ! wait_for_openai_ready "$PORT" "$SERVER_LOG" "$SERVER_PID" "SGLang (Qwen)"; then
    exit 1
fi

# --- 4. Open coding only (then kill SGLang so axial can use GPU for embed model) ---
echo "Starting Open Coding (agents.cli --open-coding-only)..."
python -m agents.cli --open-coding-only --research-question "$RESEARCH_QUESTION"
OPEN_EXIT=$?
echo "Open coding finished with exit code $OPEN_EXIT. Killing SGLang server..."
stop_sglang_server "$SERVER_PID"
if [ $OPEN_EXIT -ne 0 ]; then
    exit $OPEN_EXIT
fi

# --- 5. Resolve embed model for axial step ---
EMBED_WEIGHTS_DIR="$AGENTS_ROOT/weights/Qwen3-Embedding-0.6B"
EMBED_WEIGHTS_REL="$AGENTS_ROOT/weights/Qwen3-Embedding-0.6B"
EMBED_HF_ID="Qwen/Qwen3-Embedding-0.6B"
if [ -d "$EMBED_WEIGHTS_DIR" ]; then
    export GT_EMBED_MODEL="$EMBED_WEIGHTS_DIR"
else
    echo "Embed model not in weights/; downloading once to $EMBED_WEIGHTS_REL ..."
    if command -v huggingface-cli &>/dev/null; then
        (cd "$AGENTS_ROOT" && huggingface-cli download "$EMBED_HF_ID" --local-dir "weights/Qwen3-Embedding-0.6B")
    else
        (cd "$AGENTS_ROOT" && python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$EMBED_HF_ID', local_dir='weights/Qwen3-Embedding-0.6B', local_dir_use_symlinks=False)
")
    fi
    export GT_EMBED_MODEL="$EMBED_WEIGHTS_DIR"
fi

# --- 6. Axial step ---
export HF_HUB_OFFLINE=1
echo "Starting Axial step (agents.cli --axial-only)..."
python -m agents.cli --axial-only --research-question "$RESEARCH_QUESTION"
AXIAL_EXIT=$?
if [ $AXIAL_EXIT -ne 0 ]; then
    exit $AXIAL_EXIT
fi

# --- 7. Restart SGLang for high-level code generation ---
echo "Restarting SGLang for high-level code generation..."
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.90 \
  --context-length 8000 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
if ! wait_for_openai_ready "$PORT" "$SERVER_LOG" "$SERVER_PID" "SGLang (Qwen, phase 2)"; then
    exit 1
fi

# --- 8. High-level code generation ---
echo "Starting high-level code generation (agents.cli --high-level-only)..."
python -m agents.cli --high-level-only --research-question "$RESEARCH_QUESTION"
HL_EXIT=$?
echo "High-level step finished with exit code $HL_EXIT."
if [ $HL_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $HL_EXIT
fi

# --- 9. Refine cluster assignments ---
echo "Starting refine cluster assignments (agents.cli --refine-only)..."
python -m agents.cli --refine-only --research-question "$RESEARCH_QUESTION"
REFINE_EXIT=$?
echo "Refine step finished with exit code $REFINE_EXIT."
if [ $REFINE_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $REFINE_EXIT
fi

# --- 10. Hierarchy construction ---
echo "Starting hierarchy construction (agents.cli --hierarchy-only)..."
python -m agents.cli --hierarchy-only --research-question "$RESEARCH_QUESTION"
HIER_EXIT=$?
echo "Hierarchy step finished with exit code $HIER_EXIT."
if [ $HIER_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $HIER_EXIT
fi

# --- 11. Graph construction ---
echo "Starting graph construction (agents.cli --graph-only)..."
python -m agents.cli --graph-only --research-question "$RESEARCH_QUESTION"
GRAPH_EXIT=$?
if [ $GRAPH_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $GRAPH_EXIT
fi

# --- 12. Global graph construction ---
echo "Starting global graph construction (agents.cli --global-graph-only)..."
python -m agents.cli --global-graph-only --research-question "$RESEARCH_QUESTION"
GLOBAL_EXIT=$?
echo "Global graph step finished with exit code $GLOBAL_EXIT."
if [ $GLOBAL_EXIT -ne 0 ]; then
    echo "Stopping SGLang after global graph failure..."
    stop_sglang_server "$SERVER_PID"
    exit $GLOBAL_EXIT
fi

echo "Stopping Qwen SGLang to free GPU for Mistral research-report server..."
stop_sglang_server "$SERVER_PID"

REPORT_MODEL_PATH="$AGENTS_ROOT/weights/Mistral-7B-Instruct-v0.3"
if [ ! -d "$REPORT_MODEL_PATH" ]; then
    echo "Error: Mistral weights not found at $REPORT_MODEL_PATH"
    echo "Run: bash \"$AGENTS_ROOT/scripts/download_mistral_7b_v03.sh\""
    exit 1
fi

REPORT_SERVER_LOG="$AGENTS_ROOT/report_server.log"
echo "Starting Mistral SGLang for research report..."
python -m sglang.launch_server \
  --model-path "$REPORT_MODEL_PATH" \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.85 \
  --context-length 8192 \
  >"$REPORT_SERVER_LOG" 2>&1 &
REPORT_PID=$!
echo "Mistral report server PID: $REPORT_PID"

if ! wait_for_openai_ready "$PORT" "$REPORT_SERVER_LOG" "$REPORT_PID" "SGLang (Mistral report)"; then
    exit 1
fi

echo "Starting research report (agents.cli --report-only)..."
python -m agents.cli --report-only --research-question "$RESEARCH_QUESTION"
REPORT_STEP_EXIT=$?
echo "Research report step finished with exit code $REPORT_STEP_EXIT. Stopping Mistral SGLang..."
stop_sglang_server "$REPORT_PID"
exit $REPORT_STEP_EXIT
