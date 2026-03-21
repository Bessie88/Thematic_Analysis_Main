export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_TRITON_AWQ=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python -m vllm.entrypoints.openai.api_server \
    --model "$AGENTS_ROOT/weights/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit" \
    --served-model-name llm \
    --port 8000 \
    --max-model-len 8000 \
    --gpu-memory-utilization 0.90 \
