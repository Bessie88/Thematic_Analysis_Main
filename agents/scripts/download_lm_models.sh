#!/bin/bash
# Download GGUF models for the LM Studio Apptainer pipeline (launch_lm.sh).
#
# Uses hf / huggingface_hub (works on Alliance login nodes).
#   bash agents/scripts/download_lm_models.sh
#
# launch_lm.sh symlink-imports GGUF into LM_APPTAINER_HOME and loads by path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LM_LLM_REPO="${LM_LLM_REPO:-lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF}"
LM_LLM_QUANT="${LM_LLM_QUANT:-Q4_K_M}"
LM_EMBED_REPO="${LM_EMBED_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
LM_EMBED_QUANT="${LM_EMBED_QUANT:-Q8_0}"
LM_WEIGHTS_DIR="${LM_WEIGHTS_DIR:-$AGENTS_ROOT/weights/lmstudio}"

usage() {
    cat <<EOF
Usage: bash download_lm_models.sh [--hf-only]

  --hf-only   Accepted for compatibility (default behavior).

Env overrides:
  LM_LLM_REPO / LM_LLM_QUANT   (default: lmstudio-community/...-GGUF / Q4_K_M)
  LM_EMBED_REPO / LM_EMBED_QUANT (default: Qwen/Qwen3-Embedding-0.6B-GGUF / Q8_0)
  LM_WEIGHTS_DIR               (default: agents/weights/lmstudio)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --hf-only) shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

ensure_hf_tools() {
    if command -v hf &>/dev/null; then
        return 0
    fi
    if python -c "import huggingface_hub" 2>/dev/null; then
        return 0
    fi
    echo "huggingface_hub not found; trying pip install --user ..."
    if ! python -m pip install --user "huggingface_hub>=0.20" 2>/dev/null; then
        echo "Error: need hf CLI or huggingface_hub." >&2
        echo "  On Alliance: module load python/3.11 StdEnv/2023" >&2
        echo "  pip install --user huggingface_hub   (provides the hf command)" >&2
        echo "  Compute nodes: module load httpproxy  (for outbound HF access)" >&2
        return 1
    fi
}

download_via_huggingface() {
    ensure_hf_tools || return 1
    local repo="$1"
    local quant="$2"
    local dest="$3"
    mkdir -p "$dest"
    echo "HF download: $repo (quant pattern *${quant}*) -> $dest"

    if command -v hf &>/dev/null; then
        hf download "$repo" \
            --include "*${quant}*.gguf" \
            --local-dir "$dest"
        return 0
    fi

    python - <<PY
from huggingface_hub import hf_hub_download
import os

repo = "${repo}"
quant = "${quant}"
dest = "${dest}"
os.makedirs(dest, exist_ok=True)

from huggingface_hub import list_repo_files
files = [f for f in list_repo_files(repo) if f.endswith(".gguf") and quant in f]
if not files:
    raise SystemExit(f"No .gguf matching quant {quant!r} in {repo}")
path = hf_hub_download(repo_id=repo, filename=files[0], local_dir=dest)
print("Downloaded:", path)
PY
}

echo "=== LM Studio / GGUF model download ==="
echo "LLM:   $LM_LLM_REPO @ $LM_LLM_QUANT"
echo "Embed: $LM_EMBED_REPO @ $LM_EMBED_QUANT"
echo ""

download_via_huggingface "$LM_LLM_REPO" "$LM_LLM_QUANT" "$LM_WEIGHTS_DIR/llm"
download_via_huggingface "$LM_EMBED_REPO" "$LM_EMBED_QUANT" "$LM_WEIGHTS_DIR/embed"

echo ""
echo "Download complete. Weights under: $LM_WEIGHTS_DIR"
echo "Next: bash $SCRIPT_DIR/pull_lmstudio_sif.sh && GT_LAUNCHER=lm sbatch $SCRIPT_DIR/run.sh"
