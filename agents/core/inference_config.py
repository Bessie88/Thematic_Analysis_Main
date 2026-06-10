"""Env-driven inference settings for LLM and embedding backends."""

from __future__ import annotations

import os

from .paths import WEIGHTS_DIR

_DEFAULT_OPENAI_BASE = "http://localhost:8000/v1"
_DEFAULT_LLM_MODEL = "llm"
_DEFAULT_LMSTUDIO_EMBED_ID = "embed"


def openai_base() -> str:
    """OpenAI-compatible API base for chat and embedding requests."""
    raw = os.environ.get("GT_OPENAI_BASE", _DEFAULT_OPENAI_BASE).strip()
    base = raw.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def llm_model_name() -> str:
    """Served chat model identifier (e.g. SGLang --served-model-name or lms --identifier)."""
    return os.environ.get("GT_LLM_MODEL", _DEFAULT_LLM_MODEL).strip() or _DEFAULT_LLM_MODEL


def embed_backend() -> str:
    """``lmstudio`` (HTTP /v1/embeddings) or ``sentence_transformers`` (in-process)."""
    raw = os.environ.get("GT_EMBED_BACKEND", "sentence_transformers").strip().lower()
    if raw in {"lmstudio", "lm", "api"}:
        return "lmstudio"
    return "sentence_transformers"


def sentence_transformers_model_path() -> str:
    """Local HF path or hub id for SentenceTransformer backend."""
    model_name = os.environ.get("GT_EMBED_MODEL") or (
        str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B")
        if os.path.isdir(str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B"))
        else "Qwen/Qwen3-Embedding-0.6B"
    )
    if os.path.isdir(model_name):
        return os.path.abspath(model_name)
    return model_name


def embed_model_id() -> str:
    """Model id for LM Studio embedding API (``model`` field in /v1/embeddings)."""
    explicit = os.environ.get("GT_EMBED_MODEL", "").strip()
    if explicit and embed_backend() == "lmstudio":
        return explicit
    return _DEFAULT_LMSTUDIO_EMBED_ID
