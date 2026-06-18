"""Unified text embedding: LM Studio HTTP API or local SentenceTransformer."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .inference_config import (
    embed_backend,
    embed_model_id,
    openai_base,
    sentence_transformers_model_path,
)

_st_model = None
_st_model_path: Optional[str] = None


def _get_sentence_transformer():
    global _st_model, _st_model_path
    path = sentence_transformers_model_path()
    if _st_model is None or _st_model_path != path:
        from sentence_transformers import SentenceTransformer

        if os.path.isdir(path):
            os.environ["HF_HUB_OFFLINE"] = "1"
        _st_model = SentenceTransformer(path)
        _st_model_path = path
    return _st_model


def _encode_sentence_transformers(
    texts: List[str],
    *,
    batch_size: int,
    normalize: bool,
    show_progress: bool,
) -> np.ndarray:
    model = _get_sentence_transformer()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _encode_lmstudio(
    texts: List[str],
    *,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(base_url=openai_base(), api_key="EMPTY")
    model = embed_model_id()
    all_rows: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        all_rows.extend(item.embedding for item in ordered)

    arr = np.asarray(all_rows, dtype=np.float32)
    if normalize and len(arr) > 0:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        arr = arr / norms
    return arr


def encode_texts(
    texts: List[str],
    *,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """Return float32 embedding matrix with one row per input string."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if embed_backend() == "lmstudio":
        return _encode_lmstudio(texts, batch_size=batch_size, normalize=normalize)

    return _encode_sentence_transformers(
        texts,
        batch_size=batch_size,
        normalize=normalize,
        show_progress=show_progress,
    )
