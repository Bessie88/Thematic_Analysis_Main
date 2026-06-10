"""Tests for agents.core.embeddings (mocked backends; no live models)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.core import embeddings as emb_mod
from agents.core.embeddings import encode_texts
from agents.core.inference_config import embed_backend


@pytest.fixture(autouse=True)
def _reset_st_cache():
    emb_mod._st_model = None
    emb_mod._st_model_path = None
    yield
    emb_mod._st_model = None
    emb_mod._st_model_path = None


def test_encode_texts_empty():
    out = encode_texts([])
    assert out.shape == (0, 0)
    assert out.dtype == np.float32


def test_encode_texts_lmstudio_batches_and_normalizes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_EMBED_BACKEND", "lmstudio")
    monkeypatch.setenv("GT_EMBED_MODEL", "embed")
    monkeypatch.setenv("GT_OPENAI_BASE", "http://127.0.0.1:8000/v1")

    calls: list[list[str]] = []

    def fake_create(*, model: str, input: list[str]):
        calls.append(list(input))
        assert model == "embed"
        data = [
            SimpleNamespace(index=i, embedding=[float(i + 1), float((i + 1) * 2)])
            for i in range(len(input))
        ]
        return SimpleNamespace(data=data)

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create

    with patch("openai.OpenAI", return_value=mock_client):
        out = encode_texts(["a", "b", "c"], batch_size=2, normalize=True)

    assert len(calls) == 2
    assert calls[0] == ["a", "b"]
    assert calls[1] == ["c"]
    assert out.shape == (3, 2)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), rtol=1e-5)


def test_encode_texts_sentence_transformers_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_EMBED_BACKEND", "sentence_transformers")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        out = encode_texts(["x", "y"], normalize=True, show_progress=False)

    mock_model.encode.assert_called_once()
    assert out.shape == (2, 2)
    assert embed_backend() == "sentence_transformers"


def test_embed_backend_aliases(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_EMBED_BACKEND", "lm")
    assert embed_backend() == "lmstudio"
    monkeypatch.setenv("GT_EMBED_BACKEND", "api")
    assert embed_backend() == "lmstudio"
