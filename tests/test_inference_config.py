"""Tests for agents.core.inference_config."""

import pytest

from agents.core.inference_config import (
    embed_backend,
    embed_model_id,
    llm_model_name,
    openai_base,
)


def test_openai_base_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_OPENAI_BASE", raising=False)
    assert openai_base() == "http://localhost:8000/v1"


def test_openai_base_adds_v1_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_OPENAI_BASE", "http://host:1234")
    assert openai_base() == "http://host:1234/v1"


def test_llm_model_name_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_LLM_MODEL", "my-llm")
    assert llm_model_name() == "my-llm"


def test_embed_model_id_lmstudio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_EMBED_BACKEND", "lmstudio")
    monkeypatch.delenv("GT_EMBED_MODEL", raising=False)
    assert embed_model_id() == "embed"
    monkeypatch.setenv("GT_EMBED_MODEL", "custom-embed")
    assert embed_model_id() == "custom-embed"


def test_embed_backend_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_EMBED_BACKEND", raising=False)
    assert embed_backend() == "sentence_transformers"
