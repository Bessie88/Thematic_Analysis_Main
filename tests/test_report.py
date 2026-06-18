"""Tests for agents.core.report (tree rendering and graph text; no live LLM)."""

import json
from pathlib import Path

import pytest
from agents.core.report import (
    _count_leaves,
    _default_api_base,
    _default_model_name,
    _render_tree_node,
    build_graph_text_for_llm,
)


def test_render_tree_node_code_leaf():
    node = {"name": "laggy UI", "type": "code"}
    assert "* laggy UI" in _render_tree_node(node)


def test_count_leaves_nested():
    tree = {
        "name": "Root",
        "type": "theme",
        "children": [
            {"name": "a", "type": "code"},
            {
                "name": "sub",
                "type": "sub_theme",
                "children": [{"name": "b", "type": "code"}],
            },
        ],
    }
    assert _count_leaves(tree) == 2


def test_build_graph_text_for_llm_tree_format(fixtures_dir: Path):
    graph_path = fixtures_dir / "minimal_global_graph.json"
    text = build_graph_text_for_llm(graph_path, max_chars=50_000)
    assert "ThemeA" in text
    assert "code_a" in text
    assert "THEME HIERARCHY" in text


def test_build_graph_text_for_llm_legacy_truncates(tmp_path: Path):
    data = {
        "canonical_nodes": [f"node_{i}" for i in range(20)],
        "edges": [{"parent": f"p{i}", "child": f"c{i}"} for i in range(20)],
    }
    path = tmp_path / "legacy_graph.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    text = build_graph_text_for_llm(path, max_chars=400)
    assert "truncated" in text.lower() or len(text) <= 500


def test_default_api_base_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REPORT_OPENAI_BASE", raising=False)
    monkeypatch.delenv("GT_OPENAI_BASE", raising=False)
    assert "localhost" in _default_api_base()
    monkeypatch.setenv("GT_OPENAI_BASE", "http://lmhost:7000/v1")
    assert _default_api_base() == "http://lmhost:7000/v1"
    monkeypatch.setenv("REPORT_OPENAI_BASE", "http://custom:9000/v1/")
    assert _default_api_base() == "http://custom:9000/v1"


def test_default_model_name_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REPORT_MODEL_NAME", raising=False)
    monkeypatch.delenv("GT_LLM_MODEL", raising=False)
    assert _default_model_name() == "llm"
    monkeypatch.setenv("GT_LLM_MODEL", "qwen-chat")
    assert _default_model_name() == "qwen-chat"
    monkeypatch.setenv("REPORT_MODEL_NAME", "mistral-7b")
    assert _default_model_name() == "mistral-7b"
