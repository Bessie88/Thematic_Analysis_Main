"""Tests for agents.core.utils (parsing, extraction; no I/O except tempfile paths)."""

import json
from pathlib import Path

import pytest

from agents.core import utils


def test_remove_think_tags_strips_block():
    raw = "before <think>x</think> after"
    out = utils.remove_think_tags(raw)
    assert "<think>" not in out
    assert out.split() == ["before", "after"]


def test_remove_think_tags_multiline():
    raw = "intro\n<think>\nline\n</think>\noutro"
    assert utils.remove_think_tags(raw) == "intro\n\noutro"


def test_clean_and_parse_json_extracts_object():
    text = 'noise {"a": 1, "b": [true]} trailing'
    assert utils.clean_and_parse_json(text) == {"a": 1, "b": [True]}


def test_clean_and_parse_json_raises_without_braces():
    with pytest.raises(ValueError, match="No JSON brackets"):
        utils.clean_and_parse_json("no json here")


def test_extract_codes_empty():
    assert utils.extract_codes("") == []
    assert utils.extract_codes("   ") == []


def test_extract_codes_from_lines():
    blob = """
Some preamble
- Code: Alpha theme
- Code: Beta angle
- code: lowercase ok
"""
    assert utils.extract_codes(blob) == ["Alpha theme", "Beta angle", "lowercase ok"]


def test_summarize_llm_usage_missing_file(tmp_path: Path):
    missing = tmp_path / "none.jsonl"
    assert "not found" in utils.summarize_llm_usage(missing)


def test_summarize_llm_usage_aggregates(tmp_path: Path):
    p = tmp_path / "usage.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "tool": "t1",
                        "step": 1,
                        "skill": "s_a",
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    }
                ),
                json.dumps(
                    {
                        "tool": "t1",
                        "step": 2,
                        "skill": "s_b",
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    summary = utils.summarize_llm_usage(p)
    assert "Total LLM calls: 2" in summary
    assert "prompt_tokens=15" in summary
    assert "completion_tokens=25" in summary
    assert "total_tokens=40" in summary
    assert "t1:" in summary
    assert "s_a:" in summary
