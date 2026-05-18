"""Tests for agents.core.skills (YAML frontmatter, skill loading, invoke seam)."""

from unittest.mock import MagicMock

import pytest
from agents.core import skills
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def test_strip_yaml_frontmatter_extracts_body():
    raw = "---\ntitle: x\n---\nSkill body here.\n"
    assert skills._strip_yaml_frontmatter(raw) == "Skill body here."


def test_strip_yaml_frontmatter_no_frontmatter():
    assert skills._strip_yaml_frontmatter("plain text") == "plain text"


def test_strip_yaml_frontmatter_unclosed_delimiter():
    raw = "---\nno closing\ncontent"
    assert "no closing" in skills._strip_yaml_frontmatter(raw)


def test_load_skill_text_reads_file(gt_skills_dir, monkeypatch: pytest.MonkeyPatch):
    path = gt_skills_dir / "open_coding.md"
    path.write_text("---\nmeta: x\n---\nDo open coding well.\n", encoding="utf-8")
    skills.load_skill_text.cache_clear()
    assert skills.load_skill_text("open_coding") == "Do open coding well."


def test_load_skill_text_missing_returns_empty(gt_skills_dir):
    skills.load_skill_text.cache_clear()
    assert skills.load_skill_text("nonexistent_skill") == ""


def test_load_skill_text_disabled(monkeypatch: pytest.MonkeyPatch, gt_skills_dir):
    path = gt_skills_dir / "open_coding.md"
    path.write_text("should not read", encoding="utf-8")
    monkeypatch.setenv("GT_USE_SKILLS", "0")
    skills.load_skill_text.cache_clear()
    assert skills.load_skill_text("open_coding") == ""


def test_llm_invoke_with_skill_uses_system_message(gt_skills_dir):
    path = gt_skills_dir / "open_coding.md"
    path.write_text("System skill text.", encoding="utf-8")
    skills.load_skill_text.cache_clear()

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="model reply")

    out = skills.llm_invoke_with_skill(llm, "open_coding", "human part")
    assert out == "model reply"
    arg = llm.invoke.call_args[0][0]
    assert isinstance(arg, list)
    assert isinstance(arg[0], SystemMessage)
    assert arg[0].content == "System skill text."
    assert isinstance(arg[1], HumanMessage)


def test_llm_invoke_with_skill_plain_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_USE_SKILLS", "0")
    skills.load_skill_text.cache_clear()

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="plain")

    out = skills.llm_invoke_with_skill(llm, "open_coding", "only human")
    assert out == "plain"
    arg = llm.invoke.call_args[0][0]
    assert arg == "only human"
