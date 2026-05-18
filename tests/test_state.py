"""Tests for agents.core.state routing and tool dispatch (no LangGraph compile)."""

from unittest.mock import MagicMock, patch

from agents.core.state import (
    OPEN_CODING_MAX_RETRIES,
    _parse_validation_output,
    agent_node,
    router,
    tool_node,
)
from langgraph.graph import END


def test_parse_validation_output_pass():
    verdict, feedback = _parse_validation_output("PASS\nLooks good.")
    assert verdict == "PASS"
    assert "Looks good" in feedback


def test_parse_validation_output_fail():
    verdict, feedback = _parse_validation_output("  fail\nIssues listed")
    assert verdict == "FAIL"
    assert "fail" in feedback.lower()


def test_agent_node_schedules_open_coding():
    out = agent_node({"raw_text": "review", "research_question": "RQ?"})
    assert out["tool_call"]["tool"] == "open_coding"
    assert out["tool_call"]["args"]["text"] == "review"


def test_agent_node_schedules_validate_after_codes():
    out = agent_node(
        {
            "raw_text": "review",
            "research_question": "RQ?",
            "open_codes": "- Code: foo",
        }
    )
    assert out["tool_call"]["tool"] == "validate_open_codes"


def test_agent_node_retries_open_coding_on_fail():
    out = agent_node(
        {
            "raw_text": "review",
            "research_question": "RQ?",
            "open_codes_validation": "FAIL",
            "open_codes_validation_feedback": "too vague",
            "_open_coding_retries": 0,
        }
    )
    assert out["tool_call"]["tool"] == "open_coding"
    assert out["tool_call"]["args"]["validator_feedback"] == "too vague"
    assert out["_open_coding_retries"] == 1


def test_agent_node_schedules_high_level_on_refine():
    out = agent_node(
        {
            "research_question": "RQ?",
            "axial_mapping": "refine",
        }
    )
    assert out["tool_call"]["tool"] == "high_level_code_generation"


def test_router_returns_tool_when_tool_call_set():
    assert router({"tool_call": {"tool": "open_coding", "args": {}}}) == "tool"


def test_router_returns_end_on_validation_pass():
    assert (
        router(
            {
                "open_codes": "- Code: x",
                "open_codes_validation": "PASS",
            }
        )
        == END
    )


def test_router_returns_agent_on_fail_with_retries_left():
    assert (
        router(
            {
                "open_codes": "- Code: x",
                "open_codes_validation": "FAIL",
                "_open_coding_retries": 0,
            }
        )
        == "agent"
    )


def test_router_returns_end_when_retries_exhausted():
    assert (
        router(
            {
                "open_codes": "- Code: x",
                "open_codes_validation": "FAIL",
                "_open_coding_retries": OPEN_CODING_MAX_RETRIES,
            }
        )
        == END
    )


@patch("agents.core.state.TOOLS")
def test_tool_node_open_coding_sets_codes(mock_tools):
    tool = MagicMock()
    tool.invoke.return_value = "- Code: alpha"
    mock_tools.__getitem__.return_value = tool

    updates = tool_node(
        {
            "tool_call": {
                "tool": "open_coding",
                "args": {"text": "t", "research_question": "rq"},
            },
            "step": 1,
        }
    )
    assert updates["open_codes"] == "- Code: alpha"
    assert updates["tool_call"] is None


@patch("agents.core.state.TOOLS")
def test_tool_node_open_coding_empty_fallback(mock_tools):
    tool = MagicMock()
    tool.invoke.return_value = "   "
    mock_tools.__getitem__.return_value = tool

    updates = tool_node(
        {
            "tool_call": {
                "tool": "open_coding",
                "args": {"text": "t", "research_question": "rq"},
            },
        }
    )
    assert "Applicability: NONE" in updates["open_codes"]


@patch("agents.core.state.TOOLS")
def test_tool_node_validate_sets_verdict(mock_tools):
    tool = MagicMock()
    tool.invoke.return_value = "PASS"
    mock_tools.__getitem__.return_value = tool

    updates = tool_node(
        {
            "tool_call": {
                "tool": "validate_open_codes",
                "args": {"text": "t", "generated_codes": "c", "research_question": "rq"},
            },
        }
    )
    assert updates["open_codes_validation"] == "PASS"
