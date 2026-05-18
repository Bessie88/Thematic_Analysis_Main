"""Thin contract tests for tools with mocked LLM (no embedding models)."""

import json
from pathlib import Path
from unittest.mock import patch

from agents.core.tools import high_level_code_generation, open_coding, validate_open_codes


@patch("agents.core.tools.llm_invoke_with_skill")
def test_open_coding_returns_model_text(mock_invoke):
    mock_invoke.return_value = "- Code: laggy matchmaking"
    out = open_coding.invoke(
        {"text": "game is slow", "research_question": "What frustrates players?"}
    )
    assert "laggy matchmaking" in out
    mock_invoke.assert_called_once()


@patch("agents.core.tools.llm_invoke_with_skill")
def test_validate_open_codes_returns_pass(mock_invoke):
    mock_invoke.return_value = "PASS"
    out = validate_open_codes.invoke(
        {
            "text": "game is slow",
            "generated_codes": "- Code: lag",
            "research_question": "RQ",
        }
    )
    assert "PASS" in out


@patch("agents.core.tools.llm_invoke_with_skill")
def test_high_level_code_generation_writes_codebook(mock_invoke, tmp_path: Path):
    cluster_file = tmp_path / "gt_clustered_codes.json"
    cluster_file.write_text(
        json.dumps({"cluster_to_codes": {"1": ["code_a", "code_b"]}}),
        encoding="utf-8",
    )
    mock_invoke.return_value = json.dumps(
        {"label": "Usability issues", "confidence": 4, "rationale": "UI related"}
    )

    out = high_level_code_generation.invoke(
        {"cluster_file": str(cluster_file), "research_question": "What bugs matter?"}
    )
    codebook = json.loads(out)
    assert codebook["1"] == "Usability issues"

    codebook_path = tmp_path / "codebook.json"
    assert codebook_path.is_file()
    on_disk = json.loads(codebook_path.read_text(encoding="utf-8"))
    assert on_disk["codebook"]["1"] == "Usability issues"
