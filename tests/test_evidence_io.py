"""Tests for open-coding evidence parsing and label helpers."""

from pathlib import Path

from agents.core.evidence_io import (
    assign_open_code_ids,
    parse_code_evidence,
    short_label,
)

SAMPLE_MD = """## Review 1

- Code: chronic academic anxiety
  Evidence: "I can't even focus on homework anymore."
  Note: strong emotional tone

- Code: laggy matchmaking
  Evidence: "Servers take forever to find a game."

## Review 2

- Code: chronic academic anxiety
  Evidence: "Every test feels like a mountain."
"""


def test_short_label_strips_rich_suffix():
    raw = "emotional strain | Definition: feeling drained | Inclusion: fatigue"
    assert short_label(raw) == "emotional strain"
    assert short_label("plain label") == "plain label"


def test_parse_code_evidence(tmp_path: Path):
    md = tmp_path / "open_codes.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    evidence, notes = parse_code_evidence(md, None)

    assert "chronic academic anxiety" in evidence
    assert len(evidence["chronic academic anxiety"]) == 2
    assert "I can't even focus on homework anymore." in evidence["chronic academic anxiety"]
    assert notes["chronic academic anxiety"] == ["strong emotional tone"]
    assert "laggy matchmaking" in evidence


def test_assign_open_code_ids_stable_order():
    cluster_to_codes = {
        "1": ["b code", "a code"],
        "0": ["a code", "c code"],
    }
    mapping = assign_open_code_ids(cluster_to_codes)
    assert mapping["a code"] == "OC001"
    assert mapping["b code"] == "OC002"
    assert mapping["c code"] == "OC003"
