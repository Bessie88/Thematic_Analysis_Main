"""Tests for source memory index and example grounding."""

from pathlib import Path

from agents.core.source_memory import (
    SourceMemory,
    ground_criterion_examples,
    normalize_quote,
    parse_open_coding_snippets,
    to_grounded_example,
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

## Review 3

Applicability: NONE
Reason: not relevant
Evidence: "ignored quote"
"""


def test_parse_open_coding_snippets_review_ids(tmp_path: Path):
    md = tmp_path / "open_codes.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    snippets = parse_open_coding_snippets(md)

    assert len(snippets) == 3
    assert snippets[0].review_id == 1
    assert snippets[0].open_code == "chronic academic anxiety"
    assert snippets[0].snippet_id == "SNIP-0001"
    assert snippets[2].review_id == 2
    assert all(s.quote != "ignored quote" for s in snippets)


def test_pick_snippet_for_code_lowest_review_first(tmp_path: Path):
    md = tmp_path / "open_codes.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    mem = SourceMemory.build(md, None)

    s1 = mem.pick_snippet_for_code("chronic academic anxiety")
    assert s1 is not None
    assert s1.review_id == 1

    s2 = mem.pick_snippet_for_code("chronic academic anxiety", used={s1.snippet_id})
    assert s2 is not None
    assert s2.review_id == 2


def test_ground_quote_disambiguates_by_open_code(tmp_path: Path):
    md = tmp_path / "open_codes.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    mem = SourceMemory.build(md, None)

    g = mem.ground_quote(
        "I can't even focus on homework anymore.",
        open_code="chronic academic anxiety",
    )
    assert g is not None
    assert g["review_id"] == 1
    assert g["open_code"] == "chronic academic anxiety"


def test_save_load_roundtrip(tmp_path: Path):
    md = tmp_path / "open_codes.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    out = tmp_path / "memory.json"

    mem = SourceMemory.build(md, None)
    mem.save(out)
    loaded = SourceMemory.load(out)

    assert loaded.version == mem.version
    assert len(loaded.snippets) == len(mem.snippets)
    assert loaded.by_snippet_id["SNIP-0001"] == 0
    assert "SNIP-0001" in loaded.by_quote[normalize_quote("I can't even focus on homework anymore.")]


def test_ground_criterion_examples_converts_strings(tmp_path: Path):
    md = tmp_path / "open_codes.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    mem = SourceMemory.build(md, None)

    item = {
        "criterion": "test",
        "code_ids": ["LC001"],
        "examples": ["I can't even focus on homework anymore."],
    }
    ground_criterion_examples(
        item,
        mem,
        open_code_resolver=lambda _cid: "chronic academic anxiety",
    )
    assert len(item["examples"]) == 1
    ex = item["examples"][0]
    assert ex["snippet_id"].startswith("SNIP-")
    assert ex["review_id"] == 1
    assert ex["quote"] == "I can't even focus on homework anymore."


def test_to_grounded_example_shape():
    from agents.core.source_memory import Snippet

    s = Snippet(
        snippet_id="SNIP-0001",
        review_id=5,
        source_id="row_5",
        open_code="test code",
        open_code_id="OC001",
        quote="hello",
    )
    g = to_grounded_example(s)
    assert g["snippet_id"] == "SNIP-0001"
    assert g["source_id"] == "row_5"
    assert g["open_code_id"] == "OC001"
