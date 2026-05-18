"""Co-occurrence helpers (graph traversal + JSON I/O); no Supabase."""

import json
from pathlib import Path

import pytest
from agents.core.cooccurrence import (
    build_code_maps_from_graph,
    build_cooccurrence,
    write_cooccurrence,
)


def test_build_code_maps_from_graph_maps_codes(load_fixture):
    graph = load_fixture("minimal_global_graph.json")
    code_to_theme, code_to_meta = build_code_maps_from_graph(graph)
    assert code_to_theme["code_a"] == "ThemeA"
    assert code_to_theme["code_b"] == "ThemeB"
    assert code_to_meta["code_a"] == "MetaOne"
    assert code_to_meta["code_b"] == "MetaOne"


def test_build_code_maps_from_graph_requires_tree():
    with pytest.raises(ValueError, match="tree"):
        build_code_maps_from_graph({})


def test_build_cooccurrence_theme_pairs(tmp_path: Path, load_fixture):
    gpath = tmp_path / "gt_global_graph.json"
    cpath = tmp_path / "gt_clustered_codes.json"
    gpath.write_text(json.dumps(load_fixture("minimal_global_graph.json")), encoding="utf-8")
    cpath.write_text(json.dumps(load_fixture("minimal_clustered_codes.json")), encoding="utf-8")

    payload = build_cooccurrence(cpath, gpath)
    tm = payload["theme_matrix"]
    assert tm["ThemeA"]["ThemeB"] == 1
    assert tm["ThemeB"]["ThemeA"] == 1
    assert payload["total_reviews"] == 1
    pairs = payload["top_theme_pairs"]
    assert pairs and pairs[0]["pair"] == ["ThemeA", "ThemeB"]
    assert pairs[0]["count"] == 1


def test_build_cooccurrence_skips_unmapped_codes(tmp_path: Path, load_fixture):
    gpath = tmp_path / "gt_global_graph.json"
    cpath = tmp_path / "gt_clustered_codes.json"
    gpath.write_text(json.dumps(load_fixture("minimal_global_graph.json")), encoding="utf-8")
    cpath.write_text(
        json.dumps({"codes_per_review": [["review-1", ["code_a", "unknown"]]]}),
        encoding="utf-8",
    )
    payload = build_cooccurrence(cpath, gpath)
    assert payload["theme_matrix"] == {}


def test_write_cooccurrence_writes_file(tmp_path: Path, load_fixture):
    gpath = tmp_path / "gt_global_graph.json"
    cpath = tmp_path / "gt_clustered_codes.json"
    out_path = tmp_path / "gt_cooccurrence.json"
    gpath.write_text(json.dumps(load_fixture("minimal_global_graph.json")), encoding="utf-8")
    cpath.write_text(json.dumps(load_fixture("minimal_clustered_codes.json")), encoding="utf-8")

    meta = write_cooccurrence(cpath, gpath, out_path)
    assert out_path.is_file()
    assert meta["written"] == str(out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["total_reviews"] == 1
