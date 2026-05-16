"""Co-occurrence helpers (graph traversal + JSON I/O); no Supabase."""

import json
from pathlib import Path

import pytest

from agents.core.cooccurrence import build_cooccurrence, build_code_maps_from_graph


def _two_theme_graph() -> dict:
    """One meta-theme, two themes, two codes (for cross-theme pairs within a review)."""
    return {
        "tree": {
            "type": "root",
            "name": "Root",
            "children": [
                {
                    "type": "meta_theme",
                    "name": "MetaOne",
                    "children": [
                        {
                            "type": "theme",
                            "name": "ThemeA",
                            "children": [
                                {
                                    "type": "sub_theme",
                                    "name": "SA",
                                    "children": [{"type": "code", "name": "code_a"}],
                                }
                            ],
                        },
                        {
                            "type": "theme",
                            "name": "ThemeB",
                            "children": [
                                {
                                    "type": "sub_theme",
                                    "name": "SB",
                                    "children": [{"type": "code", "name": "code_b"}],
                                }
                            ],
                        },
                    ],
                }
            ],
        }
    }


def test_build_code_maps_from_graph_maps_codes():
    graph = _two_theme_graph()
    code_to_theme, code_to_meta = build_code_maps_from_graph(graph)
    assert code_to_theme["code_a"] == "ThemeA"
    assert code_to_theme["code_b"] == "ThemeB"
    assert code_to_meta["code_a"] == "MetaOne"
    assert code_to_meta["code_b"] == "MetaOne"


def test_build_code_maps_from_graph_requires_tree():
    with pytest.raises(ValueError, match="tree"):
        build_code_maps_from_graph({})


def test_build_cooccurrence_theme_pairs(tmp_path: Path):
    gpath = tmp_path / "gt_global_graph.json"
    cpath = tmp_path / "gt_clustered_codes.json"
    gpath.write_text(json.dumps(_two_theme_graph()), encoding="utf-8")
    cpath.write_text(
        json.dumps({"codes_per_review": [["review-1", ["code_a", "code_b"]]]}),
        encoding="utf-8",
    )

    payload = build_cooccurrence(cpath, gpath)
    tm = payload["theme_matrix"]
    assert tm["ThemeA"]["ThemeB"] == 1
    assert tm["ThemeB"]["ThemeA"] == 1
    assert payload["total_reviews"] == 1
    pairs = payload["top_theme_pairs"]
    assert pairs and pairs[0]["pair"] == ["ThemeA", "ThemeB"]
    assert pairs[0]["count"] == 1


def test_build_cooccurrence_skips_unmapped_codes(tmp_path: Path):
    gpath = tmp_path / "gt_global_graph.json"
    cpath = tmp_path / "gt_clustered_codes.json"
    gpath.write_text(json.dumps(_two_theme_graph()), encoding="utf-8")
    cpath.write_text(
        json.dumps({"codes_per_review": [["review-1", ["code_a", "unknown"]]]}),
        encoding="utf-8",
    )
    payload = build_cooccurrence(cpath, gpath)
    assert payload["theme_matrix"] == {}
