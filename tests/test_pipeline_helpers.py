"""Deterministic helpers in agents.core.pipeline_helpers (no embedding models).

Heavy deps are imported only inside embedding/cluster helpers, not at module load,
so local runs can use ``pip install -r tests/requirements.txt`` when the root
dev requirements fail on a restricted pip index (e.g. some HPC login nodes).
"""

import pytest
from agents.core.pipeline_helpers import (
    build_sub_theme_node,
    hierarchy_assign_batch,
    hierarchy_embed_drain_enabled,
    merge_two_smallest_meta,
    meta_theme_bounds,
    normalize_meta_theme_count,
    prune_hierarchy_to_valid_clusters,
    refine_llm_max_codes,
    split_largest_meta_theme,
)


def test_refine_llm_max_codes_defaults_and_clamp(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_REFINE_LLM_MAX_CODES", raising=False)
    assert refine_llm_max_codes() == 44
    monkeypatch.setenv("GT_REFINE_LLM_MAX_CODES", "200")
    assert refine_llm_max_codes() == 80
    monkeypatch.setenv("GT_REFINE_LLM_MAX_CODES", "not-int")
    assert refine_llm_max_codes() == 44


def test_hierarchy_assign_batch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_HIERARCHY_ASSIGN_BATCH", raising=False)
    assert hierarchy_assign_batch() == 40
    monkeypatch.setenv("GT_HIERARCHY_ASSIGN_BATCH", "100")
    assert hierarchy_assign_batch() == 60
    monkeypatch.setenv("GT_HIERARCHY_ASSIGN_BATCH", "x")
    assert hierarchy_assign_batch() == 40


def test_hierarchy_embed_drain_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_HIERARCHY_EMBED_DRAIN_UNGROUPED", raising=False)
    assert hierarchy_embed_drain_enabled() is False
    for v in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("GT_HIERARCHY_EMBED_DRAIN_UNGROUPED", v)
        assert hierarchy_embed_drain_enabled() is True


def test_prune_hierarchy_to_valid_clusters():
    h = {"a": {"x": 1}, "b": {"y": 2}, "orphan": {}}
    pruned, removed = prune_hierarchy_to_valid_clusters(h, {"a", "b"})
    assert pruned == {"a": {"x": 1}, "b": {"y": 2}}
    assert set(removed) == {"orphan"}


@pytest.mark.parametrize(
    "n_cids,expected",
    [
        (0, (1, 1)),
        (1, (1, 1)),
        (3, (2, 3)),
        (5, (2, 5)),
        (10, (3, 7)),
    ],
)
def test_meta_theme_bounds(n_cids: int, expected: tuple[int, int]):
    assert meta_theme_bounds(n_cids) == expected


def test_split_largest_meta_theme_noop_when_single_cluster():
    mt = [{"name": "A", "cluster_ids": ["1"]}]
    assert split_largest_meta_theme(list(mt)) == mt


def test_split_largest_meta_theme_splits():
    mt = [{"name": "Big", "cluster_ids": ["a", "b", "c", "d"]}]
    out = split_largest_meta_theme(mt)
    assert len(out) == 2
    names = {m["name"] for m in out}
    assert "Big" in names
    assert "Big (continued)" in names
    ids_flat = []
    for m in out:
        ids_flat.extend(m["cluster_ids"])
    assert sorted(ids_flat) == ["a", "b", "c", "d"]


def test_merge_two_smallest_meta():
    mt = [
        {"name": "Tiny", "cluster_ids": ["1"]},
        {"name": "Small", "cluster_ids": ["2", "3"]},
        {"name": "Med", "cluster_ids": ["4", "5", "6"]},
    ]
    out = merge_two_smallest_meta(mt)
    assert len(out) == 2
    merged_names = {m["name"] for m in out}
    assert any("Tiny" in n and "Small" in n for n in merged_names)


def test_normalize_meta_theme_count_already_in_range():
    mt = [
        {"name": "M1", "cluster_ids": ["1"]},
        {"name": "M2", "cluster_ids": ["2"]},
        {"name": "M3", "cluster_ids": ["3"]},
        {"name": "M4", "cluster_ids": ["4"]},
    ]
    assert normalize_meta_theme_count(mt, n_cids=50) == mt


def test_normalize_meta_theme_count_splits_when_below_low_bound():
    mt = [{"name": "Only", "cluster_ids": [str(i) for i in range(8)]}]
    out = normalize_meta_theme_count(mt, n_cids=50)
    assert 3 <= len(out) <= 7


def test_build_sub_theme_node_flat_codes():
    edges: list[dict[str, str]] = []
    nodes: list[str] = []
    st = {"name": "Parent Theme", "codes": ["c1", "c2"]}
    root = build_sub_theme_node(st, "cluster_x", edges, nodes)
    assert root["type"] == "sub_theme"
    assert root["name"] == "Parent Theme"
    assert len(root["children"]) == 2
    assert all(ch["type"] == "code" for ch in root["children"])
    assert {"parent": "cluster_x", "child": "Parent Theme"} in edges


def test_build_sub_theme_node_nested():
    edges: list[dict[str, str]] = []
    nodes: list[str] = []
    st = {
        "name": "Outer",
        "sub_themes": [{"name": "Inner", "codes": ["x"]}],
        "codes": [],
    }
    root = build_sub_theme_node(st, "cl", edges, nodes)
    assert root["name"] == "Outer"
    assert len(root["children"]) == 1
    inner = root["children"][0]
    assert inner["name"] == "Inner"
    assert inner["children"][0] == {"name": "x", "type": "code"}
