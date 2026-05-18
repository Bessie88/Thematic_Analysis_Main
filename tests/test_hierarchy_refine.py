"""Tests for agents.core.hierarchy_refine (deterministic refine logic; no real LLM)."""

import json

import pytest
from agents.core.hierarchy_refine import (
    _deterministic_leaves,
    _llm_max_codes_per_call,
    _max_bucket,
    _max_depth,
    _num_groups_for_split,
    _validate_split,
    maybe_refine_hierarchy,
    refine_hierarchy_json,
    refine_leaf_bucket,
)


def test_max_bucket_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_MAX_CODES_PER_BUCKET", raising=False)
    assert _max_bucket() == 48
    monkeypatch.setenv("GT_MAX_CODES_PER_BUCKET", "10")
    assert _max_bucket() == 10
    monkeypatch.setenv("GT_MAX_CODES_PER_BUCKET", "3")
    assert _max_bucket() == 4
    monkeypatch.setenv("GT_MAX_CODES_PER_BUCKET", "x")
    assert _max_bucket() == 48


def test_max_depth_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_MAX_SUBTHEME_DEPTH", raising=False)
    assert _max_depth() == 8
    monkeypatch.setenv("GT_MAX_SUBTHEME_DEPTH", "3")
    assert _max_depth() == 3
    monkeypatch.setenv("GT_MAX_SUBTHEME_DEPTH", "99")
    assert _max_depth() == 20


def test_llm_max_codes_per_call_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GT_HIERARCHY_REFINE_LLM_MAX_CODES", raising=False)
    assert _llm_max_codes_per_call() == 48
    monkeypatch.setenv("GT_HIERARCHY_REFINE_LLM_MAX_CODES", "100")
    assert _llm_max_codes_per_call() == 100
    monkeypatch.setenv("GT_HIERARCHY_REFINE_LLM_MAX_CODES", "bad")
    assert _llm_max_codes_per_call() == 48


@pytest.mark.parametrize(
    "n,cap,expected",
    [
        (10, 48, 2),
        (100, 48, 3),
        (500, 48, 11),
        (1, 48, 2),
    ],
)
def test_num_groups_for_split(n: int, cap: int, expected: int):
    assert _num_groups_for_split(n, cap) == expected


def test_validate_split_accepts_valid_groups():
    codes = ["a", "b", "c"]
    groups = [
        {"name": "G1", "codes": ["a", "b"]},
        {"name": "G2", "codes": ["c"]},
    ]
    out = _validate_split(codes, groups)
    assert out is not None
    assert sum(len(g["codes"]) for g in out) == 3


def test_validate_split_rejects_duplicate_code():
    codes = ["a", "b"]
    groups = [{"name": "G1", "codes": ["a", "a"]}]
    assert _validate_split(codes, groups) is None


def test_validate_split_rejects_missing_code():
    codes = ["a", "b"]
    groups = [{"name": "G1", "codes": ["a"]}]
    assert _validate_split(codes, groups) is None


def test_validate_split_rejects_empty_groups():
    assert _validate_split(["a"], []) is None


def test_deterministic_leaves_chunks():
    codes = [f"c{i}" for i in range(10)]
    leaves = _deterministic_leaves(codes, cap=4, name_prefix="Bucket")
    assert len(leaves) == 3
    assert leaves[0]["name"] == "Bucket (part 1)"
    assert len(leaves[0]["codes"]) == 4
    flat = [c for leaf in leaves for c in leaf["codes"]]
    assert flat == codes


def test_refine_leaf_bucket_small_list():
    def _noop_invoke(*_args, **_kwargs):
        raise AssertionError("invoke should not be called")

    out = refine_leaf_bucket("C1", "Small", ["x", "y"], "RQ?", 0, _noop_invoke)
    assert out == {"name": "Small", "codes": ["x", "y"]}


def test_refine_leaf_bucket_oversized_uses_deterministic_split(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GT_MAX_CODES_PER_BUCKET", "4")
    codes = [f"code_{i}" for i in range(10)]

    def _noop_invoke(*_args, **_kwargs):
        raise AssertionError("invoke should not be called for depth-limited oversized bucket")

    out = refine_leaf_bucket("C1", "Big", codes, "", depth=99, invoke=_noop_invoke)
    assert "sub_themes" in out
    assert len(out["sub_themes"]) >= 2


def test_refine_hierarchy_json_with_fake_invoke(load_fixture):
    hierarchy = load_fixture("minimal_hierarchy.json")

    def fake_invoke(_skill, _prompt, **_labels):
        return json.dumps(
            {
                "sub_themes": [
                    {"name": "Part A", "codes": ["c1", "c2"]},
                    {"name": "Part B", "codes": ["c3"]},
                ]
            }
        )

    out = refine_hierarchy_json(hierarchy, "test question", fake_invoke)
    assert "1" in out
    entry = out["1"]
    assert entry.get("ungrouped_codes") == []


def test_maybe_refine_hierarchy_disabled(monkeypatch: pytest.MonkeyPatch, load_fixture):
    hierarchy = load_fixture("minimal_hierarchy.json")
    monkeypatch.setenv("GT_HIERARCHY_REFINE", "0")

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("invoke should not run when refine disabled")

    assert maybe_refine_hierarchy(hierarchy, "", _should_not_run) == hierarchy
