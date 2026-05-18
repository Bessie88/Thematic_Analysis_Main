"""
Post-pipeline co-occurrence: theme / meta-theme pairs within the same review.

Reads gt_clustered_codes.json (codes_per_review) and gt_global_graph.json (tree).
No LLM. See docs/FEATURES_PLAN.md Feature 2.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Set, Tuple


def _visit_tree(
    node: Mapping[str, Any],
    meta_theme: Optional[str],
    theme: Optional[str],
    code_to_theme: Dict[str, str],
    code_to_meta: Dict[str, str],
) -> None:
    typ = node.get("type")
    name = node.get("name", "")
    children = node.get("children") or []

    if typ == "root":
        for c in children:
            if isinstance(c, dict):
                _visit_tree(c, None, None, code_to_theme, code_to_meta)
    elif typ == "meta_theme":
        for c in children:
            if isinstance(c, dict):
                _visit_tree(c, str(name), None, code_to_theme, code_to_meta)
    elif typ == "theme":
        for c in children:
            if isinstance(c, dict):
                _visit_tree(c, meta_theme, str(name), code_to_theme, code_to_meta)
    elif typ == "sub_theme":
        for c in children:
            if isinstance(c, dict):
                _visit_tree(c, meta_theme, theme, code_to_theme, code_to_meta)
    elif typ == "code":
        if meta_theme is not None and theme is not None and name:
            code_to_theme[str(name)] = theme
            code_to_meta[str(name)] = meta_theme
    else:
        for c in children:
            if isinstance(c, dict):
                _visit_tree(c, meta_theme, theme, code_to_theme, code_to_meta)


def build_code_maps_from_graph(graph: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    tree = graph.get("tree")
    if not isinstance(tree, dict):
        raise ValueError(
            "gt_global_graph.json must contain a 'tree' object (run tree assembly / --global-graph-only)."
        )
    code_to_theme: Dict[str, str] = {}
    code_to_meta: Dict[str, str] = {}
    _visit_tree(tree, None, None, code_to_theme, code_to_meta)
    return code_to_theme, code_to_meta


def _symmetric_matrix_from_pair_counts(
    pair_counts: Mapping[Tuple[str, str], int],
) -> Dict[str, Dict[str, int]]:
    """pair_counts keys are (min(a,b), max(a,b)) with a != b."""
    outer: Dict[str, MutableMapping[str, int]] = defaultdict(lambda: defaultdict(int))
    for (a, b), cnt in pair_counts.items():
        if a == b or cnt <= 0:
            continue
        outer[a][b] += cnt
        outer[b][a] += cnt
    return {k: dict(v) for k, v in outer.items()}


def _top_pairs(
    pair_counts: Counter,
    total_reviews: int,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    items = sorted(
        pair_counts.items(),
        key=lambda x: (-x[1], x[0][0], x[0][1]),
    )
    out: List[Dict[str, Any]] = []
    denom = float(total_reviews) if total_reviews > 0 else 1.0
    for (a, b), cnt in items[:limit]:
        out.append(
            {
                "pair": [a, b],
                "count": cnt,
                "pct_reviews": round(cnt / denom, 6),
            }
        )
    return out


def _compute_cooccurrence(
    clustered_path: Path,
    graph_path: Path,
) -> Tuple[Dict[str, Any], int]:
    with open(graph_path, encoding="utf-8") as f:
        graph = json.load(f)
    code_to_theme, code_to_meta = build_code_maps_from_graph(graph)

    with open(clustered_path, encoding="utf-8") as f:
        clustered = json.load(f)
    raw_cpr = clustered.get("codes_per_review", [])
    if not isinstance(raw_cpr, list):
        raw_cpr = []

    meta_pair_counts: Counter = Counter()
    theme_pair_counts: Counter = Counter()
    meta_review_hits: Counter = Counter()
    theme_review_hits: Counter = Counter()
    skipped_codes = 0

    for item in raw_cpr:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        codes = item[1]
        if not isinstance(codes, list):
            continue
        meta_set: Set[str] = set()
        theme_set: Set[str] = set()
        for code in codes:
            if not isinstance(code, str):
                continue
            mt = code_to_meta.get(code)
            th = code_to_theme.get(code)
            if mt is None or th is None:
                skipped_codes += 1
                continue
            meta_set.add(mt)
            theme_set.add(th)
        for m in meta_set:
            meta_review_hits[m] += 1
        for t in theme_set:
            theme_review_hits[t] += 1
        for a, b in combinations(sorted(meta_set), 2):
            meta_pair_counts[(a, b)] += 1
        for a, b in combinations(sorted(theme_set), 2):
            theme_pair_counts[(a, b)] += 1

    total_reviews = len(raw_cpr)
    denom = float(total_reviews) if total_reviews > 0 else 1.0

    review_coverage = {
        "meta_themes": {
            name: {"count": c, "pct": round(c / denom, 6)}
            for name, c in sorted(meta_review_hits.items(), key=lambda x: (-x[1], x[0]))
        },
        "themes": {
            name: {"count": c, "pct": round(c / denom, 6)}
            for name, c in sorted(theme_review_hits.items(), key=lambda x: (-x[1], x[0]))
        },
    }

    payload: Dict[str, Any] = {
        "meta_theme_matrix": _symmetric_matrix_from_pair_counts(meta_pair_counts),
        "theme_matrix": _symmetric_matrix_from_pair_counts(theme_pair_counts),
        "top_meta_theme_pairs": _top_pairs(meta_pair_counts, total_reviews),
        "top_theme_pairs": _top_pairs(theme_pair_counts, total_reviews),
        "review_coverage": review_coverage,
        "total_reviews": total_reviews,
    }
    return payload, skipped_codes


def build_cooccurrence(
    clustered_path: Path,
    graph_path: Path,
) -> Dict[str, Any]:
    """Build co-occurrence payload (JSON-serializable dict)."""
    payload, _skipped = _compute_cooccurrence(clustered_path, graph_path)
    return payload


def write_cooccurrence(
    clustered_path: Path,
    graph_path: Path,
    out_path: Path,
) -> Dict[str, Any]:
    data, skipped_codes = _compute_cooccurrence(clustered_path, graph_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"skipped_unmapped_codes": skipped_codes, "written": str(out_path)}
