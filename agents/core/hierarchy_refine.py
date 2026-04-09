"""
Recursively cap code fan-out under each sub-theme and cluster ungrouped_codes.

Reads/writes the same gt_hierarchy.json schema with optional nested sub_themes:
- Leaf: {"name": str, "codes": [str, ...]}
- Internal: {"name": str, "sub_themes": [ ... ]}
"""
from __future__ import annotations

import os
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, List, Optional

from .prompts import hierarchy_refine_bucket_prompt
from .utils import clean_and_parse_json, log_step


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _max_bucket() -> int:
    raw = os.environ.get("GT_MAX_CODES_PER_BUCKET", "48").strip()
    try:
        n = int(raw)
        return max(4, min(80, n))
    except ValueError:
        return 48


def _max_depth() -> int:
    raw = os.environ.get("GT_MAX_SUBTHEME_DEPTH", "8").strip()
    try:
        n = int(raw)
        return max(1, min(20, n))
    except ValueError:
        return 8


def _llm_max_codes_per_call() -> int:
    """
    Max code lines per hierarchy_refine LLM request. Larger buckets are split into
    consecutive batches; each batch is LLM-split separately, then sub-themes are
    merged by normalized name (case-insensitive).
    """
    raw = os.environ.get("GT_HIERARCHY_REFINE_LLM_MAX_CODES", "48").strip()
    try:
        n = int(raw)
        return max(8, min(200, n))
    except ValueError:
        return 48


def _num_groups_for_split(n: int, cap: int) -> int:
    """LLM groups per call: enough to approach cap-sized leaves, bounded."""
    need = (n + cap - 1) // cap
    return min(12, max(2, need))


def _deterministic_leaves(
    codes: List[str], cap: int, name_prefix: str
) -> List[Dict[str, Any]]:
    """Split codes into fixed-size chunks (neutral names)."""
    out: List[Dict[str, Any]] = []
    for i in range(0, len(codes), cap):
        chunk = codes[i : i + cap]
        part = i // cap + 1
        out.append({"name": f"{name_prefix} (part {part})", "codes": chunk})
    return out


def _validate_split(
    original: List[str], groups: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    """Return normalized list of {name, codes} if every code appears exactly once; else None."""
    if not groups:
        return None
    orig_set = Counter(original)
    seen: Counter = Counter()
    normalized: List[Dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, dict):
            return None
        name = (g.get("name") or "Unnamed").strip() or "Unnamed"
        codes = g.get("codes")
        if not isinstance(codes, list):
            return None
        clean_codes = [c for c in codes if isinstance(c, str)]
        if not clean_codes:
            continue
        normalized.append({"name": name, "codes": clean_codes})
        seen.update(clean_codes)
    if seen != orig_set:
        return None
    return normalized if normalized else None


def _llm_split_bucket_once(
    cluster_label: str,
    bucket_label: str,
    codes: List[str],
    research_question: str,
    invoke: Callable[[str, str], str],
) -> Optional[List[Dict[str, Any]]]:
    """One LLM call; caller must keep len(codes) <= _llm_max_codes_per_call()."""
    if not codes:
        return None
    cap = _max_bucket()
    num_groups = _num_groups_for_split(len(codes), cap)
    bulleted = "\n".join(f"- {c}" for c in codes)
    prompt = hierarchy_refine_bucket_prompt(
        cluster_label, bucket_label, bulleted, research_question, num_groups
    )
    try:
        raw = invoke("hierarchy_refine", prompt)
        parsed = clean_and_parse_json(raw)
    except Exception as e:
        log_step("HIERARCHY_REFINE_LLM", f"parse/call error for {bucket_label!r}: {e}")
        return None
    subs = parsed.get("sub_themes")
    if not isinstance(subs, list):
        return None
    validated = _validate_split(codes, subs)
    if not validated:
        log_step("HIERARCHY_REFINE_VALIDATION", f"LLM split failed validation for {bucket_label!r}")
        return None
    if len(validated) == 1 and len(validated[0].get("codes", [])) == len(codes):
        return None
    return validated


def _merge_group_lists(
    ordered: "OrderedDict[str, Dict[str, Any]]", groups: List[Dict[str, Any]]
) -> None:
    """Merge {name, codes} into ordered (key = lower(name)), extending codes."""
    for g in groups:
        if not isinstance(g, dict):
            continue
        name = (g.get("name") or "Unnamed").strip() or "Unnamed"
        key = name.lower()
        raw_codes = g.get("codes")
        if not isinstance(raw_codes, list):
            continue
        clean = [c for c in raw_codes if isinstance(c, str)]
        if not clean:
            continue
        if key not in ordered:
            ordered[key] = {"name": name, "codes": []}
        ordered[key]["codes"].extend(clean)


def _llm_split_bucket(
    cluster_label: str,
    bucket_label: str,
    codes: List[str],
    research_question: str,
    invoke: Callable[[str, str], str],
) -> Optional[List[Dict[str, Any]]]:
    cap = _max_bucket()
    llm_cap = _llm_max_codes_per_call()
    n = len(codes)
    if n <= llm_cap:
        return _llm_split_bucket_once(
            cluster_label, bucket_label, codes, research_question, invoke
        )

    n_batch = (n + llm_cap - 1) // llm_cap
    log_step(
        "HIERARCHY_REFINE_LLM_BATCH",
        f"{bucket_label!r}: {n} codes in {n_batch} LLM batches (max {llm_cap} codes/call).",
    )
    merged: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for bi in range(n_batch):
        start = bi * llm_cap
        batch = codes[start : start + llm_cap]
        batch_label = f"{bucket_label} (batch {bi + 1}/{n_batch})"
        got = _llm_split_bucket_once(
            cluster_label, batch_label, batch, research_question, invoke
        )
        if not got:
            fallback = _deterministic_leaves(batch, cap, f"{bucket_label} (batch {bi + 1})")
            _merge_group_lists(merged, fallback)
        else:
            _merge_group_lists(merged, got)

    combined: List[Dict[str, Any]] = [
        {"name": v["name"], "codes": v["codes"]} for v in merged.values()
    ]
    if not combined:
        return None
    orig = Counter(codes)
    seen = Counter(c for g in combined for c in g["codes"])
    if seen != orig:
        log_step(
            "HIERARCHY_REFINE_VALIDATION",
            f"Batched merge for {bucket_label!r}: code multiset mismatch (dropped/duplicated); "
            f"expected {len(orig)} unique instances, got {len(seen)}.",
        )
        return None
    if len(combined) == 1 and len(combined[0]["codes"]) == n:
        return None
    return combined


def refine_leaf_bucket(
    cluster_label: str,
    bucket_label: str,
    codes: List[str],
    research_question: str,
    depth: int,
    invoke: Callable[[str, str], str],
) -> Dict[str, Any]:
    """
    Return a sub_theme dict (leaf with codes, or internal with nested sub_themes).
    bucket_label is the display name for this node when it stays a single leaf under cap.
    """
    cap = _max_bucket()
    max_d = _max_depth()
    codes = list(dict.fromkeys(codes))  # stable dedupe

    if len(codes) <= cap or depth >= max_d:
        if len(codes) > cap:
            leaves = _deterministic_leaves(codes, cap, bucket_label)
            if len(leaves) == 1:
                return leaves[0]
            return {"name": bucket_label, "sub_themes": leaves}
        return {"name": bucket_label, "codes": codes}

    groups = _llm_split_bucket(
        cluster_label, bucket_label, codes, research_question, invoke
    )
    if not groups:
        leaves = _deterministic_leaves(codes, cap, bucket_label)
        if len(leaves) == 1:
            return leaves[0]
        return {"name": bucket_label, "sub_themes": leaves}

    children: List[Dict[str, Any]] = []
    for g in groups:
        gname = g["name"]
        gcodes = g["codes"]
        children.append(
            refine_leaf_bucket(cluster_label, gname, gcodes, research_question, depth + 1, invoke)
        )

    if len(children) == 1:
        return children[0]
    return {"name": bucket_label, "sub_themes": children}


def refine_sub_theme_node(
    st: Dict[str, Any],
    cluster_label: str,
    research_question: str,
    depth: int,
    invoke: Callable[[str, str], str],
) -> Dict[str, Any]:
    """Normalize one sub_theme entry; support legacy flat {name, codes} and nested sub_themes."""
    name = (st.get("name") or "Unnamed").strip() or "Unnamed"
    nested_in = st.get("sub_themes")
    codes_in = st.get("codes") if isinstance(st.get("codes"), list) else []

    if isinstance(nested_in, list) and nested_in:
        refined_children = [
            refine_sub_theme_node(
                child, cluster_label, research_question, depth + 1, invoke
            )
            for child in nested_in
            if isinstance(child, dict)
        ]
        refined_children = [c for c in refined_children if c]
        if codes_in:
            extra = refine_leaf_bucket(
                cluster_label, name, [str(c) for c in codes_in], research_question, depth, invoke
            )
            if refined_children:
                return {"name": name, "sub_themes": refined_children + [extra]}
            return extra
        if len(refined_children) == 1:
            return refined_children[0]
        return {"name": name, "sub_themes": refined_children}

    codes = [str(c) for c in codes_in if isinstance(c, str)]
    return refine_leaf_bucket(
        cluster_label, name, codes, research_question, depth, invoke
    )


def refine_cluster_entry(
    entry: Dict[str, Any],
    cluster_label: str,
    research_question: str,
    invoke: Callable[[str, str], str],
) -> Dict[str, Any]:
    label = entry.get("label", cluster_label)
    sub_themes_in = entry.get("sub_themes")
    if not isinstance(sub_themes_in, list):
        sub_themes_in = []
    ungrouped = entry.get("ungrouped_codes")
    if not isinstance(ungrouped, list):
        ungrouped = []
    ungrouped_str = [str(c) for c in ungrouped if isinstance(c, str)]

    new_subs: List[Dict[str, Any]] = []
    for st in sub_themes_in:
        if isinstance(st, dict):
            new_subs.append(
                refine_sub_theme_node(st, label, research_question, 0, invoke)
            )

    cap = _max_bucket()
    overflow_bucket = f"Further themes ({label})"
    if len(ungrouped_str) > cap:
        extra = refine_leaf_bucket(
            label,
            overflow_bucket,
            ungrouped_str,
            research_question,
            0,
            invoke,
        )
        if extra.get("sub_themes"):
            new_subs.extend(extra["sub_themes"])
        elif extra.get("codes"):
            new_subs.append(extra)
    elif ungrouped_str:
        new_subs.append({"name": "Ungrouped codes", "codes": ungrouped_str})

    return {"label": label, "sub_themes": new_subs, "ungrouped_codes": []}


def refine_hierarchy_json(
    hierarchy: Dict[str, Any],
    research_question: str,
    invoke: Callable[[str, str], str],
) -> Dict[str, Any]:
    """Return a new hierarchy dict with fan-out capped (ungrouped merged into sub_themes)."""
    out: Dict[str, Any] = {}
    for cid in sorted(hierarchy.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        entry = hierarchy[cid]
        if not isinstance(entry, dict):
            out[cid] = entry
            continue
        cluster_label = entry.get("label", f"Cluster {cid}")
        out[cid] = refine_cluster_entry(entry, cluster_label, research_question, invoke)
    return out


def maybe_refine_hierarchy(
    hierarchy: Dict[str, Any],
    research_question: str,
    invoke: Callable[[str, str], str],
) -> Dict[str, Any]:
    if not _truthy(os.environ.get("GT_HIERARCHY_REFINE", "1")):
        log_step("HIERARCHY_REFINE", "skipped (GT_HIERARCHY_REFINE not truthy)")
        return hierarchy
    log_step(
        "HIERARCHY_REFINE",
        f"cap={_max_bucket()} max_depth={_max_depth()} clusters={len(hierarchy)}",
    )
    return refine_hierarchy_json(hierarchy, research_question, invoke)
