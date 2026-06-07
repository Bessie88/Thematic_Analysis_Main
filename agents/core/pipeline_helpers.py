"""Non-tool implementation helpers for the GT pipeline (embed, cluster, hierarchy I/O, meta-theme repair)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Set

from .llm_clustering import USE_LLM_CLUSTERING, axial_llm_cluster, use_llm_clustering
from .paths import DATA_DIR, HIERARCHY_PATH, WEIGHTS_DIR, ensure_output_dirs
from .utils import log_step

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

REFINE_TOP_K_OTHER_CLUSTERS = 5

EMBED_DRAIN_THRESHOLD = 20


__all__ = [
    "USE_LLM_CLUSTERING",
    "axial_llm_cluster",
    "use_llm_clustering",
]


def refine_llm_max_codes() -> int:
    raw = os.environ.get("GT_REFINE_LLM_MAX_CODES", "44").strip()
    try:
        n = int(raw)
        return max(16, min(80, n))
    except ValueError:
        return 44


def hierarchy_assign_batch() -> int:
    raw = os.environ.get("GT_HIERARCHY_ASSIGN_BATCH", "40").strip()
    try:
        n = int(raw)
        return max(20, min(60, n))
    except ValueError:
        return 40


def hierarchy_embed_drain_enabled() -> bool:
    return os.environ.get("GT_HIERARCHY_EMBED_DRAIN_UNGROUPED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def prune_hierarchy_to_valid_clusters(
    hierarchy: Dict[str, Any], valid_ids: Set[str]
) -> tuple[Dict[str, Any], List[str]]:
    removed = [k for k in hierarchy if k not in valid_ids]
    pruned = {k: v for k, v in hierarchy.items() if k in valid_ids}
    return pruned, removed


def embed_model_name_for_hierarchy() -> str:
    model_name = os.environ.get("GT_EMBED_MODEL") or (
        str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B")
        if os.path.isdir(str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B"))
        else "Qwen/Qwen3-Embedding-0.6B"
    )
    if os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)
    return model_name


def drain_ungrouped_to_subthemes(
    validated_sub_themes: List[Dict[str, Any]],
    validated_ungrouped: List[str],
    embed_model: SentenceTransformer,
) -> List[str]:
    """Assign each ungrouped code to the nearest sub-theme by embedding cosine similarity."""
    import numpy as np

    if not validated_sub_themes or not validated_ungrouped:
        return validated_ungrouped
    labels = [st["name"] for st in validated_sub_themes]
    lab_emb = embed_model.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    code_emb = embed_model.encode(
        validated_ungrouped, normalize_embeddings=True, show_progress_bar=False
    )
    lab_emb = np.asarray(lab_emb, dtype=np.float32)
    code_emb = np.asarray(code_emb, dtype=np.float32)
    sims = code_emb @ lab_emb.T
    for i, code in enumerate(validated_ungrouped):
        j = int(np.argmax(sims[i]))
        validated_sub_themes[j]["codes"].append(code)
    return []


def meta_theme_bounds(n_cids: int) -> tuple[int, int]:
    if n_cids <= 1:
        return 1, 1
    if n_cids < 6:
        return 2, min(7, n_cids)
    return 3, 7


def split_largest_meta_theme(meta_themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    largest = max(meta_themes, key=lambda m: len(m.get("cluster_ids", [])))
    cids = list(largest.get("cluster_ids", []))
    if len(cids) < 2:
        return meta_themes
    mid = len(cids) // 2
    base = (largest.get("name") or "Theme").strip() or "Theme"
    largest["cluster_ids"] = cids[:mid]
    meta_themes.append({"name": f"{base} (continued)", "cluster_ids": cids[mid:]})
    return meta_themes


def merge_two_smallest_meta(meta_themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(meta_themes) < 2:
        return meta_themes
    indexed = sorted(enumerate(meta_themes), key=lambda x: len(x[1].get("cluster_ids", [])))
    i_small, a = indexed[0]
    j_small, b = indexed[1]
    name_a = (a.get("name") or "A").strip() or "A"
    name_b = (b.get("name") or "B").strip() or "B"
    merged = {
        "name": f"{name_a} / {name_b}",
        "cluster_ids": list(a.get("cluster_ids", [])) + list(b.get("cluster_ids", [])),
    }
    new_list = [m for k, m in enumerate(meta_themes) if k not in (i_small, j_small)]
    new_list.append(merged)
    return new_list


def normalize_meta_theme_count(
    meta_themes: List[Dict[str, Any]], n_cids: int
) -> List[Dict[str, Any]]:
    lo, hi = meta_theme_bounds(n_cids)
    mt = list(meta_themes)
    for _ in range(32):
        k = len(mt)
        if lo <= k <= hi:
            return mt
        if k < lo:
            prev = len(mt)
            mt = split_largest_meta_theme(mt)
            if len(mt) == prev:
                break
        elif k > hi:
            prev = len(mt)
            mt = merge_two_smallest_meta(mt)
            if len(mt) >= prev:
                break
    return mt


def deduplicate_codes(
    all_codes: List[str], model_name: str, near_dup_threshold: float = 0.95
) -> tuple:
    """Exact dedup then embedding-based near-dup merge; returns (deduped list, original→canonical map)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    norm_to_canonical: Dict[str, str] = {}
    for code in all_codes:
        normed = code.strip().lower()
        if normed not in norm_to_canonical:
            norm_to_canonical[normed] = code.strip()
    unique_codes = list(norm_to_canonical.values())

    if len(unique_codes) < 2:
        orig_map = {c: c.strip() for c in all_codes}
        return unique_codes, orig_map

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        unique_codes, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    sim_matrix = embeddings @ embeddings.T

    parent: Dict[int, int] = {i: i for i in range(len(unique_codes))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(unique_codes)):
        for j in range(i + 1, len(unique_codes)):
            if sim_matrix[i][j] >= near_dup_threshold:
                union(i, j)

    root_to_canonical: Dict[int, str] = {}
    for i in range(len(unique_codes)):
        root = find(i)
        if root not in root_to_canonical:
            root_to_canonical[root] = unique_codes[root]

    deduped = sorted(set(root_to_canonical.values()))

    idx_map = {unique_codes[i]: root_to_canonical[find(i)] for i in range(len(unique_codes))}
    orig_map: Dict[str, str] = {}
    for code in all_codes:
        normed = code.strip().lower()
        canonical_from_exact = norm_to_canonical.get(normed, code.strip())
        orig_map[code] = idx_map.get(canonical_from_exact, canonical_from_exact)

    return deduped, orig_map


def axial_embed_and_cluster(
    all_codes: List[str], model_name: str, out_dir: str = str(DATA_DIR)
) -> str:
    """Embed codes, pick K via silhouette, cluster with K-means/MiniBatch; write gt_clustered_codes.json."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    MINIBATCH_SIZE = 1000
    K_MIN, K_MAX_DIVISOR = 5, 3

    if not all_codes:
        return "No codes to cluster."

    original_count = len(all_codes)
    deduped_codes, dedup_map = deduplicate_codes(all_codes, model_name)
    log_step("DEDUP", f"Reduced {original_count} codes to {len(deduped_codes)} unique codes")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        deduped_codes, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    n = len(embeddings)
    if n < K_MIN:
        cluster_to_codes_out = {"0": deduped_codes}
        out = {
            "clustering_method": "embedding",
            "all_codes": deduped_codes,
            "labels": [0] * n,
            "k": 1,
            "cluster_to_codes": cluster_to_codes_out,
            "dedup_map": dedup_map,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "gt_clustered_codes.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return f"Axial coding: only {n} unique codes, placed in 1 cluster."
    use_minibatch = n >= 1000
    k_max = min(50, max(K_MIN + 1, n // K_MAX_DIVISOR))
    best_k, best_sil = K_MIN, -1.0

    for k in range(K_MIN, k_max + 1):
        if use_minibatch:
            km = MiniBatchKMeans(n_clusters=k, batch_size=MINIBATCH_SIZE, random_state=42, n_init=3)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, sample_size=min(5000, n))
        if sil > best_sil:
            best_sil, best_k = sil, k

    if use_minibatch:
        km = MiniBatchKMeans(
            n_clusters=best_k, batch_size=MINIBATCH_SIZE, random_state=42, n_init=3
        )
    else:
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    cluster_to_codes = defaultdict(list)
    for code, label in zip(deduped_codes, labels):
        cluster_to_codes[int(label)].append(code)

    codes_per_review = []
    codes_only_path = os.path.join(out_dir, "gt_codes_only.json")
    if os.path.isfile(codes_only_path):
        try:
            with open(codes_only_path, encoding="utf-8") as f:
                codes_only_data = json.load(f)
            raw_cpr = codes_only_data.get("codes_per_review", [])
            for item in raw_cpr:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                review_id, review_codes = item[0], item[1]
                if not isinstance(review_codes, list):
                    continue
                seen: Set[str] = set()
                row: List[str] = []
                for c in review_codes:
                    if not isinstance(c, str):
                        c = str(c)
                    canon = dedup_map.get(c, c)
                    if canon not in seen:
                        seen.add(canon)
                        row.append(canon)
                codes_per_review.append([review_id, row])
        except (json.JSONDecodeError, OSError):
            pass

    out = {
        "clustering_method": "embedding",
        "all_codes": deduped_codes,
        "labels": labels.tolist(),
        "k": best_k,
        "cluster_to_codes": {str(i): codes for i, codes in sorted(cluster_to_codes.items())},
        "dedup_map": dedup_map,
    }
    if codes_per_review:
        out["codes_per_review"] = codes_per_review

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gt_clustered_codes.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = [f"Axial coding (embed + K-means): K={best_k} clusters.", ""]
    for cid, codes in sorted(cluster_to_codes.items()):
        lines.append(f"Cluster {cid}:")
        for c in codes:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines).strip()


def save_hierarchy(hierarchy: Dict[str, Any]) -> None:
    """Write gt_hierarchy.json after each cluster (resume-friendly)."""
    ensure_output_dirs()
    with open(HIERARCHY_PATH, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2)


def build_sub_theme_node(
    st: Dict[str, Any],
    parent_name: str,
    edges: List[Dict[str, str]],
    all_nodes: List[str],
) -> Dict[str, Any]:
    """Recursively build a sub_theme node (including nested sub_themes from refinement)."""
    st_name = st.get("name", "Unnamed")
    edges.append({"parent": parent_name, "child": st_name})
    all_nodes.append(st_name)
    st_node: Dict[str, Any] = {"name": st_name, "type": "sub_theme", "children": []}
    nested = st.get("sub_themes")
    if isinstance(nested, list) and nested:
        for child in nested:
            if isinstance(child, dict):
                st_node["children"].append(build_sub_theme_node(child, st_name, edges, all_nodes))
    for code in st.get("codes") or []:
        if not isinstance(code, str):
            continue
        st_node["children"].append({"name": code, "type": "code"})
        edges.append({"parent": st_name, "child": code})
        all_nodes.append(code)
    return st_node


def assign_codes_to_subthemes_prompt(
    cluster_label: str, sub_theme_names: List[str], codes: List[str], research_question: str
) -> str:
    st_list = "\n".join(f"- {name}" for name in sub_theme_names)
    codes_list = "\n".join(f"- {c}" for c in codes)
    rq_line = f"\nResearch Question: {research_question}\n" if research_question else ""
    return f"""Cluster: "{cluster_label}"
{rq_line}
Existing sub-themes:
{st_list}

Assign each of the following codes to **one** of the sub-themes above. Prefer a **best-fit** sub-theme for every code; use "unassigned" only for codes that genuinely fit **none** of the listed sub-themes (keep "unassigned" empty or tiny).

Codes:
{codes_list}

Output ONLY valid JSON:
{{"assignments": {{"Sub-Theme Name": ["code a", "code b"], "Another Sub-Theme": ["code c"]}}, "unassigned": ["code d"]}}"""
