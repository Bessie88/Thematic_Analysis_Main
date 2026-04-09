"""LLM setup and all GT pipeline tools (open coding, axial, hierarchy, graph, etc.)."""
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .prompts import (
    open_coding_prompt,
    validate_open_codes_prompt,
    high_level_code_generation_prompt,
    refine_cluster_assignments_prompt,
    meta_theme_grouping_prompt,
    intra_cluster_subtheme_prompt,
)
from .paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_PATH,
    DATA_DIR,
    GLOBAL_GRAPH_PATH,
    HIERARCHY_PATH,
    META_THEMES_PATH,
    WEIGHTS_DIR,
    display_path,
    ensure_output_dirs,
)
from .hierarchy_refine import maybe_refine_hierarchy
from .skills import llm_invoke_with_skill
from .utils import clean_and_parse_json, log_step, remove_think_tags

# Refine step: only offer the K most similar cluster labels as MOVE targets (embedding cosine).
REFINE_TOP_K_OTHER_CLUSTERS = 5

# Chunk sizes for LLM calls (context limits); see agents/docs/PIPELINE.md
_EMBED_DRAIN_THRESHOLD = 20


def _refine_llm_max_codes() -> int:
    raw = os.environ.get("GT_REFINE_LLM_MAX_CODES", "44").strip()
    try:
        n = int(raw)
        return max(16, min(80, n))
    except ValueError:
        return 44


def _hierarchy_assign_batch() -> int:
    raw = os.environ.get("GT_HIERARCHY_ASSIGN_BATCH", "40").strip()
    try:
        n = int(raw)
        return max(20, min(60, n))
    except ValueError:
        return 40


def _hierarchy_embed_drain_enabled() -> bool:
    return os.environ.get("GT_HIERARCHY_EMBED_DRAIN_UNGROUPED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prune_hierarchy_to_valid_clusters(
    hierarchy: Dict[str, Any], valid_ids: Set[str]
) -> tuple[Dict[str, Any], List[str]]:
    removed = [k for k in hierarchy if k not in valid_ids]
    pruned = {k: v for k, v in hierarchy.items() if k in valid_ids}
    return pruned, removed


def _embed_model_name_for_hierarchy() -> str:
    model_name = os.environ.get("GT_EMBED_MODEL") or (
        str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B")
        if os.path.isdir(str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B"))
        else "Qwen/Qwen3-Embedding-0.6B"
    )
    if os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)
    return model_name


def _drain_ungrouped_to_subthemes(
    validated_sub_themes: List[Dict[str, Any]],
    validated_ungrouped: List[str],
    embed_model: SentenceTransformer,
) -> List[str]:
    """Assign each ungrouped code to the nearest sub-theme by embedding cosine similarity."""
    if not validated_sub_themes or not validated_ungrouped:
        return validated_ungrouped
    labels = [st["name"] for st in validated_sub_themes]
    lab_emb = embed_model.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    code_emb = embed_model.encode(validated_ungrouped, normalize_embeddings=True, show_progress_bar=False)
    lab_emb = np.asarray(lab_emb, dtype=np.float32)
    code_emb = np.asarray(code_emb, dtype=np.float32)
    sims = code_emb @ lab_emb.T
    for i, code in enumerate(validated_ungrouped):
        j = int(np.argmax(sims[i]))
        validated_sub_themes[j]["codes"].append(code)
    return []


def _meta_theme_bounds(n_cids: int) -> tuple[int, int]:
    if n_cids <= 1:
        return 1, 1
    if n_cids < 6:
        return 2, min(7, n_cids)
    return 3, 7


def _split_largest_meta_theme(meta_themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    largest = max(meta_themes, key=lambda m: len(m.get("cluster_ids", [])))
    cids = list(largest.get("cluster_ids", []))
    if len(cids) < 2:
        return meta_themes
    mid = len(cids) // 2
    base = (largest.get("name") or "Theme").strip() or "Theme"
    largest["cluster_ids"] = cids[:mid]
    meta_themes.append({"name": f"{base} (continued)", "cluster_ids": cids[mid:]})
    return meta_themes


def _merge_two_smallest_meta(meta_themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _normalize_meta_theme_count(meta_themes: List[Dict[str, Any]], n_cids: int) -> List[Dict[str, Any]]:
    lo, hi = _meta_theme_bounds(n_cids)
    mt = list(meta_themes)
    for _ in range(32):
        k = len(mt)
        if lo <= k <= hi:
            return mt
        if k < lo:
            prev = len(mt)
            mt = _split_largest_meta_theme(mt)
            if len(mt) == prev:
                break
        elif k > hi:
            prev = len(mt)
            mt = _merge_two_smallest_meta(mt)
            if len(mt) >= prev:
                break
    return mt


# 2. LLM Setup (vLLM)
# ==================================================
# Set a conservative completion token budget to avoid exceeding model context.
# We'll request at most `COMPLETION_TOKENS` for the model's response and
# ensure input text is truncated so input_tokens + COMPLETION_TOKENS <= CONTEXT_TOKENS.
CONTEXT_TOKENS = 8000
COMPLETION_TOKENS = 1024

llm = ChatOpenAI(
    model="llm",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    temperature=0,
    max_tokens=COMPLETION_TOKENS,
)

# ==================================================
# 3. Grounded Theory Tools (The Cognitive Steps)
# ==================================================

@tool
def open_coding(text: str, research_question: str, validator_feedback: Optional[str] = None) -> str:
    """
    Step 1: Open Coding (per review).
    Extract 0–3 reusable constructs grounded in the text and the research question.
    Use Applicability NONE when the review offers nothing relevant to the question.
    If validator_feedback is provided, use it to improve the previous attempt.
    """
    prompt = open_coding_prompt(research_question, text, validator_feedback=validator_feedback)
    return llm_invoke_with_skill(llm, "open_coding", prompt)


@tool
def validate_open_codes(text: str, generated_codes: str, research_question: str) -> str:
    """
    Review open coding output for one review (including Applicability NONE when no codes).
    Return PASS or FAIL and, if FAIL, list issues so the generator can improve.
    """
    prompt = validate_open_codes_prompt(research_question, text, generated_codes)
    return llm_invoke_with_skill(llm, "validate_open_codes", prompt)


def _deduplicate_codes(all_codes: List[str], model_name: str, near_dup_threshold: float = 0.95) -> tuple:
    """
    Deduplicate codes in two passes:
    1. Exact dedup: normalize (lowercase, strip) and keep first occurrence.
    2. Near-duplicate dedup: embed unique codes, merge pairs with cosine >= threshold via union-find.
    Returns (deduped_codes, original_to_canonical_map).
    """
    # Pass 1: exact dedup (case-insensitive, stripped)
    norm_to_canonical: Dict[str, str] = {}
    for code in all_codes:
        normed = code.strip().lower()
        if normed not in norm_to_canonical:
            norm_to_canonical[normed] = code.strip()
    unique_codes = list(norm_to_canonical.values())

    if len(unique_codes) < 2:
        orig_map = {c: c.strip() for c in all_codes}
        return unique_codes, orig_map

    # Pass 2: near-duplicate dedup via embedding + union-find
    model = SentenceTransformer(model_name)
    embeddings = model.encode(unique_codes, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
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

    # Build canonical mapping: each group maps to the first (canonical) member
    root_to_canonical: Dict[int, str] = {}
    for i in range(len(unique_codes)):
        root = find(i)
        if root not in root_to_canonical:
            root_to_canonical[root] = unique_codes[root]

    deduped = sorted(set(root_to_canonical.values()))

    # Build original -> canonical map
    idx_map = {unique_codes[i]: root_to_canonical[find(i)] for i in range(len(unique_codes))}
    orig_map: Dict[str, str] = {}
    for code in all_codes:
        normed = code.strip().lower()
        canonical_from_exact = norm_to_canonical.get(normed, code.strip())
        orig_map[code] = idx_map.get(canonical_from_exact, canonical_from_exact)

    return deduped, orig_map


def _axial_embed_and_cluster(all_codes: List[str], model_name: str, out_dir: str = str(DATA_DIR)) -> str:
    """
    Step 2: Axial Coding (embed + K-means).
    Embeds all codes, clusters them, returns a text summary of cluster -> codes.
    Uses same logic as embed_and_cluster.py (LOGOS-style).
    """
    MINIBATCH_SIZE = 1000
    K_MIN, K_MAX_DIVISOR = 5, 3

    if not all_codes:
        return "No codes to cluster."

    # Deduplicate codes before clustering
    original_count = len(all_codes)
    deduped_codes, dedup_map = _deduplicate_codes(all_codes, model_name)
    log_step("DEDUP", f"Reduced {original_count} codes to {len(deduped_codes)} unique codes")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(deduped_codes, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    n = len(embeddings)
    if n < K_MIN:
        # Not enough unique codes to cluster meaningfully
        cluster_to_codes_out = {"0": deduped_codes}
        out = {"all_codes": deduped_codes, "labels": [0] * n, "k": 1, "cluster_to_codes": cluster_to_codes_out, "dedup_map": dedup_map}
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
        km = MiniBatchKMeans(n_clusters=best_k, batch_size=MINIBATCH_SIZE, random_state=42, n_init=3)
    else:
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    cluster_to_codes = defaultdict(list)
    for code, label in zip(deduped_codes, labels):
        cluster_to_codes[int(label)].append(code)

    # Preserve codes_per_review from gt_codes_only.json so codebook_cleanup can compute datapoint frequency
    # Map original codes to their canonical (deduped) forms
    codes_per_review = []
    codes_only_path = os.path.join(out_dir, "gt_codes_only.json")
    if os.path.isfile(codes_only_path):
        try:
            with open(codes_only_path, encoding="utf-8") as f:
                codes_only_data = json.load(f)
            raw_cpr = codes_only_data.get("codes_per_review", [])
            # Each item is [review_id, [code, ...]] — not a flat list of codes
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
        "all_codes": deduped_codes,
        "labels": labels.tolist(),
        "k": best_k,
        "cluster_to_codes": {str(i): codes for i, codes in sorted(cluster_to_codes.items())},
        "dedup_map": dedup_map,  # traceability: original -> canonical
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


@tool
def axial_coding(open_codes: str) -> str:
    """
    Step 2: Axial Coding (embed + K-means).
    Input: JSON array of code strings (from all reviews).
    Embeds codes, clusters with K-means (K from Silhouette), returns cluster summary.
    """
    try:
        all_codes = json.loads(open_codes)
    except json.JSONDecodeError:
        return "axial_coding expects a JSON array of code strings."
    if not isinstance(all_codes, list) or not all(isinstance(x, str) for x in all_codes):
        return "axial_coding expects a JSON array of code strings."
    model_name = os.environ.get("GT_EMBED_MODEL") or (
        str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B") if os.path.isdir(str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B")) else "Qwen/Qwen3-Embedding-0.6B"
    )
    # Use absolute path so SentenceTransformer loads from disk only (no HF lookup; compute nodes often have no network)
    if os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)
        os.environ["HF_HUB_OFFLINE"] = "1"
    return _axial_embed_and_cluster(all_codes, model_name=model_name)


@tool
def high_level_code_generation(cluster_file: str = str(CLUSTERED_CODES_PATH), research_question: str = "") -> str:
    """
    Step 2b: High-level code generation (final part of axial).
    Reads cluster_to_codes from disk, prompts LLM once per cluster for a label, confidence (1-5), and rationale.
    Writes codebook.json (labels only, for downstream) and codebook_confidence.json (full objects). Returns codebook as JSON string.
    """
    if not os.path.isfile(cluster_file):
        return json.dumps({"error": f"Missing {cluster_file}; run axial step first."})
    with open(cluster_file, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes", {})
    if not cluster_to_codes:
        return json.dumps({"error": "No cluster_to_codes in file."})
    out_dir = os.path.dirname(cluster_file) or "."
    out_path = os.path.join(out_dir, "codebook.json")
    confidence_path = os.path.join(out_dir, "codebook_confidence.json")
    # Resume: load existing codebook if present so we can skip already-done clusters (e.g. after SGLang died)
    codebook = {}
    codebook_confidence: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            codebook = existing.get("codebook", {})
        except (json.JSONDecodeError, KeyError):
            pass
    if os.path.isfile(confidence_path):
        try:
            with open(confidence_path, encoding="utf-8") as f:
                codebook_confidence = json.load(f)
        except (json.JSONDecodeError, TypeError):
            codebook_confidence = {}

    for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0])):
        # Resume: skip if both codebook and confidence have this cluster; if codebook has it but confidence doesn't, backfill
        if cid in codebook:
            if cid not in codebook_confidence:
                codebook_confidence[cid] = {"label": codebook[cid], "confidence": 1, "rationale": ""}
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"codebook": codebook, "cluster_to_codes": cluster_to_codes}, f, indent=2)
                with open(confidence_path, "w", encoding="utf-8") as f:
                    json.dump(codebook_confidence, f, indent=2)
            continue
        codes_list = codes if isinstance(codes, list) else []
        if not codes_list:
            codebook[cid] = f"Cluster {cid}"
            codebook_confidence[cid] = {"label": codebook[cid], "confidence": 1, "rationale": ""}
        else:
            bulleted = "\n".join(f"- {c}" for c in codes_list[:30])
            if len(codes_list) > 30:
                bulleted += f"\n- ... and {len(codes_list) - 30} more"
            prompt = high_level_code_generation_prompt(bulleted, research_question)

            try:
                raw = llm_invoke_with_skill(llm, "high_level_code_generation", prompt)
                parsed = clean_and_parse_json(remove_think_tags(raw))
                label = (parsed.get("label") or "").strip().strip('"\'')
                if not label or len(label) > 80:
                    label = f"Cluster {cid}"
                confidence = parsed.get("confidence", 1)
                if not isinstance(confidence, int):
                    try:
                        confidence = int(float(confidence))
                    except (TypeError, ValueError):
                        confidence = 1
                confidence = max(1, min(5, confidence))
                rationale = (parsed.get("rationale") or "").strip()
                codebook[cid] = label
                codebook_confidence[cid] = {"label": label, "confidence": confidence, "rationale": rationale}
                if confidence <= 2:
                    log_step("LOW_CONFIDENCE_CLUSTER", f"Cluster {cid}: confidence={confidence}, label={label}")
            except Exception as e:
                log_step("HIGH_LEVEL_LLM_ERROR", f"Cluster {cid}: {e}")
                codebook[cid] = f"Cluster {cid}"
                codebook_confidence[cid] = {"label": codebook[cid], "confidence": 1, "rationale": ""}
        # Save after each cluster so we can resume if SGLang dies mid-run
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"codebook": codebook, "cluster_to_codes": cluster_to_codes}, f, indent=2)
        with open(confidence_path, "w", encoding="utf-8") as f:
            json.dump(codebook_confidence, f, indent=2)
    return json.dumps(codebook)


@tool
def refine_cluster_assignments(codebook_path: str, cluster_file: str) -> str:
    """
    After high-level labels exist, review each cluster and move codes that clearly belong in another cluster.
    Reads codebook.json and gt_clustered_codes.json, asks LLM per cluster for MOVE commands (code -> target label),
    applies moves, removes empty clusters, writes back gt_clustered_codes.json. Returns summary of moves applied.
    """
    if not os.path.isfile(codebook_path):
        return f"Error: codebook not found at {codebook_path}"
    if not os.path.isfile(cluster_file):
        return f"Error: cluster file not found at {cluster_file}"
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})
    with open(cluster_file, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes", {})
    all_codes = data.get("all_codes", [])
    codes_per_review = data.get("codes_per_review", [])

    # label -> list of cids (for ambiguous label check)
    label_to_cids: Dict[str, List[str]] = defaultdict(list)
    for cid, label in codebook.items():
        label_to_cids[label].append(cid)

    moves_applied: List[tuple] = []  # (code, from_cid, to_cid)

    sorted_oids = sorted(cluster_to_codes.keys(), key=lambda x: int(x))
    if not sorted_oids:
        return "Error: no clusters in cluster file"

    label_for_oid = {oid: codebook.get(oid, f"Cluster {oid}") for oid in sorted_oids}
    label_texts = [label_for_oid[oid] for oid in sorted_oids]

    model_name = os.environ.get("GT_EMBED_MODEL") or (
        str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B")
        if os.path.isdir(str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B"))
        else "Qwen/Qwen3-Embedding-0.6B"
    )
    if os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)
    try:
        embed_model = SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        return f"Error loading embedding model for refine: {e}"

    emb = embed_model.encode(label_texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    oid_to_pos = {oid: i for i, oid in enumerate(sorted_oids)}

    for cid in sorted_oids:
        codes = cluster_to_codes.get(cid, [])
        if not codes:
            continue
        label = codebook.get(cid, f"Cluster {cid}")
        pos = oid_to_pos[cid]
        sims = emb @ emb[pos]
        sims = sims.copy()
        sims[pos] = -np.inf

        n_others = len(sorted_oids) - 1
        k_take = min(REFINE_TOP_K_OTHER_CLUSTERS, n_others)
        if k_take <= 0:
            other_str = "(none)"
        else:
            # Top-k by cosine similarity (descending)
            part = np.argpartition(-sims, k_take - 1)[:k_take]
            top_positions = part[np.argsort(-sims[part])]
            other_labels = [label_texts[j] for j in top_positions]
            other_str = ", ".join(other_labels)

        max_chunk = _refine_llm_max_codes()
        chunk_codes_set = set(codes)
        for chunk_idx, chunk_start in enumerate(range(0, len(codes), max_chunk)):
            chunk = codes[chunk_start : chunk_start + max_chunk]
            bulleted = "\n".join(f"- {c}" for c in chunk)

            prompt = refine_cluster_assignments_prompt(label, bulleted, other_str)

            try:
                raw = remove_think_tags(
                    llm_invoke_with_skill(llm, "refine_cluster_assignments", prompt)
                )
            except Exception as e:
                log_step("REFINE_LLM_ERROR", f"Cluster {cid} chunk {chunk_idx}: {e}")
                continue
            for line in raw.splitlines():
                line = line.strip()
                if line.upper() == "NONE" or not line:
                    continue
                match = re.search(
                    r'MOVE:\s*["\']([^"\']+)["\']\s*[→>]\s*["\']([^"\']+)["\']',
                    line,
                    re.IGNORECASE,
                )
                if not match:
                    continue
                code, target_label = match.group(1).strip(), match.group(2).strip()
                if code not in chunk_codes_set:
                    continue
                target_cids = label_to_cids.get(target_label, [])
                if not target_cids:
                    continue
                if len(target_cids) > 1:
                    log_step(
                        "REFINE_SKIP_AMBIGUOUS_LABEL",
                        f"MOVE skipped: label '{target_label}' maps to multiple clusters",
                    )
                    continue
                target_cid = target_cids[0]
                if target_cid == cid:
                    continue
                moves_applied.append((code, cid, target_cid))

    # Deduplicate: keep first move per code, warn on conflict (e.g. A->B and B->C for same code)
    seen_codes: Dict[str, tuple] = {}
    deduped_moves: List[tuple] = []
    for move in moves_applied:
        code, from_cid, to_cid = move
        if code in seen_codes:
            log_step("REFINE_CONFLICTING_MOVE", f"Code '{code}' has multiple move targets; keeping first.")
        else:
            seen_codes[code] = move
            deduped_moves.append(move)
    moves_applied = deduped_moves

    # Apply moves: build code -> cid, then update
    code_to_cid: Dict[str, str] = {}
    for cid, codes in cluster_to_codes.items():
        for c in codes:
            code_to_cid[c] = cid
    for code, _from, to_cid in moves_applied:
        if code in code_to_cid:
            code_to_cid[code] = to_cid

    # Rebuild cluster_to_codes from code_to_cid
    new_cluster_to_codes: Dict[str, List[str]] = defaultdict(list)
    for code, cid in code_to_cid.items():
        new_cluster_to_codes[cid].append(code)
    # Remove empty clusters and rekey to contiguous 0..k_new-1
    non_empty = {cid: codes for cid, codes in new_cluster_to_codes.items() if codes}
    cid_list = sorted(non_empty.keys(), key=int)
    new_cluster_to_codes = {str(i): non_empty[cid_list[i]] for i in range(len(cid_list))}
    k_new = len(new_cluster_to_codes)
    code_to_idx = {c: i for i, cids in enumerate(new_cluster_to_codes.values()) for c in cids}
    labels = [code_to_idx.get(c, 0) for c in all_codes]

    out = {
        "all_codes": all_codes,
        "labels": labels,
        "k": k_new,
        "cluster_to_codes": new_cluster_to_codes,
        "codes_per_review": codes_per_review,
    }
    os.makedirs(os.path.dirname(cluster_file) or ".", exist_ok=True)
    with open(cluster_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # Keep codebook.json in sync: rekey codebook to new cluster ids 0..k_new-1
    new_codebook = {str(i): codebook.get(cid_list[i], f"Cluster {cid_list[i]}") for i in range(len(cid_list))}
    with open(codebook_path, "w", encoding="utf-8") as f:
        json.dump({"codebook": new_codebook, "cluster_to_codes": new_cluster_to_codes}, f, indent=2)
    # Rekey codebook_confidence.json so keys match new cluster ids 0..k_new-1
    confidence_path = os.path.join(os.path.dirname(codebook_path) or ".", "codebook_confidence.json")
    codebook_confidence_rekeyed: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(confidence_path):
        try:
            with open(confidence_path, encoding="utf-8") as f:
                old_confidence = json.load(f)
            for i in range(len(cid_list)):
                old_cid = cid_list[i]
                entry = old_confidence.get(str(old_cid), old_confidence.get(old_cid))
                if isinstance(entry, dict):
                    codebook_confidence_rekeyed[str(i)] = {**entry, "label": new_codebook[str(i)]}
                else:
                    codebook_confidence_rekeyed[str(i)] = {"label": new_codebook[str(i)], "confidence": 1, "rationale": ""}
        except (json.JSONDecodeError, TypeError):
            codebook_confidence_rekeyed = {str(i): {"label": new_codebook[str(i)], "confidence": 1, "rationale": ""} for i in range(len(cid_list))}
    else:
        codebook_confidence_rekeyed = {str(i): {"label": new_codebook[str(i)], "confidence": 1, "rationale": ""} for i in range(len(cid_list))}
    with open(confidence_path, "w", encoding="utf-8") as f:
        json.dump(codebook_confidence_rekeyed, f, indent=2)

    # Drop hierarchy entries for cluster IDs that no longer exist after rekey
    hier_path = str(HIERARCHY_PATH)
    if os.path.isfile(hier_path):
        try:
            with open(hier_path, encoding="utf-8") as f:
                hier = json.load(f)
            if isinstance(hier, dict):
                valid_h = set(new_cluster_to_codes.keys())
                pruned, removed = _prune_hierarchy_to_valid_clusters(hier, valid_h)
                if removed:
                    ensure_output_dirs()
                    with open(hier_path, "w", encoding="utf-8") as f:
                        json.dump(pruned, f, indent=2)
                    log_step(
                        "HIERARCHY_PRUNE_AFTER_REFINE",
                        f"Removed {len(removed)} stale cluster key(s) after rekey",
                    )
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    return f"Refined cluster assignments: {len(moves_applied)} codes moved across clusters."


@tool
def meta_theme_grouping(research_question: str = "") -> str:
    """
    Step 5a (LOGOS): Group cluster labels into a small handful of broad meta-themes (about 3–7 when many clusters).
    Reads codebook.json, asks LLM to group all cluster labels, writes gt_meta_themes.json.
    """
    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(codebook_path):
        return json.dumps({"error": "Missing codebook.json; run high-level step first."})
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})
    if not codebook:
        return json.dumps({"error": "No codebook in codebook.json."})

    labels_json = json.dumps(codebook, indent=2)
    all_cids = set(codebook.keys())
    n_cids = len(all_cids)
    lo, hi = _meta_theme_bounds(n_cids)

    def _call_meta_llm(extra: str = "") -> List[Dict[str, Any]]:
        base = meta_theme_grouping_prompt(labels_json, research_question)
        prompt = base + (extra or "")
        raw = llm_invoke_with_skill(llm, "meta_theme_grouping", prompt)
        parsed = clean_and_parse_json(raw)
        mt = parsed.get("meta_themes", [])
        if not isinstance(mt, list):
            return []
        return [m for m in mt if isinstance(m, dict)]

    try:
        meta_themes = _call_meta_llm()
    except Exception as e:
        log_step("META_THEME_LLM_ERROR", str(e))
        return json.dumps({"error": f"LLM call failed: {e}"})

    if not meta_themes:
        return json.dumps({"error": "LLM returned invalid meta_themes structure."})

    assigned_cids: set = set()
    for mt in meta_themes:
        for cid in mt.get("cluster_ids", []):
            assigned_cids.add(str(cid))

    missing = all_cids - assigned_cids
    if missing:
        largest = max(meta_themes, key=lambda m: len(m.get("cluster_ids", [])))
        for cid in missing:
            largest["cluster_ids"].append(str(cid))
        log_step(
            "META_THEME_FIXUP",
            f"Added {len(missing)} missing cluster(s) to group '{largest.get('name', '')}'",
        )

    if not (lo <= len(meta_themes) <= hi):
        reminder = (
            f"\n\nIMPORTANT: You must output between {lo} and {hi} meta_themes (inclusive), "
            f"each with a distinct name. Every cluster ID must appear exactly once."
        )
        try:
            meta_themes_retry = _call_meta_llm(reminder)
            if meta_themes_retry:
                meta_themes = meta_themes_retry
                assigned_cids = set()
                for mt in meta_themes:
                    for cid in mt.get("cluster_ids", []):
                        assigned_cids.add(str(cid))
                missing = all_cids - assigned_cids
                if missing:
                    largest = max(meta_themes, key=lambda m: len(m.get("cluster_ids", [])))
                    for cid in missing:
                        largest["cluster_ids"].append(str(cid))
                    log_step(
                        "META_THEME_FIXUP_RETRY",
                        f"Added {len(missing)} missing cluster(s) after retry",
                    )
        except Exception as e:
            log_step("META_THEME_RETRY_ERROR", str(e))

    if not (lo <= len(meta_themes) <= hi):
        before = len(meta_themes)
        meta_themes = _normalize_meta_theme_count(meta_themes, n_cids)
        log_step(
            "META_THEME_REPAIR",
            f"Adjusted meta-theme count from {before} to {len(meta_themes)} (bounds [{lo},{hi}])",
        )

    ensure_output_dirs()
    with open(META_THEMES_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta_themes": meta_themes}, f, indent=2)

    summary = (
        f"Meta-themes: {len(meta_themes)} groups covering {len(all_cids)} clusters. "
        f"See {display_path(META_THEMES_PATH)}"
    )
    return summary
@tool
def hierarchy_construction(research_question: str = "") -> str:
    """
    Step 5b (LOGOS): Intra-cluster sub-theme grouping.
    For each cluster: ask the LLM to organise codes into 2-5 sub-themes.
    Writes gt_hierarchy.json in tree format (no pairwise edges, no transitive closure).
    """
    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(codebook_path):
        return json.dumps({"error": "Missing codebook.json; run high-level step first."})
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})
    cluster_to_codes = cb_data.get("cluster_to_codes", {})
    if not cluster_to_codes:
        return json.dumps({"error": "No cluster_to_codes in codebook.json."})

    # Resume: load existing hierarchy if present
    existing_hierarchy: Dict[str, Any] = {}
    if os.path.isfile(str(HIERARCHY_PATH)):
        try:
            with open(HIERARCHY_PATH, encoding="utf-8") as f:
                existing_hierarchy = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_hierarchy = {}

    hierarchy: Dict[str, Any] = dict(existing_hierarchy)
    valid_ids = set(cluster_to_codes.keys())
    hierarchy, pruned_keys = _prune_hierarchy_to_valid_clusters(hierarchy, valid_ids)
    if pruned_keys:
        log_step(
            "HIERARCHY_PRUNE",
            f"Removed {len(pruned_keys)} stale cluster key(s) not in codebook: "
            f"{pruned_keys[:20]}{'...' if len(pruned_keys) > 20 else ''}",
        )
        _save_hierarchy(hierarchy)

    for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0])):
        if cid in hierarchy:
            log_step("HIERARCHY_SKIP", f"Cluster {cid} already done, skipping.")
            continue

        label = codebook.get(cid, f"Cluster {cid}")
        unique_codes = list(dict.fromkeys(codes))

        # Small clusters: all codes are direct children, no sub-theming needed
        if len(unique_codes) <= 5:
            hierarchy[cid] = {
                "label": label,
                "sub_themes": [],
                "ungrouped_codes": unique_codes,
            }
            log_step("HIERARCHY_CLUSTER_DONE", f"Cluster {cid} ({label}): {len(unique_codes)} codes (small, no sub-themes)")
            _save_hierarchy(hierarchy)
            continue

        # For large clusters, batch codes to avoid exceeding context window
        # Send up to 60 codes at a time; if more, do a second pass to assign extras
        BATCH_SIZE = 60
        if len(unique_codes) <= BATCH_SIZE:
            codes_for_prompt = unique_codes
        else:
            codes_for_prompt = unique_codes[:BATCH_SIZE]

        codes_list_str = "\n".join(f"- {c}" for c in codes_for_prompt)
        prompt = intra_cluster_subtheme_prompt(label, codes_list_str, research_question)

        try:
            raw = llm_invoke_with_skill(llm, "hierarchy_construction", prompt)
            parsed = clean_and_parse_json(raw)
        except Exception as e:
            log_step("HIERARCHY_LLM_ERROR", f"Cluster {cid}: {e}")
            # Fallback: all codes as ungrouped
            hierarchy[cid] = {
                "label": label,
                "sub_themes": [],
                "ungrouped_codes": unique_codes,
            }
            _save_hierarchy(hierarchy)
            continue

        sub_themes = parsed.get("sub_themes", [])
        ungrouped = parsed.get("ungrouped_codes", [])

        # Validate: ensure all codes are accounted for
        assigned_codes: set = set()
        validated_sub_themes: List[Dict[str, Any]] = []
        for st in sub_themes:
            if not isinstance(st, dict):
                continue
            st_codes = [c for c in st.get("codes", []) if c in set(codes_for_prompt)]
            assigned_codes.update(st_codes)
            if st_codes:
                validated_sub_themes.append({"name": st.get("name", "Unnamed"), "codes": st_codes})

        validated_ungrouped = [c for c in ungrouped if c in set(codes_for_prompt) and c not in assigned_codes]
        assigned_codes.update(validated_ungrouped)

        # Any codes the LLM missed go to ungrouped
        missing = [c for c in codes_for_prompt if c not in assigned_codes]
        validated_ungrouped.extend(missing)

        # If we batched, assign remaining codes in chunks (avoids one giant assign prompt)
        if len(unique_codes) > BATCH_SIZE:
            remaining = unique_codes[BATCH_SIZE:]
            st_names = [st["name"] for st in validated_sub_themes]
            assign_batch = _hierarchy_assign_batch()
            if st_names:
                for chunk_idx, chunk_start in enumerate(range(0, len(remaining), assign_batch)):
                    chunk = remaining[chunk_start : chunk_start + assign_batch]
                    assign_prompt = _assign_codes_to_subthemes_prompt(
                        label, st_names, chunk, research_question
                    )
                    try:
                        raw2 = llm_invoke_with_skill(llm, "hierarchy_construction", assign_prompt)
                        parsed2 = clean_and_parse_json(raw2)
                        assignments = parsed2.get("assignments", {})
                        assigned_chunk: set = set()
                        for st_name, st_codes in assignments.items():
                            if not isinstance(st_codes, list):
                                continue
                            for st in validated_sub_themes:
                                if st["name"] == st_name:
                                    valid_codes = [c for c in st_codes if c in set(chunk)]
                                    st["codes"].extend(valid_codes)
                                    assigned_chunk.update(valid_codes)
                                    break
                        unassigned = parsed2.get("unassigned", [])
                        if isinstance(unassigned, list):
                            for c in unassigned:
                                if isinstance(c, str) and c in chunk and c not in assigned_chunk:
                                    validated_ungrouped.append(c)
                                    assigned_chunk.add(c)
                        for c in chunk:
                            if c not in assigned_chunk:
                                validated_ungrouped.append(c)
                    except Exception as e:
                        log_step(
                            "HIERARCHY_BATCH2_ERROR",
                            f"Cluster {cid} assign chunk {chunk_idx}: {e}",
                        )
                        validated_ungrouped.extend(chunk)
            else:
                validated_ungrouped.extend(remaining)

        if (
            _hierarchy_embed_drain_enabled()
            and len(validated_ungrouped) > _EMBED_DRAIN_THRESHOLD
            and validated_sub_themes
        ):
            try:
                drain_model = SentenceTransformer(_embed_model_name_for_hierarchy(), device="cpu")
                validated_ungrouped = _drain_ungrouped_to_subthemes(
                    validated_sub_themes, validated_ungrouped, drain_model
                )
            except Exception as e:
                log_step("HIERARCHY_EMBED_DRAIN_ERROR", f"Cluster {cid}: {e}")

        hierarchy[cid] = {
            "label": label,
            "sub_themes": validated_sub_themes,
            "ungrouped_codes": validated_ungrouped,
        }
        total_codes = sum(len(st["codes"]) for st in validated_sub_themes) + len(validated_ungrouped)
        log_step("HIERARCHY_CLUSTER_DONE", f"Cluster {cid} ({label}): {len(validated_sub_themes)} sub-themes, {total_codes} codes")
        _save_hierarchy(hierarchy)

    summary = f"Hierarchy: {len(hierarchy)} clusters with sub-theme groupings. See {display_path(HIERARCHY_PATH)}"
    return summary


def _save_hierarchy(hierarchy: Dict[str, Any]) -> None:
    """Persist hierarchy to disk (called after each cluster for resume support)."""
    ensure_output_dirs()
    with open(HIERARCHY_PATH, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2)


def _build_sub_theme_node(
    st: Dict[str, Any],
    parent_name: str,
    edges: List[Dict[str, str]],
    all_nodes: List[str],
) -> Dict[str, Any]:
    """Build a sub_theme tree node; supports nested sub_themes (post hierarchy_refine)."""
    st_name = st.get("name", "Unnamed")
    edges.append({"parent": parent_name, "child": st_name})
    all_nodes.append(st_name)
    st_node: Dict[str, Any] = {"name": st_name, "type": "sub_theme", "children": []}
    nested = st.get("sub_themes")
    if isinstance(nested, list) and nested:
        for child in nested:
            if isinstance(child, dict):
                st_node["children"].append(_build_sub_theme_node(child, st_name, edges, all_nodes))
    for code in st.get("codes") or []:
        if not isinstance(code, str):
            continue
        st_node["children"].append({"name": code, "type": "code"})
        edges.append({"parent": st_name, "child": code})
        all_nodes.append(code)
    return st_node


def _assign_codes_to_subthemes_prompt(
    cluster_label: str, sub_theme_names: List[str], codes: List[str], research_question: str
) -> str:
    """Prompt to assign overflow codes to existing sub-themes."""
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


@tool
def tree_assembly(research_question: str = "") -> str:
    """
    Step 6 (LOGOS): Tree assembly.
    Read gt_meta_themes.json and gt_hierarchy.json, optionally refine hierarchy to cap code fan-out
    (GT_HIERARCHY_REFINE, rewrites gt_hierarchy.json), then build a hierarchical tree with nested
    sub_themes, write gt_global_graph.json with tree structure and flat edge list.
    """
    if not os.path.isfile(str(META_THEMES_PATH)):
        return json.dumps({"error": "Missing gt_meta_themes.json; run meta_theme_grouping step first."})
    if not os.path.isfile(str(HIERARCHY_PATH)):
        return json.dumps({"error": "Missing gt_hierarchy.json; run hierarchy step first."})
    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(codebook_path):
        return json.dumps({"error": "Missing codebook.json; run high-level step first."})

    with open(META_THEMES_PATH, encoding="utf-8") as f:
        meta_data = json.load(f)
    with open(HIERARCHY_PATH, encoding="utf-8") as f:
        hierarchy = json.load(f)

    def _invoke_skill(skill_key: str, human_prompt: str) -> str:
        return llm_invoke_with_skill(llm, skill_key, human_prompt)

    hierarchy = maybe_refine_hierarchy(hierarchy, research_question or "", _invoke_skill)
    ensure_output_dirs()
    with open(HIERARCHY_PATH, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2)

    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})

    meta_themes = meta_data.get("meta_themes", [])

    # Build tree structure
    root_name = research_question or "Thematic Analysis"
    tree: Dict[str, Any] = {"name": root_name, "type": "root", "children": []}

    # Also build flat edge list for backward compatibility
    edges: List[Dict[str, str]] = []
    all_nodes: List[str] = [root_name]

    for mt in meta_themes:
        mt_name = mt.get("name", "Unnamed Meta-Theme")
        mt_node: Dict[str, Any] = {"name": mt_name, "type": "meta_theme", "children": []}
        edges.append({"parent": root_name, "child": mt_name})
        all_nodes.append(mt_name)

        for cid in mt.get("cluster_ids", []):
            cid = str(cid)
            cluster_label = codebook.get(cid, f"Cluster {cid}")
            cluster_entry = hierarchy.get(cid, {})
            label = cluster_entry.get("label", cluster_label)

            theme_node: Dict[str, Any] = {"name": label, "type": "theme", "children": []}
            edges.append({"parent": mt_name, "child": label})
            all_nodes.append(label)

            # Sub-themes (flat or nested after hierarchy_refine)
            for st in cluster_entry.get("sub_themes", []):
                if isinstance(st, dict):
                    theme_node["children"].append(_build_sub_theme_node(st, label, edges, all_nodes))

            # Ungrouped codes: direct children of the theme
            for code in cluster_entry.get("ungrouped_codes", []):
                theme_node["children"].append({"name": code, "type": "code"})
                edges.append({"parent": label, "child": code})
                all_nodes.append(code)

            mt_node["children"].append(theme_node)

        tree["children"].append(mt_node)

    canonical_nodes = sorted(set(all_nodes))

    out = {
        "tree": tree,
        "canonical_nodes": canonical_nodes,
        "merge_groups": [],
        "edges": edges,
        "inferred_edges": [],
    }
    ensure_output_dirs()
    with open(GLOBAL_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    summary = (
        f"Tree assembled: {len(canonical_nodes)} nodes, {len(edges)} edges (strict hierarchy). "
        f"See {display_path(GLOBAL_GRAPH_PATH)} (hierarchy may have been refined; {display_path(HIERARCHY_PATH)})"
    )
    return summary

