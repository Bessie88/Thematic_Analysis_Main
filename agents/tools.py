"""LLM setup and all GT pipeline tools (open coding, axial, hierarchy, graph, etc.)."""
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_PATH,
    CROSS_CLUSTER_EDGES_PATH,
    DATA_DIR,
    GLOBAL_GRAPH_PATH,
    GRAPH_PATH,
    HIERARCHY_EDGES_PATH,
    HIERARCHY_PATH,
    WEIGHTS_DIR,
    display_path,
    ensure_output_dirs,
)
from utils import clean_and_parse_json, log_step, remove_think_tags

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
    Extract 1–5 reusable constructs grounded in the text.
    If validator_feedback is provided, use it to improve the previous attempt.
    """
    feedback_section = ""
    if validator_feedback:
        feedback_section = f"""
A reviewer found issues with the previous codes. Use this feedback to improve:
{validator_feedback}

Revise your codes accordingly. Output the same format as below.
"""
    prompt = f"""
You are doing Open Coding for thematic analysis on ONE user review.

Research Question: {research_question}
Focus on aspects of the review that are relevant to the research question above.

Rules:
- Produce 1 to 5 codes total (depending on content).
- Each code must be a short noun phrase (2–6 words).
- Codes must be distinct (no near-duplicates).
- Abstract the idea, but keep it clearly supported by the review.
- No summary, no advice.
{feedback_section}

Output exactly as bullet points:
- Code: <code>
  Evidence: "<short quote from the review>"
  Note: <one short phrase why this code fits>

Review:
{text}
"""
    return llm.invoke(prompt).content


@tool
def validate_open_codes(text: str, generated_codes: str, research_question: str) -> str:
    """
    Review open codes for one review. Check they are grounded, non-duplicate, and concise.
    Return PASS or FAIL and, if FAIL, list issues so the generator can improve.
    """
    prompt = f"""You are reviewing qualitative codes generated from user feedback for a research question.

Research Question: {research_question}

Input text (the review):
{text}

Generated codes:
{generated_codes}

Check:
1. Codes are grounded in the data (evidence in the review supports each code).
2. Codes are not duplicates or near-duplicates of each other.
3. Codes are concise concepts (short noun phrases, not vague or hallucinated).

Respond with exactly one of:
- PASS
or
- FAIL
Issues:
- <first issue>
- <second issue>
...

If PASS, you may add a single line of explanation after PASS. If FAIL, list specific issues so the coder can revise."""
    return llm.invoke(prompt).content


def _axial_embed_and_cluster(all_codes: List[str], model_name: str, out_dir: str = str(DATA_DIR)) -> str:
    """
    Step 2: Axial Coding (embed + K-means).
    Embeds all codes, clusters them, returns a text summary of cluster -> codes.
    Uses same logic as embed_and_cluster.py (LOGOS-style).
    """
    try:
        import numpy as np
        from collections import defaultdict
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import silhouette_score
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Axial step needs scikit-learn and sentence_transformers. Install with: pip install scikit-learn sentence-transformers. Original: {e}"
        ) from e

    MINIBATCH_SIZE = 1000
    K_MIN, K_MAX_DIVISOR = 2, 3

    if not all_codes:
        return "No codes to cluster."

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_codes, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    n = len(embeddings)
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
    for code, label in zip(all_codes, labels):
        cluster_to_codes[int(label)].append(code)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gt_clustered_codes.json"), "w", encoding="utf-8") as f:
        json.dump({
            "all_codes": all_codes,
            "labels": labels.tolist(),
            "k": best_k,
            "cluster_to_codes": {str(i): codes for i, codes in sorted(cluster_to_codes.items())},
        }, f, indent=2)

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
def validate_clusters(cluster_file: str, research_question: str) -> str:
    """
    Review clusters after axial coding. For each cluster, check whether codes belong to the same concept.
    Return PASS (optional: suggest label per cluster) or FAIL with issues so refinement can fix.
    """
    if not os.path.isfile(cluster_file):
        return "FAIL\nIssues:\n- Cluster file not found."
    with open(cluster_file, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes", {})
    if not cluster_to_codes:
        return "FAIL\nIssues:\n- No cluster_to_codes in file."
    clusters_text = ""
    for cid in sorted(cluster_to_codes.keys(), key=lambda x: int(x)):
        codes = cluster_to_codes.get(cid, [])
        clusters_text += f"\nCluster {cid}:\n" + "\n".join(f"  - {c}" for c in codes) + "\n"
    prompt = f"""You are reviewing qualitative code clusters for a research question.

Research Question: {research_question}

The following clusters were produced by embedding + K-means. For each cluster, check:
1. Do these codes belong in the same concept cluster (coherent theme)?
2. If any code does not fit (e.g. belongs to a different theme like performance vs interface), note it.

Clusters:
{clusters_text}

Respond with exactly one of:
- PASS
(optional: add a suggested short label per cluster, e.g. "Cluster 0: Interface usability issues")
or
- FAIL
Issues:
- Cluster <id>: <what is wrong, e.g. "Menu lag belongs to performance issues">
- ...

If PASS, you may add suggested labels after PASS. If FAIL, list specific issues so codes can be reassigned."""
    return llm.invoke(prompt).content


@tool
def cluster_refinement(cluster_file: str, validator_feedback: str, research_question: str) -> str:
    """
    Reassign codes between clusters using validator feedback (no re-embedding).
    Reads gt_clustered_codes.json, asks LLM for concrete moves, applies them, writes back, returns axial-style summary.
    """
    if not os.path.isfile(cluster_file):
        return "Error: cluster file not found."
    with open(cluster_file, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes", {})
    all_codes = data.get("all_codes", [])
    codes_per_review = data.get("codes_per_review", [])
    k_prev = data.get("k", len(cluster_to_codes))
    if not cluster_to_codes:
        return "Error: no cluster_to_codes."

    # Build current cluster text for LLM
    clusters_text = ""
    for cid in sorted(cluster_to_codes.keys(), key=lambda x: int(x)):
        codes = cluster_to_codes.get(cid, [])
        clusters_text += f"\nCluster {cid}:\n" + "\n".join(f"  - {c}" for c in codes) + "\n"

    prompt = f"""You are refining code clusters based on reviewer feedback.

Research Question: {research_question}

Reviewer feedback:
{validator_feedback}

Current clusters:
{clusters_text}

Output a list of reassignments, one per line, in this exact format:
MOVE: "code" | from_cluster_id | to_cluster_id

Example:
MOVE: "Menu lag" | 2 | 5

Only output lines that start with MOVE:. Use the exact code string as in the clusters. Cluster IDs are integers. To create a new cluster, use a new id (e.g. the current max + 1). Do not duplicate codes across clusters."""
    try:
        raw = llm.invoke(prompt).content
    except Exception as e:
        log_step("CLUSTER_REFINEMENT_LLM_ERROR", str(e))
        return _axial_summary_from_cluster_to_codes(cluster_to_codes, prefix="Axial coding (refinement failed):")

    raw = remove_think_tags(raw)
    # Parse MOVE lines
    moves = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.upper().startswith("MOVE:"):
            continue
        match = re.match(r'MOVE:\s*"([^"]+)"\s*\|\s*(\d+)\s*\|\s*(\d+)', line, re.IGNORECASE)
        if not match:
            match = re.match(r"MOVE:\s*'([^']+)'\s*\|\s*(\d+)\s*\|\s*(\d+)", line, re.IGNORECASE)
        if match:
            code, from_c, to_c = match.group(1), int(match.group(2)), int(match.group(3))
            moves.append((code, str(from_c), str(to_c)))

    # Apply moves: build code -> new cluster id (start from current assignment)
    code_to_cluster: Dict[str, str] = {}
    for cid, codes in cluster_to_codes.items():
        for c in codes:
            code_to_cluster[c] = cid
    for code, from_c, to_c in moves:
        if code not in code_to_cluster:
            continue
        if code_to_cluster[code] != from_c:
            continue
        code_to_cluster[code] = to_c

    # Rebuild cluster_to_codes from code_to_cluster
    new_cluster_to_codes: Dict[str, List[str]] = defaultdict(list)
    for code, cid in code_to_cluster.items():
        new_cluster_to_codes[cid].append(code)
    # Sort cluster ids numerically
    new_cluster_to_codes = {str(cid): codes for cid, codes in sorted(new_cluster_to_codes.items(), key=lambda x: int(x[0]))}
    k_new = len(new_cluster_to_codes)
    cid_list = sorted(new_cluster_to_codes.keys(), key=int)
    code_to_idx = {c: cid_list.index(code_to_cluster[c]) for c in code_to_cluster}
    labels = [code_to_idx.get(c, 0) for c in all_codes]

    out = {
        "all_codes": all_codes,
        "labels": labels,
        "k": k_new,
        "codes_per_review": codes_per_review,
        "cluster_to_codes": new_cluster_to_codes,
    }
    os.makedirs(os.path.dirname(cluster_file) or ".", exist_ok=True)
    with open(cluster_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return _axial_summary_from_cluster_to_codes(new_cluster_to_codes, prefix="Axial coding (refined):")


def _axial_summary_from_cluster_to_codes(cluster_to_codes: Dict[str, List[str]], prefix: str = "Axial coding:") -> str:
    """Produce the same text summary format as _axial_embed_and_cluster for axial_mapping."""
    lines = [f"{prefix} K={len(cluster_to_codes)} clusters.", ""]
    for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0])):
        lines.append(f"Cluster {cid}:")
        for c in codes:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines).strip()


@tool
def high_level_code_generation(cluster_file: str = str(CLUSTERED_CODES_PATH), research_question: str = "") -> str:
    """
    Step 2b: High-level code generation (final part of axial).
    Reads cluster_to_codes from disk, prompts LLM once per cluster for one short label
    describing the gist, writes codebook.json (after each cluster for resume), returns codebook as JSON string.
    """
    if not os.path.isfile(cluster_file):
        return json.dumps({"error": f"Missing {cluster_file}; run axial step first."})
    with open(cluster_file, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes", {})
    if not cluster_to_codes:
        return json.dumps({"error": "No cluster_to_codes in file."})
    out_path = os.path.join(os.path.dirname(cluster_file) or ".", "codebook.json")
    # Resume: load existing codebook if present so we can skip already-done clusters (e.g. after SGLang died)
    codebook = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            codebook = existing.get("codebook", {})
        except (json.JSONDecodeError, KeyError):
            pass
    for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0])):
        if cid in codebook:
            continue
        codes_list = codes if isinstance(codes, list) else []
        if not codes_list:
            codebook[cid] = f"Cluster {cid}"
        else:
            bulleted = "\n".join(f"- {c}" for c in codes_list[:30])
            if len(codes_list) > 30:
                bulleted += f"\n- ... and {len(codes_list) - 30} more"
            rq_line = f"\nResearch Question: {research_question}\nGenerate the label with the research question in mind.\n" if research_question else ""
            prompt = f"""The following open codes belong to one cluster:

{bulleted}
{rq_line}
Generate one short high-level label (2-6 words) that describes the gist of this cluster. Output only the label, no explanation."""

            try:
                raw = llm.invoke(prompt).content
            except Exception as e:
                log_step("HIGH_LEVEL_LLM_ERROR", f"Cluster {cid}: {e}")
                codebook[cid] = f"Cluster {cid}"
            else:
                label = remove_think_tags(raw).strip().split("\n")[0].strip().strip('"\'')
                if not label or len(label) > 80:
                    label = f"Cluster {cid}"
                codebook[cid] = label
        # Save after each cluster so we can resume if SGLang dies mid-run
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"codebook": codebook, "cluster_to_codes": cluster_to_codes}, f, indent=2)
    return json.dumps(codebook)


def _build_hierarchy(edges: List[Dict[str, Any]], cluster_to_codes: Dict[str, List[str]], codebook: Dict[str, str]) -> Dict[str, Any]:
    """Build per-cluster hierarchy from classified edges: apply merges (union-find), then directed edges."""
    from collections import defaultdict

    parent_map: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent_map.get(x, x) != x:
            parent_map[x] = parent_map.get(parent_map[x], parent_map[x])
            x = parent_map[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_map[rb] = ra

    cluster_edges: Dict[str, List[Dict]] = defaultdict(list)
    cluster_merges: Dict[str, List[tuple]] = defaultdict(list)

    for e in edges:
        cid = str(e["cluster_id"])
        if e["relation"] == "equivalent":
            cluster_merges[cid].append((e["node_a"], e["node_b"]))
            union(e["node_a"], e["node_b"])
        elif e["relation"] == "subsumes":
            cluster_edges[cid].append({"parent": e["node_a"], "child": e["node_b"]})
        elif e["relation"] == "subsumed_by":
            cluster_edges[cid].append({"parent": e["node_b"], "child": e["node_a"]})

    hierarchy = {}
    all_cids = set(cluster_to_codes.keys()) | set(cluster_edges.keys()) | set(cluster_merges.keys())
    for cid in sorted(all_cids, key=lambda x: int(x)):
        codes = cluster_to_codes.get(cid, [])
        rep = codebook.get(cid, f"Cluster {cid}")
        all_nodes = list(set(codes + [rep]))

        merge_groups_map: Dict[str, List[str]] = defaultdict(list)
        for n in all_nodes:
            merge_groups_map[find(n)].append(n)
        merge_groups = [g for g in merge_groups_map.values() if len(g) > 1]

        canonical = sorted(set(find(n) for n in all_nodes))

        directed = []
        for edge in cluster_edges.get(cid, []):
            cp = find(edge["parent"])
            cc = find(edge["child"])
            if cp != cc:
                directed.append({"parent": cp, "child": cc})

        hierarchy[cid] = {
            "merge_groups": merge_groups,
            "edges": directed,
            "canonical_nodes": canonical,
        }
    return hierarchy


def _infer_graph_per_cluster(
    merge_groups: List[List[str]],
    edges: List[Dict[str, Any]],
    canonical_nodes: List[str],
) -> tuple:
    """
    Apply BFS transitivity: A→B and B→C ⇒ infer A→C.
    Deduction-first conflict: do not add A→C if C→A exists or A,C in same equiv class.
    Returns (final_edges_list, inferred_edges_list).
    """
    from collections import defaultdict, deque

    equiv_rep: Dict[str, str] = {}
    for group in merge_groups:
        rep = group[0]
        for n in group:
            equiv_rep[n] = rep
    for n in canonical_nodes:
        if n not in equiv_rep:
            equiv_rep[n] = n

    def same_equiv(a: str, b: str) -> bool:
        return equiv_rep.get(a, a) == equiv_rep.get(b, b)

    out_edges: Dict[str, set] = defaultdict(set)
    edge_set: set = set()
    for e in edges:
        u, v = e["parent"], e["child"]
        out_edges[u].add(v)
        edge_set.add((u, v))

    initial_edges = set(edge_set)
    queue: deque = deque(edge_set)

    while queue:
        a, b = queue.popleft()
        for c in list(out_edges[b]):
            if (a, c) in edge_set:
                continue
            if (c, a) in edge_set:
                continue
            if same_equiv(a, c):
                continue
            edge_set.add((a, c))
            out_edges[a].add(c)
            queue.append((a, c))

    final_edges = [{"parent": u, "child": v} for u, v in sorted(edge_set)]
    inferred_edges = [{"parent": u, "child": v} for u, v in sorted(edge_set - initial_edges)]
    return final_edges, inferred_edges


def _infer_graph_global(
    merge_groups: List[List[str]],
    edges: List[Dict[str, Any]],
    canonical_nodes: List[str],
) -> tuple:
    """
    Same as _infer_graph_per_cluster but for global graph.
    Apply BFS transitivity, equivalence closure, deduction-first conflict.
    Returns (final_edges_list, inferred_edges_list).
    """
    return _infer_graph_per_cluster(merge_groups, edges, canonical_nodes)


@tool
def global_graph_construction(
    research_question: str = "",
    sim_threshold: float = 0.7,
    skip_cross_cluster: bool = False,
    cross_cluster_top_k: int = 75,
) -> str:
    """
    Step 6 (LOGOS): Global graph construction.
    Merge all per-cluster nodes and edges, optionally add cross-cluster links via LLM,
    apply global transitivity and equivalence closure, write gt_global_graph.json.
    """
    from collections import defaultdict

    graph_path = str(GRAPH_PATH)
    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(graph_path):
        return json.dumps({"error": "Missing gt_graph.json; run graph step first."})
    with open(graph_path, encoding="utf-8") as f:
        per_cluster = json.load(f)
    codebook: Dict[str, str] = {}
    cluster_to_codes: Dict[str, List[str]] = {}
    if not skip_cross_cluster:
        if not os.path.isfile(codebook_path):
            return json.dumps({"error": "Missing codebook.json; needed for cross-cluster linking."})
        with open(codebook_path, encoding="utf-8") as f:
            cb_data = json.load(f)
        codebook = cb_data.get("codebook", {})
        cluster_to_codes = cb_data.get("cluster_to_codes", {})

    # 1. Global union-find: merge equivalent nodes (same string + per-cluster merge_groups)
    parent_map: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent_map.get(x, x) != x:
            parent_map[x] = parent_map.get(parent_map[x], parent_map[x])
            x = parent_map[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_map[rb] = ra

    # Collect all node strings
    all_node_strings: set = set()
    for cid, entry in per_cluster.items():
        for n in entry.get("canonical_nodes", []):
            all_node_strings.add(n)
        for g in entry.get("merge_groups", []):
            for n in g:
                all_node_strings.add(n)
        for e in entry.get("edges", []):
            all_node_strings.add(e["parent"])
            all_node_strings.add(e["child"])

    for n in all_node_strings:
        if n not in parent_map:
            parent_map[n] = n

    # Apply per-cluster merge_groups
    for entry in per_cluster.values():
        for group in entry.get("merge_groups", []):
            if len(group) < 2:
                continue
            rep = group[0]
            for n in group[1:]:
                union(rep, n)

    # 2. Collect and canonicalize edges (adjacency sets, no NxN matrix)
    edge_set: set = set()
    for entry in per_cluster.values():
        for e in entry.get("edges", []):
            u, v = find(e["parent"]), find(e["child"])
            if u != v:
                edge_set.add((u, v))

    # 3. Optional cross-cluster linking
    cross_cluster_edges: List[Dict[str, Any]] = []
    existing_cross_count = 0
    if not skip_cross_cluster:
        import numpy as np

        model_name = os.environ.get("GT_EMBED_MODEL") or (
            str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B") if os.path.isdir(str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B")) else "Qwen/Qwen3-Embedding-0.6B"
        )
        if os.path.isdir(model_name):
            model_name = os.path.abspath(model_name)

        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(model_name, device="cpu")

        # Embed cluster reps + top-3 codes per cluster; track cluster_id for cross-cluster filter
        from collections import Counter
        labeled: List[tuple] = []  # (label, cluster_id)
        for cid in sorted(codebook.keys(), key=int):
            rep = codebook.get(cid, f"Cluster {cid}")
            labeled.append((rep, cid))
        for cid in sorted(cluster_to_codes.keys(), key=int):
            codes = cluster_to_codes.get(cid, [])
            freq = Counter(codes)
            for c, _ in freq.most_common(3):
                labeled.append((c, cid))
        # Dedup by label, keep first cluster (for embedding we just need unique labels)
        seen_label: set = set()
        candidates: List[str] = []
        label_to_cid: Dict[str, str] = {}
        for label, cid in labeled:
            if label not in seen_label:
                seen_label.add(label)
                candidates.append(label)
                label_to_cid[label] = cid
            elif label not in label_to_cid:
                label_to_cid[label] = cid
        for c in candidates:
            if c not in parent_map:
                parent_map[c] = c
                all_node_strings.add(c)

        if len(candidates) >= 2:
            embeddings = embed_model.encode(candidates, normalize_embeddings=True)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            sim_matrix = embeddings @ embeddings.T
            n = len(candidates)
            pairs: List[tuple] = []
            for i in range(n):
                for j in range(n):
                    if i >= j:
                        continue
                    ci, cj = candidates[i], candidates[j]
                    if ci == cj:
                        continue
                    # Only cross-cluster pairs
                    cid_i = label_to_cid.get(ci, "")
                    cid_j = label_to_cid.get(cj, "")
                    if cid_i == cid_j:
                        continue
                    pairs.append((ci, cj, float(sim_matrix[i, j])))
            pairs.sort(key=lambda x: -x[2])
            top_pairs = pairs[:cross_cluster_top_k]

            cross_path = str(CROSS_CLUSTER_EDGES_PATH)
            done_pairs: set = set()
            existing: List[Dict] = []
            if os.path.isfile(cross_path):
                with open(cross_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            existing.append(e)
                            done_pairs.add((e["node_a"], e["node_b"]))
                            done_pairs.add((e["node_b"], e["node_a"]))
                        except (json.JSONDecodeError, KeyError):
                            pass
            cross_cluster_edges = list(existing)
            existing_cross_count = len(existing)

            for node_a, node_b, sim in top_pairs:
                if (node_a, node_b) in done_pairs or (node_b, node_a) in done_pairs:
                    continue
                if sim < sim_threshold:
                    continue

                prompt = f"""Given two codes from a thematic analysis:
A: "{node_a}"
B: "{node_b}"

Research Question: {research_question}

Classify the relationship between A and B as exactly one of:
- "equivalent": A and B mean essentially the same thing (should be merged)
- "subsumes": A is more general/abstract than B (A contains B)
- "subsumed_by": A is more specific than B (B contains A)
- "orthogonal": A and B are distinct concepts with no hierarchical relationship

Output ONLY valid JSON: {{"relation": "<one of the four>", "reason": "<brief explanation>"}}"""

                try:
                    raw = llm.invoke(prompt).content
                    parsed = clean_and_parse_json(raw)
                    relation = parsed.get("relation", "orthogonal")
                except Exception as e:
                    log_step("CROSS_CLUSTER_LLM_ERROR", f"({node_a}, {node_b}): {e}")
                    continue

                if relation not in ("equivalent", "subsumes", "subsumed_by", "orthogonal"):
                    relation = "orthogonal"

                if relation != "orthogonal":
                    rec = {
                        "node_a": node_a,
                        "node_b": node_b,
                        "relation": relation,
                        "similarity": round(sim, 4),
                    }
                    cross_cluster_edges.append(rec)
                    if relation == "equivalent":
                        union(node_a, node_b)
                    elif relation == "subsumes":
                        u, v = find(node_a), find(node_b)
                        if u != v:
                            edge_set.add((u, v))
                    elif relation == "subsumed_by":
                        u, v = find(node_b), find(node_a)
                        if u != v:
                            edge_set.add((u, v))

                done_pairs.add((node_a, node_b))
                done_pairs.add((node_b, node_a))

                with open(cross_path, "w", encoding="utf-8") as f:
                    for rec in cross_cluster_edges:
                        f.write(json.dumps(rec) + "\n")

    # 4. Build global merge_groups and canonical_nodes
    merge_groups_map: Dict[str, List[str]] = defaultdict(list)
    for n in all_node_strings:
        merge_groups_map[find(n)].append(n)
    merge_groups = [sorted(set(g)) for g in merge_groups_map.values() if len(g) > 1]
    canonical_nodes = sorted(set(find(n) for n in all_node_strings))

    # 5. Global active inference (transitivity + equivalence closure + deduction-first)
    edges_list = [{"parent": u, "child": v} for u, v in sorted(edge_set)]
    final_edges, inferred_edges = _infer_graph_global(merge_groups, edges_list, canonical_nodes)

    out = {
        "canonical_nodes": canonical_nodes,
        "merge_groups": merge_groups,
        "edges": final_edges,
        "inferred_edges": inferred_edges,
    }
    ensure_output_dirs()
    with open(GLOBAL_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    n_cross_new = len(cross_cluster_edges) - existing_cross_count if not skip_cross_cluster else 0
    summary = f"Global graph: {len(canonical_nodes)} nodes, {len(final_edges)} edges ({len(inferred_edges)} inferred). Cross-cluster: {'skipped' if skip_cross_cluster else f'{n_cross_new} new edges'}. See {display_path(GLOBAL_GRAPH_PATH)}"
    return summary


@tool
def hierarchy_construction(research_question: str = "", sim_threshold: float = 0.6) -> str:
    """
    Step 4 (LOGOS): Relationship classification + hierarchy edges.
    For each cluster: embed codes + representative on CPU, filter pairs by cosine
    similarity, classify kept pairs via LLM, save edges to JSONL and build
    per-cluster hierarchy in gt_hierarchy.json.
    """
    import numpy as np

    codebook_path = str(CODEBOOK_PATH)
    if not os.path.isfile(codebook_path):
        return json.dumps({"error": "Missing codebook.json; run high-level step first."})
    with open(codebook_path, encoding="utf-8") as f:
        cb_data = json.load(f)
    codebook = cb_data.get("codebook", {})
    cluster_to_codes = cb_data.get("cluster_to_codes", {})
    if not cluster_to_codes:
        return json.dumps({"error": "No cluster_to_codes in codebook.json."})

    model_name = os.environ.get("GT_EMBED_MODEL") or (
        str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B") if os.path.isdir(str(WEIGHTS_DIR / "Qwen3-Embedding-0.6B")) else "Qwen/Qwen3-Embedding-0.6B"
    )
    if os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(model_name, device="cpu")

    edges_path = str(HIERARCHY_EDGES_PATH)
    done_pairs: set = set()
    existing_edges: List[Dict] = []
    if os.path.isfile(edges_path):
        with open(edges_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    existing_edges.append(e)
                    done_pairs.add((e["cluster_id"], e["node_a"], e["node_b"]))
                except (json.JSONDecodeError, KeyError):
                    pass

    all_edges = list(existing_edges)

    for cid, codes in sorted(cluster_to_codes.items(), key=lambda x: int(x[0])):
        representative = codebook.get(cid, f"Cluster {cid}")
        nodes = list(codes) + [representative]
        nodes = list(dict.fromkeys(nodes))

        if len(nodes) < 2:
            continue

        embeddings = embed_model.encode(nodes, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        sim_matrix = embeddings @ embeddings.T

        n = len(nodes)
        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= sim_threshold:
                    candidates.append((nodes[i], nodes[j], float(sim_matrix[i][j])))

        for node_a, node_b, sim in candidates:
            if (cid, node_a, node_b) in done_pairs:
                continue

            prompt = f"""Given two codes from a thematic analysis:
A: "{node_a}"
B: "{node_b}"

Research Question: {research_question}

Classify the relationship between A and B as exactly one of:
- "equivalent": A and B mean essentially the same thing (should be merged)
- "subsumes": A is more general/abstract than B (A contains B)
- "subsumed_by": A is more specific than B (B contains A)
- "orthogonal": A and B are distinct concepts with no hierarchical relationship

Output ONLY valid JSON: {{"relation": "<one of the four>", "reason": "<brief explanation>"}}"""

            try:
                raw = llm.invoke(prompt).content
                parsed = clean_and_parse_json(raw)
                relation = parsed.get("relation", "orthogonal")
                reason = parsed.get("reason", "")
            except Exception as e:
                log_step("HIERARCHY_LLM_ERROR", f"Cluster {cid}, ({node_a}, {node_b}): {e}")
                continue

            if relation not in ("equivalent", "subsumes", "subsumed_by", "orthogonal"):
                relation = "orthogonal"

            if relation != "orthogonal":
                edge = {
                    "cluster_id": cid,
                    "node_a": node_a,
                    "node_b": node_b,
                    "relation": relation,
                    "similarity": round(sim, 4),
                    "reason": reason,
                }
                all_edges.append(edge)

            done_pairs.add((cid, node_a, node_b))

            with open(edges_path, "w", encoding="utf-8") as f:
                for edge_rec in all_edges:
                    f.write(json.dumps(edge_rec) + "\n")

        log_step("HIERARCHY_CLUSTER_DONE", f"Cluster {cid}: {len(candidates)} candidates processed")

    hierarchy = _build_hierarchy(all_edges, cluster_to_codes, codebook)
    ensure_output_dirs()
    with open(HIERARCHY_PATH, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2)

    merges = sum(1 for e in all_edges if e["relation"] == "equivalent")
    subsumptions = sum(1 for e in all_edges if e["relation"] in ("subsumes", "subsumed_by"))
    summary = f"Hierarchy: {len(all_edges)} edges ({merges} merges, {subsumptions} subsumptions) across {len(cluster_to_codes)} clusters. See {display_path(HIERARCHY_EDGES_PATH)} and {display_path(HIERARCHY_PATH)}"
    return summary


@tool
def graph_construction() -> str:
    """
    Step 5 (LOGOS): Graph construction. Read gt_hierarchy.json, run BFS transitivity
    per cluster with deduction-first conflict resolution, write gt_graph.json.
    """
    hierarchy_path = str(HIERARCHY_PATH)
    if not os.path.isfile(hierarchy_path):
        return json.dumps({"error": "Missing gt_hierarchy.json; run hierarchy step first."})
    with open(hierarchy_path, encoding="utf-8") as f:
        hierarchy = json.load(f)

    result = {}
    total_edges = 0
    for cid in sorted(hierarchy.keys(), key=lambda x: int(x)):
        entry = hierarchy[cid]
        merge_groups = entry.get("merge_groups", [])
        edges = entry.get("edges", [])
        canonical_nodes = entry.get("canonical_nodes", [])
        final_edges, inferred_edges = _infer_graph_per_cluster(merge_groups, edges, canonical_nodes)
        result[cid] = {
            "merge_groups": merge_groups,
            "canonical_nodes": canonical_nodes,
            "edges": final_edges,
            "inferred_edges": inferred_edges,
        }
        total_edges += len(final_edges)

    ensure_output_dirs()
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    summary = f"Graph: {len(result)} clusters, {total_edges} total edges after inference. See {display_path(GRAPH_PATH)}"
    return summary

