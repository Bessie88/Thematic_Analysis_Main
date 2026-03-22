"""
LOGOS Step 6: Codebook clean-up.

Post-processing after global graph construction. Reduces a noisy global concept graph
to a compact codebook by: equivalence closure and representative selection, merge and
redirect, collapse low-frequency children (direct edges only), remove orphans,
rebuild canonical_nodes and direct edges, recompute inferred_edges.

Fixes applied vs previous version:
  1. Self-loop guard in _infer_edges_from_direct (a == c check).
  2. Rep chain re-resolution after collapse loop (multi-hop collapse was leaving
     intermediate nodes pointing at already-collapsed targets).
  3. Cycle-breaking pass before BFS: mutual A→B / B→A edges are resolved by
     keeping the direction where the parent has higher frequency.
  4. min_freq default lowered to 3; see scaling note on run_cleanup.
  5. Corpus frequencies use normalize_label() so graph nodes match codes_per_review
     despite quote/whitespace differences (avoids false zero-frequency collapse).
  6. Cluster representatives (codebook labels) use review-level cluster support:
     frequency is at least the count of reviews containing any code from that cluster.
"""

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.core.paths import (
    CLEANED_GLOBAL_GRAPH_PATH,
    CLUSTERED_CODES_PATH,
    CODEBOOK_PATH,
    GLOBAL_GRAPH_PATH,
    GT_CODES_ONLY_PATH,
)


def normalize_label(s: str) -> str:
    """
    Normalize theme labels so graph JSON strings align with codes_per_review keys.

    Strips leading/trailing whitespace, collapses internal runs of whitespace to a
    single space, and removes outer layers of matching single/double quotes
    (handles e.g. '\\\"frustrating difficulty\\\"' vs 'frustrating difficulty').
    """
    if not isinstance(s, str):
        s = str(s)
    t = s.strip()
    t = re.sub(r"\s+", " ", t)
    quote_chars = "\"'"
    while len(t) >= 2 and t[0] == t[-1] and t[0] in quote_chars:
        t = t[1:-1].strip()
        t = re.sub(r"\s+", " ", t)
    return t


def load_graph(path: Path) -> Tuple[List[str], List[List[str]], List[Dict[str, str]]]:
    """Load gt_global_graph.json. Returns (canonical_nodes, merge_groups, edges). Uses direct edges only."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("canonical_nodes", [])
    merge_groups = data.get("merge_groups", [])
    edges = data.get("edges", [])
    return nodes, merge_groups, edges


def load_datapoint_frequency(
    clustered_path: Path,
    codes_only_path: Path,
) -> Dict[str, int]:
    """
    Compute datapoint frequency keyed by **normalize_label(code)**.

    Counts each normalized code at most once per review. This matches graph
    nodes that differ only by quoting/whitespace from corpus strings.
    """
    freq: Dict[str, int] = defaultdict(int)

    for path in (clustered_path, codes_only_path):
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        codes_per_review = data.get("codes_per_review", [])
        for item in codes_per_review:
            _review_id, codes = item[0], item[1]
            seen_in_review: Set[str] = set()
            for c in codes:
                key = normalize_label(c)
                if key not in seen_in_review:
                    seen_in_review.add(key)
                    freq[key] += 1
        if codes_per_review:
            break
    return dict(freq)


def load_codes_per_review_list(
    clustered_path: Path,
    codes_only_path: Path,
) -> List[Any]:
    """Return codes_per_review from the first file that has a non-empty list."""
    for path in (clustered_path, codes_only_path):
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cpr = data.get("codes_per_review", [])
        if cpr:
            return cpr
    return []


def _build_code_to_cluster_map(cluster_to_codes: Dict[str, List[str]]) -> Dict[str, str]:
    """Map raw code string and normalized form → cluster id (string)."""
    m: Dict[str, str] = {}
    for cid, codes in cluster_to_codes.items():
        scid = str(cid)
        for c in codes:
            m[c] = scid
            nk = normalize_label(c)
            if nk not in m:
                m[nk] = scid
    return m


def cluster_review_support_counts(
    cluster_to_codes: Dict[str, List[str]],
    codes_per_review: List[Any],
) -> Dict[str, int]:
    """
    For each cluster id, number of reviews that contain at least one open code
    assigned to that cluster (via cluster_to_codes).
    """
    code_to_cid = _build_code_to_cluster_map(cluster_to_codes)
    support: Dict[str, int] = defaultdict(int)
    for item in codes_per_review:
        codes = item[1]
        touched: Set[str] = set()
        for c in codes:
            cid = code_to_cid.get(c)
            if cid is None:
                cid = code_to_cid.get(normalize_label(c))
            if cid is not None:
                touched.add(cid)
        for cid in touched:
            support[cid] += 1
    return dict(support)


def _rep_normalized_to_cid(codebook: Dict[str, str]) -> Dict[str, str]:
    """normalized rep label → cluster id string."""
    out: Dict[str, str] = {}
    for cid, rep in codebook.items():
        out[normalize_label(rep)] = str(cid)
    return out


def apply_representative_support(
    datapoint_freq: Dict[str, int],
    all_labels: Set[str],
    codebook_path: Path,
    clustered_codes_path: Path,
    codes_only_path: Path,
    verbose: bool = False,
) -> None:
    """
    In-place: cluster high-level representatives (codebook labels) rarely appear
    verbatim in reviews. Set their frequency to at least the number of reviews
    that contain any code from that cluster, so cleanup does not treat them as
    unsupported (max with any existing literal normalized count).
    """
    if not codebook_path.is_file():
        if verbose:
            print(f"[codebook_cleanup] representative support: skip (no codebook at {codebook_path})")
        return
    with open(codebook_path, encoding="utf-8") as f:
        cb = json.load(f)
    codebook = cb.get("codebook", {})
    cluster_to_codes = cb.get("cluster_to_codes", {})
    if not codebook or not cluster_to_codes:
        if verbose:
            print("[codebook_cleanup] representative support: skip (empty codebook/cluster_to_codes)")
        return
    cpr = load_codes_per_review_list(clustered_codes_path, codes_only_path)
    if not cpr:
        if verbose:
            print("[codebook_cleanup] representative support: skip (no codes_per_review)")
        return
    support = cluster_review_support_counts(cluster_to_codes, cpr)
    rep_norm_to_cid = _rep_normalized_to_cid(codebook)
    boosted = 0
    for label in all_labels:
        nl = normalize_label(label)
        cid = rep_norm_to_cid.get(nl)
        if cid is None:
            continue
        sup = support.get(cid, 0)
        prev = datapoint_freq.get(label, 0)
        new_v = max(prev, sup)
        datapoint_freq[label] = new_v
        if new_v > prev:
            boosted += 1
    if verbose:
        n_reps = len(codebook)
        print(
            f"[codebook_cleanup] representative support: {n_reps} clusters in codebook; "
            f"{boosted} graph labels matched a rep and got frequency ≥ cluster review count"
        )


def equivalence_components(merge_groups: List[List[str]]) -> List[List[str]]:
    """
    Compute full connected components under equivalence closure.
    Build undirected graph from merge_groups (each list connects all pairs), then union-find.
    Uses an iterative find to avoid Python recursion limits on large graphs.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while root in parent and parent[root] != root:
            root = parent[root]
        if root not in parent:
            parent[root] = root
        while x in parent and parent[x] != root:
            parent_x = parent[x]
            parent[x] = root
            x = parent_x
        if x not in parent:
            parent[x] = root
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in merge_groups:
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                union(a, b)

    components_map: Dict[str, List[str]] = defaultdict(list)
    for x in list(parent.keys()):
        root = find(x)
        components_map[root].append(x)
    return list(components_map.values())


def _in_degree_from_edges(edges: List[Dict[str, str]]) -> Dict[str, int]:
    """In-degree (number of direct edges whose child is this node)."""
    indeg: Dict[str, int] = defaultdict(int)
    for e in edges:
        indeg[e["child"]] += 1
    return dict(indeg)


def _break_cycles(
    direct_edges: List[Dict[str, str]],
    datapoint_freq: Dict[str, int],
) -> List[Dict[str, str]]:
    """
    Remove one direction of any mutual A→B / B→A pair in direct_edges.
    Keeps the direction where the parent has strictly higher frequency.
    When frequencies are equal, keeps the lexicographically smaller parent
    so the result is deterministic.
    Returns the cleaned edge list.
    """
    edge_set = {(e["parent"], e["child"]) for e in direct_edges}
    kept: List[Dict[str, str]] = []
    for e in direct_edges:
        p, c = e["parent"], e["child"]
        if (c, p) in edge_set:
            fp = datapoint_freq.get(p, 0)
            fc = datapoint_freq.get(c, 0)
            # Keep p→c only if p has higher freq, or equal freq and p < c lexicographically.
            if fp > fc or (fp == fc and p < c):
                kept.append(e)
            # else drop — the reverse direction will be kept when we process that edge
        else:
            kept.append(e)
    return kept


def run_cleanup(
    graph_path: Path,
    clustered_codes_path: Path,
    codes_only_path: Path,
    output_path: Path,
    # min_freq scaling guide (approximate — adjust based on your corpus size):
    #   ~100  reviews  → min_freq = 2
    #   ~1000 reviews  → min_freq = 3   ← current default
    #   ~5000 reviews  → min_freq = 5
    #   ~50k  reviews  → min_freq = 10
    # Rule of thumb: a concept should appear in at least 0.3–0.5% of the corpus
    # to be considered stable enough to keep in the final codebook.
    min_freq: int = 3,
    w_freq: float = 1.0,
    w_indeg: float = 1.0,
    include_inferred_edges: bool = True,
    verbose: bool = False,
    codebook_path: Optional[Path] = None,
    use_representative_support: bool = True,
) -> Dict[str, Any]:
    """
    Run LOGOS Step 6 codebook clean-up and return the cleaned graph dict.
    """
    nodes, merge_groups, direct_edges = load_graph(graph_path)
    freq_norm = load_datapoint_frequency(clustered_codes_path, codes_only_path)
    cb_path = codebook_path if codebook_path is not None else CODEBOOK_PATH

    all_labels: Set[str] = set(nodes)
    for g in merge_groups:
        all_labels.update(g)
    for e in direct_edges:
        all_labels.add(e["parent"])
        all_labels.add(e["child"])

    datapoint_freq: Dict[str, int] = {}
    for label in all_labels:
        datapoint_freq[label] = freq_norm.get(normalize_label(label), 0)

    if use_representative_support:
        apply_representative_support(
            datapoint_freq,
            all_labels,
            cb_path,
            clustered_codes_path,
            codes_only_path,
            verbose=verbose,
        )

    if verbose:
        n_labels = len(all_labels)
        n_zero = sum(1 for v in datapoint_freq.values() if v == 0)
        raw_freq: Dict[str, int] = defaultdict(int)
        for path in (clustered_codes_path, codes_only_path):
            if not path.is_file():
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            cpr = data.get("codes_per_review", [])
            for item in cpr:
                seen: Set[str] = set()
                for c in item[1]:
                    if c not in seen:
                        seen.add(c)
                        raw_freq[c] += 1
            if cpr:
                break
        n_zero_raw = sum(1 for lab in all_labels if raw_freq.get(lab, 0) == 0)
        n_recovered_norm = sum(
            1
            for lab in all_labels
            if raw_freq.get(lab, 0) == 0 and freq_norm.get(normalize_label(lab), 0) > 0
        )
        print(
            f"[codebook_cleanup] frequency alignment: {n_labels} graph labels, "
            f"{n_zero} with zero count after normalized corpus keys + rep support"
        )
        print(
            f"[codebook_cleanup] exact raw-key match would leave {n_zero_raw} labels at zero; "
            f"normalized corpus keys alone recover {n_recovered_norm} of those"
        )

    # --- Step 1: Equivalence closure and representative per component ---
    components = equivalence_components(merge_groups)
    in_deg = _in_degree_from_edges(direct_edges)

    rep: Dict[str, str] = {}
    for n in nodes:
        rep[n] = n
    for comp in components:
        best = None
        best_score = -1.0
        for c in comp:
            score = w_freq * datapoint_freq.get(c, 0) + w_indeg * in_deg.get(c, 0)
            if score > best_score or (score == best_score and (best is None or c < best)):
                best_score = score
                best = c
        for c in comp:
            rep[c] = best

    # --- Step 2: Merge equivalents, redirect direct edges only, aggregate frequency ---
    new_edges: Set[Tuple[str, str]] = set()
    for e in direct_edges:
        p, q = rep[e["parent"]], rep[e["child"]]
        if p != q:
            new_edges.add((p, q))
    direct_edges = [{"parent": p, "child": q} for p, q in sorted(new_edges)]

    agg_freq: Dict[str, int] = defaultdict(int)
    for n in nodes:
        agg_freq[rep[n]] += datapoint_freq.get(n, 0)
    datapoint_freq = dict(agg_freq)

    in_deg = _in_degree_from_edges(direct_edges)
    current_nodes = set(rep[n] for n in nodes)

    # --- Step 3: Collapse low-frequency children (only direct edges) ---
    direct_parents: Dict[str, List[str]] = defaultdict(list)
    for e in direct_edges:
        direct_parents[e["child"]].append(e["parent"])

    orphan_set: Set[str] = set()

    def _parent_sort_key(p: str) -> Tuple[int, int, str]:
        return (-datapoint_freq.get(p, 0), -in_deg.get(p, 0), p)

    for n in list(current_nodes):
        if datapoint_freq.get(n, 0) >= min_freq:
            continue
        parents = direct_parents.get(n, [])
        if not parents:
            orphan_set.add(n)
            continue
        chosen = min(parents, key=_parent_sort_key)
        for x in nodes:
            if rep[x] == n:
                rep[x] = chosen
        datapoint_freq[chosen] = datapoint_freq.get(chosen, 0) + datapoint_freq.get(n, 0)

    # FIX: re-resolve rep chains — multi-hop collapses (A→B→C) leave A pointing
    # at B even after B was itself collapsed into C. Flatten all chains here.
    for x in nodes:
        while rep[rep[x]] != rep[x]:
            rep[x] = rep[rep[x]]

    # Redirect edges after collapse
    new_edges = set()
    for e in direct_edges:
        p, q = rep[e["parent"]], rep[e["child"]]
        if p != q:
            new_edges.add((p, q))
    direct_edges = [{"parent": p, "child": q} for p, q in sorted(new_edges)]

    # --- Step 4: Remove orphans from node set and drop edges referencing them ---
    current_nodes = set(rep[n] for n in nodes) - orphan_set
    direct_edges = [e for e in direct_edges if e["parent"] in current_nodes and e["child"] in current_nodes]

    # --- Step 4b: Break mutual cycles before BFS ---
    # The hierarchy step can produce contradictory edges (A subsumes B in one
    # cluster, B subsumes A in another). Both survive into the global graph and
    # cause self-loops in BFS transitivity. Resolve by keeping the direction
    # where the parent has higher datapoint frequency.
    direct_edges = _break_cycles(direct_edges, datapoint_freq)

    # --- Step 5: Build provenance map (raw code → final node) ---
    # Saved in the output so the evidence panel can trace which original open
    # codes collapsed into each final concept node.
    provenance: Dict[str, List[str]] = defaultdict(list)
    for original_code in nodes:
        final = rep.get(original_code, original_code)
        if final in current_nodes:
            provenance[final].append(original_code)

    # --- Step 6: Rebuild canonical_nodes, direct edges, optional inferred_edges ---
    canonical_nodes = sorted(current_nodes)
    if include_inferred_edges:
        _, inferred_edges = _infer_edges_from_direct(direct_edges, canonical_nodes)
    else:
        inferred_edges = []

    out = {
        "canonical_nodes": canonical_nodes,
        "edges": direct_edges,
        "inferred_edges": inferred_edges,
        "node_frequencies": {n: datapoint_freq.get(n, 0) for n in canonical_nodes},
        "code_provenance": {n: sorted(set(provenance[n])) for n in canonical_nodes},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _infer_edges_from_direct(
    edges: List[Dict[str, str]],
    canonical_nodes: List[str],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Apply BFS transitivity: A→B and B→C ⇒ infer A→C.
    No equivalence classes (post-cleanup). Returns (all_edges, inferred_edges).
    """
    out_edges: Dict[str, Set[str]] = defaultdict(set)
    edge_set: Set[Tuple[str, str]] = set()
    for e in edges:
        u, v = e["parent"], e["child"]
        out_edges[u].add(v)
        edge_set.add((u, v))
    initial = set(edge_set)
    queue: deque = deque(edge_set)
    while queue:
        a, b = queue.popleft()
        for c in list(out_edges[b]):
            if a == c:          # FIX: guard against self-loops from cycles
                continue
            if (a, c) in edge_set:
                continue
            if (c, a) in edge_set:
                continue
            edge_set.add((a, c))
            out_edges[a].add(c)
            queue.append((a, c))
    final_edges = [{"parent": u, "child": v} for u, v in sorted(edge_set)]
    inferred_edges = [{"parent": u, "child": v} for u, v in sorted(edge_set - initial)]
    return final_edges, inferred_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="LOGOS Step 6: Codebook clean-up (post-process global graph).")
    parser.add_argument("--graph", default=None, help="Path to gt_global_graph.json")
    parser.add_argument("--clustered", default=None, help="Path to gt_clustered_codes.json")
    parser.add_argument("--codes-only", default=None, help="Path to gt_codes_only.json")
    parser.add_argument("--out", default=None, help="Output path for cleaned graph JSON")
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help=(
            "Min datapoint frequency to keep a node (default 3 for ~1000 reviews). "
            "Scale with corpus size: ~100 reviews→2, ~1k→3, ~5k→5, ~50k→10. "
            "Rule of thumb: concept should appear in at least 0.3-0.5%% of reviews."
        ),
    )
    parser.add_argument("--w-freq", type=float, default=1.0, help="Weight for datapoint frequency in representative score")
    parser.add_argument("--w-indeg", type=float, default=1.0, help="Weight for in-degree in representative score")
    parser.add_argument("--no-inferred", action="store_true", help="Do not compute inferred_edges")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print frequency alignment stats (raw vs normalized corpus keys vs graph labels)",
    )
    parser.add_argument(
        "--codebook",
        default=None,
        help="Path to codebook.json (default: agents outputs/data/codebook.json)",
    )
    parser.add_argument(
        "--no-rep-support",
        action="store_true",
        help="Do not boost cluster representative labels with per-cluster review counts",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph) if args.graph else GLOBAL_GRAPH_PATH
    clustered_path = Path(args.clustered) if args.clustered else CLUSTERED_CODES_PATH
    codes_only_path = Path(args.codes_only) if args.codes_only else GT_CODES_ONLY_PATH
    output_path = Path(args.out) if args.out else CLEANED_GLOBAL_GRAPH_PATH
    codebook_path = Path(args.codebook) if args.codebook else CODEBOOK_PATH

    if not graph_path.is_file():
        raise SystemExit(f"Graph not found: {graph_path}")

    result = run_cleanup(
        graph_path=graph_path,
        clustered_codes_path=clustered_path,
        codes_only_path=codes_only_path,
        output_path=output_path,
        min_freq=args.min_freq,
        w_freq=args.w_freq,
        w_indeg=args.w_indeg,
        include_inferred_edges=not args.no_inferred,
        verbose=args.verbose,
        codebook_path=codebook_path,
        use_representative_support=not args.no_rep_support,
    )
    n_nodes = len(result["canonical_nodes"])
    n_edges = len(result["edges"])
    n_inferred = len(result.get("inferred_edges", []))
    n_provenance_entries = sum(len(v) for v in result.get("code_provenance", {}).values())
    print(
        f"Wrote {output_path}: {n_nodes} nodes, {n_edges} direct edges, "
        f"{n_inferred} inferred edges, {n_provenance_entries} provenance mappings."
    )


if __name__ == "__main__":
    main()