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
"""

import json
import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from agents.core.paths import (
    CLEANED_GLOBAL_GRAPH_PATH,
    CLUSTERED_CODES_PATH,
    GLOBAL_GRAPH_PATH,
    GT_CODES_ONLY_PATH,
)


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
    Compute datapoint_frequency: for each code, number of reviews (datapoints) that contain it.
    Tries clustered_path first (codes_per_review), then codes_only_path.
    """
    freq: Dict[str, int] = defaultdict(int)

    for path in (clustered_path, codes_only_path):
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        codes_per_review = data.get("codes_per_review", [])
        for item in codes_per_review:
            review_id, codes = item[0], item[1]
            seen_in_review: Set[str] = set()
            for c in codes:
                if c not in seen_in_review:
                    seen_in_review.add(c)
                    freq[c] += 1
        if codes_per_review:
            break
    return dict(freq)


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
) -> Dict[str, Any]:
    """
    Run LOGOS Step 6 codebook clean-up and return the cleaned graph dict.
    """
    nodes, merge_groups, direct_edges = load_graph(graph_path)
    datapoint_freq = load_datapoint_frequency(clustered_codes_path, codes_only_path)
    for n in nodes:
        if n not in datapoint_freq:
            datapoint_freq[n] = 0

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
    args = parser.parse_args()

    graph_path = Path(args.graph) if args.graph else GLOBAL_GRAPH_PATH
    clustered_path = Path(args.clustered) if args.clustered else CLUSTERED_CODES_PATH
    codes_only_path = Path(args.codes_only) if args.codes_only else GT_CODES_ONLY_PATH
    output_path = Path(args.out) if args.out else CLEANED_GLOBAL_GRAPH_PATH

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