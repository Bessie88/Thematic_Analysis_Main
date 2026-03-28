"""
LOGOS Step 6: Codebook clean-up (tree-aware version).

Post-processing after tree assembly. Prunes low-frequency codes from the
hierarchical tree, removes empty sub-themes, and optionally collapses
sub-themes with too few codes into their parent theme.

Also supports the legacy flat-graph format for backward compatibility.
"""

import argparse
import json
import re
from collections import defaultdict
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
    (handles e.g. '\"frustrating difficulty\"' vs 'frustrating difficulty').
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


def load_graph(path: Path) -> Dict[str, Any]:
    """Load gt_global_graph.json. Returns the full data dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    """Map raw code string and normalized form -> cluster id (string)."""
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


def _prune_tree(
    tree: Dict[str, Any],
    datapoint_freq: Dict[str, int],
    min_freq: int,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """
    Prune low-frequency codes from the tree. Returns (pruned_tree, num_pruned).

    Only leaf nodes (type == "code") are candidates for pruning.
    After pruning codes, remove empty sub-themes. After removing sub-themes,
    if a theme has no children, keep it anyway (it's still a valid cluster label).
    """
    pruned_count = 0

    def prune_node(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal pruned_count
        node_type = node.get("type", "")
        children = node.get("children", [])

        # Leaf code: check frequency
        if node_type == "code":
            name = node.get("name", "")
            freq = datapoint_freq.get(normalize_label(name), 0)
            if freq < min_freq:
                pruned_count += 1
                if verbose:
                    print(f"  [prune] Removing low-freq code: '{name}' (freq={freq})")
                return None
            return node

        # Non-leaf: recurse on children
        new_children = []
        for child in children:
            pruned = prune_node(child)
            if pruned is not None:
                new_children.append(pruned)

        # Remove empty sub-themes (but keep themes/meta-themes even if empty)
        if node_type == "sub_theme" and not new_children:
            return None

        result = dict(node)
        if new_children or node_type != "sub_theme":
            result["children"] = new_children
        else:
            result["children"] = []
        return result

    pruned_tree = prune_node(tree)
    return pruned_tree or tree, pruned_count


def _rebuild_edges_from_tree(tree: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Rebuild flat canonical_nodes and edges from the tree structure."""
    nodes: List[str] = []
    edges: List[Dict[str, str]] = []

    def walk(node: Dict[str, Any]) -> None:
        name = node.get("name", "")
        nodes.append(name)
        for child in node.get("children", []):
            edges.append({"parent": name, "child": child.get("name", "")})
            walk(child)

    walk(tree)
    return sorted(set(nodes)), edges


def _build_provenance(tree: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build provenance map: for each non-code node, list all code descendants."""
    provenance: Dict[str, List[str]] = {}

    def collect_codes(node: Dict[str, Any]) -> List[str]:
        if node.get("type") == "code":
            return [node.get("name", "")]
        codes: List[str] = []
        for child in node.get("children", []):
            codes.extend(collect_codes(child))
        name = node.get("name", "")
        if codes:
            provenance[name] = sorted(set(codes))
        return codes

    collect_codes(tree)
    return provenance


def run_cleanup(
    graph_path: Path,
    clustered_codes_path: Path,
    codes_only_path: Path,
    output_path: Path,
    min_freq: int = 3,
    verbose: bool = False,
    codebook_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run codebook clean-up on the tree-structured global graph.
    Prunes low-frequency codes, removes empty sub-themes, rebuilds edges.
    """
    data = load_graph(graph_path)
    freq_norm = load_datapoint_frequency(clustered_codes_path, codes_only_path)

    # Build datapoint_freq mapping (normalized label -> freq)
    datapoint_freq: Dict[str, int] = {}
    for node_name in data.get("canonical_nodes", []):
        datapoint_freq[normalize_label(node_name)] = freq_norm.get(normalize_label(node_name), 0)

    # Boost cluster representatives with review-level support
    cb_path = codebook_path if codebook_path is not None else CODEBOOK_PATH
    if cb_path.is_file():
        with open(cb_path, encoding="utf-8") as f:
            cb = json.load(f)
        codebook = cb.get("codebook", {})
        cluster_to_codes = cb.get("cluster_to_codes", {})
        if codebook and cluster_to_codes:
            cpr = load_codes_per_review_list(clustered_codes_path, codes_only_path)
            if cpr:
                support = cluster_review_support_counts(cluster_to_codes, cpr)
                for cid, rep_label in codebook.items():
                    nl = normalize_label(rep_label)
                    sup = support.get(str(cid), 0)
                    prev = datapoint_freq.get(nl, 0)
                    datapoint_freq[nl] = max(prev, sup)

    if "tree" in data:
        # New tree format
        tree = data["tree"]
        pruned_tree, n_pruned = _prune_tree(tree, datapoint_freq, min_freq, verbose)

        canonical_nodes, edges = _rebuild_edges_from_tree(pruned_tree)
        provenance = _build_provenance(pruned_tree)

        # Compute node frequencies for the output
        node_frequencies = {}
        for n in canonical_nodes:
            node_frequencies[n] = datapoint_freq.get(normalize_label(n), 0)

        out = {
            "tree": pruned_tree,
            "canonical_nodes": canonical_nodes,
            "merge_groups": [],
            "edges": edges,
            "inferred_edges": [],
            "node_frequencies": node_frequencies,
            "code_provenance": provenance,
        }
    else:
        # Legacy flat format — minimal cleanup (just prune low-freq nodes from edge list)
        nodes = data.get("canonical_nodes", [])
        direct_edges = data.get("edges", [])

        surviving = set()
        for n in nodes:
            freq = datapoint_freq.get(normalize_label(n), 0)
            if freq >= min_freq:
                surviving.add(n)

        edges = [e for e in direct_edges if e["parent"] in surviving and e["child"] in surviving]
        canonical_nodes = sorted(surviving)

        out = {
            "canonical_nodes": canonical_nodes,
            "merge_groups": [],
            "edges": edges,
            "inferred_edges": [],
            "node_frequencies": {n: datapoint_freq.get(normalize_label(n), 0) for n in canonical_nodes},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LOGOS Step 6: Codebook clean-up (post-process global graph / tree).")
    parser.add_argument("--graph", default=None, help="Path to gt_global_graph.json")
    parser.add_argument("--clustered", default=None, help="Path to gt_clustered_codes.json")
    parser.add_argument("--codes-only", default=None, help="Path to gt_codes_only.json")
    parser.add_argument("--out", default=None, help="Output path for cleaned graph JSON")
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help=(
            "Min datapoint frequency to keep a code node (default 3 for ~1000 reviews). "
            "Scale with corpus size: ~100 reviews->2, ~1k->3, ~5k->5, ~50k->10."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print pruning details",
    )
    parser.add_argument(
        "--codebook",
        default=None,
        help="Path to codebook.json (default: agents outputs/data/codebook.json)",
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
        verbose=args.verbose,
        codebook_path=codebook_path,
    )
    n_nodes = len(result["canonical_nodes"])
    n_edges = len(result["edges"])
    has_tree = "tree" in result
    print(
        f"Wrote {output_path}: {n_nodes} nodes, {n_edges} edges"
        f"{' (tree structure preserved)' if has_tree else ''}."
    )


if __name__ == "__main__":
    main()
