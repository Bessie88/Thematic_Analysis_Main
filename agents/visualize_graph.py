"""Produce gt_graph.html from gt_global_graph.json (or gt_graph.json)."""
import json
import sys

from paths import GLOBAL_GRAPH_PATH, GRAPH_PATH, GRAPH_HTML_PATH, ensure_output_dirs, display_path


def load_graph():
    """Load global graph if present, else per-cluster graph."""
    if GLOBAL_GRAPH_PATH.is_file():
        with open(GLOBAL_GRAPH_PATH, encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("canonical_nodes", [])
        edges = data.get("edges", [])
        return nodes, edges, "global"
    if GRAPH_PATH.is_file():
        with open(GRAPH_PATH, encoding="utf-8") as f:
            per_cluster = json.load(f)
        nodes = []
        edges = []
        for cid, entry in per_cluster.items():
            nodes.extend(entry.get("canonical_nodes", []))
            for e in entry.get("edges", []):
                edges.append(e)
        nodes = sorted(set(nodes))
        return nodes, edges, "per-cluster"
    return [], [], None


def build_html(nodes: list, edges: list) -> str:
    """Simple HTML listing nodes and edges; can be extended with D3/Cytoscape later."""
    rows = []
    for e in edges:
        rows.append(f"  <tr><td>{e.get('parent', '')}</td><td>→</td><td>{e.get('child', '')}</td></tr>")
    table = "<table border='1'><tr><th>Parent</th><th></th><th>Child</th></tr>\n" + "\n".join(rows) + "\n</table>"
    node_list = "<ul>\n" + "\n".join(f"  <li>{n}</li>" for n in nodes) + "\n</ul>"
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>GT Graph</title></head>
<body>
<h1>Nodes ({len(nodes)})</h1>
{node_list}
<h1>Edges ({len(edges)})</h1>
{table}
</body>
</html>
"""


def main():
    ensure_output_dirs()
    nodes, edges, kind = load_graph()
    if kind is None:
        print("Error: neither gt_global_graph.json nor gt_graph.json found.", file=sys.stderr)
        sys.exit(1)
    html = build_html(nodes, edges)
    GRAPH_HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {display_path(GRAPH_HTML_PATH)} ({kind}: {len(nodes)} nodes, {len(edges)} edges).")


if __name__ == "__main__":
    main()
