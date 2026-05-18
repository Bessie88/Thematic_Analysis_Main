"""Research report generation from the global graph via a dedicated OpenAI-compatible LLM (e.g. Mistral on SGLang)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI

from .paths import ensure_output_dirs
from .prompts import research_report_prompt
from .skills import llm_invoke_with_skill
from .utils import log_step, remove_think_tags

# Align with ~8k context: rough 4 chars/token, leave room for prompt template + RQ text.
REPORT_CONTEXT_TOKENS = 8000
REPORT_COMPLETION_TOKENS = 1024
# Graph-only budget inside the user message (prompt wrapper + question consume the rest).
REPORT_GRAPH_MAX_CHARS = max(4000, int(0.55 * REPORT_CONTEXT_TOKENS * 4) - 3500)

_TRUNK_EDGES_NOTE = (
    "\nNote: edges truncated for length; synthesis uses listed nodes and partial edges.\n"
)
_TRUNK_NODES_NOTE = "\nNote: nodes truncated for length; only a prefix of themes is included.\n"

_DEFAULT_API_BASE = "http://localhost:8000/v1"
_DEFAULT_MODEL = "llm"


def _default_api_base() -> str:
    return os.environ.get("REPORT_OPENAI_BASE", _DEFAULT_API_BASE).rstrip("/")


def _default_model_name() -> str:
    return os.environ.get("REPORT_MODEL_NAME", _DEFAULT_MODEL)


def _render_tree_node(node: Dict[str, Any], indent: int = 0) -> str:
    """Recursively render a tree node into indented text for the LLM."""
    prefix = "  " * indent
    name = node.get("name", "Unnamed")
    node_type = node.get("type", "")
    children = node.get("children", [])

    if node_type == "code":
        return f"{prefix}* {name}\n"

    # For non-leaf nodes, show name and type
    type_label = f" [{node_type}]" if node_type and node_type != "root" else ""
    child_count = _count_leaves(node)
    count_label = f" ({child_count} codes)" if children and node_type != "code" else ""
    line = f"{prefix}- {name}{type_label}{count_label}\n"

    for child in children:
        line += _render_tree_node(child, indent + 1)
    return line


def _count_leaves(node: Dict[str, Any]) -> int:
    """Count leaf (code) nodes under a tree node."""
    children = node.get("children", [])
    if not children:
        return 1 if node.get("type") == "code" else 0
    return sum(_count_leaves(c) for c in children)


def build_graph_text_for_llm(graph_path: Path, max_chars: int) -> str:
    """
    Load gt_global_graph.json and build a compact text block for the LLM.
    Supports both the new tree format and the legacy flat edge-list format.
    """
    with open(graph_path, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # New tree format
    if "tree" in data:
        tree = data["tree"]
        nodes: List[str] = list(data.get("canonical_nodes") or [])
        edges = data.get("edges") or []

        header = f"Thematic hierarchy: {len(nodes)} nodes, {len(edges)} edges (strict tree)\n\n"
        header += "THEME HIERARCHY:\n"
        tree_text = header + _render_tree_node(tree)

        if len(tree_text) > max_chars:
            tree_text = tree_text[: max_chars - 80] + "\n...[hierarchy truncated to char limit]\n"
        return tree_text

    # Legacy flat format (backward compat)
    nodes = list(data.get("canonical_nodes") or [])
    raw_edges = data.get("edges") or []
    e_list: List[Dict[str, Any]] = [e for e in raw_edges if isinstance(e, dict)]

    file_edge_count = len(e_list)

    def pack(nn: List[str], ee: List[Dict[str, Any]]) -> str:
        node_lines = "\n".join(f"- {x}" for x in nn) if nn else "(none)"
        edge_lines = "\n".join(f"  {e.get('parent')} -> {e.get('child')}" for e in ee)
        return "\n".join(
            [
                f"graph_file_nodes: {len(nodes)}",
                f"graph_file_edges: {file_edge_count}",
                "",
                "NODES:",
                node_lines,
                "",
                "EDGES (parent -> child):",
                edge_lines if edge_lines else "(none listed)",
            ]
        )

    note_reserve = len(_TRUNK_EDGES_NOTE) + len(_TRUNK_NODES_NOTE) + 64
    budget = max(512, max_chars - note_reserve)

    n_work = list(nodes)
    e_work = list(e_list)

    while len(pack(n_work, e_work)) > budget and e_work:
        e_work.pop()
    edges_truncated = len(e_work) < file_edge_count

    while len(pack(n_work, e_work)) > budget and n_work:
        n_work.pop()
    nodes_truncated = len(n_work) < len(nodes)

    out = pack(n_work, e_work)
    if edges_truncated:
        out += _TRUNK_EDGES_NOTE
    if nodes_truncated:
        out += _TRUNK_NODES_NOTE
    if len(out) > max_chars:
        out = out[: max_chars - 80] + "\n...[payload hard truncated to char limit]\n"
    return out


def generate_research_report(
    research_question: str,
    graph_path: Path,
    out_path: Path,
    *,
    api_base: str | None = None,
    model: str | None = None,
    max_tokens: int = REPORT_COMPLETION_TOKENS,
) -> None:
    """
    Call the report LLM and write markdown to out_path.
    Does not use the global tools.llm (Qwen); builds a dedicated ChatOpenAI client.
    """
    ensure_output_dirs()
    openai_base, mname = resolve_report_client_config(api_base, model)

    graph_text = build_graph_text_for_llm(graph_path, REPORT_GRAPH_MAX_CHARS)
    if "edges truncated for length" in graph_text or "nodes truncated for length" in graph_text:
        log_step(
            "RESEARCH_REPORT_GRAPH",
            "Graph payload was truncated to fit context budget.",
        )

    llm = ChatOpenAI(
        model=mname,
        openai_api_key="EMPTY",
        openai_api_base=openai_base,
        temperature=0,
        max_tokens=max_tokens,
    )
    prompt = research_report_prompt(research_question, graph_text)
    raw = llm_invoke_with_skill(llm, "research_report", prompt, phase="report") or ""
    cleaned = remove_think_tags(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned.strip() + "\n")


def resolve_report_client_config(
    api_base: str | None,
    model: str | None,
) -> Tuple[str, str]:
    """Merge CLI overrides with env defaults."""
    base = (api_base or _default_api_base()).rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base, model or _default_model_name()
