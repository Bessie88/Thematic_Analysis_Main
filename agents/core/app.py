import sqlite3

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from .codebook_review_gate import codebook_review_gate
from .paths import GRAPH_CHECKPOINT_PATH, ensure_output_dirs
from .state import GTState, agent_node, router, router_after_tool, tool_node
from .utils import log_step

ensure_output_dirs()


def _make_checkpointer():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(GRAPH_CHECKPOINT_PATH), check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        log_step(
            "CHECKPOINTER_FALLBACK",
            "langgraph-checkpoint-sqlite not installed; using InMemorySaver "
            "(interrupt resume will not survive process restart — use manual mode on HPC).",
        )
        return InMemorySaver()


_checkpointer = _make_checkpointer()

graph = StateGraph(GTState)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)
graph.add_node("codebook_review_gate", codebook_review_gate)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    router,
    {"tool": "tool", "codebook_review": "codebook_review_gate", END: END, "agent": "agent"},
)
graph.add_conditional_edges(
    "tool",
    router_after_tool,
    {"agent": "agent", "codebook_review": "codebook_review_gate"},
)
graph.add_edge("codebook_review_gate", "agent")

app = graph.compile(
    checkpointer=_checkpointer,
    interrupt_before=["codebook_review_gate"],
)
