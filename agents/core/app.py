from langgraph.graph import END, StateGraph

from .state import GTState, agent_node, router, tool_node

graph = StateGraph(GTState)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", router, {"tool": "tool", END: END, "agent": "agent"})
graph.add_edge("tool", "agent")

app = graph.compile()  # gt_agents.py calls this with app.invoke(state, config={...})
