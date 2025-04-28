from langgraph.graph import StateGraph
from agents.router import router_agent
from agents.retriever import retriever_agent
from agents.general_insights_agent import general_insights_agent
from agents.specialized_topic_agent import specific_answer_agent
from agents.presentation import presentation_agent
from typing import TypedDict, List, Optional

class AviationState(TypedDict):
    query: str
    retrieved_docs: List
    retrieved_scores: List
    route_decision: str
    final_summary: Optional[str]
    final_answer: Optional[str]

def build_multi_agent_graph():
    """
    Build a simple multi-agent graph: router, retriever, general/specific agents, presenter.
    """
    graph = StateGraph(AviationState)

    # Nodes
    graph.add_node("router", router_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("general_insights", general_insights_agent)
    graph.add_node("specific_answer", specific_answer_agent)
    graph.add_node("presenter", presentation_agent)
    graph.set_entry_point("router")

    # Edges
    graph.add_edge("router", "retriever")

    # Based on router decision select agent
    graph.add_conditional_edges(source="retriever",
                                path=lambda x: x["route_decision"],
                                path_map={
                                    "general_insights": "general_insights",
                                    "specific_answer": "specific_answer"})

    # Pass to presenter in all cases
    graph.add_edge("general_insights", "presenter")
    graph.add_edge("specific_answer", "presenter")

    return graph.compile()