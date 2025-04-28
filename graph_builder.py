from langgraph.graph import StateGraph
from agents.router import router_agent
from agents.retriever import retriever_agent
from agents.trend_analyzer import trend_analyzer_agent
from agents.regulatory_sme import regulatory_expert_agent
from agents.opportunity_identifier import opportunity_identifier_agent
from agents.summarization import summarization_agent
from agents.presentation import presentation_agent

def build_multi_agent_graph():
    graph = StateGraph()
    graph.add_node("router", router_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("trend", trend_analyzer_agent)
    graph.add_node("regulation", regulatory_expert_agent)
    graph.add_node("opportunity", opportunity_identifier_agent)
    graph.add_node("summarizer", summarization_agent)
    graph.add_node("presenter", presentation_agent)
    graph.set_entry_point("router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "trend")
    graph.add_edge("retriever", "regulation")
    graph.add_edge("retriever", "opportunity")
    graph.add_edge("trend", "summarizer")
    graph.add_edge("regulation", "summarizer")
    graph.add_edge("opportunity", "summarizer")
    graph.add_edge("summarizer", "presenter")
    return graph.compile()
