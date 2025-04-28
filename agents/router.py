from utils import get_llm

def router_agent(state):
    """
    The router agent decides which specialized agent to invoke based on the user query.
    Inputs:
        state: The state of the graph
    Returns:
        state: The updated state with the routing decision
    """
    # Load the LLM
    llm = get_llm()
    query = state["query"]
    prompt = f"""You are a router agent for aviation industry questions.

User query:
"{query}"

Decide if the query is:
- General about the aviation industry
- Specific targeted question targeting a specific aspect of the aviation industry. E.g. economics, regulations, safety, sustainability, passenger experience, airline retail, financial services, etc.

Return one of these Python strings:
- "general_insights"
- "specific_answer"
"""
    route_decision = llm.predict(prompt).strip().replace('"', '').replace("'", '')
    return {"route_decision": route_decision}