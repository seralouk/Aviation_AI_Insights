from utils import get_llm

def opportunity_identifier_agent(state):
    if "Opportunity Identifier" not in state["selected_agents"]:
        return state
    llm = get_llm()
    context = "\n".join(doc.page_content for doc in state["retrieved_docs"])
    prompt = f"Identify business opportunities from the following context:\n{context}"
    state["opportunity_summary"] = llm.predict(prompt)
    return state
