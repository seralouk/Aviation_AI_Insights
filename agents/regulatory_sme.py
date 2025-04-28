from utils import get_llm

def regulatory_expert_agent(state):
    if "Regulatory Expert" not in state["selected_agents"]:
        return state
    llm = get_llm()
    context = "\n".join(doc.page_content for doc in state["retrieved_docs"])
    prompt = f"Summarize key regulatory changes and impacts from the following context:\n{context}"
    state["regulation_summary"] = llm.predict(prompt)
    return state
