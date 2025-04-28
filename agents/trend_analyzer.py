from utils import get_llm

def trend_analyzer_agent(state):
    if "Trend Analyzer" not in state["selected_agents"]:
        return state
    llm = get_llm()
    context = "\n".join(doc.page_content for doc in state["retrieved_docs"])
    prompt = f"Summarize aviation industry trends from the following context:\n{context}"
    state["trend_summary"] = llm.predict(prompt)
    return state
