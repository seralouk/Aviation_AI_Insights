from utils import get_llm

def summarization_agent(state):
    llm = get_llm()
    sections = []
    if "trend_summary" in state:
        sections.append(f"**Trends:** {state['trend_summary']}")
    if "regulation_summary" in state:
        sections.append(f"**Regulations:** {state['regulation_summary']}")
    if "opportunity_summary" in state:
        sections.append(f"**Opportunities:** {state['opportunity_summary']}")
    combined = "\n\n".join(sections)
    prompt = f"Summarize this aviation strategic analysis clearly:\n{combined}"
    state["final_summary"] = llm.predict(prompt)
    return state
