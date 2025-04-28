def presentation_agent(state):
    sources = set(doc.metadata.get('source', 'Unknown') for doc in state["retrieved_docs"])
    state["final_answer"] = f"""### Aviation Industry Strategic Report

{state['final_summary']}

---
Sources: {', '.join(sources)}""" 
    return state
