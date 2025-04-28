from utils import get_llm

def specific_answer_agent(state):
    """
    The specific answer agent generates a structured strategic briefing based on the retrieved documents.
    Inputs:
        state: The state of the graph
    Returns:
        insights: A dictionary containing the structured insights
    """
    if state.get("route_decision") != "specific_answer":
        return {}

    llm = get_llm()
    context = "\n".join(doc.page_content for doc in state["retrieved_docs"])
    question = state["query"]
    
    prompt = f"""You are working as a Data Science Consultant for an entrepreneur exploring investment opportunities in the aviation industry.

Analyze the provided context and deliver a focused strategic briefing tailored to the specific topic asked.

Organize your answer under relevant themes among:
- Economics
- Regulations
- Environment & Sustainability
- Safety
- Passenger Experience
- Modern Airline Retailing
- Financial Services

Context:
{context}

Question:
{question}

Answer:
"""
    
    return {"final_summary": llm.predict(prompt)}
