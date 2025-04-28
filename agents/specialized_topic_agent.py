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
    
    prompt = f"""You are working as an experienced Data Science Consultant for an entrepreneur who is exploring investment opportunities in the aviation industry.

The entrepreneur has limited knowledge of the aviation industry.  
Your task is to analyze the provided context and deliver a structured, professional response that:
- Explains the evolution of the aviation industry across key thematic areas
- Offers investment-relevant recommendations tailored to the entrepreneur

**Focus the insights and recommendations mainly on the topic asked, but still structure under the standard categories if applicable.**

**Focus your analysis and suggestions under these key thematic areas** (only include those that apply):

- Economics
- Regulations
- Environment & Sustainability
- Safety
- Passenger Experience
- Modern Airline Retailing
- Financial Services

**Instructions:**
- Use bullet points (not numbered lists) grouped under each relevant category.
- After the insights, create a clear "Investment Recommendations" section listing 3–5 actionable investment ideas.
- Keep the tone professional, concise, and investor-friendly — as if writing a strategic briefing to a non-expert.
- Avoid technical jargon, abbreviations, or acronyms. Always spell terms fully (e.g., say "Sustainable Aviation Fuel" instead of "SAF").
- If the information is insufficient, state it clearly and suggest logical next steps or areas for further research.

---

**Context:**
{context}

**Question:**
{question}

**Answer:**
"""
    
    return {"final_summary": llm.predict(prompt)}
