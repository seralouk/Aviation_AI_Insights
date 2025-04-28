from utils import get_llm

def router_agent(state):
    llm = get_llm()
    query = state["query"]
    prompt = f"""Decide which agents should process the following question:
"{query}"
Options: ["Trend Analyzer", "Regulatory Expert", "Opportunity Identifier"]
Return a Python list of selected agent names.""" 
    answer = llm.predict(prompt)
    state["selected_agents"] = eval(answer)
    return state
