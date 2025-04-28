
def presentation_agent(state):
    """
    Presentation agent: formats the final report for presentation.
    Inputs:
        state: The state of the graph
    Returns:
        state: The updated state with formatted final report
    """
    sources = set(doc.metadata.get('source', 'Unknown') for doc in state["retrieved_docs"])
    final_answer = f"""### Aviation Industry Insights

                            {state['final_summary']}

                    """ 
    return {"final_answer": final_answer}
