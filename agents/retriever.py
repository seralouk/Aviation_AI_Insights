from utils import get_vectorstore

def retriever_agent(state):
    """
    The main retriever (get chunk matchings from vectorDB based on input query).
    Inputs:
        state: The state of the graph
    Returns:
        state: The updated state with retrieved documents
    """
    query = state["query"]
    # Load the vectorDB and do similarity search
    vectorstore = get_vectorstore()
    docs_with_scores  = vectorstore.similarity_search_with_score(query, k=5)
    docs = [doc for doc, score in docs_with_scores]
    scores = [score for doc, score in docs_with_scores]

    return {"retrieved_docs": docs, "retrieved_scores": scores}
