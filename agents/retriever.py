from utils import get_vectorstore

def retriever_agent(state):
    query = state["query"]
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=5)
    state["retrieved_docs"] = docs
    return state
