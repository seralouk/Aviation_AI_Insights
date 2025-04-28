import streamlit as st
import pandas as pd
from graph_builder import build_multi_agent_graph
from utils import display_chunk, get_vectorstore


# Config: Setup landing page of streamlit app
st.set_page_config(page_title="Aviation Industry Insights")
st.title("Aviation Industry Insights - Virtual Consultant")
st.markdown("### Your Aviation Industry Trusted AI Advisor!")
st.markdown("Note: AI-generated content may not always be accurate. Please cross-verify with reliable sources.")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# Display chat past messages if exist
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# RAG query input: Starting query of system
query = st.chat_input("What would you like to know about the aviation industry? ")

# Load vectorstore
vectorstore = get_vectorstore()

# Capture user inputs
if query:
    # Clear old chunks
    st.session_state.last_chunks = []

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Build RAG chain and get LLM completion on input query
    lang_graph = build_multi_agent_graph()
    lang_graph.get_graph().print_ascii()
    result = lang_graph.invoke({"query": query})
    print(result['route_decision']) # for debugging and testing
    response = result["final_answer"]

    # Show assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

        # Retrieve and sort similarity chunks for context display
        retrieved_docs = result.get("retrieved_docs", [])
        retrieved_scores = result.get("retrieved_scores", [])

        # Pair documents and scores
        st.session_state.last_chunks = list(zip(retrieved_docs[:3], retrieved_scores[:3]))

        # # Show top 3 chunks inside expander
        # with st.expander("Show Retrieved Chunks"):
        #     for i, (doc, score) in enumerate(docs_with_scores[:3]):
        #         display_chunk(i, doc, score)

    # Save chat exchange to CSV download history
    sources = [
        f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'N/A')}), Score: {1.0 - score:.3f}"
        for doc, score in st.session_state.last_chunks
    ]
    st.session_state.qa_history.append({
        "Question": query,
        "Answer": response,
        "Sources": ", ".join(sources)
    })

# Show retrieved chunks
if st.session_state.last_chunks:
    with st.expander("Show Top Retrieved Chunks"):
        for i, (doc, score) in enumerate(st.session_state.last_chunks):
            display_chunk(i, doc, score)
        
# Download button for Q&A history
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("### Download Chat History")
    st.download_button(
        label="Download CSV file",
        data=pd.DataFrame(st.session_state.qa_history).to_csv(index=False).encode("utf-8"),
        file_name="aviation_qa_history.csv",
        mime="text/csv"
    )