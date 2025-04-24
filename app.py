import streamlit as st
import pandas as pd
from rag_chain import build_rag_chain
from utils import get_relevance_level, display_chunk

# --- Config ---
# Setup landing page of streamlit app
st.set_page_config(page_title="Aviation Industry Insights")
st.title("Aviation Industry Insights - Virtual Consultant")
st.markdown("### Your Aviation Industry Trusted AI Advisor!")
st.markdown("Note: AI-generated content may not always be accurate. Please cross-verify with reliable sources.")

# --- Session state init ---
# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- Display past messages ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- RAG query input ---
# Starting query of the RAG system
query = st.chat_input("What would you like to know about the aviation industry? ")

if query:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Build RAG chain and get response
    rag_chain, vectorstore = build_rag_chain()
    llm_completion = rag_chain(query)

    response = llm_completion["result"]

    # Show assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

        # Retrieve and sort similarity chunks for context display
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
        docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])

        # Show top 3 chunks inside expander
        with st.expander("Show Retrieved Chunks"):
            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                display_chunk(i, doc, score)

    # Save this exchange to CSV download history
    sources = [
        f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'N/A') + 1})"
        for doc in llm_completion["source_documents"]
    ]
    st.session_state.qa_history.append({
        "Question": query,
        "Answer": response,
        "Sources": ", ".join(sources)
    })


# Download button for Q&A history
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("### Download Q&A History")
    st.download_button(
        label="Download QA History",
        data=pd.DataFrame(st.session_state.qa_history).to_csv(index=False).encode("utf-8"),
        file_name="aviation_qa_history.csv",
        mime="text/csv"
    )

# if st.session_state.qa_history:
#     # Build the text content
#     txt_lines = []
#     for entry in st.session_state.qa_history:
#         txt_lines.append(f"Q: {entry['Question']}")
#         txt_lines.append(f"A: {entry['Answer']}")
#         txt_lines.append(f"Sources: {entry['Sources']}")
#         txt_lines.append("-" * 40)  # separator between entries

#     full_txt = "\n".join(txt_lines)

#     st.markdown("---")
#     st.markdown("### Download Q&A History")
#     st.download_button(
#         label="Download QA History as TXT",
#         data=full_txt.encode("utf-8"),
#         file_name="aviation_qa_history.txt",
#         mime="text/plain"
#     )