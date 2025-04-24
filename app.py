import streamlit as st
import pandas as pd
from rag_chain import build_rag_chain
from utils import get_relevance_level, display_chunk

# --- Config ---
# Setup landing page of streamlit app
st.set_page_config(page_title="Aviation Industry Insights")
st.title("Aviation Industry Insights - Virtual Consultant")
st.markdown("### Your AI-powered Aviation Industry Trusted Advisor!")

# --- Session state init ---
# Initialize session state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- RAG query input ---
# Starting query of the RAG system
query = st.text_input("What would you like to know about the aviation industry? ")

if query:
    rag_chain, vectorstore = build_rag_chain()
    llm_completion = rag_chain(query)

    st.markdown("### Answer (**AI-generated, use with care**):")
    st.write(llm_completion["result"])

    # Store LLM completion and sources to history
    sources = [
        f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'N/A')+1})"
        for doc in llm_completion["source_documents"]
    ]
    st.session_state.qa_history.append({
        "Question": query,
        "Answer": llm_completion["result"],
        "Sources": ", ".join(sources)
    })
    #st.write("Session state history:", st.session_state.qa_history) # debug 

    # Retrieve & sort relevant chunks for visualization
    # the lower the score, the more relevant the chunk is to the query 
    # (see https://api.python.langchain.com/en/latest/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.similarity_search_with_score)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1]) # lower score = higher similarity

    # Show top 3
    st.markdown("---")
    st.markdown("### Relevant Knowledge Base Extracts Used for Answering")
    for i, (doc, score) in enumerate(docs_with_scores[:3]):  # show top 3
        #print(score, doc.metadata.get("source"))
        display_chunk(i, doc, score)


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