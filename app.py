import streamlit as st
import pandas as pd
from rag_chain import build_rag_chain


# Setup landing page of streamlit app
st.set_page_config(page_title="Aviation Industry Insights", layout="wide")
st.title("Aviation Industry Insights Virtual Consultant")
st.markdown("### Your AI-powered Aviation Industry Trusted Advisor!")

# Initialize session state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Starting query of the RAG system
query = st.text_input("What would you like to know about the aviation industry? ")

if query:
    rag_chain = build_rag_chain()
    llm_completion = rag_chain(query)

    st.markdown("### Answer (**AI-generated**):")
    st.write(llm_completion["result"])

    # Store Q&A and sources
    sources = [
        f"{doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', 'N/A')})"
        for doc in llm_completion["source_documents"]
    ]
    st.session_state.qa_history.append({
        "Question": query,
        "Answer": llm_completion["result"],
        "Sources": "; ".join(sources)
    })

    st.markdown("---")
    st.markdown("### Retrieved Knowledge Base Chunks:")
    for i, doc in enumerate(llm_completion["source_documents"][:3]):
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "N/A")
        text_preview = doc.page_content[:500]
        highlight = "🟢 Most Relevant" if i == 0 else f"Chunk {i+1} (First 250 characters)"

        # reformat chunks before displaying in the streamlit app
        st.markdown(f"**{highlight}** — *{source.split('/')[-1]}, page {page+1}*")
        formatted_text = text_preview.replace('\n', '  ')
        st.markdown(f"> {formatted_text}")


# Download button for Q&A history
if st.session_state.qa_history:
    df = pd.DataFrame(st.session_state.qa_history)
    st.markdown("---")
    st.markdown("### Download Q&A History")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="aviation_qa_history.csv",
        mime="text/csv"
    )

