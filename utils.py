import streamlit as st
import os

DATA_DIR = os.path.join(os.getcwd(), "data")


def get_relevance_level(score):
    """
    Determine the relevance level based on the cosine similarity and return a colored title.
    Inputs:
        score (float): The cosine similarity score
    Returns:
        str: A colored title according to the relevance level.
    """
    # Colored titles according to the score
    if score >= 0.5:
        level = "🟢 Highly Relevant"
    elif 0.3 <= score < 0.5:
        level = "🟡 Moderately Relevant"
    else:
        level = "🔴 Less Relevant"
    return level


def display_chunk(i, doc, raw_score):
    """
    Display a chunk of text with its relevance score and source information.
    Inputs:
        i (int): The index of the chunk
        doc (Document): The document object containing the chunk
        raw_score (float): The raw distance score (L2 distance)
    Returns:
        None (but alters streamlit app state)
    """
    cosine_score = 1.0 - raw_score
    level = get_relevance_level(cosine_score)

    # Get source and page from metadata
    source = doc.metadata.get("source", "Unknown file")
    page = doc.metadata.get("page", "N/A")

    subtitle = f"{level} (Top {i+1}) Retrieved Chunk (first 500 characters)"
    formatted_text = doc.page_content[:500].replace('\n', ' ').strip()
    styled_text = f"<p style='font-style: italic; border-left: 4px solid #ccc; padding-left: 10px;'>«{formatted_text}»</p>"

    st.markdown(f"**{subtitle}** — *Source: {source.split('/')[-1]}, page {page + 1}*")
    st.markdown(f"**Relevance Score:** {cosine_score:.3f} (higher score = more relevant)")
    
    # cosine similarity value as colored bar for maximum UX
    bar_html = f"""
    <div style="background: linear-gradient(to right, red 0%, orange 25%, yellow 50%, lightgreen 75%, green 100%);
                border-radius: 5px; height: 20px; width: 100%;">
    <div style="background-color: rgba(0, 0, 0, 0); height: 100%;
                width: {cosine_score * 100:.1f}%;
                border-radius: 5px;
                border: 2px solid black;">
    </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 12px;">
    <span>0 (Low)</span>
    <span>1 (High)</span>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
    st.markdown(styled_text, unsafe_allow_html=True)