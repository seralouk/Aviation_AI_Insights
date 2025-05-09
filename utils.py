import streamlit as st

def get_relevance_level(score):
    """
    Determine the relevance level based on the score and return a colored title.
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

    source = doc.metadata.get("source", "Unknown file")
    page = doc.metadata.get("page", "N/A")
    page_display = page + 1 if isinstance(page, int) else page

    subtitle = f"{level} (Top {i+1}) Retrieved Chunk (first 500 characters)"
    formatted_text = doc.page_content[:500].replace('\n', ' ').strip()
    styled_text = f"<p style='font-style: italic; border-left: 4px solid #ccc; padding-left: 10px;'>«{formatted_text}»</p>"

    st.markdown(f"**{subtitle}** — *Source: {source.split('/')[-1]}, page {page_display}*")
    st.markdown(f"**Relevance Score:** {cosine_score:.3f} (higher score = more relevant)")
    st.progress(cosine_score)
    st.markdown(styled_text, unsafe_allow_html=True)