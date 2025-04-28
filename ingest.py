import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load .env OpenAI API key for embeddings
load_dotenv()

# Path to knowledge base documents
DATA_DIR = os.path.join(os.getcwd(), 'data')

# Chuck size and overlap constants
CHUNK_SIZE=500
CHUNK_OVERLAP=100

def ingest_KB():
    """
    Loads and processes all PDF Annual review docs from the knowledge base directory.
    Splits them into chunks, generates vector embeddings, and stores them in a FAISS vector database.

    Inputs:
        None
    Returns:
        None (internally creates KB in faiss_db directory)
    """
    documents = []
    try:
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            print("No PDF files found in the data directory.")
            return

        for file in pdf_files:
            print(f"\n Reading and ingesting knowledge base file: {file}")
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            pages = loader.load()
            # Filter pages with content less than 150 characters
            for page in pages:
                clean_text = page.page_content.strip()
                if len(clean_text) > 150: documents.append(page)

        # Chunk splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(documents)

        # Embed chunks, build vectorstore and save it locally
        vectorstore = FAISS.from_documents(splits, embedding=OpenAIEmbeddings())
        vectorstore.save_local("faiss_db")

        print("\n Ingestion done. Documents successfully ingested, chunked, and vectorized.\n")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")


if __name__ == "__main__":
    ingest_KB()