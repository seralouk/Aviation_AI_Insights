import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Constant path for knowledge base
DATA_DIR = os.path.join(os.getcwd(), 'data')

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
            documents.extend(loader.load())

        # Chunk splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        # Embed chunks
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)
        vectorstore.save_local("faiss_db")  # Save locally
        # Uncomment below if you want to use Chroma instead
        # vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory="chroma_db")

        print("\n Ingestion done. Documents successfully ingested, chunked, and vectorized.\n")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")

if __name__ == "__main__":
    ingest_KB()