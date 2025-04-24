import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Define constant path for knowledge base
DATA_DIR = os.path.join(os.getcwd(), 'data')

def ingest_docs():
    """
    Loads and processes all PDF Annual review docs from the knowledge base directory.
    Splits them into chunks, generates vector embeddings and stores them in a Chroma vector database.

    Inputs:
        None
    Returns:
        None (internally creates KB in chroma_db directory)
    """
    # Only process PDF files
    documents = []
    for file in [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]:
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        documents.extend(loader.load())

    # Chunk splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    # Embed chunks
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist() # save to local disk for later usage by the rag app

    print("\n \n Ingestion done. Documents successfully ingested, chunked and vectorized.\n \n ")

if __name__ == "__main__":
    ingest_docs()
