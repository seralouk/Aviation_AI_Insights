from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from prompt import prompt_template

load_dotenv()

def build_rag_chain():
    """
    Builds a RAG QA chain using our preconstructed vectorDB.

    Returns:
        RetrievalQA: A LangChain RAG chain ready for use in question answering over embedded documents.
    """
    # Build a retriever of the top 3 most relevant chunks
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name="gpt-4", temperature=0) # we want to minimize hallucinations so temp=0
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type="stuff",
                                           chain_type_kwargs={"prompt": prompt_template})
    return qa_chain
