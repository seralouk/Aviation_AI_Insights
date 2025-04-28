from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from prompt import prompt_template, prompt_template_CoT

# Load API key
load_dotenv()


def build_rag_chain():
    """
    Builds a RAG QA chain using our preconstructed faiss vectorDB.

    Returns:
        tuple: (RetrievalQA chain, vectorstore)
    """
    # Build retriever from vectorstore
    vectorstore = FAISS.load_local(folder_path="faiss_db", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name="gpt-4", temperature=0) # temp=0 to minimize hallucinations & maximize groudedness
    # Build the RAG chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type="stuff",
                                           chain_type_kwargs={"prompt": prompt_template_CoT})
    return qa_chain, vectorstore
