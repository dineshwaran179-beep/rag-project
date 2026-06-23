# rag_engine.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load the API key from .env file
load_dotenv()

def load_and_process_pdf(pdf_path):
    """STEP 1: Load PDF and split into chunks"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """STEP 2: Convert chunks to embeddings and store in FAISS"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_qa_chain(vector_store):
    """STEP 3: Connect vector store with Groq LLM"""
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
    )
    return qa_chain

def get_answer(qa_chain, question):
    """STEP 4: Get answer for user question"""
    response = qa_chain.invoke({"query": question})
    return response["result"]