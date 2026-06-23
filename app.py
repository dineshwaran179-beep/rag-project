# app.py - This is the Streamlit Web UI

import streamlit as st
import tempfile
import os
from rag_engine import (
    load_and_process_pdf,
    create_vector_store,
    create_qa_chain,
    get_answer
)

# Page title
st.set_page_config(page_title="📄 Document Q&A", page_icon="📄")
st.title("📄 Ask Questions From Your PDF")
st.write("Upload a PDF and ask anything about it!")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your PDF file", 
    type=["pdf"]
)

# When PDF is uploaded
if uploaded_file is not None:
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Process the PDF (show spinner while loading)
    with st.spinner("📚 Reading and processing your PDF..."):
        chunks = load_and_process_pdf(tmp_path)
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)
    
    st.success(f"✅ PDF processed! {len(chunks)} chunks created.")
    
    # Question input
    question = st.text_input("💬 Ask a question about your PDF:")
    
    # When question is asked
    if question:
        with st.spinner("🤔 Finding answer..."):
            answer = get_answer(qa_chain, question)
        
        st.subheader("📝 Answer:")
        st.write(answer)
    
    # Cleanup temp file
    os.unlink(tmp_path)