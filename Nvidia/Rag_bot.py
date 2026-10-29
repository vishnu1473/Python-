# app.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from openai import OpenAI
import tempfile
from langchain_community.document_loaders import PyPDFLoader


# -----------------------
# 1. Page Config
# -----------------------
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📘 CNC Troubleshooting RAG Assistant (PDF Upload)")

# -----------------------
# 2. API Key
# -----------------------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")

# -----------------------
# 3. Upload PDF
# -----------------------
uploaded_files = st.file_uploader(
    "Upload CNC Troubleshooting PDF(s)", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s). Processing...")

    docs = []
    for uploaded_file in uploaded_files:
        # ✅ Save each uploaded file into a real temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # ✅ Use the temp file path with PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Embeddings
    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        nvidia_api_key=NVIDIA_API_KEY
    )

    # Build FAISS index
    vdb = FAISS.from_documents(split_docs, embedder)
    retriever = vdb.as_retriever(search_kwargs={"k": 2})

    st.success("✅ PDF processed and FAISS index created!")

    # -----------------------
    # 4. Query Box
    # -----------------------
    query = st.text_input("Ask a question about CNC machines:")

    if query.strip():  # ✅ Runs immediately when Enter is pressed
        # Retrieve docs
        docs = retriever.get_relevant_documents(query)

        # Build prompt
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Answer the following based on the context:\n\n{context}\n\nQuestion: {query}"

        # Call LLM
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )

        response = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": prompt}]
        )

        # Show only the answer
        st.subheader("💡 Answer")
        st.write(response.choices[0].message.content)
