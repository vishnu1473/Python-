# app.py Latest Version which accepts pptx and xlsx
#Need to update of diff other files- csv,
#make pdf into png for clip

import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from openai import OpenAI
import tempfile
import pyttsx3
import av
import speech_recognition as sr
import pandas as pd
from pptx import Presentation
from langchain.schema import Document   

# -----------------------
# 1. Page Config
# -----------------------
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("ASM AI Assistant")

# -----------------------
# 2. API Key
# -----------------------
NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "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"
)

# -----------------------
# 3. Initialize Session State
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "spoken_query" not in st.session_state:
    st.session_state.spoken_query = ""

if "audio_paths" not in st.session_state:
    st.session_state.audio_paths = {}

if "submit_query" not in st.session_state:
    st.session_state.submit_query = False

# -----------------------
# 4. Upload Files
# -----------------------
from langchain.schema import Document  # Add this import at the top of your file
import pandas as pd
from pptx import Presentation

uploaded_files = st.file_uploader(
    "Upload PDF, Excel or PPT files", type=["pdf", "xlsx", "pptx"], accept_multiple_files=True
)

if uploaded_files and st.session_state.retriever is None:
    st.success(f"Uploaded {len(uploaded_files)} file(s). Processing...")

    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())

        elif file_ext == "xlsx":
            df = pd.read_excel(tmp_path)
            text = df.astype(str).apply(' '.join, axis=1).str.cat(sep='\n')
            docs.append(Document(page_content=text, metadata={}))

        elif file_ext == "pptx":
            prs = Presentation(tmp_path)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            docs.append(Document(page_content=text, metadata={}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        nvidia_api_key=NVIDIA_API_KEY
    )

    vdb = FAISS.from_documents(split_docs, embedder)
    st.session_state.retriever = vdb.as_retriever(search_kwargs={"k": 2})

    st.success("✅ Files processed and FAISS index created!")

# -----------------------
# 5. Speech-to-Text
# -----------------------
st.subheader("🎤 Speak Your Question")
if st.button("Record Question"):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio_data = recognizer.listen(source, timeout=4, phrase_time_limit=100)
            st.info("Recognizing...")
            text = recognizer.recognize_google(audio_data)
            st.session_state.spoken_query = text
            st.success(f"Recognized: {text}")
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"STT service error: {e}")

# -----------------------
# 6. Query Box (Chat Mode)
# -----------------------
query = st.text_input(
    "Ask a question (or follow-up):",
    value=st.session_state.spoken_query,
    key="query_input"
)

if st.button("Submit Question"):
    st.session_state.submit_query = True

if st.session_state.submit_query and query.strip() and st.session_state.retriever:
    retriever = st.session_state.retriever
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    if not context.strip() or len(context.strip()) < 50:
        answer = "I’m sorry, the information you requested is not available in the uploaded document(s)."
    else:
        history_text = ""
        for q_hist, a_hist in st.session_state.chat_history:
            history_text += f"User: {q_hist}\nAssistant: {a_hist}\n"

        prompt = f"""
You are a helpful PDF, Excel and PPT Question Answering Assistant. Be concise and helpful.
Give answers in 4-5 sentences max and only give answers in english.

Conversation so far:
{history_text}

Context from documents:
{context}

User: {query}
Assistant:
"""

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[{"role": "user", "content": prompt}]
            )

        answer = response.choices[0].message.content.strip()
        st.session_state.chat_history.append((query, answer))
        st.session_state.submit_query = False

# -----------------------
# 7. Display Chat + TTS
# -----------------------
if st.session_state.chat_history:
    st.subheader("💬 Chat")

    for display_idx, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        actual_idx = len(st.session_state.chat_history) - 1 - display_idx

        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")

        tts_button_key = f"tts_button_{actual_idx}"
        audio_key = f"audio_{actual_idx}"

        if audio_key not in st.session_state.audio_paths:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[0].id)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                audio_path = tmp_file.name

            engine.save_to_file(a, audio_path)
            engine.runAndWait()
            st.session_state.audio_paths[audio_key] = audio_path

        if st.button(f"🔊 Listen to Answer {actual_idx}", key=tts_button_key):
            st.audio(st.session_state.audio_paths[audio_key], format="audio/mp3")

        st.markdown("---")
