# app.py Latest Version with metrics

import os
import time  # For measuring durations
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

# -----------------------
# 1. Page Config
# -----------------------
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📘 PDF RAG Assistant (Upload & Ask Questions)")

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
    st.session_state.chat_history = []   # [(user, assistant), ...]

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "spoken_query" not in st.session_state:
    st.session_state.spoken_query = ""

if "audio_paths" not in st.session_state:
    st.session_state.audio_paths = {}

if "submit_query" not in st.session_state:
    st.session_state.submit_query = False

if "embedding_time" not in st.session_state:
    st.session_state.embedding_time = 0

if "last_input_tokens" not in st.session_state:
    st.session_state.last_input_tokens = 0

if "last_output_tokens" not in st.session_state:
    st.session_state.last_output_tokens = 0

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0

# -----------------------
# 4. Upload PDF
# -----------------------
uploaded_files = st.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

if uploaded_files and st.session_state.retriever is None:
    st.success(f"Uploaded {len(uploaded_files)} file(s). Processing...")

    start_embedding_time = time.time()  # Start timer

    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        nvidia_api_key=NVIDIA_API_KEY
    )

    vdb = FAISS.from_documents(split_docs, embedder)
    st.session_state.retriever = vdb.as_retriever(search_kwargs={"k": 2})

    embedding_time = time.time() - start_embedding_time  # End timer
    st.session_state.embedding_time = embedding_time  # Store time

    st.success("✅ PDF(s) processed and FAISS index created!")

# -----------------------
# 5. Speech-to-Text
# -----------------------
st.subheader("🎤 Speak Your Question")
if st.button("Record Question"):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio_data = recognizer.listen(source, timeout=3, phrase_time_limit=100)
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
        st.session_state.last_input_tokens = len(query.split())
        st.session_state.last_output_tokens = len(answer.split())
        st.session_state.last_response_time = 0
    else:
        history_text = ""
        for q_hist, a_hist in st.session_state.chat_history:
            history_text += f"User: {q_hist}\nAssistant: {a_hist}\n"

        prompt = f"""
You are a helpful PDF Question Answering Assistant. Be concise and helpful.
Give answers in 4-5 sentences max.

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

        start_response_time = time.time()  # Start timer

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[{"role": "user", "content": prompt}]
            )

        end_response_time = time.time()  # End timer
        response_time = end_response_time - start_response_time

        answer = response.choices[0].message.content.strip()

        # Estimate token counts using word count × 1.3
        input_tokens = int(len(prompt.split()) * 1.3)
        output_tokens = int(len(answer.split()) * 1.3)

        st.session_state.last_input_tokens = input_tokens
        st.session_state.last_output_tokens = output_tokens
        st.session_state.last_response_time = response_time

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

# -----------------------
# 8. Display Metrics
# -----------------------
st.subheader("📊 Metrics")

embedding_time = st.session_state.get("embedding_time", 0)
st.write(f"**PDF Embedding Time:** {embedding_time:.2f} seconds")

input_tokens = st.session_state.get("last_input_tokens", 0)
output_tokens = st.session_state.get("last_output_tokens", 0)
response_time = st.session_state.get("last_response_time", 0)

st.write(f"**Last Query Input Tokens:** {input_tokens}")
st.write(f"**Last Query Output Tokens:** {output_tokens}")
st.write(f"**Last Query Response Time:** {response_time:.2f} seconds")
