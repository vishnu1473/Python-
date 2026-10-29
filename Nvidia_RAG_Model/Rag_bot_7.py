# app.py Latest Version with new UI

import os
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
# 1. Page Config & Theme
# -----------------------
st.set_page_config(page_title="RAG Assistant", layout="wide", page_icon="🤖")

# Apply Custom CSS for colors and layout
st.markdown("""
    <style>
            
    .stApp {
        background: linear-gradient(160deg, #d0e8f2, #b0cfe7);
        font-family: 'Segoe UI', sans-serif;
        color: #1a3c5a;
    }

    .block-container {
        padding-top: 2rem;
    }

    .title-style {
        color: #0d47a1; /* strong navy blue */
        font-weight: 800;
        font-size: 2.5rem;
        text-align: center;
        text-shadow: 1px 1px 2px #81d4fa;
        margin-bottom: 1.5rem;
    }

    .sidebar .sidebar-content {
        background: #063970; /* dark navy blue */
        color: #c3d9f5;
    }

    h2, h3 {
        color: #0d47a1;
    }

    .stTextInput > div > div > input {
        border: 2px solid #4fc3f7; /* bright cyan */
        background-color: #e3f2fd;
        color: #0d47a1;
    }

    .stTextInput > div > div > input::placeholder {
        color: #82b1ff;
    }

    .stButton > button {
        color: white;
        background-color: #0288d1;  /* medium blue */
        border-radius: 10px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #01579b; /* darker blue */
    }

    .stSuccess {
        border: 2px solid #0288d1;
        border-radius: 8px;
        background-color: #bbdefb;
        color: #0d47a1;
    }

    .stError {
        border: 2px solid #c62828;
        border-radius: 8px;
        background-color: #ffcdd2;
        color: #b71c1c;
    }

    .stMarkdown hr {
        border-top: 1px dashed #0277bd; /* blue */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-style">🤖 Nvidia Rag Chatbot</div>', unsafe_allow_html=True)

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
# 4. Upload PDF
# -----------------------
st.sidebar.header("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

if uploaded_files and st.session_state.retriever is None:
    st.success(f"Uploaded {len(uploaded_files)} file(s). Processing...")

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

    st.success("✅ PDF(s) processed and FAISS index created!")


# -----------------------
# 5. Speech-to-Text
# -----------------------
st.subheader("🎤 Speak Your Question")
if st.button("🎙️ Record Question"):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2  # default is 0.8 sec
    recognizer.energy_threshold = 300  # lower if your mic is quiet
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=500)
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
    "💬 Ask a question (or follow-up):",
    value=st.session_state.spoken_query,
    key="query_input"
)

if st.button("🚀 Submit Question"):
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
You are a helpful PDF Question Answering Assistant. Be concise and helpful.
It can be up to 10 sentences based on the question.

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

        with st.spinner("Thinking... 🤔"):
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}]
            )

        answer = response.choices[0].message.content.strip()
        st.session_state.chat_history.append((query, answer))
        st.session_state.submit_query = False


# -----------------------
# 7. Display Chat + TTS
# -----------------------
if st.session_state.chat_history:
    st.subheader("💬 Chat History")

    for display_idx, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        actual_idx = len(st.session_state.chat_history) - 1 - display_idx

        st.markdown(f"**🧠 You:** {q}")
        st.markdown(f"<div style='color:#004d40; font-weight:500;'>🤖 Assistant: {a}</div>", unsafe_allow_html=True)

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

        if st.button(f"🔊 Listen to Answer", key=tts_button_key):
            st.audio(st.session_state.audio_paths[audio_key], format="audio/mp3")

        st.markdown("---")
