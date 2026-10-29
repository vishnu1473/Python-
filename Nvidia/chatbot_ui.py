import os
import time
import requests
import streamlit as st

# --- API Key ---
API_KEY = os.getenv("NVIDIA_API_KEY", "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")

# --- API endpoint ---
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# --- Init session state ---
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "system", "content": "You are a helpful AI assistant."}]
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_latency": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_response_length": 0,
        "start_time": time.time()
    }

# --- Chat function ---
def chat_with_model(user_input):
    m = st.session_state.metrics
    m["total_requests"] += 1
    st.session_state.conversation.append({"role": "user", "content": user_input})

    payload = {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "messages": st.session_state.conversation,
        "temperature": 0.7,
        "max_tokens": 512
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    start = time.time()
    response = requests.post(API_URL, headers=headers, json=payload)
    latency = time.time() - start
    m["total_latency"] += latency

    if response.status_code == 200:
        data = response.json()
        reply = data["choices"][0]["message"]["content"]

        # metrics update
        m["successful_requests"] += 1
        usage = data.get("usage", {})
        m["total_input_tokens"] += usage.get("prompt_tokens", 0)
        m["total_output_tokens"] += usage.get("completion_tokens", 0)
        m["total_response_length"] += len(reply)

        st.session_state.conversation.append({"role": "assistant", "content": reply})
        return reply
    else:
        m["failed_requests"] += 1
        return f"Error: {response.status_code}, {response.text}"

# --- Metrics ---
def get_metrics():
    m = st.session_state.metrics
    elapsed = time.time() - m["start_time"]
    return {
        "Total Requests": m["total_requests"],
        "Success Rate (%)": round((m["successful_requests"]/m["total_requests"]*100), 2) if m["total_requests"] else 0,
        "Avg Latency (s)": round(m["total_latency"]/m["total_requests"], 2) if m["total_requests"] else 0,
        "Input Tokens": m["total_input_tokens"],
        "Output Tokens": m["total_output_tokens"],
        "Runtime (s)": round(elapsed, 2)
    }

# --- Streamlit UI ---
st.set_page_config(page_title="NVIDIA Chatbot", page_icon="🤖")
st.title("🤖 NVIDIA Llama Chatbot")

# Sidebar
st.sidebar.header("📊 Metrics")
for k, v in get_metrics().items():
    st.sidebar.write(f"**{k}:** {v}")
if st.sidebar.button("Reset Chat"):
    st.session_state.conversation = [{"role": "system", "content": "You are a helpful AI assistant."}]
    st.session_state.metrics = {
        "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
        "total_latency": 0.0, "total_input_tokens": 0, "total_output_tokens": 0,
        "total_response_length": 0, "start_time": time.time()
    }
    st.sidebar.success("Chat reset!")

# Display chat messages
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").markdown(msg["content"])

# Input box
user_input = st.chat_input("Say something...")
if user_input:
    st.chat_message("user").markdown(user_input)
    reply = chat_with_model(user_input)
    st.chat_message("assistant").markdown(reply)
