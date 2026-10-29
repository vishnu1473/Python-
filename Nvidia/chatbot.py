import time
import streamlit as st
from openai import OpenAI

# --- Initialize NVIDIA client ---
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

"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

# --- Performance metrics ---
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

metrics = st.session_state.metrics

# --- Function to chat with model ---
def chat_with_model(user_input):
    metrics["total_requests"] += 1
    start = time.time()

    try:
        stream = client.chat.completions.create(
            model="nvidia/usdcode-llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.2,
            max_tokens=512,
            extra_body={"expert_type": "auto"},
            stream=True,
        )

        reply = ""
        response_placeholder = st.empty()

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                piece = chunk.choices[0].delta.content
                reply += piece
                response_placeholder.markdown(reply)

        latency = time.time() - start
        metrics["total_latency"] += latency
        metrics["successful_requests"] += 1
        metrics["total_response_length"] += len(reply)

        return reply.strip()

    except Exception as e:
        metrics["failed_requests"] += 1
        st.error(f"⚠️ Error: {e}")
        return None


# --- Function to show performance metrics ---
def show_metrics():
    elapsed = time.time() - metrics["start_time"]
    avg_lat = metrics["total_latency"] / metrics["total_requests"] if metrics["total_requests"] else 0
    avg_len = metrics["total_response_length"] / max(1, metrics["successful_requests"])
    success_rate = metrics["successful_requests"] / metrics["total_requests"] * 100 if metrics["total_requests"] else 0

    st.subheader("📊 Performance Metrics")
    st.write(f"**Total Requests:** {metrics['total_requests']}")
    st.write(f"**Success Rate:** {success_rate:.2f}%")
    st.write(f"**Avg Latency:** {avg_lat:.2f} sec")
    st.write(f"**Avg Response Length:** {avg_len:.1f} chars")
    st.write(f"**Input Tokens:** {metrics['total_input_tokens']}")
    st.write(f"**Output Tokens:** {metrics['total_output_tokens']}")
    st.write(f"**Total Runtime:** {elapsed:.2f} sec")


# --- Streamlit UI ---
st.set_page_config(page_title="NVIDIA USDCode Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 NVIDIA USD Code Chatbot")
st.markdown("Ask me about **OpenUSD** or request **Python-USD code**.")

# Conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        reply = chat_with_model(user_input)
        if reply:
            st.session_state.history.append(("assistant", reply))

# Show history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# Sidebar for metrics
with st.sidebar:
    show_metrics()
