import time
from openai import OpenAI

# --- Initialize NVIDIA client ---
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# --- Performance metrics ---
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_response_length": 0,
    "start_time": time.time()
}

# --- Function to chat with NVIDIA USDCode model ---
def chat_with_model(user_input):
    metrics["total_requests"] += 1
    start = time.time()

    try:
        # Create a streaming completion
        stream = client.chat.completions.create(
            model="nvidia/usdcode-llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.2,
            max_tokens=512,
            extra_body={"expert_type": "auto"},
            stream=True,
        )

        reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_piece = chunk.choices[0].delta.content
                reply += content_piece
                print(content_piece, end="", flush=True)  # live streaming output

        print()  # new line after reply finishes
        latency = time.time() - start
        metrics["total_latency"] += latency
        metrics["successful_requests"] += 1
        metrics["total_response_length"] += len(reply)

        # Try to update tokens (if provided in usage metadata)
        # Note: with streaming, tokens usually come at the end
        if hasattr(stream, "usage"):
            usage = getattr(stream, "usage", {})
            metrics["total_input_tokens"] += usage.get("prompt_tokens", 0)
            metrics["total_output_tokens"] += usage.get("completion_tokens", 0)

        return reply.strip()

    except Exception as e:
        metrics["failed_requests"] += 1
        print(f"\n[Error] {e}")
        return None

# --- Function to print performance metrics ---
def print_final_metrics():
    elapsed = time.time() - metrics["start_time"]
    avg_lat = metrics["total_latency"] / metrics["total_requests"] if metrics["total_requests"] else 0
    avg_len = metrics["total_response_length"] / max(1, metrics["successful_requests"])
    success_rate = metrics["successful_requests"] / metrics["total_requests"] * 100 if metrics["total_requests"] else 0

    print("\n📊 Performance Metrics:")
    print(f"  Total Requests: {metrics['total_requests']}")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Avg Latency: {avg_lat:.2f} sec")
    print(f"  Avg Response Length: {avg_len:.1f} chars")
    print(f"  Input Tokens: {metrics['total_input_tokens']}")
    print(f"  Output Tokens: {metrics['total_output_tokens']}")
    print(f"  Total Runtime: {elapsed:.2f} sec")

# --- Run Chatbot Loop ---
print("🤖 NVIDIA USD Code Chatbot")
print("Ask me about OpenUSD or request Python-USD code. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("\nChat ended.")
        print_final_metrics()
        break
    print("AI: ", end="", flush=True)
    chat_with_model(user_input)
