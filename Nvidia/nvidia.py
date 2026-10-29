import os
import requests

API_KEY = os.getenv("NVIDIA_API_KEY", "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")  

# --- API endpoint ---
url = "https://integrate.api.nvidia.com/v1/chat/completions"

# --- Initialize conversation history ---
conversation = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

def chat_with_model(user_input):
    # Add user message to history
    conversation.append({"role": "user", "content": user_input})

    payload = {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "messages": conversation,
        "temperature": 0.7,
        "max_tokens": 512
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        # Add assistant reply to history
        conversation.append({"role": "assistant", "content": reply})
        return reply
    else:
        return f"Error: {response.status_code}, {response.text}"


# --- Live chatbot loop ---
print("🤖 NVIDIA Llama-3.3-Nemotron-Super-49B Chatbot")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    reply = chat_with_model(user_input)
    print("AI:", reply, "\n")
