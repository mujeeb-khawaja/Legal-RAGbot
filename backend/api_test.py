import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from root folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# 🔐 Read API key from environment variable
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is not set")

print("--- 🧠 Starting Reasoning Test (Step 1) ---")

# First API call with reasoning
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AFG-RAG-Reasoning-Test",
    },
    data=json.dumps({
        "model": "arcee-ai/trinity-large-preview:free",
        "messages": [
            {
                "role": "user",
                "content": "How many r's are in the word 'strawberry'?"
            }
        ],
        "reasoning": {"enabled": True}
    })
)

if response.status_code != 200:
    print(f"Error Step 1 (Status {response.status_code}):")
    print(response.text)
    exit()

# Extract the assistant message with reasoning_details
result_json = response.json()
assistant_message = result_json['choices'][0]['message']

print("\n--- 📝 Reasoning Details Received ---")
print(assistant_message.get('reasoning_details', "No reasoning_details found."))
print("\n--- 💬 Content Received ---")
print(assistant_message.get('content'))

# Preserve the assistant message with reasoning_details
messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": assistant_message.get('content'),
        "reasoning_details": assistant_message.get('reasoning_details')  # Pass back unmodified
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]

print("\n--- 🧠 Continuing Reasoning (Step 2) ---")

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AFG-RAG-Reasoning-Test",
    },
    data=json.dumps({
        "model": "openai/gpt-oss-120b:free",
        "messages": messages,  # Includes preserved reasoning_details
        "reasoning": {"enabled": True}
    })
)

print("\nStatus Code Step 2:", response2.status_code)
try:
    final_result = response2.json()
    print(json.dumps(final_result, indent=2))
except Exception:
    print("Raw response Step 2:")
    print(response2.text)