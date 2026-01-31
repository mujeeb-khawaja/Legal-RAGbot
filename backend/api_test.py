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

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",

        # Optional but recommended by OpenRouter
        "HTTP-Referer": "http://localhost",      # or your site URL
        "X-Title": "AFG-RAG-Test",                # any project name
    },
    data=json.dumps({
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [
            {
                "role": "user",
                "content": "What are the rules regarding judicial independence of afghanistan?"
            }
        ],
        "temperature": 0
    })
)

print("Status Code:", response.status_code)

try:
    result = response.json()
    print(json.dumps(result, indent=2))
except Exception:
    print("Raw response:")
    print(response.text)