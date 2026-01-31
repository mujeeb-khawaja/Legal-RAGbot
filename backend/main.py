import os
import json
import hashlib
import requests
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- CONFIGURATION ---
API_KEY = os.getenv("API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- SETUP CACHE DIRECTORIES ---
BASE_DIR = os.path.dirname(__file__)
HF_CACHE = os.path.join(BASE_DIR, "hf_cache")
QUERY_CACHE_DIR = os.path.join(BASE_DIR, "query_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(QUERY_CACHE_DIR, exist_ok=True)

# Force the library to use our clean project-level cache
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_HOME"] = HF_CACHE

# --- SETUP MODELS ---
print(f"Initializing embedding model (Disk Cache: {HF_CACHE})...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder=HF_CACHE)

# --- LLM ENGINE (Manual Requests to Bypass Buggy Library) ---
def call_ai_model(query: str, top_chunk: str):
    """Uses Liquid LFM 2.5 Instruct to turn the top chunk into a professional legal answer."""
    prompt = (
        "You are an expert Afghan Legal Assistant.\n"
        "Task: Rewrite the following legal text into a professional, human-like answer.\n"
        f"--- SOURCE TEXT ---\n{top_chunk}\n\n"
        f"--- USER QUESTION ---\n{query}\n\n"
        "Output Format:\n"
        "1. Professional Summary (in a helpful, easy-to-read tone)\n"
        "2. DIRECT QUOTE: (Include the specific sentence from the source that supports this answer)\n"
    )
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "liquid/lfm-2.5-1.2b-instruct:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000 # Prevents the '65535 tokens' credit error
            }),
            timeout=60
        )
        
        if response.status_code != 200:
            return f"AI Error: {response.status_code} - {response.text}"
            
        result = response.json()
        if 'choices' not in result:
            return f"AI Error: No choices in response. JSON: {json.dumps(result)}"
            
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"AI Exception: {str(e)}"

# --- CACHE HELPERS ---
def get_cache_path(query: str):
    query_id = hashlib.md5(query.strip().lower().encode()).hexdigest()
    return os.path.join(QUERY_CACHE_DIR, f"{query_id}_ai.json")

def get_cached_ai_response(query: str):
    path = get_cache_path(query)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # If the cached answer is an error, ignore it
            answer = data.get("answer", "")
            if "AI Error:" in answer or "AI Exception:" in answer:
                return None
            return data
    return None

def save_to_cache(query: str, answer: str, sources: list):
    path = get_cache_path(query)
    data = {"query": query, "answer": answer, "sources": sources}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main_chat():
    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name="afghan_doc_local")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = index.as_retriever(similarity_top_k=5)
    
    print("\n--- 🇦🇫 Afghan Legal Expert (RAG System) ---")
    print("(Type 'exit' to quit)")
    
    while True:
        q = input("\n🔍 Query: ")
        if q.lower() in ["exit", "quit", "bye"]: break
        if not q.strip(): continue
            
        # 1. Try Cache
        cached = get_cached_ai_response(q)
        if cached:
            print("⚡ [CACHE HIT] Loaded verified result from disk...")
            print(f"\n{cached['answer']}")
            continue

        # 2. Retrieve Path
        print("Searching legal database for evidence...")
        nodes = retriever.retrieve(q)
        if not nodes:
            print("No legal documents found matching your query.")
            continue
            
        # 3. Handle Chunks
        source_list = [n.node.get_content() for n in nodes]
        top_chunk = source_list[0] # Take only the 1st chunk for the AI
        
        # 4. Generate with AI
        print("AI is rewriting the answer using the top source...")
        answer = call_ai_model(q, top_chunk)
        
        print("\n" + "="*50)
        print("🤖 AI EXPERT ANSWER")
        print("="*50)
        print(f"\n{answer}")
        
        print("\n\n" + "="*50)
        print("🔍 FULL SOURCE REFERENCES (Top 5 Chunks)")
        print("="*50)
        for i, text in enumerate(source_list, 1):
            print(f"\n[REFERENCE {i}]")
            print(text)
            print("-" * 30)

        # 5. Save Cache (ONLY if it's not an error)
        if "AI Error:" not in answer and "AI Exception:" not in answer:
            save_to_cache(q, answer, source_list)
            print("\n💾 Saved to verified cache.")
        else:
            print("\n⚠️ Results NOT saved to cache due to AI error.")

if __name__ == "__main__":
    main_chat()
