import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openrouter import OpenRouter
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import requests
import json

# Load env vars from root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- CONFIGURATION ---
API_KEY = os.getenv("API_KEY") or os.getenv("API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not API_KEY:
    print("❌ ERROR: No API Key found in .env (tried API_KEY and API_KEY)")

os.environ["API_KEY"] = API_KEY or ""

# --- SETUP APP ---
app = FastAPI(title="AfghanLegalGPT API")

# Allow React (Frontend) to talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace * with your specific URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL VARIABLES ---
retriever = None

# --- AI ENGINE (Bypassing Library Bug) ---
def call_ai_model(query: str, top_chunk: str):
    """Uses Liquid LFM 2.5 Instruct to turn the top chunk into a professional legal answer."""
    prompt = (
        "You are an expert Afghan Legal Assistant.\n"
        "Task: Rewrite the following legal text into a professional, human-like answer.\n"
        f"--- SOURCE TEXT ---\n{top_chunk}\n\n"
        f"--- USER QUESTION ---\n{query}\n\n"
        "Output Format:\n"
        "1. Professional Summary (in a helpful, easy-to-read tone)\n\n"
        "2. DIRECT QUOTE:\n"
        "> [Insert exact sentence from source here]\n"
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
                "max_tokens": 1000
            }),
            timeout=60
        )
        if response.status_code != 200:
            return f"Error from AI Provider: {response.status_code} - {response.text}"
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Exception in AI call: {str(e)}"

# --- INITIALIZATION (Runs once when server starts) ---
@app.on_event("startup")
def startup_event():
    global query_engine
    print("🚀 Starting Server... Loading Models...")

    # 1. Force a clean cache directory for the embedding model
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "hf_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR

    # 1. Load Local Embeddings (Using library with absolute cache folder)
    print(f"Initializing embedding model (Cache: {CACHE_DIR})...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder=CACHE_DIR)

    # Load LLM
    print("Loading LLM...")
    if not API_KEY:
        print("❌ ERROR: API_KEY is None at startup")
    
    # Check if this is an OpenRouter key
    if API_KEY and API_KEY.startswith("sk-or-"):
        print("Detected OpenRouter Key. Using DeepSeek R1 (Free) via OpenRouter.")
        Settings.llm = OpenRouter(
            api_key=API_KEY,
            model="deepseek/deepseek-r1-0528:free"
        )
    else:
        print("Using standard GoogleGenAI library.")
        Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=API_KEY)

    # 3. Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name="afghan_doc_local")
    
    # 4. Create Retriever (Not Query Engine)
    global retriever
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = index.as_retriever(similarity_top_k=5)
    print("✅ System Ready!")

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# --- ENDPOINTS ---
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="System is initializing")

    try:
        # 1. Retrieve relevant chunks manually
        print(f"Retrieving context for: {request.query}")
        nodes = retriever.retrieve(request.query)
        
        if not nodes:
            return QueryResponse(
                answer="I could not find any legal documents matching your query.",
                sources=[]
            )

        # 2. Extract information
        source_texts = [n.node.get_content() for n in nodes]
        top_chunk = source_texts[0]
        
        # 3. Generate AI Answer
        print("Drafting expert response...")
        expert_answer = call_ai_model(request.query, top_chunk)

        return QueryResponse(
            answer=expert_answer,
            sources=source_texts
        )
    except Exception as e:
        print(f"🔥 Retrieval Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "running"}
