import os
import re
import time
import json
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# --- CONFIG ---
BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

load_dotenv(dotenv_path=os.path.join(BASE_DIR, '..', '.env'))
API_KEY = os.getenv("API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

app = FastAPI(title="AfghanLegalGPT API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Globals
retriever = None
reranker = None
executor = ThreadPoolExecutor(max_workers=4)
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- HELPER FUNCTIONS ---
def rewrite_query_for_legal_search(query: str) -> str:
    prompt = (
        "You are an expert legal search query generator for the Civil Code of Afghanistan (1977).\n"
        "Your Goal: Translate layman user questions into precise legal terminology and concepts found in the Civil Code to maximize vector retrieval accuracy.\n\n"
        
        "GUIDELINES FOR TRANSLATION:\n"
        "1. IDENTIFY THE LEGAL DOMAIN:\n"
        "   - If Money/Debts/Agreements -> Use 'Obligations', 'Contracts', 'Debt Discharge'.\n"
        "   - If Land/Houses -> Use 'Real Rights', 'Real Estate', 'Ownership', 'Preemption (Shufa)', 'Mortgage'.\n"
        "   - If Family/Death -> Use 'Personal Status', 'Inheritance', 'Will (Wasiyat)', 'Marriage', 'Custody'.\n\n"

        "2. MAP LAYMAN TERMS TO CIVIL CODE JARGON:\n"
        "   - 'Breaking a deal' -> 'Rescission' or 'Dissolution of Contract'.\n"
        "   - 'Cheating/Lying' -> 'Fraud', 'Deception', or 'Lesion'.\n"
        "   - 'Forcing someone' -> 'Duress' or 'Coercion'.\n"
        "   - 'Buying together' or 'Contributing money' -> 'Common Ownership (Shirkat)', 'Company', 'Division of Property'.\n"
        "   - 'Neighbor rights' -> 'Preemption', 'Easement Rights'.\n"
        "   - 'Giving for free' -> 'Donation' or 'Endowment (Waqf)'.\n\n"

        "3. HANDLE FAMILY CONTEXT INTELLIGENTLY:\n"
        "   - If the query is about business, debts, or property purchase between relatives (father/son) BUT no one has died -> FOCUS on 'Contract' and 'Ownership' terms. IGNORE Inheritance terms.\n"
        "   - Only use 'Inheritance' or 'Bequeath' if the query explicitly mentions death or passing away.\n\n"

        f"User Query: {query}\n\n"
        "Output: A single line of high-value legal search keywords and phrases."
    )
    
    models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "arcee-ai/trinity-large-preview:free",
        "google/gemma-3-4b-it:free"
    ]
    
    for model in models:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 100
                }),
                timeout=5
            )
            if response.status_code == 200:
                rewritten = response.json()['choices'][0]['message']['content'].strip()
                print(f"🔄 Rewritten: {rewritten}")
                return rewritten
        except Exception:
            continue
    return query

def call_ai_model(query: str, context: str) -> str:
    prompt = (
        "You are an expert Afghan Legal Assistant.\n"
        "Your Task: Answer the user's question using ONLY the provided Legal Articles.\n"
        "Instructions:\n"
        "1. Read all provided Articles carefully.\n"
        "2. If the user asks about a specific rule (e.g. 'stranger'), look for legal equivalents (e.g. 'non-heir').\n"
        "3. Explain the rule and any exceptions found in the text.\n"
        "4. Cite the Article Number explicitly.\n\n"
        f"--- LEGAL ARTICLES ---\n{context}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        "Answer:"
    )
    
    # Using a slightly more robust model if possible, fallback to Llama
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            data=json.dumps({
                "model": "arcee-ai/trinity-large-preview:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            }),
            timeout=30  # Increased timeout for stability
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
             print(f"AI Error Status: {response.status_code}")
    except Exception as e:
        print(f"AI Connection Error: {e}")
    return "AI Unavailable. Please check the sources below."

@app.on_event("startup")
def startup_event():
    global retriever, reranker
    print("🚀 Starting AfghanLegalGPT (Balanced Accuracy Mode)...")

    # 1. Embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=CACHE_DIR
    )

    # 2. Re-ranker
    reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=512, cache_folder=CACHE_DIR)

    # 3. Hybrid Retriever (BALANCED)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name="afghan_doc_local",
        enable_hybrid=True
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # alpha=0.4: Better balance for Semantic (Vector) and Precision (Keyword)
    # top_k=30: Retrieve a wider pool so we don't miss the article.
    retriever = index.as_retriever(
        similarity_top_k=30, 
        vector_store_query_mode="hybrid", 
        alpha=0.4 
    ) 
    print("✅ System ready!")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    t_start = time.time()
    
    # 1. Rewrite for Search Precision
    search_query = rewrite_query_for_legal_search(request.query)
    
    # 2. Retrieve (Wide Net)
    nodes = retriever.retrieve(search_query)
    if not nodes:
        return QueryResponse(answer="No documents found.", sources=[])
    
    article_chunks = [n.node.get_content() for n in nodes]
    
    # 2. Re-rank (Intelligent Filtering)
    pairs = [[request.query, doc] for doc in article_chunks]
    scores = reranker.predict(pairs)
    ranked_results = sorted(zip(scores, article_chunks), key=lambda x: x[0], reverse=True)
    
    # 3. Context Selection
    # Take top 4 to ensure we catch Rule + Exception (e.g. Partnership death)
    top_chunks = [text for _, text in ranked_results[:4]]
    combined_context = "\n\n".join(top_chunks)
    
    # 4. Generate Answer
    answer = call_ai_model(request.query, combined_context)
    
    print(f"⏱ Total Latency: {time.time()-t_start:.2f}s")
    return QueryResponse(answer=answer, sources=top_chunks)