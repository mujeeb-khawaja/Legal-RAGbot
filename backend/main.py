import os
import re
import time
import json
import requests
from dotenv import load_dotenv

# --- CACHE CONFIG ---
BASE_DIR = os.path.dirname(__file__)
HF_CACHE = os.path.join(BASE_DIR, 'hf_cache')
os.makedirs(HF_CACHE, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE
os.environ['HF_HOME'] = HF_CACHE
os.environ['SENTENCE_TRANSFORMERS_HOME'] = HF_CACHE

# Llama Index & Qdrant
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# Load env
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '..', '.env'))

API_KEY = os.getenv('API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Setup Embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name='BAAI/bge-small-en-v1.5', 
    cache_folder=HF_CACHE
)

# --- HELPER FUNCTIONS ---
def generate_hyde_query(user_query: str) -> str:
    prompt = (
        "You are an expert Afghan Legal Scholar.\n"
        "Write a SHORT hypothetical legal paragraph (2-3 sentences) that would answer this question.\n"
        "Use formal Civil Code vocabulary: 'non-heir', 'one-third', 'dissolution', 'preemption', 'rescission', etc.\n"
        "Do NOT worry about being correct. Focus on using the RIGHT LEGAL WORDS.\n\n"
        f"Question: {user_query}\n\n"
        "Hypothetical Legal Paragraph:"
    )
    models = [
        "google/gemma-3-12b-it:free",
        "google/gemma-3-4b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "arcee-ai/trinity-large-preview:free"
    ]
    for model in models:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 150
                }),
                timeout=8
            )
            if response.status_code == 200:
                hyde_text = response.json()['choices'][0]['message']['content'].strip()
                print(f"👻 HyDE: {hyde_text[:100]}...")
                return hyde_text
        except Exception:
            continue
    return user_query  # fallback

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
    try:
        response = requests.post(
            url='https://openrouter.ai/api/v1/chat/completions',
            headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
            data=json.dumps({
                'model': 'arcee-ai/trinity-large-preview:free',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.1,
                'max_tokens': 300,
            }),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"⚠️ AI Gen failed: {e}")
    return "AI generation unavailable. Most relevant provision provided below."

def format_answer(best_article: str, score: float) -> str:
    article_match = re.search(r'(Article \d+)', best_article)
    article_ref = article_match.group(1) if article_match else "Relevant Provision"
    return (
        f"📋 {article_ref}\n"
        f"{'─'*50}\n\n"
        f"{best_article}\n\n"
        f"{'─'*50}\n"
        f"Confidence Score: {score:.2f}"
    )

def main_chat():
    print('Connecting to Qdrant Cloud (Hybrid Search)...')
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name='afghan_doc_local',
        enable_hybrid=True
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    retriever = index.as_retriever(
        similarity_top_k=30, 
        vector_store_query_mode="hybrid", 
        alpha=0.4
    )

    print(f'Loading fast re-ranker ({RERANKER_MODEL_ID})...')
    reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=512, cache_folder=HF_CACHE)

    print("\n--- Afghan Legal Expert (RAG Hybrid Optimized) ---")
    print("(Type 'exit' to quit)")

    while True:
        q = input('\nQuery: ')
        if q.lower() in ['exit', 'quit', 'bye']:
            break
        if not q.strip():
            continue

        t_start = time.time()
        
        # 1. HyDE Transformation
        search_query = generate_hyde_query(q)
        
        # 2. Retrieve (Hybrid)
        print(f"🔍 Searching using HyDE query...")
        nodes = retriever.retrieve(search_query)
        if not nodes:
            print('No documents found.')
            continue
        
        # 2. Re-rank (Fast MiniLM)
        article_chunks = [n.node.get_content() for n in nodes]
        pairs = [[q, doc] for doc in article_chunks]
        
        t_rerank = time.time()
        scores = reranker.predict(pairs)
        print(f"⏱ Re-ranking {len(pairs)} articles took: {time.time()-t_rerank:.4f}s")
        
        ranked_results = sorted(zip(scores, article_chunks), key=lambda x: x[0], reverse=True)
        best_score = float(ranked_results[0][0])
        best_article = ranked_results[0][1]

        # 3. Fast Path
        FAST_PATH_THRESHOLD = 0.5
        if best_score > FAST_PATH_THRESHOLD:
            print(f"🚀 FAST PATH TRIGGERED (Score: {best_score:.2f})")
            answer = format_answer(best_article, best_score)
        else:
            print(f"🧠 SLOW PATH TRIGGERED (Score: {best_score:.2f})")
            combined_context = "\n\n".join([text for _, text in ranked_results[:3]])
            answer = call_ai_model(q, combined_context)

        print('\n' + '=' * 50)
        print('LEGAL EXPERT RESPONSE')
        print('=' * 50)
        print(f'\n{answer}')
        print(f'\nTOTAL LATENCY: {time.time()-t_start:.2f}s')

if __name__ == '__main__':
    main_chat()
