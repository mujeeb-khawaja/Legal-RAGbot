import os
import re
import time
import json
import requests
import argparse
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
                timeout=12
            )
            if response.status_code == 200:
                hyde_text = response.json()['choices'][0]['message']['content'].strip()
                return hyde_text
        except Exception:
            continue
    return user_query

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
                'max_tokens': 500,
            }),
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception:
        pass
    return "AI generation failed."

TEST_CASES = [
    # Initial 10
    {"q": "What is the legal age for marriage for boys and girls?", "expected_source": "70"},
    {"q": "Under what conditions is a man allowed to marry more than one wife?", "expected_source": "86"},
    {"q": "Until what age does a mother keep custody of her son and daughter?", "expected_source": "249"},
    {"q": "How much inheritance does a husband get if his wife dies and they have children?", "expected_source": "2007"},
    {"q": "Can a person make a will for their entire property to go to a stranger?", "expected_source": "2137"},
    {"q": "If I buy a stolen item from a market in good faith, do I have to give it back to the original owner?", "expected_source": "2290"},
    {"q": "What is the maximum period for a lease if the person administering the property is not the owner (e.g. a guardian)?", "expected_source": "1325"},
    {"q": "Does a partnership end if one of the partners dies?", "expected_source": "1250"},
    {"q": "What is the right of Preemption (Shufa)?", "expected_source": "2213"},
    {"q": "Is preemption allowed if the property was given as a gift or inheritance?", "expected_source": "2226"},
    
    # 20 New
    {"q": "If a man and woman are engaged and they exchange gifts, but then one of them cancels the engagement, can they get the gifts back?", "expected_source": "65"},
    {"q": "If a wife donates her dowry (Mahr) to her husband but they get divorced before the marriage is consummated, can the husband claim half of it back?", "expected_source": "111"},
    {"q": "Can a mother demand payment (wages) for breastfeeding her own child?", "expected_source": "230"},
    {"q": "If a divorced woman is pregnant, when does her waiting period (Iddah) end?", "expected_source": "206"},
    {"q": "What happens to the right of custody if the mother marries a stranger (someone not related to the child)?", "expected_source": "245"},
    {"q": "How many years must a person possess a piece of land to claim ownership of it through 'Lapse of Time' (Adverse Possession)?", "expected_source": "2279"},
    {"q": "If someone finds buried ancient relics or treasure on their own private land, who owns it?", "expected_source": "1988"},
    {"q": "Can a person own public property, like a bridge or public park, by possessing it for a long time?", "expected_source": "482"},
    {"q": "If a person plants crops on another person's land without permission, who do the crops belong to?", "expected_source": "2208"},
    {"q": "If I own the upper floor of a building and you own the lower floor, do I have the right to live on the roof of your floor?", "expected_source": "1979"},
    {"q": "In a sale contract, if the buyer buys something without seeing it (Option of Sight), when can he reject it?", "expected_source": "1044"},
    {"q": "What is the limitation period for filing a lawsuit regarding a defect in a purchased item?", "expected_source": "1107"},
    {"q": "If an architect designs a building plan but does not supervise the construction, is he liable for defects in the building?", "expected_source": "1492"},
    {"q": "If a person hires a worker (Labour Contract) but does not specify the wage, how is the wage determined?", "expected_source": "1535"},
    {"q": "Can an agent appointed to buy a specific house buy it for himself instead?", "expected_source": "1583"},
    {"q": "If a person intentionally kills their father, can they inherit from him?", "expected_source": "1999"},
    {"q": "If two people die in the same accident (e.g., a fire or drowning) and we don't know who died first, do they inherit from each other?", "expected_source": "1996"},
    {"q": "How is the inheritance divided if the deceased leaves behind a daughter and no sons?", "expected_source": "2008"},
    {"q": "Who has a stronger right to Preemption (Shufa): a partner in the property or a neighbor?", "expected_source": "2221"},
    {"q": "Can a neighbor open a window looking directly into my house if the distance is less than one meter?", "expected_source": "1929"},
]

def run_evaluation(start, end):
    print(f'Worker starting range {start}-{end}...')
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name='afghan_doc_local', enable_hybrid=True)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = index.as_retriever(similarity_top_k=30, vector_store_query_mode="hybrid", alpha=0.4)
    reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=512, cache_folder=HF_CACHE)

    subset = TEST_CASES[start:end]
    results = []
    
    for i, test in enumerate(subset):
        idx = start + i + 1
        print(f"[{idx}/{len(TEST_CASES)}] Testing: {test['q']}")
        t_start = time.time()
        
        # HyDE
        hyde_query = generate_hyde_query(test['q'])
        
        # Retrieve
        nodes = retriever.retrieve(hyde_query)
        if not nodes:
            results.append({"q": test['q'], "pass": False, "reason": "No retrieval", "ai_answer": "N/A", "found_source": "N/A"})
            continue
            
        # Re-rank
        article_chunks = [n.node.get_content() for n in nodes]
        pairs = [[test['q'], doc] for doc in article_chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, article_chunks, nodes), key=lambda x: x[0], reverse=True)
        
        top_context = "\n\n".join([text for _, text, _ in ranked[:4]])
        ai_answer = call_ai_model(test['q'], top_context)
        
        found_expected_source = test['expected_source'] in ai_answer or any(test['expected_source'] in str(n.node.metadata.get('article_id', '')) for _, _, n in ranked[:3])
        
        results.append({
            "q": test['q'],
            "expected_source": test['expected_source'],
            "ai_answer": ai_answer,
            "found_expected_source": found_expected_source,
            "top_ranked_article": ranked[0][2].node.metadata.get('article_id', 'Unknown'),
            "latency": time.time() - t_start
        })
        
    filename = f'evaluation_results_{start}_{end}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Worker {start}-{end} complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=30)
    args = parser.parse_args()
    run_evaluation(args.start, args.end)
