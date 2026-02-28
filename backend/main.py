import os
import json
import hashlib
import requests
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv('API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

BASE_DIR = os.path.dirname(__file__)
HF_CACHE = os.path.join(BASE_DIR, 'hf_cache')
QUERY_CACHE_DIR = os.path.join(BASE_DIR, 'query_cache')
os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(QUERY_CACHE_DIR, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = HF_CACHE
os.environ['HF_HOME'] = HF_CACHE

print(f'Initializing embedding model (disk cache: {HF_CACHE})...')
Settings.embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_folder=HF_CACHE)


def call_ai_model(query: str, context_text: str) -> str:
    """Generate a legal answer using the re-ranked context."""
    prompt = (
        'You are an expert Afghan Legal Assistant.\\n'
        'You have been given multiple legal text chunks from Afghan legal materials.\\n'
        'Task: search across ALL chunks and identify the most relevant article(s) to answer directly.\\n'
        'Ignore irrelevant chunks.\\n'
        f'--- LEGAL CONTEXT (MULTIPLE CHUNKS) ---\\n{context_text}\\n\\n'
        f'--- USER QUESTION ---\\n{query}\\n\\n'
        'Output Format:\\n'
        '1. Professional Summary (directly answering the question)\\n\\n'
        '2. DIRECT QUOTE:\\n'
        '> [Insert exact sentence from the most relevant article]\\n\\n'
        '3. Source: Article [number if available]\\n'
    )

    try:
        response = requests.post(
            url='https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
            },
            data=json.dumps({
                'model': 'meta-llama/llama-3.2-3b-instruct:free',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.2,
                'max_tokens': 1000,
            }),
            timeout=60,
        )

        if response.status_code != 200:
            return f'AI Error: {response.status_code} - {response.text}'

        result = response.json()
        if 'choices' not in result:
            return f'AI Error: No choices in response. JSON: {json.dumps(result)}'

        return result['choices'][0]['message']['content']
    except Exception as e:
        return f'AI Exception: {str(e)}'


def rewrite_query_for_legal_search(user_query: str) -> str:
    """Rewrite user query in legal terminology to improve retrieval recall."""
    prompt = (
        'You are an Afghan legal search assistant.\\n'
        'Rewrite the user question using formal Afghan Civil Code legal terminology.\\n'
        'Return only one rewritten query sentence.\\n'
        f'User Question: {user_query}'
    )

    try:
        response = requests.post(
            url='https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
            },
            data=json.dumps({
                'model': 'meta-llama/llama-3.2-3b-instruct:free',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.1,
                'max_tokens': 100,
            }),
            timeout=30,
        )

        if response.status_code != 200:
            return user_query

        result = response.json()
        rewritten = result['choices'][0]['message']['content'].strip()
        return rewritten or user_query
    except Exception:
        return user_query


def extract_articles(chunk: str) -> list:
    """Split a chunk into individual articles."""
    # Split on "Article [number]:" pattern
    parts = re.split(r'(?=Article \d+[:\s])', chunk)
    # Clean and filter empty parts
    articles = [p.strip() for p in parts if p.strip() and re.match(r'Article \d+', p.strip())]
    return articles


def find_best_article(query: str, chunks: list, reranker) -> tuple:
    """
    From all retrieved chunks, extract individual articles,
    re-rank them, and return the single most relevant one.
    Returns (best_article_text, score)
    """
    all_articles = []
    
    # Step 1: Split every chunk into individual articles
    for chunk in chunks:
        articles = extract_articles(chunk)
        all_articles.extend(articles)
    
    if not all_articles or not reranker:
        return chunks[0] if chunks else "No content found.", 0.0
    
    print(f"📋 Total individual articles extracted: {len(all_articles)}")
    
    # Step 2: Re-rank all individual articles against the query
    pairs = [[query, article] for article in all_articles]
    scores = reranker.predict(pairs)
    
    # Step 3: Sort and return the best one
    ranked = sorted(zip(scores, all_articles), key=lambda x: x[0], reverse=True)
    
    best_score = ranked[0][0]
    best_article = ranked[0][1]
    
    print(f"🏆 Best article score: {best_score:.3f}")
    print(f"🏆 Best article preview: {best_article[:80]}...")
    
    return best_article, float(best_score)


def format_answer(query: str, best_article: str, confidence: float) -> str:
    """Format the best article into a clean readable answer."""
    # Extract article number for display
    article_match = re.search(r'(Article \d+)', best_article)
    article_ref = article_match.group(1) if article_match else "Relevant Article"
    
    confidence_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.3 else "Low"
    
    return (
        f"📋 Most Relevant Legal Provision\\n"
        f"{'='*50}\\n\\n"
        f"{best_article}\\n\\n"
        f"{'='*50}\\n"
        f"Source: {article_ref} | Confidence: {confidence_label} ({confidence:.2f})"
    )


def get_cache_path(query: str):
    query_id = hashlib.md5(query.strip().lower().encode()).hexdigest()
    return os.path.join(QUERY_CACHE_DIR, f'{query_id}_ai.json')


def get_cached_ai_response(query: str):
    path = get_cache_path(query)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            answer = data.get('answer', '')
            if 'AI Error:' in answer or 'AI Exception:' in answer:
                return None
            return data
    return None


def save_to_cache(query: str, answer: str, sources: list):
    path = get_cache_path(query)
    data = {'query': query, 'answer': answer, 'sources': sources}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main_chat():
    print('Connecting to Qdrant Cloud...')
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name='afghan_doc_local')
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = index.as_retriever(similarity_top_k=10)

    print('Loading re-ranker model...')
    if CrossEncoder is None:
        print('WARNING: sentence-transformers import failed. Running without re-ranker.')
        reranker = None
    else:
        reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)

    print("\\n--- Afghan Legal Expert (RAG System) ---")
    print("(Type 'exit' to quit)")

    while True:
        q = input('\\nQuery: ')
        if q.lower() in ['exit', 'quit', 'bye']:
            break
        if not q.strip():
            continue

        cached = get_cached_ai_response(q)
        if cached:
            print('[CACHE HIT] Loaded verified result from disk...')
            print(f"\\n{cached['answer']}")
            continue

        rewritten_query = rewrite_query_for_legal_search(q)
        print(f'Searching legal database using: {rewritten_query}')
        nodes = retriever.retrieve(rewritten_query)
        if not nodes:
            print('No legal documents found matching your query.')
            continue

        source_list = [n.node.get_content() for n in nodes]
        if reranker:
            pairs = [[q, chunk] for chunk in source_list]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, source_list), key=lambda x: x[0], reverse=True)
            top_chunks = [text for _, text in ranked[:3]]
            print(f'Re-ranker top score: {ranked[0][0]:.3f} | Bottom score: {ranked[-1][0]:.3f}')
        else:
            top_chunks = source_list[:3]

        # Step 4: Article-level re-ranking within top 3 chunks
        best_article, confidence = find_best_article(q, top_chunks, reranker)

        # Step 5: Format clean answer
        answer = format_answer(q, best_article, confidence)

        print('\\n' + '=' * 50)
        print('AI EXPERT ANSWER')
        print('=' * 50)
        print(f'\\n{answer}')

        print('\\n\\n' + '=' * 50)
        print('FULL SOURCE REFERENCES (TOP 3 RERANKED CHUNKS)')
        print('=' * 50)
        for i, text in enumerate(top_chunks, 1):
            print(f'\\n[REFERENCE {i}]')
            print(text)
            print('-' * 30)

        if 'AI Error:' not in answer and 'AI Exception:' not in answer:
            save_to_cache(q, answer, top_chunks)
            print('\\nSaved to verified cache.')
        else:
            print('\\nResults NOT saved to cache due to AI error.')


if __name__ == '__main__':
    main_chat()
