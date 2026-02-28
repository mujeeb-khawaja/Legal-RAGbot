import os
import json
import requests
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# Load env vars from root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv('API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

if not API_KEY:
    print('ERROR: API_KEY not found in .env')

os.environ['API_KEY'] = API_KEY or ''

app = FastAPI(title='AfghanLegalGPT API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Restrict in production
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

retriever = None
reranker = None


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
                'model': 'google/gemma-3-12b-it:free',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.2,
                'max_tokens': 1000,
            }),
            timeout=60,
        )
        if response.status_code != 200:
            return f'Error from AI Provider: {response.status_code} - {response.text}'

        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f'Exception in AI call: {str(e)}'


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


def find_best_article(query: str, chunks: list) -> tuple:
    """
    From all retrieved chunks, extract individual articles,
    re-rank them, and return the single most relevant one.
    Returns (best_article_text, score)
    """
    global reranker
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


@app.on_event('startup')
def startup_event():
    global retriever, reranker
    print('Starting server and loading models...')

    cache_dir = os.path.join(os.path.dirname(__file__), 'hf_cache')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir

    print(f'Initializing embedding model (cache: {cache_dir})...')
    Settings.embed_model = HuggingFaceEmbedding(
        model_name='BAAI/bge-small-en-v1.5',
        cache_folder=cache_dir,
    )

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
    print('System ready.')


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="System is initializing")

    try:
        # Step 1: Rewrite query
        rewritten_query = rewrite_query_for_legal_search(request.query)
        print(f"Rewritten: {rewritten_query}")

        # Step 2: Retrieve 10 chunks
        nodes = retriever.retrieve(rewritten_query)
        if not nodes:
            return QueryResponse(answer="No documents found.", sources=[])

        source_texts = [n.node.get_content() for n in nodes]

        # Step 3: Re-rank chunks, take top 3
        if reranker:
            pairs = [[request.query, chunk] for chunk in source_texts]
            scores = reranker.predict(pairs)
            ranked_chunks = sorted(zip(scores, source_texts), key=lambda x: x[0], reverse=True)
            top_3_chunks = [text for _, text in ranked_chunks[:3]]
        else:
            top_3_chunks = source_texts[:3]

        # Step 4: Article-level re-ranking within top 3 chunks
        best_article, confidence = find_best_article(request.query, top_3_chunks)

        # Step 5: Format clean answer
        answer = format_answer(request.query, best_article, confidence)

        return QueryResponse(answer=answer, sources=top_3_chunks)

    except Exception as e:
        print(f"🔥 Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
def health_check():
    return {'status': 'running'}
