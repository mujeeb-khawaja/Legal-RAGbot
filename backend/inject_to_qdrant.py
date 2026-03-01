import os
import re
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "hf_cache")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    cache_folder=CACHE_DIR
)

# --- CONFIGURATION ---
KEYWORD_BRIDGE = {
    "2137": "stranger entire property will limit one third non-heir bequest maximum",
    "1325": "maximum period lease guardian administrator three years",
    "1044": "option of sight buyer reject unseen purchase",
}

def clean_and_structurally_chunk(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Clean Table of Contents
    # Make sure your text file doesn't actually have the TOC before Article 1
    text = re.sub(r"Table of Contents.*?(?=Article 1[:\s])", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Split by Article (Flexible Regex: Handles 'Article 1:' and 'Article 1 ')
    # This keeps the "Article X" part in the chunk
    chunks = re.split(r'(?=Article \d+)', text)
    
    legal_documents = []
    for chunk in chunks:
        clean_chunk = chunk.strip()
        if not clean_chunk: continue
        
        # Filter out tiny chunks
        if len(clean_chunk) < 20: continue
            
        # Extract Article Number for Metadata
        match = re.search(r'Article (\d+)', clean_chunk)
        article_num = match.group(1) if match else "Unknown"
        
        # KEYWORD BRIDGE: Inject layman terms for articles with vocabulary gaps
        bridge_keywords = KEYWORD_BRIDGE.get(article_num, "")
        if bridge_keywords:
            clean_chunk = f"Keywords: {bridge_keywords}\n\n{clean_chunk}"
        
        doc = Document(
            text=clean_chunk,
            metadata={
                "article_id": article_num,
                "source": "Civil Code"
            }
        )
        legal_documents.append(doc)
        
    print(f"✅ Structural Parsing: Created {len(legal_documents)} article-chunks.")
    return legal_documents

def upload_hybrid():
    print("--- 🚀 Starting Robust Hybrid Upload ---")
    file_path = os.path.join(os.path.dirname(__file__), "verified_data.txt")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    if client.collection_exists("afghan_doc_local"):
        client.delete_collection("afghan_doc_local")
        print("🗑️ Deleted old collection.")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="afghan_doc_local",
        enable_hybrid=True, 
        batch_size=20
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = clean_and_structurally_chunk(file_path)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ Upload Complete!")

if __name__ == "__main__":
    upload_hybrid()