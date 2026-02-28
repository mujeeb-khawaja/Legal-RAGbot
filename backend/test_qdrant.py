import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "afghan_doc_local"

def test_qdrant_connection():
    print("--- 🔍 Qdrant Database Test ---")
    
    # 1. Connect
    print(f"1. Connecting to Qdrant at: {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        print("✅ Connection Successful!")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 2. Check Collection
    print(f"2. Checking for collection: '{COLLECTION_NAME}'...")
    collection_names = [c.name for c in collections.collections]
    if COLLECTION_NAME in collection_names:
        print(f"✅ Collection '{COLLECTION_NAME}' found.")
        
        # 3. Get Stats
        info = client.get_collection(COLLECTION_NAME)
        print(f"   - Point count: {info.points_count}")
        print(f"   - Status: {info.status}")
        print(f"   - Vector size: {info.config.params.vectors.size}")
        
    else:
        print(f"❌ Collection '{COLLECTION_NAME}' NOT found.")
        print(f"   Available collections: {collection_names}")
        return

    # 4. Test Search (Optional but recommended)
    print(f"4. Performing sample search test...")
    try:
        # Setup embedding model (same as upload/api)
        CACHE_DIR = os.path.join(os.path.dirname(__file__), "hf_cache")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder=CACHE_DIR)
        
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        retriever = index.as_retriever(similarity_top_k=2)
        results = retriever.retrieve("judicial independence")
        
        if results:
            print(f"✅ Search Successful! Retrieved {len(results)} chunks.")
            for i, res in enumerate(results):
                print(f"   [{i+1}] {res.text[:100]}...")
        else:
            print("⚠️ Search returned NO results. Check if data was uploaded correctly.")
            
    except Exception as e:
        print(f"❌ Search Test Failed: {e}")

if __name__ == "__main__":
    test_qdrant_connection()
