import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from root folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- 1. CONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- 2. SETUP MODELS ---
# Force a clean local cache for the embedding model
CACHE_DIR = os.path.join(os.path.dirname(__file__), "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR

# Use the official HuggingFace name with local cache folder
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder=CACHE_DIR)

# 2. Chunk Settings
Settings.chunk_size = 512
Settings.chunk_overlap = 50

def upload_word_file():
    print("--- 🚀 Starting Upload Process (Local Embeddings) ---")
    
    file_path = os.path.join(os.path.dirname(__file__), "verified_data.docx")
    if not os.path.exists(file_path):
        print(f"❌ Error: '{file_path}' not found.")
        return

    # 3. CONNECT TO QDRANT
    print("1. Connecting to Qdrant Cloud...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # We use a NEW collection name because local models produce different numbers than Google
        vector_store = QdrantVectorStore(client=client, collection_name="afghan_doc_local")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 4. READ FILE
    print("2. Reading verified_data.docx...")
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    # 5. UPLOAD
    print("3. Generating Embeddings (Locally) & Uploading...")
    
    try:
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print("\n✅ SUCCESS! Data uploaded using Local Embeddings.")
    except Exception as e:
        print(f"\n❌ Upload Failed: {e}")

if __name__ == "__main__":
    upload_word_file()