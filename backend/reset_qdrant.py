import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load keys
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# The name of the collection you created earlier
# Check your upload_word_to_qdrant.py file if you aren't sure. 
# In our last step, we named it "afghan_doc_local"
COLLECTION_NAME = "afghan_doc_local" 

def wipe_database():
    print(f"Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Check if it exists
    collections = client.get_collections()
    exists = any(c.name == COLLECTION_NAME for c in collections.collections)
    
    if exists:
        print(f"⚠️ Found collection '{COLLECTION_NAME}'. Deleting it now...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' has been DELETED. Database is clean.")
    else:
        print(f"❌ Collection '{COLLECTION_NAME}' does not exist. Nothing to delete.")

if __name__ == "__main__":
    wipe_database()