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

# Setup Embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    cache_folder=CACHE_DIR
)

def clean_and_contextual_chunk(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    legal_documents = []
    
    # State variables to hold context
    current_book = ""
    current_title = ""
    current_chapter = ""
    current_topic = ""
    
    current_article_text = ""
    current_article_num = ""

    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. Capture Context Headers
        if line.lower().startswith("book"):
            current_book = line
        elif line.lower().startswith("title"):
            current_title = line
        elif line.lower().startswith("chapter"):
            current_chapter = line
        elif line.lower().startswith("topic") or line.lower().startswith("section"):
            current_topic = line
            
        # 2. Detect Article Start
        elif line.lower().startswith("article"):
            # Save previous article if exists
            if current_article_num:
                # INJECT CONTEXT INTO THE CHUNK
                full_context = f"{current_book} > {current_title} > {current_chapter} > {current_topic}"
                final_text = f"Context: {full_context}\n\nArticle {current_article_num}:\n{current_article_text}"
                
                doc = Document(
                    text=final_text,
                    metadata={
                        "article_id": current_article_num,
                        "context": full_context
                    }
                )
                legal_documents.append(doc)

            # Start new article
            match = re.search(r'(\d+)', line)
            if match:
                current_article_num = match.group(1)
                current_article_text = ""
        
        # 3. Accumulate Text
        else:
            current_article_text += line + " "

    # Save the last article
    if current_article_num:
        full_context = f"{current_book} > {current_title} > {current_chapter} > {current_topic}"
        final_text = f"Context: {full_context}\n\nArticle {current_article_num}:\n{current_article_text}"
        doc = Document(text=final_text, metadata={"article_id": current_article_num})
        legal_documents.append(doc)

    print(f"✅ Contextual Parsing: Created {len(legal_documents)} enriched chunks.")
    return legal_documents

def upload_hybrid():
    print("--- 🚀 Starting Context-Aware Hybrid Upload ---")
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

    documents = clean_and_contextual_chunk(file_path)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ Upload Complete!")

if __name__ == "__main__":
    upload_hybrid()