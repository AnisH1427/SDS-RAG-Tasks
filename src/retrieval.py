import os
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pathlib import Path
import asyncio

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'Apple-knowledge-base').replace('"', '')


def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )

def get_query_embedding(query: str) -> List[float]:
    embedder = get_embeddings()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return embedder.embed_query(query)

def get_top_k_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks from Qdrant for a given query.
    Returns a list of dicts with text and metadata.
    """
    client = get_qdrant_client()
    query_vector = get_query_embedding(query)
    search_result = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )
    results = []
    for hit in search_result:
        payload = hit.payload or {}
        results.append({
            "score": hit.score,
            "text": payload.get("text", ""),
            "page": payload.get("page"),
            "chunk_id": payload.get("chunk_id")
        })
    return results

# if __name__ == "__main__":
#     query = input("Enter your query: ")
#     top_chunks = get_top_k_chunks(query)
#     for i, chunk in enumerate(top_chunks, 1):
#         print(f"\nResult {i} (Score: {chunk['score']:.4f}):")
#         print(f"Page: {chunk['page']}, Chunk ID: {chunk['chunk_id']}")
#         print(chunk['text'])

