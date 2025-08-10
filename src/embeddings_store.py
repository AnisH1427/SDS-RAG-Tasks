import os
import json
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'Apple-knowledge-base').replace('"', '')

CHUNKED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '../data/processed/chunked_pdf_data.json'
)

BATCH_SIZE = 100  # Number of points to upsert in each batch

def load_chunked_data():
    with open(CHUNKED_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )

def create_collection_if_not_exists(client, vector_size):
    if QDRANT_COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def upsert_to_qdrant_in_batches(client, points: List[PointStruct], batch_size: int = BATCH_SIZE):
    """
    Upsert points to Qdrant in batches to avoid timeouts.
    """
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=batch
        )
        print(f"Upserted batch {i//batch_size + 1} ({len(batch)} points)")

def delete_qdrant_collection(client):
    client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model= EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )

def get_google_embeddings(texts):
    embedder = get_embeddings()
    vectors = embedder.embed_documents(texts)
    return vectors

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true', help='Delete the Qdrant collection and exit')
    args = parser.parse_args()

    client = get_qdrant_client()

    if args.delete:
        delete_qdrant_collection(client)
        print("Qdrant collection deleted successfully.")
    else:
        data = load_chunked_data()
        texts = [chunk["text"] for chunk in data]
        vectors = get_google_embeddings(texts)
        points = [
            PointStruct(
                id=i,  # Use integer index as point ID
                vector=vectors[i],
                payload={
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],  # Keep original chunk_id in payload
                    "text": chunk["text"]
                }
            )
            for i, chunk in enumerate(data)
        ]
        create_collection_if_not_exists(client, vector_size=len(vectors[0]))
        upsert_to_qdrant_in_batches(client, points)
        print("All points upserted to Qdrant successfully.")
