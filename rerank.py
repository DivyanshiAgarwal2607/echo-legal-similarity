import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, util

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-cases"   # 👈 keep this same as your Pinecone index name

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ✅ Use a model that outputs 1024-dimension embeddings
MODEL_NAME = "intfloat/e5-large-v2"
sim_model = SentenceTransformer(MODEL_NAME)

def semantic_search_and_rerank(query, top_k=3):
    """
    Search for semantically similar cases in Pinecone and rerank them.
    """
    # 1️⃣ Encode the query into 1024-dimensional vector
    query_vector = sim_model.encode(query).tolist()

    # 2️⃣ Query Pinecone index
    search_response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 3️⃣ Extract and structure the results
    matches = search_response.get("matches", [])
    if not matches:
        return [{"text": "No similar cases found", "score": 0.0}]

    results = []
    for match in matches:
        metadata = match.get("metadata", {})
        results.append({
            "text": metadata.get("text", "No text found"),
            "score": match.get("score", 0.0)
        })

    # 4️⃣ Sort high → low by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results
