import os
from utils import pdf_to_text, chunk_document, get_embeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def index_pdf(file_path):
    print(f"Indexing: {file_path}")
    text = pdf_to_text(file_path)
    chunks = chunk_document(text)
    embeddings = get_embeddings(chunks)

    for i, emb in enumerate(embeddings):
        vector_id = f"{os.path.basename(file_path)}_{i}"
        index.upsert(vectors=[(vector_id, emb, {"text": chunks[i]})])
    print("✅ Indexed:", file_path)

if __name__ == "__main__":
    folder = "uploads"
    for pdf in os.listdir(folder):
        if pdf.endswith(".pdf"):
            index_pdf(os.path.join(folder, pdf))
