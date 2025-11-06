import os
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from pinecone import Pinecone
from dotenv import load_dotenv

# Setup NLTK + environment
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
load_dotenv()

# Initialize Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

def pdf_to_text(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def chunk_document(text, chunk_size=600, overlap=100):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())

    # add overlap
    out = []
    for i in range(len(chunks)):
        start = max(0, i - 1)
        merged = " ".join(chunks[start:i + 1])
        out.append(merged)
    return out

def get_embeddings(chunks):
    """
    Generate embeddings using Pinecone’s native embed API.
    """
    response = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=chunks,
        parameters={"input_type": "passage"}
    )
    return [item.values for item in response.data]
