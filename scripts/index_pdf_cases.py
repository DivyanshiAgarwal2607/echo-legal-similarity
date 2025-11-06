import os
import fitz  # PyMuPDF for reading PDFs
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

# --------------------------------------------------
# STEP 1: Load environment variables
# --------------------------------------------------
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "legal-cases")
model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

if not api_key:
    raise ValueError("⚠️ Set your PINECONE_API_KEY in environment variables first.")

# --------------------------------------------------
# STEP 2: Initialize Pinecone
# --------------------------------------------------
pc = Pinecone(api_key=api_key)

if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    print(f"🆕 Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"✅ Index '{index_name}' already exists!")

index = pc.Index(index_name)

# --------------------------------------------------
# STEP 3: Load embedding model
# --------------------------------------------------
print("🔹 Loading embedding model...")
embedder = SentenceTransformer(model_name)
print(f"✅ Loaded model: {model_name}")

# --------------------------------------------------
# STEP 4: PDF text extraction helper
# --------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file using PyMuPDF"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()

# --------------------------------------------------
# STEP 5: Choose your dataset folder
# --------------------------------------------------
pdf_folder = os.path.join("uploads", "filessssss")  # update this to your folder name

if not os.path.exists(pdf_folder):
    raise FileNotFoundError(f"⚠️ Folder not found: {pdf_folder}")

# Recursively find all PDF files inside subfolders
pdf_files = []
for root, dirs, files in os.walk(pdf_folder):
    for f in files:
        if f.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(root, f))

if not pdf_files:
    raise FileNotFoundError(f"⚠️ No PDF files found in '{pdf_folder}'")

print(f"📂 Found {len(pdf_files)} PDF files across all subfolders. Starting indexing...")

# --------------------------------------------------
# STEP 6: Process PDFs and upload embeddings
# --------------------------------------------------
batch = []
batch_size = 20  # adjustable for speed vs stability

for i, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"⚠️ Skipping empty file: {pdf_path}")
            continue

        embedding = embedder.encode(text).tolist()
        case_id = os.path.splitext(os.path.basename(pdf_path))[0]
        year_folder = os.path.basename(os.path.dirname(pdf_path))

        batch.append({
            "id": f"{year_folder}_{case_id}",
            "values": embedding,
            "metadata": {
                "filename": os.path.basename(pdf_path),
                "year": year_folder,
                "path": pdf_path
            }
        })

        # Upload in batches
        if len(batch) >= batch_size:
            index.upsert(vectors=batch)
            print(f"✅ Uploaded {len(batch)} embeddings...")
            batch = []
            time.sleep(1)  # avoid rate limits

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")

# Upload any remaining
if batch:
    index.upsert(vectors=batch)
    print(f"✅ Uploaded remaining {len(batch)} embeddings.")

print("🎉 All PDF cases have been indexed successfully!")
