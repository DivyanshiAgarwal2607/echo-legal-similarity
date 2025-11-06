import os
import time
import pdfplumber
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# =====================================
# CONFIG
# =====================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "legal-cases")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

# ⚠️ Update this to the EXACT path where your PDFs are
BASE_DIR = r"C:\Users\Lenovo\echo-legal-similarity\uploads\filesssss"

CHUNK_LIMIT = 5000  # store only first 5000 chars in metadata
BATCH_SIZE = 20     # how many files to upload to Pinecone per batch

# =====================================
# INIT CONNECTIONS
# =====================================
print("🔹 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("🔹 Loading model...")
model = SentenceTransformer(MODEL_NAME)

# =====================================
# FUNCTIONS
# =====================================
def extract_text_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""

# =====================================
# MAIN INDEXING LOGIC
# =====================================
print(f"🚀 Starting full reindexing in folder: {os.path.abspath(BASE_DIR)}\n")

indexed = 0
skipped = 0
batch = []
start_time = time.time()

# Debug: print every found file
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(root, file)
        print(f"🔍 Found file: {pdf_path}")

        text = extract_text_pdf(pdf_path)

        if not text or len(text.strip()) < 50:
            print(f"⚠️ Skipped (no text): {file}")
            skipped += 1
            continue

        try:
            vector = model.encode(text).tolist()
            doc_id = os.path.splitext(file)[0]

            batch.append({
                "id": doc_id,
                "values": vector,
                "metadata": {"text": text[:CHUNK_LIMIT]}
            })

            if len(batch) >= BATCH_SIZE:
                index.upsert(batch)
                print(f"📤 Uploaded batch of {len(batch)} files to Pinecone...")
                batch.clear()

            print(f"✅ Ready to index: {file}")
            indexed += 1

        except Exception as e:
            print(f"❌ Failed to index {file}: {e}")
            skipped += 1

# Upload remaining
if batch:
    index.upsert(batch)
    print(f"📤 Uploaded final batch of {len(batch)} files to Pinecone...")

elapsed = round(time.time() - start_time, 2)
print("\n===============================")
print(f"🎉 Reindexing completed in {elapsed} seconds.")
print(f"✅ Indexed files: {indexed}")
print(f"⚠️ Skipped (no text): {skipped}")
print("===============================")
