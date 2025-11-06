import os
import time
import gzip
import base64
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ------------------------------
# 1️⃣ Load Environment
# ------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "legal-cases")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
UPLOADS_DIR = "uploads/filesssss"

# ------------------------------
# 2️⃣ Initialize Model + Pinecone
# ------------------------------
print("🔹 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("🔹 Loading model...")
model = SentenceTransformer(MODEL_NAME)

# ------------------------------
# 3️⃣ Helper Functions
# ------------------------------
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception:
        return ""

def compress_text(text):
    """Compress extracted text to save bandwidth."""
    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("utf-8")

def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    doc_id = os.path.splitext(filename)[0]
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None
    vector = model.encode(text).tolist()

    compressed_preview = compress_text(text[:3000])  # compress first 3k chars
    metadata = {
        "filename": filename,
        "text_preview": compressed_preview,
        "local_path": pdf_path.replace("\\", "/"),
    }
    return {"id": doc_id, "values": vector, "metadata": metadata}

# ------------------------------
# 4️⃣ Collect all PDFs
# ------------------------------
all_pdfs = []
for root, _, files in os.walk(UPLOADS_DIR):
    for f in files:
        if f.lower().endswith(".pdf"):
            all_pdfs.append(os.path.join(root, f))
print(f"📁 Found {len(all_pdfs)} PDFs to index.")

# ------------------------------
# 5️⃣ Fast Multi-thread + Batching
# ------------------------------
start = time.time()
batch_size = 100
to_upsert, done, skipped = [], 0, 0

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for record in tqdm(executor.map(process_pdf, all_pdfs), total=len(all_pdfs)):
        if record:
            to_upsert.append(record)
            if len(to_upsert) >= batch_size:
                index.upsert(vectors=to_upsert)
                done += len(to_upsert)
                to_upsert = []
        else:
            skipped += 1

if to_upsert:
    index.upsert(vectors=to_upsert)
    done += len(to_upsert)

end = time.time()
print("\n===============================")
print(f"🎉 FAST Reindex completed in {end - start:.2f} sec")
print(f"✅ Indexed: {done}")
print(f"⚠️ Skipped (no text): {skipped}")
print("===============================")
