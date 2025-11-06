import os
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm  # progress bar

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "legal-cases")
UPLOADS_DIR = "uploads"

if not PINECONE_API_KEY:
    raise ValueError("❌ Missing Pinecone API key in .env file")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print(f"\n🧹 Fast cleaning Pinecone index: {INDEX_NAME}\n")

# 1️⃣ Collect all local PDF names (without .pdf)
local_ids = set()
for root, _, files in os.walk(UPLOADS_DIR):
    for f in files:
        if f.lower().endswith(".pdf"):
            local_ids.add(os.path.splitext(f)[0])

print(f"📁 Found {len(local_ids)} local PDF files.")

# 2️⃣ Get Pinecone stats
stats = index.describe_index_stats()
total_vectors = stats.get("total_vector_count", 0)
print(f"📊 Pinecone currently has {total_vectors} entries.\n")

# 3️⃣ Fetch all Pinecone IDs (generator)
print("🔍 Fetching all Pinecone IDs from index (please wait)...")
all_ids = []
for ids_page in index.list():  # ✅ yields list of IDs
    all_ids.extend(ids_page)

print(f"✅ Retrieved {len(all_ids)} IDs from Pinecone.\n")

# 4️⃣ Find IDs that are NOT present locally
to_delete = [pid for pid in tqdm(all_ids, desc="🧩 Checking IDs") if pid not in local_ids]

# 5️⃣ Delete missing ones in batches
if to_delete:
    print(f"\n🗑️ Deleting {len(to_delete)} entries missing locally...")
    batch_size = 100
    for i in tqdm(range(0, len(to_delete), batch_size), desc="🧹 Deleting"):
        batch = to_delete[i:i + batch_size]
        index.delete(ids=batch)
    print(f"\n✅ Cleaned {len(to_delete)} invalid vectors successfully.\n")
else:
    print("✅ No missing or invalid files found!\n")

print("🎯 Pinecone index is now fully in sync with your uploads folder.\n")
