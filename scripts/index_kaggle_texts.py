import os
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -------------------------------
# STEP 1: Load environment variables
# -------------------------------
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX", "legal-cases")
model_name = os.getenv("MODEL_NAME", "llama-text-embed-v2")

if not api_key:
    raise ValueError("⚠️ Set your PINECONE_API_KEY in environment variables first.")

# -------------------------------
# STEP 2: Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=api_key)

# Check if index exists, if not — create one
if index_name not in [i["name"] for i in pc.list_indexes()]:
    print(f"🆕 Creating Pinecone index '{index_name}' ...")
    pc.create_index(
        name=index_name,
        dimension=1024,  # since you’re using llama-text-embed-v2 (1024-dim)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"✅ Index '{index_name}' already exists!")

index = pc.Index(index_name)

# -------------------------------
# STEP 3: Load your Kaggle dataset
# -------------------------------
dataset_path = os.path.join(
    os.path.expanduser("~"),
    "kagglehub/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024"
)

# Try to detect CSV
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("❌ No CSV files found in dataset path.")
csv_path = os.path.join(dataset_path, csv_files[0])

print(f"📄 Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

# Just pick a sample subset for now (to test)
df = df.sample(n=min(100, len(df)), random_state=42).reset_index(drop=True)

# -------------------------------
# STEP 4: Load embedding model
# -------------------------------
print("⚙️ Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

# -------------------------------
# STEP 5: Generate embeddings & push to Pinecone
# -------------------------------
print("🚀 Generating embeddings and uploading to Pinecone...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row.get("judgment_text") or row.get("text") or "")
    if not text.strip():
        continue

    # Create embedding vector (1024-dimension expected)
    vector = embedder.encode(text).tolist()

    # If dimension mismatch, pad/truncate to 1024
    if len(vector) < 1024:
        vector = vector + [0.0] * (1024 - len(vector))
    elif len(vector) > 1024:
        vector = vector[:1024]

    # Upsert to Pinecone
    index.upsert(vectors=[
        {
            "id": f"case-{i}",
            "values": vector,
            "metadata": {
                "title": str(row.get("case_title") or f"Case {i}"),
                "date": str(row.get("date") or ""),
                "court": str(row.get("court") or ""),
                "text": text[:1000]  # store first 1000 chars as preview
            }
        }
    ])

print("✅ All data uploaded successfully to Pinecone!")
