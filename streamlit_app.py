import os
import gzip
import base64
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# ------------------------------
# 1️⃣ Setup
# ------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "legal-cases")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

st.set_page_config(page_title="⚖️ Legal Case Similarity Finder", layout="wide")
st.title("⚖️ Legal Case Similarity Finder")
st.markdown("Upload a legal case PDF to automatically find similar judgments from the database.")

# ------------------------------
# 2️⃣ Connect to Pinecone
# ------------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"🚨 Pinecone connection failed: {e}")
    st.stop()

# ------------------------------
# 3️⃣ Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# ------------------------------
# 4️⃣ Upload and Process PDF
# ------------------------------
uploaded_file = st.file_uploader("📄 Upload a case PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Reading PDF..."):
        pdf_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        reader = PdfReader(pdf_path)
        text = " ".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        st.error("❌ No readable text found in PDF.")
    else:
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")
        query_vector = model.encode(text).tolist()

        with st.spinner("🔍 Searching for similar cases..."):
            results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        st.subheader("📚 Top 5 Similar Cases")
        for i, match in enumerate(results.get("matches", []), 1):
            metadata = match.get("metadata", {})
            filename = metadata.get("filename", f"Case_{i}")
            score = match.get("score", 0)
            preview_compressed = metadata.get("text_preview")
            preview_text = ""
            if preview_compressed:
                try:
                    preview_text = gzip.decompress(base64.b64decode(preview_compressed)).decode("utf-8")
                except Exception:
                    preview_text = ""

            st.markdown(f"### ⚖️ Match {i}: `{filename}`")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(f"**Similarity Score:** {score:.2%}")
            st.write(preview_text[:1000] + "..." if preview_text else "🔹 No preview text available.")
            
            local_path = metadata.get("local_path")
            if local_path and os.path.exists(local_path):
                with open(local_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Case PDF",
                        data=pdf_file,
                        file_name=filename,
                        mime="application/pdf"
                    )
            else:
                pass  # silently skip unavailable
