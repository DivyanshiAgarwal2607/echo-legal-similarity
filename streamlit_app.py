import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# ------------------------------
# CONFIG
# ------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "legal-cases")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
UPLOADS_DIR = "uploads"

st.set_page_config(page_title="⚖️ Legal Case Similarity Finder", layout="wide")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ------------------------------
# INIT
# ------------------------------
if not PINECONE_API_KEY:
    st.error("❌ Missing Pinecone API key.")
    st.stop()

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"🚨 Pinecone connection failed: {e}")
    st.stop()

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# ------------------------------
# UI
# ------------------------------
st.title("⚖️ Legal Case Similarity Finder")
st.markdown("Upload a **legal case PDF** to automatically find similar judgments.")

uploaded_file = st.file_uploader("📄 Upload a case PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Extracting text..."):
        pdf_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            reader = PdfReader(pdf_path)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
            st.stop()

        if not text.strip():
            st.error("⚠️ No text found in this PDF.")
            st.stop()

        st.success(f"✅ {uploaded_file.name} uploaded and processed successfully!")

        # Query Pinecone
        st.info("🔍 Searching for similar cases...")
        query_vector = model.encode(text).tolist()
        try:
            res = index.query(vector=query_vector, top_k=5, include_metadata=True)
        except Exception as e:
            st.error(f"🚨 Error querying Pinecone: {e}")
            st.stop()

        # ------------------------------
        # DISPLAY RESULTS
        # ------------------------------
        if res and "matches" in res and len(res["matches"]) > 0:
            st.subheader("📚 Top 5 Similar Cases")

            for i, match in enumerate(res["matches"], start=1):
                filename = match["id"]
                score = match["score"]
                meta_text = match["metadata"].get("text", "").strip() if match.get("metadata") else ""

                st.markdown(f"### ⚖️ Match {i}: `{filename}`")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption(f"**Similarity Score:** {score:.2%}")

                # If metadata has some text, show a short summary only
                if meta_text:
                    st.write(meta_text[:600] + "...")

                # File path (either .pdf or filename only)
                pdf_path = os.path.join(UPLOADS_DIR, f"{filename}.pdf")
                pdf_path_alt = os.path.join(UPLOADS_DIR, filename)

                # Show only Download button if file exists
                if os.path.exists(pdf_path) or os.path.exists(pdf_path_alt):
                    file_to_open = pdf_path if os.path.exists(pdf_path) else pdf_path_alt
                    with open(file_to_open, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    st.download_button(
                        label="📥 Download Case PDF",
                        data=pdf_data,
                        file_name=os.path.basename(file_to_open),
                        mime="application/pdf"
                    )
                # If file not found, hide message entirely
                else:
                    pass

        else:
            st.warning("⚠️ No similar cases found in database.")
