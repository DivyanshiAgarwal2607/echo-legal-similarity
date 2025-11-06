import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# -----------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "legal-cases")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# -----------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="⚖️ Legal Case Similarity Finder", layout="wide")
st.title("⚖️ Legal Case Similarity Finder")
st.markdown("Upload a legal case PDF to find **similar judgments** from the database instantly.")

# -----------------------------------
# CONNECT TO PINECONE
# -----------------------------------
if not PINECONE_API_KEY:
    st.error("❌ Pinecone API key missing. Add it to your `.env` file.")
    st.stop()

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    st.error(f"🚨 Pinecone connection failed: {e}")
    st.stop()

# -----------------------------------
# LOAD MODEL (cached for performance)
# -----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# -----------------------------------
# FILE UPLOAD SECTION
# -----------------------------------
uploaded_file = st.file_uploader("📄 Upload a legal case PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("📖 Reading and analyzing PDF..."):
        pdf_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        reader = PdfReader(pdf_path)
        text = " ".join(page.extract_text() or "" for page in reader.pages)

        if not text.strip():
            st.warning("⚠️ This PDF contains no readable text. Please upload a searchable PDF.")
        else:
            st.success(f"✅ {uploaded_file.name} uploaded successfully!")

            # Create query embedding
            query_vector = model.encode(text).tolist()

            # Query Pinecone
            st.info("🔍 Finding top 5 most similar cases...")
            try:
                res = index.query(vector=query_vector, top_k=5, include_metadata=True)
            except Exception as e:
                st.error(f"🚨 Pinecone query failed: {e}")
                st.stop()

            # Display results
            if res and "matches" in res and len(res["matches"]) > 0:
                st.subheader("📚 Top 5 Similar Cases")
                for i, match in enumerate(res["matches"], start=1):
                    filename = match.get("id", "Unknown Case")
                    score = match.get("score", 0.0)
                    meta = match.get("metadata", {})
                    summary = meta.get("text", "")[:600].strip()

                    st.markdown(f"### ⚖️ Match {i}: `{filename}`")
                    st.progress(min(max(score, 0.0), 1.0))
                    st.caption(f"**Similarity Score:** {score:.2%}")

                    if summary:
                        st.write(summary + "...")
                    else:
                        st.write("_Preview unavailable for this case._")

                    # Download button (only if file exists locally)
                    local_pdf = None
                    for root, _, files in os.walk(UPLOADS_DIR):
                        if filename in files:
                            local_pdf = os.path.join(root, filename)
                            break

                    if local_pdf and os.path.exists(local_pdf):
                        with open(local_pdf, "rb") as pdf_data:
                            st.download_button(
                                label="📥 Download Case PDF",
                                data=pdf_data,
                                file_name=filename,
                                mime="application/pdf"
                            )
                    else:
                        # Hide missing file warnings on Streamlit
                        pass
            else:
                st.warning("⚠️ No similar cases found.")
