import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ----------------------------------------------
# Load environment variables
# ----------------------------------------------
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("❌ Set your PINECONE_API_KEY in the .env file!")

# ----------------------------------------------
# Flask setup
# ----------------------------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------------------------
# Pinecone setup
# ----------------------------------------------
index_name = "legal-cases"
pc = Pinecone(api_key=api_key)

if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    print(f"🆕 Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ----------------------------------------------
# Embedding model
# ----------------------------------------------
print("🔹 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Loaded model: all-MiniLM-L6-v2")

# ----------------------------------------------
# File save folder
# ----------------------------------------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads", "user_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------------------------
# PDF text extraction (with OCR fallback)
# ----------------------------------------------
def extract_pdf_text(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
        if not text.strip():
            print("🔍 Using OCR fallback...")
            images = convert_from_path(file_path)
            for img in images:
                text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"⚠️ PDF extraction failed: {e}")
    return text.strip()

# ----------------------------------------------
# Upload + Match route
# ----------------------------------------------
@app.route("/upload_and_match", methods=["POST"])
def upload_and_match():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    print(f"📂 File saved: {save_path}")

    # Extract text
    text = extract_pdf_text(save_path)
    if not text:
        print("❌ No text extracted from PDF.")
        return jsonify({"error": "Unable to extract readable text from PDF"}), 400

    # Create embeddings
    embedding = embedder.encode(text).tolist()

    # Query Pinecone for similar cases
    try:
        res = index.query(vector=embedding, top_k=5, include_metadata=True)
    except Exception as e:
        print(f"❌ Pinecone query failed: {e}")
        return jsonify({"error": "Failed to query Pinecone"}), 500

    # Format results
    results = []
    for match in res["matches"]:
        file_name = match["id"]
        score = match["score"]
        results.append({
            "file": file_name,
            "score": score
        })

    return jsonify({"message": "Matches retrieved successfully!", "results": results})

# ----------------------------------------------
# File download route
# ----------------------------------------------
@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

# ----------------------------------------------
# List indexed PDFs
# ----------------------------------------------
@app.route("/list", methods=["GET"])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"files": files})

# ----------------------------------------------
# Run app
# ----------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
