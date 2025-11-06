import kagglehub
import os
import zipfile
from tqdm import tqdm
import PyPDF2

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("adarshsingh0903/legal-dataset-sc-judgments-india-19502024")
print("✅ Downloaded dataset at:", path)

os.makedirs("kaggle_data", exist_ok=True)
os.makedirs("kaggle_texts", exist_ok=True)

# Extract PDFs if zipped
for file in os.listdir(path):
    if file.endswith(".zip"):
        zip_path = os.path.join(path, file)
        print("Extracting:", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall("kaggle_data")

# Convert PDFs → text
pdf_dir = "kaggle_data"
txt_dir = "kaggle_texts"

for pdf_file in tqdm(os.listdir(pdf_dir), desc="Converting PDFs to text"):
    if pdf_file.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        txt_path = os.path.join(txt_dir, pdf_file.replace(".pdf", ".txt"))
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            with open(txt_path, "w", encoding="utf-8") as out:
                out.write(text)
        except Exception as e:
            print(f"❌ Failed {pdf_file}: {e}")

print("✅ All PDFs converted to text in:", txt_dir)
