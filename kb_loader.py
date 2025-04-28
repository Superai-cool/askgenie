# kb_loader.py

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", "?", "!"]
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    text = load_pdf("BANK OF PUNE SOP 1.pdf")
    chunks = split_text(text)
    print(f"✅ Loaded {len(chunks)} chunks successfully.")
