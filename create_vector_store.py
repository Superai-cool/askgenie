# create_vector_store.py

import faiss
import openai
import numpy as np
import pickle
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# ----------------- Load OpenAI API Key -----------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OpenAI API Key not found! Set it as an environment variable.")
openai.api_key = api_key

# ----------------- Load PDF -----------------
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ----------------- Split Text -----------------
def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", "?", "!"]
    )
    return splitter.split_text(text)

# ----------------- Get Embedding -----------------
def get_embedding(text, model="text-embedding-ada-002"):
    result = openai.Embedding.create(input=[text], model=model)
    return result['data'][0]['embedding']

# ----------------- Create Vector Store -----------------
def create_vector_store(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    pdf_path = "BANK OF PUNE SOP 1.pdf"  # 📄 Your SOP file
    print(f"📚 Loading {pdf_path}...")
    text = load_pdf(pdf_path)
    print("✅ PDF Loaded!")

    print("✂️ Splitting into chunks...")
    chunks = split_text(text)

    print(f"✅ {len(chunks)} chunks created!")

    print("🔎 Creating embeddings and vector index...")
    index, chunks = create_vector_store(chunks)

    # Save FAISS index and chunks
    faiss.write_index(index, "kb.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Successfully created 'kb.index' and 'chunks.pkl' files!")
    print("🎯 Now upload these two files to your GitHub along with your app.")
