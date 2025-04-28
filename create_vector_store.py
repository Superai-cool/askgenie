# ✅ Final Corrected create_vector_store.py for Railway Deployment

import openai
import faiss
import numpy as np
import pickle
import fitz  # PyMuPDF
import os
from langdetect import detect

# ✅ Make sure you have set your OpenAI API key in Railway Variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the PDF file
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into smaller chunks
def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Get embedding using OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    result = openai.embeddings.create(input=[text], model=model)
    return result.data[0].embedding

# Build the FAISS index and save it
def build_knowledge_base():
    print("🔵 Step 1: Loading PDF...")
    text = load_pdf("BANK OF PUNE SOP 1.pdf")

    print("🔵 Step 2: Splitting text into chunks...")
    chunks = split_text(text)

    print("🔵 Step 3: Generating embeddings...")
    embeddings = [get_embedding(chunk) for chunk in chunks]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    print("🔵 Step 4: Adding embeddings to FAISS index...")
    index.add(np.array(embeddings).astype('float32'))

    print("🔵 Step 5: Saving FAISS index and chunks...")
    faiss.write_index(index, "kb.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Knowledge base created and saved successfully!")

if __name__ == "__main__":
    build_knowledge_base()
