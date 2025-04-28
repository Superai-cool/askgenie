import faiss
import openai
import numpy as np
import pickle
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai.api_key = os.getenv("OPENAI_API_KEY")

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

def get_embedding(text, model="text-embedding-ada-002"):
    result = openai.Embedding.create(input=[text], model=model)
    return result['data'][0]['embedding']

def build_knowledge_base():
    print("🔵 Building knowledge base from scratch...")
    text = load_pdf("BANK OF PUNE SOP 1.pdf")
    chunks = split_text(text)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save files for future use
    faiss.write_index(index, "kb.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("✅ Knowledge base created successfully!")
    return index, chunks

# 🛠 Always Rebuild Fresh (Best for Streamlit + Railway)
if os.path.exists("kb.index") and os.path.exists("chunks.pkl"):
    try:
        index = faiss.read_index("kb.index")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        print("✅ Loaded existing knowledge base!")
    except:
        index, chunks = build_knowledge_base()
else:
    index, chunks = build_knowledge_base()

def search_kb(question, top_k=3):
    question_embedding = np.array([get_embedding(question)]).astype('float32')
    D, I = index.search(question_embedding, top_k)
    return [chunks[i] for i in I[0]]

def answer_from_kb(question):
    related_chunks = search_kb(question)

    prompt = f"""
You are Ask Genie, an internal banking assistant.

Answer the user's question strictly using ONLY the below information:

{''.join(related_chunks)}

Question: {question}

If information not found, reply: "Information not available in the SOP."

Always reply in the user's question language.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message["content"].strip()
