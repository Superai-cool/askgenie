import faiss
import os
import numpy as np
import pickle
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# ✅ Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------- Load and Split PDF ----------------------
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

# ---------------------- Get Embeddings ----------------------
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# ---------------------- Build Knowledge Base ----------------------
def build_knowledge_base():
    print("🔵 Building knowledge base from scratch...")
    text = load_pdf("BANK OF PUNE SOP 1.pdf")
    chunks = split_text(text)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save the index and chunks
    faiss.write_index(index, "kb.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("✅ Knowledge base created successfully!")
    return index, chunks

# ---------------------- Load or Create KB ----------------------
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

# ---------------------- Search and Answer Functions ----------------------
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

If the information is NOT available in the above content, reply with EXACTLY:
"Information not available in the SOP."

⚠️ Do NOT translate or change this fallback message. Always keep it in English.

If information is available, then answer normally, matching the user's input language.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
