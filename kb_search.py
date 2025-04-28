# ✅ FINAL kb_search.py with STRICT COPY-ONLY prompt

import openai
import faiss
import numpy as np
import pickle
import fitz  # PyMuPDF
from langdetect import detect

# Load FAISS index and chunks
index = faiss.read_index("kb.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Helper function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Load PDF text (if rebuilding)
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into smaller chunks (if rebuilding)
def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Get OpenAI Embedding
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    result = openai.embeddings.create(input=[text], model=model)
    return result.data[0].embedding

# Build Knowledge Base from PDF (optional)
def build_knowledge_base():
    print("🔵 Building knowledge base from scratch...")
    text = load_pdf("BANK OF PUNE SOP 1.pdf")
    chunks = split_text(text)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # Save index and chunks
    faiss.write_index(index, "kb.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Knowledge base created successfully!")
    return index, chunks

# Strict Copy-Only Answering from KB
def answer_from_kb(question, answer_type="short"):
    embedded_question = get_embedding(question)
    D, I = index.search(np.array([embedded_question]).astype('float32'), k=3)
    related_chunks = [chunks[i] for i in I[0] if i != -1]

    system_prompt = f"""
You are Ask Genie, an internal banking assistant.

Strictly answer ONLY by copying the text exactly from the information provided below:

{''.join(related_chunks)}

Question: {question}

⚠️ Do not summarize, shorten, edit, add new points, or paraphrase.
⚠️ Only copy exactly what is written in the provided information.

If the information is not available, reply exactly:
"Information not available in the SOP."
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    final_answer = response.choices[0].message.content.strip()

    detected_lang = detect_language(question)

    # Translate back if needed (optional, you can add)
    return final_answer
