# kb_search.py

import faiss
import openai
import numpy as np
import pickle

openai.api_key = "your-openai-api-key"

def get_embedding(text, model="text-embedding-ada-002"):
    result = openai.Embedding.create(input=[text], model=model)
    return result['data'][0]['embedding']

# Load FAISS index and chunks
index = faiss.read_index("kb.index")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

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
