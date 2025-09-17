# day3_rag.py
# Day 3: Document Q&A (ingest PDF/text -> chunk -> embeddings -> FAISS -> query)
import os
import re
import math
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Optional OpenAI generation (if you have a key)
try:
    import openai
except Exception:
    openai = None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

# ----- Config -----
DOC_PATH = "DAY3/DOCS/doc_testing.pdf"   # put your PDF here
EMBED_MODEL = "all-MiniLM-L6-v2"     # light & fast
CHUNK_SIZE = 800    # characters per chunk (adjustable)
CHUNK_OVERLAP = 200 # overlap between chunks
TOP_K = 3           # how many chunks to retrieve

# ----- Helpers -----
def read_pdf(path):
    text_pages = []
    reader = PdfReader(path)
    for p in reader.pages:
        txt = p.extract_text() or ""
        text_pages.append(txt)
    return "\n".join(text_pages)

def clean_text(text):
    # simple cleanup
    text = text.replace("\x0c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks

# ----- Main pipeline -----
print("Loading/reading document...")
if DOC_PATH.lower().endswith(".pdf"):
    raw = read_pdf(DOC_PATH)
else:
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

raw = clean_text(raw)
print(f"Document length (chars): {len(raw)}")

print("Chunking document...")
chunks = chunk_text(raw, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

print("Encoding chunks (this may take a bit)...")
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

d = embeddings.shape[1]  # dimension
print("Embedding dim:", d)

# Normalize embeddings (good for cosine using inner product)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / (norms + 1e-10)

# Build FAISS index (IndexFlatIP uses inner product; with normalized vectors it's cosine)
index = faiss.IndexFlatIP(d)
index.add(embeddings)
print("FAISS index built. Number of vectors:", index.ntotal)

# Simple interactive loop: ask questions
def retrieve(query, k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = index.search(q_emb.astype("float32"), k)
    scores = D[0]
    indices = I[0]
    results = []
    for score, idx in zip(scores, indices):
        results.append((float(score), chunks[idx]))
    return results

def generate_answer_with_openai(query, contexts):
    if not OPENAI_API_KEY or not openai:
        return None
    prompt = "You are an assistant. Use the following extracted document pieces to answer the question truthfully and concisely.\n\n"
    for i, c in enumerate(contexts, 1):
        prompt += f"Context {i}:\n{c}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        max_tokens=256,
        temperature=0.2
    )
    return resp.choices[0].message['content'].strip()

print("\nReady. Ask questions about the document. Type 'exit' to quit.\n")
while True:
    q = input("Question: ").strip()
    if q.lower() in ("exit", "quit"):
        print("Bye.")
        break
    retrieved = retrieve(q, k=TOP_K)
    print("\nTop retrieved chunks (score, snippet):")
    for score, snippet in retrieved:
        print(f"Score: {score:.4f}  Snippet: {snippet[:300].strip()}...\n")
    # If OpenAI key available, generate final answer using retrieved chunks
    if OPENAI_API_KEY and openai:
        contexts = [s for _, s in retrieved]
        answer = generate_answer_with_openai(q, contexts)
        print("=== Final answer (OpenAI) ===")
        print(answer)
    else:
        print("=== No generation (OpenAI key missing). Use the retrieved snippets to answer manually or add your API key in .env ===")
    print("\n---\n")
