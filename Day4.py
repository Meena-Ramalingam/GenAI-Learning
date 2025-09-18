"""
Day 4 — Document Chatbot with smarter chunking, model swap, and persistent FAISS index.

How to use:
1. Put your document file at: day4/docs/document.pdf  OR day4/docs/document.txt
2. (Optional) Create .env with OPENAI_API_KEY=sk-...
3. Run: python day4_chatbot.py
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Optional OpenAI usage for final answer generation
try:
    import openai
except Exception:
    openai = None

# ----- Config -----
BASE_DIR = Path("day4")
DOC_PATH = BASE_DIR / "docs" / "document.pdf"   # or document.txt
EMBED_MODEL = "all-mpnet-base-v2"               # change model here: all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, etc.
CHUNK_TYPE = "sentences"                        # options: "fixed", "sentences", "paragraphs"
CHUNK_SIZE = 800         # used for "fixed" chunking (chars)
CHUNK_OVERLAP = 200      # used for "fixed" chunking (chars)
TOP_K = 3
INDEX_PATH = BASE_DIR / "faiss.index"
META_PATH = BASE_DIR / "meta.json"   # stores chunks & metadata
EMB_PATH = BASE_DIR / "embeddings.npy"
PERSIST = True  # whether to save/load index and embeddings

# load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

# ensure folders exist
(BASE_DIR / "docs").mkdir(parents=True, exist_ok=True)

# ----- Utilities -----
def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    return "\n".join(pages)

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def clean_text(s: str) -> str:
    s = s.replace("\x0c", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Sentence tokenizer (nltk) with auto-download if needed
def ensure_nltk():
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except Exception:
        import nltk
        nltk.download("punkt")

def chunk_by_sentences(text: str, max_chars=600):
    # use nltk sentence tokenizer and group sentences until approx max_chars
    ensure_nltk()
    import nltk
    sents = nltk.tokenize.sent_tokenize(text)
    chunks = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

def chunk_by_paragraphs(text: str):
    # split on double newlines
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paras

def chunk_fixed(text: str, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ----- Main pipeline -----
def load_document(doc_path: Path):
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found at {doc_path}. Place your file there.")
    if doc_path.suffix.lower() == ".pdf":
        raw = read_pdf(doc_path)
    else:
        raw = read_txt(doc_path)
    return clean_text(raw)

def create_chunks(text: str, chunk_type: str):
    if chunk_type == "sentences":
        return chunk_by_sentences(text, max_chars=CHUNK_SIZE)
    elif chunk_type == "paragraphs":
        return chunk_by_paragraphs(text)
    else:
        return chunk_fixed(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

def build_embeddings_and_index(chunks, model_name=EMBED_MODEL):
    print("Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)
    print("Encoding chunks...")
    embs = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-10)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype("float32"))
    return model, embs, index

def save_state(index: faiss.IndexFlatIP, chunks, embs, idx_path=INDEX_PATH, meta_path=META_PATH, emb_path=EMB_PATH):
    print("Saving FAISS index ->", idx_path)
    faiss.write_index(index, str(idx_path))
    print("Saving meta (chunks) ->", meta_path)
    meta = {"chunks": chunks}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print("Saving embeddings ->", emb_path)
    np.save(str(emb_path), embs)

def load_state(model_name=EMBED_MODEL, idx_path=INDEX_PATH, meta_path=META_PATH, emb_path=EMB_PATH):
    if not idx_path.exists() or not meta_path.exists() or not emb_path.exists():
        return None
    print("Loading saved FAISS index from disk...")
    index = faiss.read_index(str(idx_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = meta["chunks"]
    embs = np.load(str(emb_path))
    # model must be same to encode queries reliably
    model = SentenceTransformer(model_name)
    return model, chunks, embs, index

def retrieve(index, model, query, chunks, k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
    D, I = index.search(q_emb.astype("float32"), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((float(score), chunks[idx]))
    return results

def generate_with_openai(query, contexts):
    if not OPENAI_API_KEY or not openai:
        return None
    prompt = "You are an assistant. Use the following extracted document pieces to answer the question truthfully and concisely.\n\n"
    for i, c in enumerate(contexts, 1):
        prompt += f"Context {i}:\n{c}\n\n"
    prompt += f"Question: {query}\nAnswer:"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        max_tokens=300,
        temperature=0.2
    )
    return resp.choices[0].message['content'].strip()

# ----- Run sequence -----
def main():
    # Try load existing state
    state = None
    if PERSIST:
        state = load_state(model_name=EMBED_MODEL)
    if state:
        model, chunks, embs, index = state
        print("Loaded existing index and chunks. Number of chunks:", len(chunks))
    else:
        print("Reading document...")
        text = load_document(DOC_PATH)
        print(f"Document length (chars): {len(text)}")
        print("Creating chunks using method:", CHUNK_TYPE)
        chunks = create_chunks(text, CHUNK_TYPE)
        print("Number of chunks created:", len(chunks))
        model, embs, index = build_embeddings_and_index(chunks, model_name=EMBED_MODEL)
        if PERSIST:
            save_state(index, chunks, embs)

    print("\nReady. Ask questions about the document. Type 'exit' to quit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            print("Bye.")
            break
        retrieved = retrieve(index, model, q, chunks, k=TOP_K)
        print("\nTop retrieved chunks (score, snippet):")
        for score, snippet in retrieved:
            print(f"Score: {score:.4f}\n{snippet[:500].strip()}...\n")
        # if OpenAI key present, create final answer
        if OPENAI_API_KEY and openai:
            contexts = [s for _, s in retrieved]
            ans = generate_with_openai(q, contexts)
            print("=== Final answer (OpenAI) ===")
            print(ans)
        else:
            print("=== No generation (OpenAI key missing) ===")
        print("\n---\n")

if __name__ == "__main__":
    main()
