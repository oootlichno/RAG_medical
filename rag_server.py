import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

DB_DIR = os.getenv("CHROMA_DIR", "./chroma_db_v_export")
COLL = os.getenv("CHROMA_COLL", "langchain")
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/oootlichno/Library/Mobile Documents/com~apple~CloudDocs/Projects/Личная/Projects/RAG_medical/models/mistral-7b-instruct-v0.2.Q6_K.gguf")

# Load once at startup
client = chromadb.PersistentClient(DB_DIR)
col = client.get_collection(COLL)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=8192, verbose=False)

app = FastAPI(title="Medical RAG API", version="1.0")

class Query(BaseModel):
    question: str
    k: int = 4
    temperature: float = 0.1
    max_tokens: int = 256

class Answer(BaseModel):
    answer: str
    sources: List[Dict]

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/health", "/rag"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rag", response_model=Answer)
def rag(q: Query):
    q_vec = embedder.encode(q.question).tolist()
    hits = col.query(
        query_embeddings=[q_vec],
        n_results=q.k,
        include=["documents", "distances", "metadatas"]  # ← no "ids" here
    )
    if not hits["documents"] or not hits["documents"][0]:
        return {"answer": "I don't know.", "sources": []}

    docs = hits["documents"][0]
    context = "\n\n".join(docs)

    prompt = (
        "Answer strictly from the context. If the answer is not in the context, "
        "say: I don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q.question}\nAnswer:"
    )

    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are concise and cautious."},
            {"role": "user", "content": prompt}
        ],
        temperature=q.temperature,
        max_tokens=q.max_tokens
    )
    answer = out["choices"][0]["message"]["content"]

    # IDs are returned by default even if not requested in include
    sources = []
    for _id, doc, dist in zip(hits["ids"][0], hits["documents"][0], hits["distances"][0]):
        sources.append({"id": _id, "distance": float(dist), "snippet": doc[:300]})

    return {"answer": answer, "sources": sources}
