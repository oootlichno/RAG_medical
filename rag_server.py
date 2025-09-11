import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

""" DB_DIR = os.getenv("CHROMA_DIR", "/data/chroma_db_v_export")
COLL = os.getenv("CHROMA_COLLECTION", "langchain")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.gguf")
 """
DB_DIR     = os.getenv("CHROMA_DIR", "/opt/data/chroma_db_v_export")
COLL       = os.getenv("CHROMA_COLL", "langchain")
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/data/models/model.gguf")
API_KEY = os.getenv("RAG_API_KEY", "")

client   = chromadb.PersistentClient(DB_DIR)
col      = client.get_collection(COLL)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm      = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4960, verbose=False)

from fastapi import Header, HTTPException

app = FastAPI()

class Query(BaseModel):
    question: str
    k: int = 4
    temperature: float = 0.1
    max_tokens: int = 256

class Answer(BaseModel):
    answer: str
    sources: List[Dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rag", response_model=Answer)
def rag(q: Query, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    q_vec = embedder.encode(q.question).tolist()
    hits = col.query(
        query_embeddings=[q_vec],
        n_results=q.k,
        include=["documents", "distances", "metadatas"]  # no "ids"
    )
    docs = hits["documents"][0]
    context = "\n\n".join(docs)

    prompt = (
        "Answer strictly from the context. If the answer is not in the context, "
        "say: I don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q.question}\nAnswer:"
    )

    out = llm.create_chat_completion(
        messages=[
            {"role":"system","content":"You are concise and cautious."},
            {"role":"user","content":prompt}
        ],
        temperature=q.temperature,
        max_tokens=q.max_tokens
    )
    answer = out["choices"][0]["message"]["content"]

    sources = []
    for meta, doc, dist in zip(hits["metadatas"][0], hits["documents"][0], hits["distances"][0]):
        sources.append({
            "distance": float(dist),
            "title": meta.get("source") if isinstance(meta, dict) else None,
            "snippet": doc[:300]
        })

    return {"answer": answer, "sources": sources}
