import os
import shutil
from typing import List, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    from llama_cpp import Llama 
except Exception:
    from llama_cpp.llama import Llama 

from starlette.middleware.cors import CORSMiddleware
import numpy as np


# -----------------------------
# Config
# -----------------------------
def _expand(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p))

DB_DIR = _expand(os.getenv("CHROMA_DIR", "/opt/data/chroma_db_v_export"))
COLL = os.getenv("CHROMA_COLL", "langchain")
MODEL_PATH = _expand(os.getenv("MODEL_PATH", "/opt/data/models/model.gguf"))
API_KEY = os.getenv("RAG_API_KEY", "")

N_THREADS = int(os.getenv("LLAMA_THREADS", str(max(1, (os.cpu_count() or 2)))))
N_CTX = int(os.getenv("LLAMA_CTX", "4096"))
N_BATCH = int(os.getenv("LLAMA_N_BATCH", "128"))

CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]

# controls startup forced index rebuild 
CHROMA_FORCE_REINDEX = os.getenv("CHROMA_FORCE_REINDEX", "0") == "1"
# silence noisy telemetry
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
# retrieval mode: auto | bruteforce | chroma
RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "auto").lower()


# -----------------------------
# Helpers
# -----------------------------
def _nuke_all_indexes() -> int:
    """
    Recursively delete *all* 'index' dirs inside DB_DIR (safe: does not touch sqlite or doc data).
    Returns number of deleted index dirs.
    """
    count = 0
    if os.path.isdir(DB_DIR):
        for root, dirs, _ in os.walk(DB_DIR):
            if "index" in dirs:
                idxp = os.path.join(root, "index")
                shutil.rmtree(idxp, ignore_errors=True)
                count += 1
    return count


def _safe_reindex() -> bool:
    """Compatibility shim: delete nested index dirs; return True if anything was removed."""
    return _nuke_all_indexes() > 0


def _init_client_and_collection():
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
    col = client.get_collection(name=COLL)
    return client, col


# -----------------------------
# Initialize Chroma, Embeddings, and LLM
# -----------------------------
if CHROMA_FORCE_REINDEX:
    _safe_reindex()

client, col = _init_client_and_collection()

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMBED_MODEL_ID, device="cpu")

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=0,
    n_threads=N_THREADS,
    n_ctx=N_CTX,
    n_batch=N_BATCH,
    verbose=False,
)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Medical RAG API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return {"service": "rag-medical", "status": "ok"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": os.path.exists(MODEL_PATH),
        "db_dir_exists": os.path.isdir(DB_DIR),
        "collection": COLL,
        "retrieval_mode": RETRIEVAL_MODE,
    }


@app.get("/debug/versions")
def debug_versions():
    import numpy
    return {
        "chromadb": chromadb.__version__,
        "numpy": numpy.__version__,
        "db_dir": DB_DIR,
        "collection": COLL,
        "embed_model": EMBED_MODEL_ID,
    }


@app.get("/debug/collections")
def debug_collections():
    try:
        cols = client.list_collections()
        out = []
        for c in cols:
            try:
                cc = client.get_collection(c.name)
                cnt = cc.count()
            except Exception:
                cnt = None
            out.append({"name": c.name, "count": cnt})
        return {"collections": out, "active": COLL}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/db-list")
def debug_db_list():
    items = []
    if os.path.isdir(DB_DIR):
        for name in sorted(os.listdir(DB_DIR)):
            p = os.path.join(DB_DIR, name)
            items.append({"name": name, "is_dir": os.path.isdir(p), "size": os.path.getsize(p) if os.path.isfile(p) else None})
    return {"db_dir": DB_DIR, "items": items}


@app.get("/debug/embed-dim")
def debug_embed_dim():
    v = embedder.encode("test")
    dim = int(v.shape[-1]) if hasattr(v, "shape") else len(v)
    return {"embedding_model": EMBED_MODEL_ID, "dim": dim}


@app.get("/debug/count")
def debug_count():
    try:
        return {"collection": COLL, "count": col.count()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/debug/peek")
def debug_peek():
    try:
        got = col.get(limit=3, include=["documents", "metadatas"])
        docs = []
        doc_list = got.get("documents") or []
        meta_list = got.get("metadatas") or []
        for d, m in zip(doc_list, meta_list):
            title = None
            if isinstance(m, dict):
                title = m.get("title") or m.get("source")
            docs.append({
                "id": (m or {}).get("id") if isinstance(m, dict) else None,
                "title": title,
                "snippet": (d or "")[:300],
            })
        return {"ok": True, "sample": docs}
    except Exception as e:
        return {"ok": False, "error": f"peek failed: {e}"}


@app.post("/debug/reindex")
def debug_reindex(x_api_key: Optional[str] = Header(default=None)):
    global client, col
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    changed = _safe_reindex()
    client, col = _init_client_and_collection()
    return {"ok": True, "reindexed": changed}


def _bruteforce_knn(question_vec, k: int):
 
    q = np.asarray(question_vec, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-12)

    total = col.count()
    if total == 0:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    batch = int(os.getenv("BRUTE_BATCH", "512"))
    best = []  # (-sim, idx_str, meta, doc)

    offset = 0
    while offset < total:
        got = col.get(
            limit=min(batch, total - offset),
            offset=offset,
            include=["documents", "metadatas"],   # IMPORTANT: no "embeddings" here
        )
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        if not docs:
            break

        # Re-embed docs locally to avoid hitting vector segment
        E = embedder.encode(docs, convert_to_numpy=True).astype(np.float32)

        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        sims = (En @ qn)

        for i, sim in enumerate(sims.tolist()):
            best.append((-sim, f"offset:{offset+i}", metas[i] if i < len(metas) else None, docs[i] if i < len(docs) else ""))

        # occasional prune to keep memory bounded
        if len(best) > max(5000, 8*k):
            best.sort()
            best = best[:max(5000, 8*k)]

        offset += len(docs)

    if not best:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    best.sort()
    top = best[:k]
    ids_out = [b[1] for b in top]
    metas_out = [b[2] for b in top]
    docs_out = [b[3] for b in top]
    dists_out = [float(1.0 - (-b[0])) for b in top] 

    # Mark that we used doc re-embedding (handy for debugging)
    for i, m in enumerate(metas_out):
        if isinstance(m, dict):
            m["_retrieval"] = "docs_reembedded"
        else:
            metas_out[i] = {"_retrieval": "docs_reembedded"}

    return {
        "ids": [ids_out],
        "distances": [dists_out],
        "metadatas": [metas_out],
        "documents": [docs_out],
    }


def _query_auto(q_vec, k: int, include=None):
    """
    Robust retrieval:
      - 'bruteforce'  => always brute-force KNN (bypasses Chroma index completely)
      - 'chroma'      => try Chroma ANN only (may raise)
      - 'auto' (default) => try Chroma once, then reindex+retry, then brute-force
    """
    global client, col
    include = include or ["documents", "metadatas"]

    mode = RETRIEVAL_MODE
    if mode == "bruteforce":
        return _bruteforce_knn(q_vec, k)

    if mode == "chroma":
        return col.query(query_embeddings=[q_vec], n_results=k, include=include)

    # AUTO
    try:
        return col.query(query_embeddings=[q_vec], n_results=k, include=include)
    except Exception:
        _safe_reindex()
        client, col = _init_client_and_collection()
        try:
            return col.query(query_embeddings=[q_vec], n_results=k, include=include)
        except Exception:
            return _bruteforce_knn(q_vec, k)


@app.get("/debug/test-retrieval")
def debug_test_retrieval(q: str = "What is appendicitis?", k: int = 4, mode: Optional[str] = None):
    """
    Quick check endpoint:
      - mode can override env (auto|bruteforce|chroma) 
    """
    question = (q or "").strip()
    if not question:
        return {"ok": False, "error": "Empty query"}

    try:
        q_vec = embedder.encode(question)
        q_vec = q_vec.tolist() if hasattr(q_vec, "tolist") else list(q_vec)
    except Exception as e:
        return {"ok": False, "error": f"Embedding error: {e}"}

    global RETRIEVAL_MODE
    prev = RETRIEVAL_MODE
    if mode:
        RETRIEVAL_MODE = mode.lower()

    try:
        hits = _query_auto(q_vec, max(1, min(int(k), 10)), include=["documents", "metadatas"])
        return {"ok": True, "mode": RETRIEVAL_MODE, "preview": {
            "docs": (hits.get("documents") or [[]])[0][:2],
            "metas": (hits.get("metadatas") or [[]])[0][:2],
        }}
    finally:
        RETRIEVAL_MODE = prev


@app.post("/rag", response_model=Answer)
def rag(q: Query, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    question = (q.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    k = max(1, min(int(q.k), 10))

    try:
        q_vec = embedder.encode(question)
        q_vec = q_vec.tolist() if hasattr(q_vec, "tolist") else list(q_vec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # ALWAYS protected by robust fallbacks
    hits = _query_auto(q_vec, k, include=["documents", "metadatas"])

    docs = (hits.get("documents") or [[]])[0]
    metas = (hits.get("metadatas") or [[]])[0]
    dists = (hits.get("distances") or [[]])[0] if "distances" in hits else []

    if not docs:
        return {"answer": "I don't know.", "sources": []}

    context = "\n\n".join(docs)
    prompt = (
        "Answer strictly from the context. If the answer is not in the context, say: I don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are concise and cautious."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(q.temperature),
            max_tokens=int(q.max_tokens),
        )
        answer = out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    sources: List[Dict] = []
    for i, (meta, doc) in enumerate(zip(metas, docs)):
        dist = float(dists[i]) if i < len(dists) else None
        title = None
        if isinstance(meta, dict):
            title = meta.get("title") or meta.get("source")
        src = {"title": title, "snippet": (doc or "")[:300]}
        if dist is not None:
            src["distance"] = dist
        sources.append(src)

    return {"answer": answer, "sources": sources}
