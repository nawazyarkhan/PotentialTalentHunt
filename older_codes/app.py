import os
import pickle
from typing import Optional, List

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from typing import Optional, Union

# from fastapi.middleware.cors import CORSMiddleware


# ---------------- Config (env-friendly) ----------------
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/e5-base-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "jd.index")
META_PATH  = os.getenv("META_PATH", "jd_meta.pkl")
DEFAULT_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.6"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "10"))

app = FastAPI(title="JD Semantic Search API", version="1.0.0")

model: Optional[SentenceTransformer] = None
index = None
meta = None


# ---------------- Request/Response Schemas ----------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search text (e.g., job description or profile)")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=100, description="Max candidates to retrieve from FAISS")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=-1.0, le=1.0, description="Minimum similarity score to return")


class SearchResult(BaseModel):
    rank: int                      # added (1 = best)
    faiss_id: int                  #  added (row in FAISS index)
    score: float
    id: Optional[Union[int,str]] = None
    job_title: Optional[str] = None
    location: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    threshold: float
    count: int
    results: List[SearchResult]


# ---------------- Startup: load once ----------------
@app.on_event("startup")
def load_assets():
    global model, index, meta

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"FAISS index not found at {INDEX_PATH}")
    if not os.path.exists(META_PATH):
        raise RuntimeError(f"Metadata pickle not found at {META_PATH}")

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    # Basic sanity check: meta length should match index size (often does)
    # Not strictly required, but helpful
    if hasattr(index, "ntotal") and isinstance(meta, list) and len(meta) < index.ntotal:
        # It's okay if meta is longer, but if it's shorter you'll index out of bounds.
        raise RuntimeError(f"Meta has {len(meta)} rows but FAISS index has {index.ntotal} vectors")


# ---------------- Core search logic ----------------
def _search_jds(query: str, top_k: int = 10, threshold: float = 0.6):
    if model is None or index is None or meta is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    q = "query: " + query
    qv = model.encode([q], normalize_embeddings=True).astype(np.float32)

    # Retrieve more candidates to allow for threshold filtering
    retrieve_k = max(top_k * 3, top_k)
    scores, idxs = index.search(qv, retrieve_k)

    results = []
    rank = 0

    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        if float(score) < threshold:
            continue

        rank += 1
        results.append({
            "rank": rank,
            "faiss_id": int(i),
            "score": round(float(score), 4),
            "id": meta[i].get("id"),
            "job_title": meta[i].get("job_title"),
            "location": meta[i].get("location"),
        })

        # ✅ break MUST be inside the loop
        if len(results) >= top_k:
            break

    return results

# ---------------- API endpoints ----------------
@app.get("/health")
def health():
    ok = (model is not None) and (index is not None) and (meta is not None)
    return {"ok": ok, "model": MODEL_NAME, "index_path": INDEX_PATH, "meta_path": META_PATH}


@app.post("/search_jds", response_model=SearchResponse)
def search_jds(req: SearchRequest):
    results = _search_jds(req.query, req.top_k, req.threshold)
    return {
        "query": req.query,
        "top_k": req.top_k,
        "threshold": req.threshold,
        "count": len(results),
        "results": results,
    }

# # ---------------- CORS Middleware ----------------
# app = FastAPI(title="JD Semantic Search API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost",
#         "http://localhost:3000",
#         "http://127.0.0.1",
#         "http://127.0.0.1:8000",
#     ],  # 👈 add frontend URLs here
#     allow_credentials=True,
#     allow_methods=["*"],   # GET, POST, OPTIONS, etc.
#     allow_headers=["*"],   # Authorization, Content-Type, etc.
# )
