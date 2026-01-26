import os
import pickle
from typing import Optional, List

### import functions from functions.py
from functions import *

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

app = FastAPI(title="Job Description Semantic Search API", version="1.0.0")

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
        raise RuntimeError(f"index not found at {INDEX_PATH}")
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

## ---------------- Core search logic ----------------
# Use the search_by_vector function from functions.py
# Use normalize from functions.py

# ---------------- API endpoints ----------------
@app.get("/health")
def health():
    ok = (model is not None) and (index is not None) and (meta is not None)
    return {"ok": ok, "model": MODEL_NAME, "index_path": INDEX_PATH, "meta_path": META_PATH}
@app.post("/search_jds", response_model=SearchResponse)
def search_jds(req: SearchRequest):
    # index = faiss.read_index(INDEX_PATH)
    # with open(META_PATH, "rb") as f:
    # meta = pickle.load(f)
    # Encode query
    qv = model.encode(req.query, convert_to_numpy=True, normalize_embeddings=True)

    results = search_by_vector(
        qv,
        top_k=req.top_k,
        min_score=req.threshold
    )

    return {
        "query": req.query,
        "top_k": req.top_k,
        "threshold": req.threshold,
        "count": len(results),
        "results": results,
    }
@app.post("/star_rank", response_model=SearchResponse)
def star_rank(req: SearchRequest):
    qv = model.encode(req.query, convert_to_numpy=True, normalize_embeddings=True)

    results = search_by_vector(
        qv,
        top_k=req.top_k,
        min_score=req.threshold,
    )

    return {
        "query": req.query,
        "top_k": req.top_k,
        "threshold": req.threshold,
        "count": len(results),
        "results": results,
    }