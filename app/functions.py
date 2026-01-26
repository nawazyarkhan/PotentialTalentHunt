import faiss
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/e5-base-v2"   # great for retrieval; or "all-mpnet-base-v2"
INDEX_PATH = "../data/jd_model_e5.index"
META_PATH  = "../data/jd_meta_model_e5.pkl"

# Load default index and meta (can be overridden in function calls)
# try:
#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "rb") as f:
#         meta = pickle.load(f)
# except Exception as e:
#     print(f"Warning: Could not load index/meta at module level: {e}")
#     index = None
#     meta = None


## vector search with optional reranking
def search_by_vector(
    qv,
    index_obj=None,          # pass custom index, or uses module-level default
    meta_obj=None,           # pass custom meta, or uses module-level default
    top_k=10,
    retrieve_k=200,
    min_score=None,          # e.g. 0.35; or None to disable
    rel_drop=None,           # e.g. 0.10 keeps results within 0.10 of best
    location_preference=None,# e.g. "london" (already normalized)
    location_penalty=0.08,   # subtract if mismatch
    location_boost=0.02,     # add if match
    title_terms=None,        # list[str] tokens to boost when present
    title_boost=0.02
):
    # Use provided index/meta or fall back to module-level
    idx = index_obj if index_obj is not None else index
    metadata = meta_obj if meta_obj is not None else meta
    
    if idx is None or metadata is None:
        raise ValueError("Index and meta must be provided or loaded at module level")
    
    # Ensure qv shape (1, d) and normalized for cosine/IP
    qv = normalize(qv).reshape(1, -1).astype("float32")
    scores, idxs = idx.search(qv, retrieve_k)

    scores = scores[0]
    idxs = idxs[0]

    # If nothing found
    if len(scores) == 0:
        return []

    best = float(scores[0]) if idxs[0] != -1 else None

    candidates = []
    for score, i in zip(scores, idxs):
        if i == -1:
            continue

        s = float(score)
        row = metadata[int(i)]

        # Soft min_score (optional)
        if min_score is not None and s < float(min_score):
            continue

        # Relative drop cutoff (optional)
        if rel_drop is not None and best is not None and (best - s) > float(rel_drop):
            continue

        # --- lightweight rerank adjustments ---
        adj = 0.0

        # Location preference
        if location_preference:
            job_loc = (row.get("location") or "").strip().lower()
            # if you have location_norm in meta, use that instead
            if location_preference in job_loc:
                adj += location_boost
            else:
                adj -= location_penalty

        # Title term boost
        if title_terms:
            jt = (row.get("job_title") or "").lower()
            if any(t.lower() in jt for t in title_terms):
                adj += title_boost

        final_score = s + adj

        candidates.append((final_score, s, int(i), row))

    # Sort by reranked score
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Build output
    results = []
    for rank, (final_score, raw_score, i, row) in enumerate(candidates[:top_k], start=1):
        results.append({
            "rank": rank,
            "id": row.get("id"),
            "faiss_id": i,
            "score": round(float(final_score), 4),
            "raw_score": round(float(raw_score), 4),
            "job_title": row.get("job_title"),
            "location": row.get("location"),
        })

    return results

import numpy as np

### normalize vector to unit length
def normalize(v: np.ndarray) -> np.ndarray:
     v = v.astype(np.float32)
     norm = np.linalg.norm(v)
     if norm == 0:
         return v
     return v / norm

#### text preprocessing function
import re
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text