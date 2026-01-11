import os
import pickle
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------- Config ----------------
MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/e5-base-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "jd.index")
META_PATH  = os.getenv("META_PATH", "jd_meta.pkl")

st.set_page_config(page_title="JD Semantic Search (FAISS)", layout="wide")

# ---------------- Load once (cached) ----------------
@st.cache_resource
def load_assets(model_name: str, index_path: str, meta_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata pickle not found: {meta_path}")

    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Guard against meta shorter than FAISS index
    if hasattr(index, "ntotal") and isinstance(meta, list) and len(meta) < index.ntotal:
        raise RuntimeError(f"Meta has {len(meta)} rows but FAISS index has {index.ntotal} vectors")

    return model, index, meta

def search_jds(model, index, meta, query: str, top_k: int, threshold: float, retrieve_k: int):
    # E5 best practice: prefix query
    q = "query: " + query
    qv = model.encode([q], normalize_embeddings=True).astype(np.float32)

    scores, idxs = index.search(qv, retrieve_k)

    results = []
    rank = 0

    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        if float(score) < threshold:
            continue

        rank += 1
        row = meta[int(i)]

        results.append({
            "rank": rank,                 # 1..top_k (after filtering)
            "faiss_id": int(i),           # row position in FAISS
            "score": round(float(score), 4),
            "id": row.get("id"),
            "job_title": row.get("job_title"),
            "location": row.get("location"),
        })

        if len(results) >= top_k:
            break

    return results

# ---------------- UI ----------------
st.title("JD Semantic Search (FAISS + Sentence Transformers)")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Return results (top_k)", min_value=1, max_value=50, value=10, step=1)
    threshold = st.slider("Score threshold", min_value=-1.0, max_value=1.0, value=0.6, step=0.01)

    # Retrieve more candidates than we return, so threshold filtering doesn’t starve results
    retrieve_k_default = max(int(top_k) * 5, int(top_k))
    retrieve_k = st.number_input("FAISS retrieve_k (candidates)", min_value=int(top_k), max_value=500, value=retrieve_k_default, step=5)

    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Index: `{INDEX_PATH}`")
    st.caption(f"Meta: `{META_PATH}`")

query = st.text_area(
    "Enter a job description / profile text to search similar job descriptions:",
    value="Aspiring Human Resources Professional",
    #height=140
    height=100
)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("Search", type="primary", use_container_width=True)
with col2:
    st.write("")

# Load assets
try:
    model, index, meta = load_assets(MODEL_NAME, INDEX_PATH, META_PATH)
except Exception as e:
    st.error(f"Failed to load assets: {e}")
    st.stop()

if run:
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("Searching..."):
        results = search_jds(model, index, meta, query.strip(), int(top_k), float(threshold), int(retrieve_k))

    st.subheader(f"Results ({len(results)} found)")
    if not results:
        st.info("No results above the threshold. Try lowering the threshold or increasing retrieve_k.")
    else:
        st.dataframe(results, use_container_width=True)

         # Optional: pretty cards
        st.divider()
        for r in results:
             with st.container():
                 st.markdown(
                     f"**#{r['rank']} — {r.get('job_title') or 'Untitled'}**  \n"
                     f"Score: `{r['score']}` | FAISS ID: `{r['faiss_id']}` | ID: `{r.get('id')}` | Location: `{r.get('location')}`"
                 )
