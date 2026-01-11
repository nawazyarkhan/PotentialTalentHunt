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

def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _get_vector_from_faiss(index, faiss_id: int) -> np.ndarray:
    """
    Returns the stored vector for a given FAISS id.
    Works for many index types that support reconstruct().
    """
    if not hasattr(index, "reconstruct"):
        raise RuntimeError(
            "This FAISS index does not support reconstruct(). "
            "To rerank using starring, save embeddings (e.g., embeddings.npy) or use an index that supports reconstruct."
        )
    v = index.reconstruct(int(faiss_id))
    return _normalize(np.array(v, dtype=np.float32))


def search_by_vector(index, meta, qv: np.ndarray, top_k: int, threshold: float, retrieve_k: int):
    qv = _normalize(qv).reshape(1, -1).astype(np.float32)
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
            "rank": rank,
            "faiss_id": int(i),
            "score": round(float(score), 4),
            "id": row.get("id"),
            "job_title": row.get("job_title"),
            "location": row.get("location"),
        })

        if len(results) >= top_k:
            break

    return results

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
        results = search_by_vector(index, meta, model.encode([query.strip()], normalize_embeddings=True).astype(np.float32)[0], int(top_k), float(threshold), int(retrieve_k))

    st.subheader(f"Results ({len(results)} found)")
    if not results:
        st.info("No results above the threshold. Try lowering the threshold or increasing retrieve_k.")
    else:
        st.dataframe(results, use_container_width=True)

        #  # Optional: pretty cards
        # st.divider()
        # for r in results:
        #      with st.container():
        #          st.markdown(
        #              f"**#{r['rank']} — {r.get('job_title') or 'Untitled'}**  \n"
        #              f"Score: `{r['score']}` | FAISS ID: `{r['faiss_id']}` | ID: `{r.get('id')}` | Location: `{r.get('location')}`"
        #          )

# --- rerank logic -------------------
# --- Session state initialization ---
if "base_query_text" not in st.session_state:
    st.session_state.base_query_text = ""
if "base_qv" not in st.session_state:
    st.session_state.base_qv = None  # np.ndarray
if "starred_faiss_ids" not in st.session_state:
    st.session_state.starred_faiss_ids = []  # list[int]
if "prototype_vec" not in st.session_state:
    st.session_state.prototype_vec = None  # np.ndarray
if "alpha" not in st.session_state:
    st.session_state.alpha = 0.5  # how strongly to trust the star

with st.sidebar:
    st.subheader("Rerank settings")
    st.session_state.alpha = st.slider("Star influence (alpha)", 0.0, 1.0, st.session_state.alpha, 0.05)
    if st.button("Clear stars / reset rerank", use_container_width=True):
        st.session_state.starred_faiss_ids = []
        st.session_state.prototype_vec = None
        st.session_state.base_qv = None
        st.experimental_rerun()

# --- When user clicks Search: compute base query vector and reset stars for this query ---
if run:
    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    # New query => reset stars (important: stars are per-role/query)
    if query.strip() != st.session_state.base_query_text:
        st.session_state.base_query_text = query.strip()
        st.session_state.starred_faiss_ids = []
        st.session_state.prototype_vec = None

        # Encode base query vector (normalized)
        base_q = "query: " + st.session_state.base_query_text
        st.session_state.base_qv = model.encode([base_q], normalize_embeddings=True)[0].astype(np.float32)

# --- If base_qv exists, compute current query vector (reranked or not) and search ---
if st.session_state.base_qv is not None:
    base_qv = st.session_state.base_qv

    # If we have a prototype (mean of starred vectors), mix it with base query
    if st.session_state.prototype_vec is not None:
        alpha = float(st.session_state.alpha)
        qv_current = _normalize((1 - alpha) * base_qv + alpha * st.session_state.prototype_vec)
        st.info(f"Reranking active: {len(st.session_state.starred_faiss_ids)} starred candidate(s).")
    else:
        qv_current = _normalize(base_qv)

    # Run search
    with st.spinner("Searching..."):
        results = search_by_vector(index, meta, qv_current, int(top_k), float(threshold), int(retrieve_k))

    st.subheader(f"Results ({len(results)} shown)")
    if not results:
        st.info("No results above threshold. Lower threshold or increase retrieve_k.")
        st.stop()

    # --- Render results with ⭐ buttons ---
    st.caption("Click ⭐ to set an 'ideal candidate' and rerank the list instantly.")
    for r in results:
        c1, c2, c3, c4, c5 = st.columns([0.6, 1.2, 3.5, 1.5, 1.2])

        with c1:
            st.write(f"**#{r['rank']}**")

        with c2:
            st.code(r["score"])

        with c3:
            st.write(f"**{r.get('job_title') or 'Untitled'}**  \nID: {r.get('id')} | FAISS: {r['faiss_id']}")

        with c4:
            st.write(r.get("location") or "")

        with c5:
            already_starred = r["faiss_id"] in st.session_state.starred_faiss_ids
            label = "⭐ Starred" if already_starred else "☆ Star"
            if st.button(label, key=f"star_{r['faiss_id']}", use_container_width=True, disabled=already_starred):
                # Get the starred candidate vector from FAISS
                v_star = _get_vector_from_faiss(index, r["faiss_id"])

                # Update prototype = mean of starred vectors (running mean)
                if st.session_state.prototype_vec is None:
                    st.session_state.prototype_vec = v_star
                else:
                    # running mean of starred vectors
                    n = len(st.session_state.starred_faiss_ids)
                    st.session_state.prototype_vec = _normalize((st.session_state.prototype_vec * n + v_star) / (n + 1))

                st.session_state.starred_faiss_ids.append(r["faiss_id"])
                st.experimental_rerun()

