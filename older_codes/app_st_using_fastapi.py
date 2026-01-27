import json
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

top_k = 10
threshold = 0.6
endpoint = "/search_jds"
clear = False
check_health = False

st.set_page_config(page_title="JD Semantic Search GUI", layout="wide")

st.title("Job Description Semantic Search — GUI (Streamlit + FastAPI)")


# ----------------------------
# Helpers
# ----------------------------
def safe_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError:
        # show server-provided error details if available
        try:
            detail = r.json()
        except Exception:
            detail = {"detail": r.text}
        raise RuntimeError(f"HTTP {r.status_code} error from API: {detail}") from None
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from None


def safe_get(url: str, timeout: int = 30) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError:
        try:
            detail = r.json()
        except Exception:
            detail = {"detail": r.text}
        raise RuntimeError(f"HTTP {r.status_code} error from API: {detail}") from None
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}") from None


def to_df(resp: Dict[str, Any]) -> pd.DataFrame:
    results = resp.get("results", []) or []
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)

    # Put most useful columns first if they exist
    preferred = ["rank", "score", "raw_score", "job_title", "location", "id", "faiss_id"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


# ----------------------------
# Sidebar controls
# ----------------------------
clear = False
check_health = False
with st.sidebar:
    st.header("API Settings")

    api_base = st.text_input("FastAPI base URL", value="http://54.170.13.169:8000")
    endpoint = st.radio(
        "Endpoint",
        options=["/search_jds", "/star_rank"],
        index=0,
        help="Pick which FastAPI route to call."
    )
    # after endpoint = st.radio(...)
if "prev_endpoint" not in st.session_state:
    st.session_state["prev_endpoint"] = endpoint

if endpoint != st.session_state["prev_endpoint"]:
    st.session_state.pop("last_resp", None)
    st.session_state.pop("last_df", None)
    st.session_state["prev_endpoint"] = endpoint

    st.divider()
    st.subheader("Search params")
    top_k = st.slider("top_k", min_value=1, max_value=100, value=10, step=1)
    threshold = st.slider("threshold", min_value=-1.0, max_value=1.0, value=0.6, step=0.01)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        check_health = st.button("Check /health", use_container_width=True)
    with col_b:
        clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state.pop("last_resp", None)
    st.session_state.pop("last_df", None)
    st.toast("Cleared.")


# ----------------------------
# Health check
# ----------------------------
if check_health:
    try:
        health = safe_get(f"{api_base.rstrip('/')}/health")
        st.success("API is reachable.")
        st.json(health)
    except Exception as e:
        st.error(str(e))


# ----------------------------
# Main input
# ----------------------------
st.subheader("Query")
query = st.text_area(
    "Paste a job description / profile text to search against the FAISS index:",
    height=220,
    placeholder="Type or paste your query here..."
)

run = st.button("Run search", type="primary", use_container_width=True)

if run:
    if not query.strip():
        st.warning("Please enter a non-empty query.")
    else:
        payload = {"query": query, "top_k": int(top_k), "threshold": float(threshold)}
        url = f"{api_base.rstrip('/')}{endpoint}"
        with st.spinner(f"Calling {url} ..."):
            try:
                resp = safe_post(url, payload)
                df = to_df(resp)
                st.session_state["last_resp"] = resp
                st.session_state["last_df"] = df
                st.success(f"Done. Returned {resp.get('count', 0)} result(s).")
            except Exception as e:
                st.error(str(e))


# ----------------------------
# Output
# ----------------------------
resp: Optional[Dict[str, Any]] = st.session_state.get("last_resp")
df: Optional[pd.DataFrame] = st.session_state.get("last_df")

if resp:
    st.divider()
    st.subheader("Results")

    left, right = st.columns([2, 1])

    with left:
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download results
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="search_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No results matched your threshold/top_k settings.")

    with right:
        st.markdown("**Request params**")
        st.code(
            json.dumps(
                {"query": resp.get("query"), "top_k": resp.get("top_k"), "threshold": resp.get("threshold")},
                indent=2
            ),
            language="json"
        )
        st.markdown("**Raw JSON response**")
        st.json(resp)
else:
    st.info("Run a search to see results here.")
