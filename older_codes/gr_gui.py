import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

# Import functions from app
from app.functions import search_by_vector, normalize

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
MODEL_NAME = "intfloat/e5-base-v2"
INDEX_PATH = str(DATA_DIR / "jd_model_e5.index")
META_PATH = str(DATA_DIR / "jd_meta_model_e5.pkl")

# Global variables
model = None
index = None
meta = None
starred_candidates = {}

def load_resources():
    """Load model, index, and metadata"""
    global model, index, meta
    
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    
    if index is None:
        if os.path.exists(INDEX_PATH):
            index = faiss.read_index(INDEX_PATH)
        else:
            raise RuntimeError(f"Index not found at {INDEX_PATH}")
    
    if meta is None:
        if os.path.exists(META_PATH):
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
        else:
            raise RuntimeError(f"Metadata not found at {META_PATH}")

def search_candidates(job_description: str, top_k: int = 10, threshold: float = 0.6) -> str:
    """Search for candidates matching job description"""
    try:
        load_resources()
        
        # Encode query
        qv = model.encode(job_description, convert_to_numpy=True, normalize_embeddings=True)
        
        # Search using FAISS
        results = search_by_vector(
            qv,
            index_obj=index,
            meta_obj=meta,
            top_k=top_k,
            min_score=threshold
        )
        
        # Format results as HTML table
        if not results:
            return "<p>No candidates found matching the job description.</p>"
        
        html_output = "<table border='1' cellpadding='10'><tr><th>Rank</th><th>Job Title</th><th>Location</th><th>Score</th><th>Action</th></tr>"
        
        for result in results:
            faiss_id = result.get("faiss_id", "")
            rank = result.get("rank", "")
            job_title = result.get("job_title", "N/A")
            location = result.get("location", "N/A")
            score = result.get("score", 0)
            
            is_starred = "⭐" if faiss_id in starred_candidates else "☆"
            
            html_output += f"""
            <tr>
                <td>{rank}</td>
                <td>{job_title}</td>
                <td>{location}</td>
                <td>{score:.4f}</td>
                <td><button onclick="alert('Starred: {job_title}')">{is_starred}</button></td>
            </tr>
            """
        
        html_output += "</table>"
        return html_output
    
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

def star_candidate(faiss_id: int, job_title: str) -> str:
    """Star a candidate"""
    try:
        if faiss_id in starred_candidates:
            del starred_candidates[faiss_id]
            status = f"Unstarred: {job_title}"
        else:
            starred_candidates[faiss_id] = {"title": job_title}
            status = f"Starred: {job_title}"
        
        return f"<p style='color:green;'>{status}</p><p>Total starred: {len(starred_candidates)}</p>"
    
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

def get_starred_candidates() -> str:
    """Get all starred candidates"""
    if not starred_candidates:
        return "<p>No candidates starred yet.</p>"
    
    html_output = "<table border='1' cellpadding='10'><tr><th>Candidate ID</th><th>Job Title</th></tr>"
    
    for faiss_id, info in starred_candidates.items():
        html_output += f"<tr><td>{faiss_id}</td><td>{info.get('title', 'N/A')}</td></tr>"
    
    html_output += "</table>"
    return html_output

# Create Gradio interface
with gr.Blocks(title="Potential Talent Search & Ranking") as demo:
    gr.Markdown("# 🎯 Potential Talent Search & Ranking")
    gr.Markdown("Find the best matching candidates from the database based on job descriptions.")
    
    with gr.Tabs():
        # Tab 1: Job Search
        with gr.TabItem("🔍 Search Candidates"):
            with gr.Row():
                with gr.Column(scale=3):
                    job_desc = gr.Textbox(
                        label="Job Description",
                        placeholder="Enter job description, requirements, skills needed...",
                        lines=6
                    )
                with gr.Column(scale=1):
                    top_k = gr.Slider(
                        label="Top K Results",
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1
                    )
                    threshold = gr.Slider(
                        label="Similarity Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05
                    )
            
            search_btn = gr.Button("🔍 Search Candidates", variant="primary", size="lg")
            results_output = gr.HTML(label="Search Results")
            
            search_btn.click(
                fn=search_candidates,
                inputs=[job_desc, top_k, threshold],
                outputs=results_output
            )
        
        # Tab 2: Starred Candidates
        with gr.TabItem("⭐ Starred Candidates"):
            gr.Markdown("### Your Starred Candidates")
            starred_output = gr.HTML(label="Starred Candidates List")
            refresh_btn = gr.Button("🔄 Refresh", size="lg")
            
            refresh_btn.click(
                fn=get_starred_candidates,
                outputs=starred_output
            )
            
            # Initial load
            demo.load(fn=get_starred_candidates, outputs=starred_output)
        
        # Tab 3: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ### About Potential Talent Search
            
            This application uses **Semantic Search** to find the best matching candidates from a database
            based on job descriptions.
            
            **Features:**
            - 🔍 Vector-based semantic search using E5 embeddings
            - ⭐ Star/bookmark your favorite candidates
            - 📊 Similarity scoring for ranking
            - ⚙️ Adjustable search parameters
            
            **How it works:**
            1. Enter a job description with required skills and qualifications
            2. Adjust the number of results and similarity threshold
            3. Click "Search Candidates" to find matches
            4. Star candidates you're interested in
            5. View all starred candidates in the "Starred Candidates" tab
            
            **Model:** Sentence Transformers (intfloat/e5-base-v2)
            """)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

