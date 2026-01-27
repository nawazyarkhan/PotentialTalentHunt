import gradio as gr
import json
from typing import List, Dict, Optional
import requests
from urllib.parse import urljoin
#from app.functions import search_by_vector, normalize

# Configuration
FASTAPI_BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = "/health"
SEARCH_ENDPOINT = "/search_jds"
STAR_ENDPOINT = "/star_rank"

# Global variables
starred_candidates = {}
last_candidates = {}  # faiss_id (str) -> {title, location, id}
api_available = False

def check_api_health():
    """Check if FastAPI server is running and healthy"""
    global api_available
    try:
        response = requests.get(urljoin(FASTAPI_BASE_URL, HEALTH_ENDPOINT), timeout=2)
        api_available = response.status_code == 200
        return api_available
    except requests.exceptions.RequestException:
        api_available = False
        return False

def search_candidates(job_description: str, top_k: int = 10, threshold: float = 0.6) -> tuple:
    """Search for candidates matching job description via FastAPI endpoint"""
    try:
        if not job_description.strip():
            return "<p style='color:orange;'>Please enter a job description.</p>", []
        
        if not api_available and not check_api_health():
            msg = """<p style='color:red;'>
            ❌ FastAPI server is not running. Please start it with:<br>
            <code>cd C:\\Users\\My-PC\\OneDrive\\apziva\\PotentialTalent && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload</code>
            </p>"""
            return msg, []
        
        # Call FastAPI endpoint
        payload = {
            "query": job_description,
            "top_k": top_k,
            "threshold": threshold
        }
        
        response = requests.post(
            urljoin(FASTAPI_BASE_URL, SEARCH_ENDPOINT),
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            return f"<p style='color:red;'>Error: API returned status {response.status_code}</p>", []
        
        data = response.json()
        results = data.get("results", [])
        
        # Format results as HTML table
        if not results:
            return "<p>No candidates found matching the job description.</p>", []
        
        html_output = """
        <table border='1' cellpadding='10' style='border-collapse: collapse; width: 100%;'>
        <tr style='background-color: #f0f0f0;'>
            <th>Rank</th><th>Job Title</th><th>Location</th><th>Score</th><th>Candidate ID</th>
        </tr>
        """
        
        results_list = []
        for result in results:
            faiss_id = result.get("faiss_id", "")
            rank = result.get("rank", "")
            job_title = result.get("job_title", "N/A")
            location = result.get("location", "N/A")
            score = result.get("score", 0)
            
            is_starred = "⭐" if str(faiss_id) in starred_candidates else ""
            star_indicator = f" {is_starred}" if is_starred else ""
            
            html_output += f"""
            <tr>
                <td>{rank}</td>
                <td>{job_title}{star_indicator}</td>
                <td>{location}</td>
                <td style='text-align: center;'><b>{score:.4f}</b></td>
                <td style='text-align: center;'><code>{faiss_id}</code></td>
            </tr>
            """
            
            # cache for later lookup
            last_candidates[str(faiss_id)] = {
                "title": job_title,
                "location": location,
                "id": faiss_id
            }

            results_list.append({
                "Rank": rank,
                "Job Title": job_title,
                "Location": location,
                "Score": f"{score:.4f}",
                "ID": faiss_id
            })
        
        html_output += "</table>"
        return html_output, results_list
    
    except requests.exceptions.Timeout:
        return "<p style='color:red;'>⏱️ Request timeout. FastAPI server may be slow or offline.</p>", []
    except requests.exceptions.ConnectionError:
        msg = f"""<p style='color:red;'>
        ❌ Cannot connect to FastAPI server at {FASTAPI_BASE_URL}. 
        Please ensure it's running on port 8000.
        </p>"""
        return msg, []
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>", []

def star_candidate(faiss_id: int, job_title: str) -> str:
    """Star a candidate"""
    try:
        faiss_id_str = str(faiss_id)
        if faiss_id_str in starred_candidates:
            del starred_candidates[faiss_id_str]
            status = f"Unstarred: {job_title}"
            icon = "☆"
        else:
            starred_candidates[faiss_id_str] = {"title": job_title, "id": faiss_id}
            status = f"Starred: {job_title}"
            icon = "⭐"
        
        return f"<p style='color:green;'>{icon} {status}</p><p>Total starred: {len(starred_candidates)}</p>"
    
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

def get_starred_candidates() -> str:
    """Get all starred candidates with their info"""
    if not starred_candidates:
        return "<p style='color:#888;'>No candidates starred yet. Use the Search tab to find and star candidates.</p>"
    
    html_output = """
    <table border='1' cellpadding='10' style='border-collapse: collapse; width: 100%;'>
    <tr style='background-color: #f0f0f0;'>
        <th>Candidate ID</th><th>Job Title</th><th>Location</th><th>Status</th>
    </tr>
    """
    
    for faiss_id, info in starred_candidates.items():
        html_output += f"""
        <tr>
            <td><code>{faiss_id}</code></td>
            <td>{info.get('title', 'N/A')}</td>
            <td>{info.get('location', 'N/A')}</td>
            <td style='text-align: center;'>⭐</td>
        </tr>
        """
    
    html_output += f"</table><p style='margin-top: 10px;'>Total starred: {len(starred_candidates)}</p>"
    return html_output

def get_reranked_candidates(top_k: int = 10, threshold: float = 0.6) -> str:
    """Get 10 reranked candidates based on starred selections via star_rank endpoint"""
    if not starred_candidates:
        return "<p style='color:#888;'>No candidates starred. Star candidates to see reranked results.</p>"
    
    try:
        if not api_available and not check_api_health():
            return "<p style='color:red;'>❌ FastAPI server is not running.</p>"
        
        # Build a query string from starred candidate titles
        starred_titles = [info.get('title', '') for info in starred_candidates.values()]
        query = " ".join(starred_titles)
        
        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold
        }
        
        response = requests.post(
            urljoin(FASTAPI_BASE_URL, STAR_ENDPOINT),
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            return f"<p style='color:red;'>Error: API returned status {response.status_code}</p>"
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return "<p>No reranked candidates found.</p>"
        
        html_output = """
        <table border='1' cellpadding='10' style='border-collapse: collapse; width: 100%;'>
        <tr style='background-color: #e8f5e9;'>
            <th>Rank</th><th>Job Title</th><th>Location</th><th>Score</th><th>Candidate ID</th>
        </tr>
        """
        
        for result in results:
            faiss_id = result.get("faiss_id", "")
            rank = result.get("rank", "")
            job_title = result.get("job_title", "N/A")
            location = result.get("location", "N/A")
            score = result.get("score", 0)

            # cache reranked candidates too
            last_candidates[str(faiss_id)] = {
                "title": job_title,
                "location": location,
                "id": faiss_id
            }
            
            html_output += f"""
            <tr>
                <td>{rank}</td>
                <td>{job_title}</td>
                <td>{location}</td>
                <td style='text-align: center;'><b>{score:.4f}</b></td>
                <td style='text-align: center;'><code>{faiss_id}</code></td>
            </tr>
            """
        
        html_output += "</table>"
        return html_output
    
    except requests.exceptions.Timeout:
        return "<p style='color:red;'>⏱️ Request timeout.</p>"
    except requests.exceptions.ConnectionError:
        return f"<p style='color:red;'>❌ Cannot connect to FastAPI server.</p>"
    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p>"

# Create Gradio interface
with gr.Blocks(title="Potential Talent Search & Ranking") as demo:
    gr.Markdown("""
    # 🎯 Potential Talent Search & Ranking
    ### Find the best matching candidates using semantic search
    """)
    
    # Display API status
    api_status = gr.HTML(value="<p>Checking API status...</p>")
    
    with gr.Tabs():
        # Tab 1: Job Search
        with gr.TabItem("🔍 Search Candidates"):
            with gr.Row():
                with gr.Column(scale=3):
                    job_desc = gr.Textbox(
                        label="Job Description",
                        placeholder="Enter job description, required skills, qualifications, experience needed...",
                        lines=6,
                        info="Be detailed for better matching results"
                    )
                with gr.Column(scale=1):
                    top_k = gr.Slider(
                        label="Top K Results",
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        info="Number of candidates to return"
                    )
                    threshold = gr.Slider(
                        label="Similarity Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        info="Minimum similarity score (0-1)"
                    )
            
            search_btn = gr.Button("🔍 Search Candidates", variant="primary", size="lg")
            results_output = gr.HTML(label="Search Results")
            results_data = gr.State([])
            
            def do_search(job_desc, top_k, threshold):
                html, data = search_candidates(job_desc, top_k, threshold)
                return html, data
            
            search_btn.click(
                fn=do_search,
                inputs=[job_desc, top_k, threshold],
                outputs=[results_output, results_data]
            )
            
            # Star candidate section
            gr.Markdown("### Star a Candidate")
            with gr.Row():
                candidate_id_input = gr.Number(
                    label="Candidate ID",
                    info="Enter the ID from the 'Candidate ID' column",
                    precision=0
                )
                star_action = gr.Dropdown(
                    choices=["⭐ Star", "☆ Unstar"],
                    value="⭐ Star",
                    label="Action"
                )
                star_btn = gr.Button("Apply", variant="secondary")
            
            star_status = gr.HTML()
            
            # Starred candidates list (will be referenced later in Tab 2)
            starred_output_ref = gr.State(None)

            # Inline starred list under Search tab
            gr.Markdown("### ⭐ Starred Candidates (this session)")
            starred_output_inline = gr.HTML(label="Starred Candidates List", value=get_starred_candidates())
            
            def do_star_action(candidate_id, action):
                if candidate_id is None or candidate_id == "":
                    return "<p style='color:orange;'>Please enter a Candidate ID</p>", get_starred_candidates()
                try:
                    faiss_id = int(candidate_id)
                    # Get job title from results
                    job_title = "Candidate"
                    location = "N/A"

                    # Prefer cached candidates from last search / rerank
                    if str(faiss_id) in last_candidates:
                        job_title = last_candidates[str(faiss_id)].get("title", "Candidate")
                        location = last_candidates[str(faiss_id)].get("location", "N/A")
                    elif results_data.value:
                        for r in results_data.value:
                            rid = r.get("ID")
                            try:
                                rid_int = int(rid)
                            except Exception:
                                rid_int = rid
                            if rid_int == faiss_id:
                                job_title = r.get("Job Title", "Candidate")
                                location = r.get("Location", "N/A")
                                break
                    elif str(faiss_id) in starred_candidates:
                        job_title = starred_candidates[str(faiss_id)].get("title", "Candidate")
                        location = starred_candidates[str(faiss_id)].get("location", "N/A")
                    
                    if "Star" in action:
                        starred_candidates[str(faiss_id)] = {
                            "title": job_title,
                            "location": location,
                            "id": faiss_id
                        }
                        status = f"<p style='color:green;'>✅ ⭐ Starred: {job_title}</p>"
                    else:
                        if str(faiss_id) in starred_candidates:
                            del starred_candidates[str(faiss_id)]
                            status = f"<p style='color:green;'>✅ ☆ Unstarred: {job_title}</p>"
                        else:
                            status = f"<p style='color:orange;'>⚠️ Candidate not in starred list</p>"
                    
                    # Return both status and updated starred list
                    return status, get_starred_candidates()
                except ValueError:
                    return "<p style='color:red;'>❌ Invalid Candidate ID</p>", get_starred_candidates()
                except Exception as e:
                    return f"<p style='color:red;'>❌ Error: {str(e)}</p>", get_starred_candidates()
            
        
        # Tab 2: Starred Candidates
        with gr.TabItem("⭐ Starred Candidates"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📌 Your Starred Candidates")
                    starred_output = gr.HTML(label="Starred Candidates List")
                with gr.Column(scale=1):
                    rerank_top_k = gr.Slider(
                        label="Top K Results",
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        info="Candidates to return"
                    )
                    rerank_threshold = gr.Slider(
                        label="Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        info="Min score"
                    )
            
            refresh_btn = gr.Button("🔄 Generate Reranked Candidates (Based on Starred)", variant="primary", size="lg")
            reranked_output = gr.HTML(label="Reranked Candidates List")
            
            def do_refresh_starred():
                return get_starred_candidates()
            
            def do_get_reranked(top_k, threshold):
                return get_reranked_candidates(int(top_k), float(threshold))
            
            refresh_btn.click(
                fn=do_get_reranked,
                inputs=[rerank_top_k, rerank_threshold],
                outputs=reranked_output
            )
            
            # Initial load
            demo.load(fn=get_starred_candidates, outputs=starred_output)
            
            # Connect star button to update starred output
            star_btn.click(
                fn=do_star_action,
                inputs=[candidate_id_input, star_action],
                outputs=[star_status, starred_output]
            )
            
            # Also update inline starred list in Search tab
            star_btn.click(
                fn=do_star_action,
                inputs=[candidate_id_input, star_action],
                outputs=[star_status, starred_output_inline]
            )
            
            # Update reranked when star button is clicked
            star_btn.click(
                fn=do_get_reranked,
                inputs=[rerank_top_k, rerank_threshold],
                outputs=reranked_output
            )
        
        # Tab 3: About & Instructions
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About Potential Talent Search
            
            This application uses **Semantic Search** with embeddings to find the best matching candidates 
            from a database based on job descriptions and requirements.
            
            ### Features:
            - 🔍 **Vector-based semantic search** using E5 embeddings
            - ⭐ **Star/bookmark candidates** for future reference
            - 📊 **Similarity scoring** for ranking candidates
            - ⚙️ **Adjustable parameters** for fine-tuned searches
            - 🚀 **FastAPI backend** for reliable, scalable performance
            
            ### How to Use:
            1. **Go to Search Tab**: Enter a detailed job description with required skills and qualifications
            2. **Adjust Parameters**: 
               - *Top K Results*: How many candidates to show (1-50)
               - *Similarity Threshold*: Minimum match score (0-1, higher = stricter)
            3. **Search**: Click "Search Candidates" to find matches
            4. **Star Candidates**: Mark candidates you're interested in
            5. **View Starred**: Check "Starred Candidates" tab to see your bookmarks
            
            ### Technical Details:
            - **Model**: Sentence Transformers (intfloat/e5-base-v2)
            - **Backend**: FastAPI with FAISS vector index
            - **Server**: Running at """ + FASTAPI_BASE_URL + """
            
            ### Tips for Best Results:
            - ✅ Be detailed in your job description
            - ✅ Include key skills and qualifications
            - ✅ Mention experience requirements
            - ✅ Adjust threshold if you get too few/many results
            - ✅ Start with higher threshold for stricter matching
            """)

def update_api_status():
    """Update API status display"""
    if check_api_health():
        return "<p style='color:green;'>✅ FastAPI server is running and healthy</p>"
    else:
        return f"""<p style='color:red;'>❌ FastAPI server is not accessible at {FASTAPI_BASE_URL}</p>
        <p>To start the server, run:</p>
        <code>cd PotentialTalent && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload</code>"""

# Launch the app
if __name__ == "__main__":
    # Check API status on startup
    api_status_html = update_api_status()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

