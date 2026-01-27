# PotentialTalentHunt

An **end-to-end AI-powered talent search and ranking system** built with **FastAPI**, **FAISS**, **Gradio**, and modern MLOps practices. The project demonstrates how to go from **data analysis and experimentation** to a **production-ready ML application** deployed using **Docker** and **AWS ECS**.

The system enables recruiters or analysts to search and rank potential candidates (job profiles / resumes) using semantic vector similarity and configurable ranking logic (including a STAR-style ranking endpoint).

---
Apziva project code: AbTK0iyp4MBSTNLb
---
## 🚀 Project Overview

This repository showcases a complete ML application lifecycle:

- 📊 **Exploratory analysis & modeling** using Jupyter notebooks
- 🧠 **Semantic search** with embeddings + FAISS
- 🔌 **FastAPI backend** for model inference and ranking APIs
- 🖥️ **Gradio GUI** for interactive user access
- 🐳 **Dockerized application** for reproducible builds
- ☁️ **AWS ECS deployment** for scalable, production-grade serving

---

## 🚀 Features

- 🔍 **Semantic Search** over job descriptions / candidate profiles
- 🧠 **Vector Embeddings + FAISS Index** for fast similarity search
- ⭐ **STAR Ranking Endpoint** (extensible ranking logic)
- 📊 **Notebook-based analysis & experimentation**
- 🖥️ **Gradio-based GUI** for interactive querying
- 🐳 **Dockerized FastAPI app** for deployment
- ☁️ **AWS ECS deployment** enabling an end-to-end ML system

---

## 🧱 Tech Stack

- **Python 3.10**
- **FastAPI** – REST API backend
- **FAISS** – Vector similarity search
- **Sentence Transformers / Embeddings**
- **Gradio** – Lightweight web-based GUI
- **Pandas / NumPy** – Data processing
- **Docker** – Containerization
- **AWS ECS** – Cloud deployment
- **Uvicorn** – ASGI server

---

## 📁 Project Structure

```text
PotentialTalentHunt/
│
├── app/
│   ├── main.py                     # FastAPI application (API entry point)
│   ├── functions.py                # Core ML logic: FAISS search & ranking
│   ├── Dockerfile                  # Docker configuration for deployment
│   └── __init__.py
│
├── notebooks/
│   ├── gr_gui.py                   # Gradio GUI application
│   ├── potentialtalent_V1_2.ipynb  # End-to-end analysis & experimentation
│   └── experiments.ipynb           # Additional modeling / ranking tests
│
├── data/
│   ├── raw/                        # Raw candidate / JD data
│   ├── processed/                  # Cleaned / vectorized data
│   └── index/                      # FAISS index files
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── images/
    └── screenshots/                # UI / architecture screenshots
```

---

## 📊 Notebooks

- **`potentialtalent_V1_2.ipynb`**  \
  Contains exploratory data analysis, embedding generation, FAISS indexing, and ranking experiments.

- **`gr_gui.py`**  \
  Implements a **Gradio-based GUI** that connects to the FastAPI backend, allowing users to run semantic searches and view ranked candidates interactively.

- **`expermiments.ipynb`**  \
  Testing FASTAPI endpoints at various stage i.e. locally, docker and on ECS (post deployment)

---

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns API status.

---

### Semantic Search
```
POST /search_jds
```
**Payload:**
```json
{
  "query": "text to search",
  "top_k": 10,
  "threshold": 0.6
}
```

---

### STAR Rank Search
```
POST /star_rank
```
Same payload structure as `/search_jds`, with ranking logic designed to be extended.

---

## 🖥️ Running the Project Locally

### 1️⃣ Create & Activate Environment
```bash
conda create -n tf310 python=3.10
conda activate tf310
pip install -r requirements.txt
```

---

### 2️⃣ Run FastAPI Backend
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at:
```
http://localhost:8000/docs
```

---

### 3️⃣ Run Gradio GUI
```bash
python notebooks/gr_gui.py
```

Gradio will launch a local web UI (typically at `http://127.0.0.1:7860`) that communicates with the FastAPI backend.

---

## 🏗️ System Architecture

### End-to-End Architecture (Local → Docker → AWS ECS)

```text
┌────────────────────────────┐
│        Local Machine       │
│                            │
│  Jupyter Notebook          │
│  (analysis & experiments)  │
│        │                   │
│        ▼                   │
│  Gradio UI (gr_gui.py)     │
│        │  HTTP requests    │
│        ▼                   │
│  FastAPI Backend           │
│  (FAISS + ML logic)        │
│        │                   │
└ ────────┼──────────────────┘
         │ Docker Image
         ▼
┌────────────────────────────┐
│        Docker Container    │
│                            │
│  FastAPI App               │
│  + FAISS Index             │
│                            │
└ ────────┼──────────────────┘
         │ Push Image
         ▼
┌────────────────────────────┐
│        Docker Hub          │
│  (Container Registry)      │
└ ────────┼──────────────────┘
         │ Deploy
         ▼
┌────────────────────────────┐
│        AWS ECS             │
│  (Fargate / EC2)           │
│                            │
│  FastAPI Service           │
│  Scalable Inference API    │
└────────────────────────────┘
```

This architecture demonstrates how the project transitions from **local experimentation** to a **cloud-deployed, production-ready ML service**.

---

## 🖥️ Gradio + FastAPI Interaction Flow

```text
User (Browser)
      │
      ▼
Gradio UI (gr_gui.py)
      │  REST calls
      ▼
FastAPI Backend (app/main.py)
      │
      ▼
FAISS Index + Embedding Model
      │
      ▼
Ranked Candidate Results
```

Gradio acts as a lightweight frontend, while FastAPI serves as the core inference and ranking engine.

---

## 🏗️ Architecture

### End-to-End Flow (Local → Docker → AWS ECS)

```text
┌───────────────────────────┐
│        Local Dev          │
│  (Notebook / Scripts)     │
└─────────────┬─────────────┘
              │ 1) Build embeddings + FAISS index
              ▼
┌───────────────────────────┐
│   FAISS Index + Metadata   │
│ (data/index + processed)   │
└─────────────┬─────────────┘
              │ 2) Serve model/search via API
              ▼
┌───────────────────────────┐
│      FastAPI Backend       │
│        app/main.py         │
│  /health /search /star_rank│
└─────────────┬─────────────┘
              │ 3) Containerize
              ▼
┌───────────────────────────┐
│          Docker            │
│  Image: fastapi + faiss    │
└─────────────┬─────────────┘
              │ 4) Push image
              ▼
┌───────────────────────────┐
│     Docker Hub            │
└─────────────┬─────────────┘
              │ 5) Deploy service
              ▼
┌───────────────────────────┐
│   Amazon ECS (Fargate/EC2) │
│  Task/Service + Load Bal.  │
└─────────────┬─────────────┘
              │ 6) User access
              ▼
┌───────────────────────────┐
│   Client UI / Consumers    │
│  Gradio UI + other clients │
└───────────────────────────┘
```

### Gradio + FastAPI Interaction

```text
User (Browser)
   │
   │ HTTP
   ▼
Gradio UI (notebooks/gr_gui.py)
   │
   │ REST calls (requests)
   ▼
FastAPI API (app/main.py)
   │
   │ vectorize + search
   ▼
FAISS Index + Candidate Metadata
   │
   ▼
Ranked Results (JSON) → Gradio table
```

---

## 🐳 Docker & AWS ECS Deployment

- The FastAPI application is containerized using **Docker**
- The Docker image is pushed to **Amazon ECR**
- The image is deployed on **AWS ECS** (Fargate or EC2-backed) behind a load balancer for scalable inference

---

## 🚢 Deployment Guide (ECR → ECS)

Below is a practical, high-level deployment flow for pushing your container to **Amazon ECR** and running it on **Amazon ECS**.

### 1) Prerequisites
- AWS account + IAM permissions for **ECR** and **ECS**
- AWS CLI installed and configured:
  ```bash
  aws configure
  ```
- Docker installed

### 2) Create a repository on Docker Hub


### 3) Authenticate/Logon to  Docker 

### 4) Build the Docker image
From the repo root (or where your Dockerfile lives):
```bash
docker build -t potential-talent-hunt -f app/Dockerfile .
```

### 5) Tag the image for Docker Hub
```bash
docker tag potential-talent-hunt:latest \
  <userid>/potential-talent-hunt:latest
```

### 6) Push the image to Docker Hub
```bash
docker push <userid>/potential-talent-hunt:latest
```

### 7) Create an ECS Cluster
- In AWS Console → **ECS** → **Clusters** → Create
- Choose **Fargate** (recommended for simplicity)

### 8) Create a Task Definition
- Container image: your Docker Hub image URI
- Port mapping: `8000` (FastAPI)
- CPU/Memory: choose based on FAISS/model size
- Environment variables: (optional) `MODEL_PATH`, `INDEX_PATH`, etc.

### 9) Create a Service
- Launch type: **Fargate**
- Desired tasks: 1+ (scale as needed)
- Networking: select VPC/subnets, enable public IP if required
- Load balancing: optional but recommended for production

### 10) Verify
- Open the service endpoint
- Confirm:
  - `/health` returns OK
  - `/docs` loads Swagger UI
  - `/search_jds` and `/star_rank` respond

---


## 🙌 Acknowledgements

Built as part of engagement at APZIVA as an applied AI & MLOps exploration project , demonstrating how to productionize semantic search systems end to end.

Please feel free to reach out to me over linkedin for any questions/discussions with regards to this project. 

