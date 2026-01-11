# LM Match Service

**[简体中文](README.zh-CN.md)** | **English**

## Project Overview

LM Match Service is a FastAPI-based job-resume matching service. The project is currently in the M7 stage, building upon explainable ranking, RAG explanations, evaluation systems, and Streamlit interactive interface, with the addition of a complete Learning to Rank (LTR) system that optimizes ranking effectiveness through machine learning for more accurate job recommendations.

### Current Features

#### M1: Basic Matching Features
- ✅ Health check endpoint (`/health`)
- ✅ Job-resume matching endpoint (`/match`) - Returns structured matching results
- ✅ Data models defined using Pydantic (JobPosting, Resume, MatchResponse)
- ✅ Skill-set based matching algorithm (no LLM)
- ✅ Provides match score, matched skills, skill gaps, and learning suggestions

#### M2: Semantic Recommendation Features
- ✅ Job recommendation endpoint (`/recommend_jobs`) - Top-K recommendations based on semantic similarity
- ✅ Text embedding using local sentence-transformers model
- ✅ Cosine similarity calculation and ranking
- ✅ Batch job dataset (jobs.jsonl) and resume dataset (resumes.jsonl)
- ✅ Fully local operation, no paid API required

#### M3: Explainable Ranking Features
- ✅ Lightweight ranking layer - Introduces multi-dimensional scoring on top of embedding retrieval
- ✅ Skills vocabulary (180+ skills) - Standardized skill matching
- ✅ YAML configuration - Adjust ranking weights without modifying code
- ✅ Multi-dimensional features:
  - `embedding`: Semantic similarity (embedding score)
  - `skill_overlap`: Skill coverage rate
  - `keyword_bonus`: Keyword matching bonus
  - `gap_penalty`: Missing critical skills penalty
- ✅ Explainability - Automatically generates detailed explanations for the top-ranked job

#### M4: RAG Explanation Layer
- ✅ Evidence construction - Extracts structured evidence from jobs and resumes
- ✅ Intelligent retrieval - Selects the most relevant evidence fragments based on semantic similarity
- ✅ LLM generation - Uses large language models to generate evidence-based explanations
- ✅ Three-dimensional analysis - Provides for each recommended job:
  - `explanation`: Why this job suits the candidate
  - `gap_analysis`: Which key skills or qualifications the candidate lacks
  - `improvement_suggestions`: Specific actionable improvement suggestions
- ✅ Prevents hallucination - Strictly evidence-based generation, LLM only used for explanation layer, not for ranking
- ✅ **Automatic skill extraction and merging** - Automatically extracts skills from resume text (education/projects/experience) to avoid overly strict matching
- ✅ **Soft skill filtering** - Missing soft skills (e.g., Communication, Leadership) are not counted in gap_penalty

#### M5: Evaluation and Weak Supervision Label Generation (Old Version, Partially Replaced by M7)
- ✅ Data ID alignment - Added job_id and resume_id to jobs.jsonl and resumes.jsonl
- ⚠️ LLM-assisted label generation - Uses GPT-4o-mini to generate 0-3 graded labels for Top-15 recommendations (**M7 upgraded to full 1-5 labels**)
- ✅ Weak labels - Quickly generates large-scale annotation data
- ✅ Evaluation metrics implementation:
  - Precision@K - Measures recommendation accuracy
  - NDCG@K - Measures ranking quality (considers position weights)
- ⚠️ Evaluation method - Simple label validation (**M7 upgraded to LOOCV + Ablation Study**)
- ❌ **Deprecated files**: labels_final.csv (manual correction template), run_eval.py (evaluation script), eval_results.json (evaluation results)

#### M6: Streamlit Interactive Interface
- ✅ Streamlit Web interface - Lightweight interactive frontend
- ✅ Multiple resume input methods - Text box input or upload TXT file
- ✅ Job selection - Choose from jobs.jsonl database
- ✅ Top-K parameter configuration - Flexibly adjust recommendation count
- ✅ One-click matching - Calls backend `/recommend_jobs` endpoint
- ✅ Visualized result display - Job information, match scores, skill comparison
- ✅ Detailed explanation generation - Click button to call `/explain` endpoint
- ✅ Backend status monitoring - Real-time check of backend service availability

#### M7: Learning to Rank (LTR) System
- ✅ Full weak labels (1-5 scale) - Covers all resume×job combinations (750 pairs: 15 resumes × 50 jobs)
- ✅ Pairwise LTR training - Ranking model based on Logistic Regression
- ✅ LOOCV evaluation - Leave-One-Out Cross-Validation (essential for small data)
- ✅ Ablation study - Compares three ranking methods: embedding_only, heuristic, LTR
- ✅ Evaluation metrics - NDCG@5/10, Precision@5/10
- ✅ FastAPI use_ltr switch - Frontend can toggle LTR ranking on/off
- ✅ Streamlit LTR toggle - UI one-click enable/disable LTR feature
- ✅ Model persistence - joblib save/load LTR model

#### General Features
- ✅ RESTful API design
- ✅ Auto-generated API documentation (Swagger UI / ReDoc)

## Project Structure

```
lm/
├── backend/
│   ├── main.py              # FastAPI main application file
│   ├── schemas.py           # Pydantic data model definitions
│   ├── test_match.py        # Match endpoint test file
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Environment variable configuration example (M4)
│   ├── services/            # Business logic services
│   │   ├── __init__.py         # Service package initialization
│   │   ├── embedding.py        # Text embedding service (M2)
│   │   ├── retrieval.py        # Retrieval and ranking service (M2)
│   │   ├── ranking.py          # Explainable ranking service (M3)
│   │   ├── rag.py              # RAG explanation layer service (M4)
│   │   └── utils.py            # Utility functions (skill extraction and merging) (M4.1)
│   ├── src/                 # LTR source code module (M7 added)
│   │   └── ranking/
│   │       ├── __init__.py     # Ranking package initialization
│   │       ├── features.py     # Feature extraction and vectorization
│   │       ├── pairwise.py     # Pairwise training data construction
│   │       └── ltr_logreg.py   # Pairwise Logistic Regression model
│   ├── scripts/             # Scripts directory (M7 added)
│   │   └── eval_ablation.py    # LOOCV + Ablation evaluation script
│   ├── models/              # Model save directory (M7 added)
│   │   └── ltr_logreg.joblib   # Trained LTR model
│   ├── results/             # Evaluation results directory (M7 added)
│   │   └── ablation_results.json  # Ablation study results
│   ├── config/              # Configuration files (M3 added)
│   │   └── ranking_config.yaml # Ranking weight configuration
│   ├── eval/                # Evaluation module (M5/M7 updated)
│   │   ├── generate_labels.py  # Full 1-5 weak labels generation script (M7 updated)
│   │   ├── labels_suggested.jsonl  # Full 1-5 labels (M7: 750 pairs)
│   │   ├── labels_final.csv    # Manual correction template (deprecated)
│   │   ├── metrics.py          # Evaluation metrics (Precision@K, NDCG@K)
│   │   ├── run_eval.py         # Evaluation run script (deprecated, use scripts/eval_ablation.py)
│   │   ├── eval_results.json   # Evaluation results (deprecated)
│   │   └── eval_report.md      # Evaluation report (M7 updated)
│   └── data/
│       ├── sample_job.json        # Sample job data
│       ├── sample_resume.json     # Sample resume data
│       ├── jobs.jsonl             # Batch job data (50 items, with job_id) (M5/M7)
│       ├── resumes.jsonl          # Batch resume data (15 items, with resume_id) (M5/M7)
│       └── skills_vocabulary.txt  # Skills vocabulary (180+ skills) (M3)
├── frontend/                # Frontend interface (M6 added)
│   ├── streamlit_app.py     # Streamlit interactive interface
│   └── requirements.txt     # Frontend dependencies (Streamlit, requests)
├── .gitignore               # Git ignore file configuration
└── README.md                # Project documentation
```

## How to Run

### 1. Environment Requirements

- Python 3.8+
- pip

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure Environment Variables (M4 Added)

To use the RAG explanation layer feature, you need to configure the OpenAI API Key:

```bash
# Copy environment variable example file
cp .env.example .env

# Edit .env file, fill in your OpenAI API Key
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

**How to get OpenAI API Key:**
1. Visit https://platform.openai.com/api-keys
2. Login or register OpenAI account
3. Create a new API Key
4. Fill the API Key into the `.env` file

**Note:** If you don't configure the API Key, the recommendation endpoint will still work normally, but the `explanation`, `gap_analysis`, and `improvement_suggestions` fields for each recommended job will be `null`.

### 5. Start Service

```bash
# Method 1: Using uvicorn command
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Method 2: Run main.py directly
python main.py
```

After the service starts, access http://localhost:8000

### 6. View API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Data Model Description

### JobPosting (Job Information)

```json
{
  "title": "Job Title",
  "responsibilities": "Job responsibilities description",
  "requirements_text": "Job requirements description",
  "skills": ["Skill1", "Skill2", "..."],
  "company": "Company Name (optional)",
  "location": "Location (optional)",
  "level": "Job Level (optional)"
}
```

### Resume (Resume Information)

```json
{
  "education": "Educational Background",
  "projects": "Project Experience",
  "skills": ["Skill1", "Skill2", "..."],
  "experience": "Work Experience"
}
```

### MatchResponse (Match Result)

```json
{
  "match_score": 57,
  "matched_skills": ["Python", "FastAPI", "Docker"],
  "gaps": ["PostgreSQL", "Kubernetes", "Redis", "AWS"],
  "suggestions": [
    "Consider learning PostgreSQL to better match this position",
    "Consider learning Kubernetes to better match this position",
    "..."
  ]
}
```

## Sample Data

### Sample Job Data (backend/data/sample_job.json)

```json
{
  "title": "Senior Backend Engineer",
  "responsibilities": "Design and implement scalable backend services, lead technical architecture decisions, mentor junior developers, and collaborate with cross-functional teams to deliver high-quality software solutions.",
  "requirements_text": "5+ years of backend development experience, strong knowledge of Python and web frameworks, experience with databases and cloud platforms, excellent problem-solving skills.",
  "skills": [
    "Python",
    "FastAPI",
    "PostgreSQL",
    "Docker",
    "Kubernetes",
    "Redis",
    "AWS"
  ],
  "company": "TechCorp Inc.",
  "location": "San Francisco, CA / Remote",
  "level": "Senior"
}
```

### Sample Resume Data (backend/data/sample_resume.json)

```json
{
  "education": "Bachelor of Science in Computer Science, Stanford University, 2015-2019. Relevant coursework: Data Structures, Algorithms, Database Systems, Distributed Systems.",
  "projects": "1) E-commerce Platform - Built a scalable e-commerce backend using Python and FastAPI, serving 100k+ daily users. Implemented RESTful APIs, payment integration, and order management system. 2) Real-time Chat Application - Developed a real-time messaging system using WebSocket, Redis pub/sub, and MongoDB for message persistence. 3) DevOps Automation - Created CI/CD pipelines using Docker and GitHub Actions to automate deployment processes.",
  "skills": [
    "Python",
    "FastAPI",
    "Django",
    "Docker",
    "MongoDB",
    "Git",
    "Linux"
  ],
  "experience": "Software Engineer at StartupXYZ (2019-2023): Developed and maintained backend services using Python and FastAPI. Designed database schemas and optimized query performance. Collaborated with frontend team to integrate APIs. Implemented automated testing and deployment pipelines using Docker. Mentored 2 junior developers."
}
```

### Batch Test Datasets (JSONL Format)

To support subsequent top-k recommendation feature testing, we provide two JSON Lines format datasets:

#### backend/data/jobs.jsonl
- Contains 50 real job postings (M7 expanded)
- Covers skill domains: recommendation systems, search, NLP, LLM, CV, data engineering, backend development, machine learning, etc.
- Each line is a JSON object conforming to `JobPosting` schema

#### backend/data/resumes.jsonl
- Contains 15 resumes with different backgrounds (M7 expanded)
- Skills overlap with job data to varying degrees, suitable for testing matching algorithms
- Each line is a JSON object conforming to `Resume` schema

#### How to Load JSONL Files

Loading these files in Python for testing:

```python
import json
from schemas import JobPosting, Resume

# Load all jobs
jobs = []
with open('backend/data/jobs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        job_data = json.loads(line)
        jobs.append(JobPosting(**job_data))

print(f"Loaded {len(jobs)} jobs")

# Load all resumes
resumes = []
with open('backend/data/resumes.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        resume_data = json.loads(line)
        resumes.append(Resume(**resume_data))

print(f"Loaded {len(resumes)} resumes")
```

#### Top-k Recommendation Test Example

```python
# Example: Find top-5 matching jobs for a resume
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Load first resume (recommendation systems background)
with open('backend/data/resumes.jsonl', 'r', encoding='utf-8') as f:
    resume_data = json.loads(f.readline())

# Load all jobs and calculate match scores
matches = []
with open('backend/data/jobs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        job_data = json.loads(line)

        # Call /match endpoint
        response = client.post("/match", json={
            "job": job_data,
            "resume": resume_data
        })

        result = response.json()
        matches.append({
            "job_title": job_data["title"],
            "match_score": result["match_score"],
            "matched_skills": result["matched_skills"],
            "gaps": result["gaps"]
        })

# Sort by match score, take top-5
top_5 = sorted(matches, key=lambda x: x["match_score"], reverse=True)[:5]

print("\nTop 5 best matching jobs:")
for i, match in enumerate(top_5, 1):
    print(f"{i}. {match['job_title']} - Match Score: {match['match_score']}%")
    print(f"   Matched Skills: {', '.join(match['matched_skills'])}")
    print(f"   Skill Gaps: {', '.join(match['gaps'])}\n")
```

#### Expected Use Cases

These JSONL datasets will be used in subsequent Milestones for:
1. **Batch matching tests**: Test system performance processing multiple jobs and resumes
2. **Top-k recommendations**: Recommend the most matching k jobs for a given resume (or reverse recommendation)
3. **Ranking algorithm verification**: Verify sorting logic based on match scores
4. **Performance benchmarking**: Test response time and accuracy for large-scale matching

## How to Test Endpoints

### Test Health Check Endpoint

**Using curl:**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "message": "Service is healthy and running"
}
```

### Test Match Endpoint

#### Method 1: Using Swagger UI (Recommended)

1. Visit http://localhost:8000/docs
2. Find `POST /match` endpoint
3. Click **"Try it out"** button
4. Paste the following JSON in the Request body:

```json
{
  "job": {
    "title": "Senior Backend Engineer",
    "responsibilities": "Design and implement scalable backend services, lead technical architecture decisions, mentor junior developers, and collaborate with cross-functional teams to deliver high-quality software solutions.",
    "requirements_text": "5+ years of backend development experience, strong knowledge of Python and web frameworks, experience with databases and cloud platforms, excellent problem-solving skills.",
    "skills": [
      "Python",
      "FastAPI",
      "PostgreSQL",
      "Docker",
      "Kubernetes",
      "Redis",
      "AWS"
    ],
    "company": "TechCorp Inc.",
    "location": "San Francisco, CA / Remote",
    "level": "Senior"
  },
  "resume": {
    "education": "Bachelor of Science in Computer Science, Stanford University, 2015-2019. Relevant coursework: Data Structures, Algorithms, Database Systems, Distributed Systems.",
    "projects": "1) E-commerce Platform - Built a scalable e-commerce backend using Python and FastAPI, serving 100k+ daily users. Implemented RESTful APIs, payment integration, and order management system. 2) Real-time Chat Application - Developed a real-time messaging system using WebSocket, Redis pub/sub, and MongoDB for message persistence. 3) DevOps Automation - Created CI/CD pipelines using Docker and GitHub Actions to automate deployment processes.",
    "skills": [
      "Python",
      "FastAPI",
      "Django",
      "Docker",
      "MongoDB",
      "Git",
      "Linux"
    ],
    "experience": "Software Engineer at StartupXYZ (2019-2023): Developed and maintained backend services using Python and FastAPI. Designed database schemas and optimized query performance. Collaborated with frontend team to integrate APIs. Implemented automated testing and deployment pipelines using Docker. Mentored 2 junior developers."
  }
}
```

5. Click **"Execute"** button to execute the request
6. View the response results

**Expected Response Example:**
```json
{
  "match_score": 42,
  "matched_skills": [
    "Python",
    "FastAPI",
    "Docker"
  ],
  "gaps": [
    "PostgreSQL",
    "Kubernetes",
    "Redis",
    "AWS"
  ],
  "suggestions": [
    "Consider learning PostgreSQL to better match this position",
    "Consider learning Kubernetes to better match this position",
    "Consider learning Redis to better match this position",
    "Consider learning AWS to better match this position"
  ]
}
```

#### Method 2: Using curl

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "job": {
      "title": "Senior Backend Engineer",
      "responsibilities": "Design and implement scalable backend services",
      "requirements_text": "5+ years of backend development experience",
      "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
      "company": "TechCorp Inc.",
      "location": "Remote",
      "level": "Senior"
    },
    "resume": {
      "education": "BS Computer Science",
      "projects": "Built e-commerce platform",
      "skills": ["Python", "FastAPI", "MongoDB"],
      "experience": "4 years backend development"
    }
  }'
```

#### Method 3: Using Python Test Script

Run the project's built-in test script:

```bash
cd backend
python test_match.py
```

This script contains multiple test cases covering complete match, partial match, no match scenarios, etc.

### Test Job Recommendation Endpoint (M2 Added)

#### Method 1: Using Swagger UI (Recommended)

1. Visit http://localhost:8000/docs
2. Find `POST /recommend_jobs` endpoint
3. Click **"Try it out"** button
4. Paste the following JSON in the Request body (using sample resume data):

```json
{
  "resume": {
    "education": "Master of Science in Natural Language Processing, Carnegie Mellon University, 2019-2021. Bachelor of Science in Linguistics and Computer Science, University of Washington, 2015-2019. Relevant coursework: Deep Learning for NLP, Statistical NLP, Computational Semantics, Machine Translation.",
    "projects": "1) Conversational AI System - Built chatbot using GPT-4 and RAG, serving 500K+ users with 90% satisfaction rate. Implemented custom fine-tuning pipeline and prompt engineering framework. 2) Multilingual NER System - Developed named entity recognition system supporting 15 languages using BERT and mBERT. 3) Text Summarization Tool - Created abstractive summarization model fine-tuned on domain-specific data, deployed to production with FastAPI backend. 4) LLM Evaluation Framework - Built comprehensive evaluation pipeline for testing LLM outputs across multiple dimensions.",
    "skills": [
      "NLP",
      "LLM",
      "Transformers",
      "BERT",
      "GPT",
      "Claude",
      "Prompt Engineering",
      "RAG",
      "Fine-tuning",
      "Python",
      "spaCy",
      "Langchain",
      "PyTorch",
      "FastAPI"
    ],
    "experience": "NLP Engineer at AI Startup (2021-2024): Built LLM-powered products, implemented RAG systems, fine-tuned models for domain adaptation. NLP Research Intern at Microsoft (Summer 2020): Worked on transformer models for multilingual understanding, contributed to internal NLP libraries."
  },
  "top_k": 3
}
```

5. Click **"Execute"** button to execute the request
6. View the response results

**Expected Response Example:**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "title": "NLP Engineer - Conversational AI",
      "company": "ChatBot Solutions",
      "location": "Austin, TX",
      "level": "Mid-level",
      "similarity_score": 0.682073712348938,
      "matched_skills": [
        "Python",
        "Transformers",
        "Prompt Engineering",
        "NLP",
        "LLM",
        "BERT",
        "spaCy"
      ],
      "gap_skills": [],
      "features": {
        "embedding": 0.682073712348938,
        "skill_overlap": 1,
        "keyword_bonus": 0.85,
        "gap_penalty": 0,
        "final_score": 0.7428294849395752
      }
    },
    {
      "rank": 2,
      "title": "LLM Engineer",
      "company": "AI Startup",
      "location": "Remote",
      "level": null,
      "similarity_score": 0.6174665093421936,
      "matched_skills": [
        "Python",
        "Prompt Engineering",
        "LLM",
        "Claude",
        "RAG",
        "GPT",
        "Fine-tuning",
        "Langchain"
      ],
      "gap_skills": [
        "Vector Databases"
      ],
      "features": {
        "embedding": 0.6174665093421936,
        "skill_overlap": 0.8888888888888888,
        "keyword_bonus": 0.9,
        "gap_penalty": 0.1,
        "final_score": 0.6836532704035442
      }
    },
    {
      "rank": 3,
      "title": "NLP Research Scientist",
      "company": "AI Research Lab",
      "location": "Remote",
      "level": "Senior",
      "similarity_score": 0.6600039005279541,
      "matched_skills": [
        "Python",
        "PyTorch",
        "Transformers",
        "NLP",
        "GPT",
        "BERT"
      ],
      "gap_skills": [
        "Deep Learning",
        "Research"
      ],
      "features": {
        "embedding": 0.6600039005279541,
        "skill_overlap": 0.75,
        "keyword_bonus": 0.7,
        "gap_penalty": 0.2,
        "final_score": 0.6090015602111816
      }
    }
  ],
  "total_jobs_searched": 50,
  "explanation": "【NLP Engineer - Conversational AI】Ranked #1 for the following reasons:\n\n1. Semantic Similarity: 0.682 (Weight: 0.4)\n   - The job description is highly semantically aligned with the resume content\n\n2. Skill Coverage: 1.000 (Weight: 0.3)\n   - Matched skills (7): Python, Transformers, Prompt Engineering, NLP, LLM\n   - Missing skills (0): None\n\n3. Keyword Bonus: 0.850 (Weight: 0.2)\n   - Matches high-priority skills\n\n4. Gap Penalty: 0.000 (Weight: 0.1)\n   - Penalty applied for missing critical skills\n\nOverall Score: 0.743"
}
```

**Description (M4 Updated):**
- `similarity_score`: Cosine similarity based on semantic embeddings (between 0-1, equivalent to embedding_score)
- `matched_skills`: Intersection of resume skills and job required skills (based on standardized skills vocabulary)
- `gap_skills`: Skills required by the job but missing from the resume (M3 added)
- `features`: Explainable ranking features (M3 added)
  - `embedding`: Semantic similarity (0-1)
  - `skill_overlap`: Skill coverage rate (0-1)
  - `keyword_bonus`: Keyword bonus (0-1)
  - `gap_penalty`: Missing penalty (0-1)
  - `final_score`: Comprehensive score (weighted calculation)
- `explanation`: Detailed explanation for the top-ranked job (M3 added)
- **M4 Added Fields (for each recommended job):**
  - `explanation`: Why this job suits the candidate (evidence-based explanation)
  - `gap_analysis`: Which key skills or qualifications the candidate lacks
  - `improvement_suggestions`: Specific actionable improvement suggestions
- `total_jobs_searched`: Total number of jobs loaded from jobs.jsonl

**M4 Response Example (single recommended job):**
```json
{
  "rank": 1,
  "title": "NLP Engineer - Conversational AI",
  "company": "ChatBot Solutions",
  "location": "Austin, TX",
  "level": "Mid-level",
  "similarity_score": 0.682,
  "matched_skills": ["Python", "Transformers", "NLP", "LLM"],
  "gap_skills": [],
  "features": {
    "embedding": 0.682,
    "skill_overlap": 1.0,
    "keyword_bonus": 0.85,
    "gap_penalty": 0.0,
    "final_score": 0.743
  },
  "explanation": "This position is an excellent fit for you because your experience building conversational AI systems with GPT-4 and RAG directly aligns with the job's core requirements. Your projects demonstrate practical expertise in NLP and LLM applications, particularly in handling large-scale user interactions (500K+ users).",
  "gap_analysis": "While you have strong NLP fundamentals, the position requires experience with dialogue systems and intent recognition frameworks which are not explicitly mentioned in your resume. Additionally, production-scale deployment experience with specific chatbot frameworks would strengthen your candidacy.",
  "improvement_suggestions": "- Build a dialogue management system using Rasa or similar frameworks to demonstrate intent recognition capabilities\n- Complete a project focusing on multi-turn conversation handling and context management\n- Document your experience with A/B testing and performance optimization in production chatbot environments"
}
```

#### Method 2: Using curl

```bash
curl -X POST http://localhost:8000/recommend_jobs \
  -H "Content-Type: application/json" \
  -d '{
    "resume": {
      "education": "BS Computer Science",
      "projects": "Built recommendation systems and ML models",
      "skills": ["Python", "Machine Learning", "TensorFlow", "Recommendation Systems"],
      "experience": "3 years as ML Engineer"
    },
    "top_k": 3
  }'
```

#### Recommendation Endpoint Features (M3 Enhanced)

- **Semantic Matching (M2)**: Uses sentence-transformers local model (all-MiniLM-L6-v2) for text embedding
- **Multi-dimensional Ranking (M3)**: Comprehensive scoring combining semantic similarity, skill coverage rate, keyword bonus, and missing penalty
- **Explainability (M3)**: Automatically generates detailed explanations for the top-ranked job, explaining why it's the best match
- **Flexible Configuration (M3)**: Adjust ranking weights through YAML configuration file without modifying code
- **Standardized Skills (M3)**: Standardized matching based on 180+ skills vocabulary
- **No Paid API Required**: Fully local operation, no external API calls
- **Skill Overlap Information**: Provides precise matched skills and missing skills lists

## Technology Stack

- **FastAPI**: Modern, high-performance Python web framework
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server
- **Sentence-Transformers**: Local text embedding model (M2)
- **NumPy**: Vector computation and similarity calculation (M2)
- **PyYAML**: Configuration file management (M3)
- **OpenAI API**: LLM-generated explanation text (M4)

## Matching Algorithm Description

### M1: Skill Set-Based Exact Matching

Uses set operations for skill matching:

1. **Matched Skills** (matched_skills): Intersection of candidate skills and job required skills
2. **Skill Gaps** (gaps): Skills required by the job that the candidate doesn't have
3. **Match Score** (match_score): Percentage of matched skills relative to total job required skills
   - Formula: `match_score = (len(matched_skills) / len(job.skills)) * 100`
   - If the job has no skill requirements, returns 0
4. **Learning Suggestions** (suggestions): Provides learning suggestions for each skill gap

### M2: Semantic Embedding-Based Recommendation System

Uses sentence-transformers for semantic similarity matching:

1. **Text Embedding**:
   - Model: all-MiniLM-L6-v2 (384-dimensional vectors, local operation)
   - Job text: Concatenates title + responsibilities + requirements_text + skills
   - Resume text: Concatenates education + projects + experience + skills

2. **Similarity Calculation**:
   - Uses Cosine Similarity to calculate semantic similarity between resume and jobs
   - Similarity range: 0-1, closer to 1 indicates more similar

3. **Top-K Recommendations**:
   - Sorts by similarity score in descending order
   - Returns top-k most matching jobs
   - Includes precise skill overlap information (reuses M1 logic)

### M3: Explainable Lightweight Ranking Layer

Introduces multi-dimensional scoring mechanism on top of M2 embedding retrieval:

#### 1. Ranking Features

- **embedding (Semantic Similarity)**:
  - From M2's text embedding cosine similarity
  - Range: 0-1

- **skill_overlap (Skill Coverage Rate)**:
  - Matching rate based on standardized skills vocabulary (180+ skills)
  - Formula: `matched_skills / job_required_skills`
  - Range: 0-1

- **keyword_bonus (Keyword Bonus)**:
  - Bonus for matching high-priority skills (e.g., Python, Machine Learning, LLM, etc.)
  - High-priority skill weight 1.5x
  - Normalized to 0-1 range

- **gap_penalty (Missing Penalty)**:
  - Penalty for missing critical skills (e.g., Python, SQL, etc.)
  - Critical skill missing weight 2.0x
  - Normalized to 0-1 range

#### 2. Scoring Formula

```
final_score = w1 * embedding
            + w2 * skill_overlap
            + w3 * keyword_bonus
            - w4 * gap_penalty
```

Default weight configuration (adjustable via YAML):
- `w1 (embedding)`: 0.4
- `w2 (skill_overlap)`: 0.3
- `w3 (keyword_bonus)`: 0.2
- `w4 (gap_penalty)`: 0.1

#### 3. Configuration File

Ranking weights configured through `config/ranking_config.yaml`, supporting:
- Adjust feature weights
- Define high-priority keyword list
- Define critical skills list
- Adjust reward/penalty multipliers
- **Adjust ranking strategy without modifying code**

#### 4. Explainability

The system automatically generates detailed explanations for the top-ranked job, including:
- Feature scores for each dimension
- Matched skills list
- Missing skills list
- Comprehensive score calculation process

Example explanation output:
```
【NLP Engineer - Conversational AI】Ranked #1 for the following reasons:

1. Semantic Similarity: 0.682 (Weight: 0.4)
   - Job description highly aligned with resume content

2. Skill Coverage: 0.875 (Weight: 0.3)
   - Matched skills (7): NLP, Prompt Engineering, Python, ...
   - Missing skills (1): Dialogue Systems

3. Keyword Bonus: 0.650 (Weight: 0.2)
   - Matches high-priority skills

4. Gap Penalty: 0.100 (Weight: 0.1)
   - Penalty for missing critical skills

Overall Score: 0.723
```

### M4.1: Skills Auto-Extract & Merge

#### Background

In traditional skill matching, the system only relies on skills explicitly listed in the `resume.skills` list. This leads to the following issues:

1. **Overly Strict Matching**: Many skills are actually mentioned in `experience`, `projects`, or `education`, but not listed in the `skills` list
2. **False Skill Gaps**: For example, the resume mentions "conducted NER research" or "published papers on entity extraction," but because "NER" or "Entity Extraction" is not in the `skills` list, it's judged as a missing skill

#### Solution

The system automatically extracts skills from resume text and merges them with the user-provided skills list:

**Core Logic:**
```
merged_skills = union(
    user_provided_resume.skills,
    extracted_skills_from_resume_text
)
```

**Extraction Process:**
1. **Text Assembly**: Combines `resume.education`, `resume.projects`, `resume.experience` into one text
2. **Vocabulary Matching**: Matches based on `skills_vocabulary.txt` (contains 180+ skill words)
3. **Smart Boundary Detection**: Uses regex word boundaries (`\b`) to avoid false matches (e.g., "C" won't match "Cloud", "React" won't match "Reactivity")
4. **Special Character Handling**: Correctly handles skills with special characters like "C++", "C#", ".NET"
5. **Case Normalization**: Ignores case during matching but preserves original case from vocabulary
6. **Deduplication and Merging**: Merges extracted skills with user-provided skills, deduplicates, and returns

**Example:**
```python
# User-provided skills
resume.skills = ["Python", "Machine Learning"]

# Content mentioned in resume text
resume.projects = "Conducted research on NER and entity extraction..."
resume.experience = "Published papers on Named Entity Recognition..."

# Automatically extracted skills
extracted_skills = ["NER", "Entity Extraction", "Research", "Publication"]

# Final merged skills (used for matching)
merged_skills = ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication"]
```

#### Soft Skills Filtering

To avoid over-penalizing candidates, the system **filters out soft skills** when calculating `gap_penalty`:

**Soft Skills List** (not counted in missing penalty):
- Communication
- Leadership
- Collaboration
- Teamwork
- Problem Solving
- Critical Thinking
- Time Management
- Adaptability
- etc...

**Why filter soft skills?**
- Soft skills are important, but missing them should not be penalized as severely as technical skills
- Soft skills are difficult to quantify in resumes and easily omitted
- Soft skills are more assessed during interviews rather than resume screening hard requirements

**Note:** Soft skills will still:
- ✅ Appear in `matched_skills` (if matched)
- ✅ Appear in `gap_skills` (if missing)
- ✅ Be used for `keyword_bonus` bonus
- ✅ Appear in RAG explanation evidence
- ❌ **NOT** counted in `gap_penalty` deduction

#### Implementation Location

**New File:** `backend/services/utils.py`
- `extract_skills_from_text(text, vocab)` - Extract skills from text
- `merge_resume_skills(resume, vocab)` - Merge user skills with extracted skills
- `filter_soft_skills(skills)` - Filter soft skills
- `SOFT_SKILLS` - Soft skills constant set

**Calling Location:** `backend/services/ranking.py`'s `rank_jobs_with_features` function
```python
# === SKILLS AUTO-EXTRACT & MERGE ===
# Line 247-255
vocab_list = list(vocab)
merged_skills = merge_resume_skills(resume, vocab_list)
resume_skills_normalized = normalize_skills(merged_skills, vocab)
```

**Usage Locations:**
- ✅ `matched_skills` calculation - Uses merged skills
- ✅ `gap_skills` calculation - Uses merged skills
- ✅ `skill_overlap` calculation - Uses merged skills
- ✅ `keyword_bonus` calculation - Uses merged skills
- ✅ `gap_penalty` calculation - Uses merged skills (after filtering soft skills)

#### Acceptance Example

**Scenario:** Resume mentions NER research but doesn't list it in the skills list

```json
{
  "resume": {
    "skills": ["Python", "Machine Learning"],
    "projects": "Built NER system for entity extraction in medical texts",
    "experience": "Conducted research on Named Entity Recognition, published 2 papers",
    "education": "Thesis: Literature review of state-of-the-art NER methods"
  }
}
```

**Old Behavior (Problem):**
- `matched_skills`: ["Python", "Machine Learning"]
- `gap_skills`: ["NER", "Entity Extraction", "Research", "Publication"]  ❌ Falsely judged as missing

**New Behavior (Fixed):**
- `merged_skills`: ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication", "Literature Review"]
- `matched_skills`: ["Python", "Machine Learning", "NER", "Entity Extraction", "Research", "Publication"]
- `gap_skills`: []  ✅ Correctly identified

### M4: RAG Explanation Layer Architecture

#### RAG's Position in the System

The RAG (Retrieval-Augmented Generation) layer is a **pure explanation layer**, positioned after ranking, **not involved in job ranking logic**. The complete recommendation flow is as follows:

```
1. [M2 Semantic Retrieval] Calculate similarity between all jobs and resume using embedding
           ↓
2. [M3 Explainable Ranking] Calculate final score based on multi-dimensional features (embedding + skill + keyword + gap) and rank
           ↓
3. [M3 Top-K Selection] Select top K jobs (ranking is fixed, won't change)
           ↓
4. [M4 RAG Explanation Layer] Generate evidence-based explanations for each Top-K job
   ├─ Evidence Construction: Extract structured evidence from jobs and resumes
   ├─ Intelligent Retrieval: Select most relevant evidence fragments
   └─ LLM Generation: Generate explanation / gap_analysis / improvement_suggestions based on evidence
           ↓
5. [Return Results] Complete recommendation results including ranking, features, and RAG explanations
```

**Key Constraints:**
- M4's RAG layer **only used to generate explanation text**
- **Does not change** M3's `final_score` and ranking order
- LLM output must be evidence-based, hallucination prohibited

#### What RAG Retrieves

RAG retrieves **text fragments (chunks) from jobs and resumes**, specifically including:

**Job Evidence:**
- `title`: Job title
- `responsibilities`: Job responsibilities
- `requirements_text`: Job requirements
- `skills`: Required skills list

**Resume Evidence:**
- `education`: Educational background
- `projects`: Project experience
- `experience`: Work experience
- `skills`: Skills list

**Retrieval Process:**
1. **Text Chunking**: Split job descriptions and resume content into small fragments by sentences (about 200 characters)
2. **Semantic Embedding**: Calculate vector representations for all chunks using sentence-transformers model
3. **Similarity Calculation**: Calculate cross-similarity between job chunks and resume chunks
4. **Top-K Selection**: Select the most relevant 3 job chunks and 3 resume chunks as evidence

**Example:**
- Job chunk: `[responsibilities] Design and implement scalable NLP systems for production chatbots.`
- Resume chunk: `[projects] Built chatbot using GPT-4 and RAG, serving 500K+ users with 90% satisfaction rate.`
- These two chunks have high semantic similarity and will be selected as evidence passed to LLM

#### LLM's Role in the System

LLM (Large Language Model) **only assumes "explanation generation" role**, not involved in any ranking or recommendation decisions:

**LLM's Responsibilities:**
1. **Read Evidence**: Receive the most relevant job and resume fragments retrieved
2. **Generate Explanation**: Answer "why this job suits the candidate" based on evidence
3. **Analyze Gaps**: Point out key skills the candidate lacks based on evidence
4. **Provide Suggestions**: Give specific actionable improvement suggestions

**What LLM Does Not Do:**
- ❌ Does not calculate match scores (completed by M3 ranking layer)
- ❌ Does not decide job ranking (determined by M3 final_score)
- ❌ Does not retrieve jobs (completed by M2 embedding)
- ❌ Does not evaluate skill matching (completed by M3 skill_overlap)

**LLM Model Used:**
- Default: `gpt-4o-mini` (OpenAI)
- Advantages: Low cost, fast speed, suitable for generating short explanations
- Temperature setting: 0.3 (low temperature ensures stable, fact-based output)

#### How to Prevent LLM Fabrication

To prevent LLM hallucination, we adopt multi-layer protection measures:

**1. Evidence Constraint (Evidence Grounding)**
- LLM can only see evidence fragments selected through retrieval
- Prompt explicitly requires: "Based ONLY on the evidence provided below"
- Prohibits LLM from adding information not present in evidence

**2. Structured Prompt**
- Provides clear job evidence and resume evidence
- Explicitly lists `matched_skills` and `gap_skills` (calculated by M3)
- Requires LLM to cite specific evidence content

**3. Low Temperature Generation**
- Set `temperature=0.3` (default is 1.0)
- Low temperature makes output more deterministic and fact-based
- Reduces creative elaboration, enhances factual accuracy

**4. Formatted Output**
- Requires LLM to output in fixed format (EXPLANATION / GAP_ANALYSIS / IMPROVEMENT_SUGGESTIONS)
- Automatically parses and validates output format
- Falls back to rule-based simple explanations on failure

**5. Retrieval Quality Assurance**
- Uses same sentence-transformers model as M2 for retrieval
- Selects most relevant evidence based on cosine similarity
- Ensures evidence passed to LLM has high job-resume match

**Prompt Example Snippet:**
```
CRITICAL RULES:
- Base your analysis ONLY on the evidence provided above
- Reference specific details from the job and resume evidence
- Do not make assumptions or add information not present in the evidence
- Keep each section concise and focused
```

**Fallback Mechanism:**
If LLM API call fails (network issues, API key not set, etc.), the system falls back to rule-based simple explanations:
```python
{
    "explanation": "This position matches 4 of your skills: Python, NLP, LLM, Transformers. The overall compatibility score is 0.68.",
    "gap_analysis": "You may need to develop these skills: Dialogue Systems, Intent Recognition.",
    "improvement_suggestions": "- Review the job requirements carefully\n- Consider online courses for missing skills"
}
```

## M3 Configuration Description

### Ranking Weight Configuration

Edit `backend/config/ranking_config.yaml` to adjust ranking strategy:

```yaml
weights:
  embedding: 0.4        # Semantic similarity weight
  skill_overlap: 0.3    # Skill coverage rate weight
  keyword_bonus: 0.2    # Keyword bonus weight
  gap_penalty: 0.1      # Missing penalty weight

keywords:
  high_priority:        # High-priority keywords
    - "Python"
    - "Machine Learning"
    - "LLM"
    # ... more
  high_priority_multiplier: 1.5  # Bonus multiplier

gap_penalty:
  critical_skills:      # Critical skills
    - "Python"
    - "SQL"
  critical_penalty_multiplier: 2.0  # Penalty multiplier
```

### Skills Vocabulary

`backend/data/skills_vocabulary.txt` contains 180+ standardized skills, covering:
- Programming languages (Python, Java, JavaScript, ...)
- Web frameworks (FastAPI, Django, React, ...)
- ML/AI (Machine Learning, Deep Learning, TensorFlow, ...)
- NLP/LLM (Transformers, BERT, GPT, RAG, ...)
- Recommendation/Search (Recommendation Systems, Elasticsearch, ...)
- Data engineering (Spark, Airflow, ETL, ...)
- Cloud/Infrastructure (AWS, Docker, Kubernetes, ...)

Can add new skills to vocabulary as needed.

## M5 Evaluation Description (Old Version, Replaced by M7)

> ⚠️ **Notice**: M5 is the initial evaluation method, and its main functionality has been replaced by M7's Learning to Rank (LTR) system. M7 uses full 1-5 labels (750 pairs) and LOOCV + Ablation evaluation, which is more comprehensive and rigorous than M5's Top-15 partial labels (105 pairs, 0-3 scale). The following content is for historical reference only.

### Evaluation Objectives (M5 Old Version)

M5 introduces an initial evaluation system to quantify job recommendation system performance:
- **Data Alignment**: Added unique IDs (job_id, resume_id) to jobs.jsonl and resumes.jsonl
- **Weak Supervision Labels**: Uses LLM (GPT-4o-mini) to generate 0-3 graded labels for Top-15 recommendations (**Replaced by M7's full 1-5 labels**)
- **Quantitative Metrics**: Precision@K and NDCG@K measure recommendation quality
- **Manual Correction**: Supports manual review and correction of LLM-generated labels (**Deprecated in M7**)

### Label System (0-3 Grading) (M5 Old Version, M7 Changed to 1-5 Scale)

> ⚠️ **Deprecated**: M7 uses 1-5 label system to replace this 0-3 system.

| Label | Name | Definition |
|------|------|------|
| **0** | Not Matching | Obviously irrelevant or inconsistent direction |
| **1** | Weak Match | Some relevant points but missing key skills or direction deviation |
| **2** | Medium Match | Consistent direction, some skills met, with some skill gaps |
| **3** | Strong Match | Highly consistent direction, high key skill coverage, few skill gaps |

**Relevance Threshold**: Label ≥ 2 (medium or strong match) is considered "relevant job"

### Evaluation Metrics

**Precision@K** (Precision):
- Definition: Proportion of relevant jobs in Top-K recommendations
- Formula: `Precision@K = (Number of relevant jobs in Top-K) / K`
- Range: 0.0 - 1.0, higher is better

**NDCG@K** (Normalized Discounted Cumulative Gain):
- Definition: Quality score considering ranking position
- Formula: `NDCG@K = DCG@K / IDCG@K`
- Range: 0.0 - 1.0, higher is better
- Feature: Jobs ranked higher have more weight

### How to Run Evaluation (M5 Old Version, Deprecated)

> ⚠️ **Deprecated**: The following M5 evaluation process has been replaced by M7's LOOCV + Ablation Study. Please refer to the **M7: Learning to Rank (LTR) Complete Pipeline** section for the new evaluation method.

#### 1. Generate LLM Labels (Deprecated, M7 uses generate_labels.py to generate full 1-5 labels)

```bash
cd backend/eval
python generate_labels.py
```

~~This will generate:
- `labels_suggested.jsonl` - LLM-generated labels (JSONL format)~~ (**M7 overwrote with 750 pairs of 1-5 labels**)
- ~~`labels_final.csv` - Manual correction template (CSV format)~~ (**M7 deprecated**)

#### 2. Manual Correction (Deprecated, M7 no longer needs)

~~Open `backend/eval/labels_final.csv`, fill in corrected labels in the `final_label` column~~ (**M7 deprecated this file**)

#### 3. Run Evaluation (Deprecated, M7 uses scripts/eval_ablation.py)

```bash
cd backend/eval
python run_eval.py  # Deprecated
```

~~Evaluation results will be saved to:
- `eval_results.json` - Detailed results (JSON format)~~ (**M7 deprecated, replaced with results/ablation_results.json**)
- ~~Console output summary metrics~~

#### 4. View Evaluation Report (M7 still retained, but content updated)

```bash
cat backend/eval/eval_report.md
```

~~The report includes~~ (**M7 updated report content**):
- Data scale and distribution (M7: 750 pairs vs M5: 105 pairs)
- Label system description (M7: 1-5 scale vs M5: 0-3 scale)
- Evaluation metrics definitions (Same)
- Results interpretation guide (M7: LOOCV + Ablation vs M5: Simple validation)
- Weak Labels explanation and improvement suggestions

### Evaluation Data Scale

Current evaluation based on:
- **M5 Old Version (Deprecated)**: 7 resumes × Top-15 jobs = 105 annotation pairs (0-3 scale)
- **M7 New Version (Current)**: 15 resumes × 50 jobs = **750 annotation pairs** (1-5 scale)
- **Label Source**: LLM (GPT-4o-mini) independently generated (no information leakage)
- **Coverage**: Full coverage (all resume×job combinations)

### Evaluation Fairness Guarantee

**Prevent Evaluation Bias (Label Leakage Prevention)**:

To avoid evaluation bias, the LLM annotation stage does not expose any system ranking or scoring information; all labels are generated independently based on original JD and Resume.

Specific measures:
- ✅ LLM only receives original resume and job description text
- ✅ Does not provide system-calculated matched_skills, gap_skills, final_score
- ✅ LLM is explicitly told its role is "independent human evaluator"
- ✅ Ensures labels reflect true judgment, not system output paraphrase

### Weak Labels Description

**What are Weak Labels?**
- Labels automatically generated by LLM, not manually annotated gold standard
- Advantages: Fast, low-cost, scalable
- Limitations: Less accurate than manual annotation, recommend spot-checking and corrections

**Recommended Process:**
1. LLM quickly generates suggested_label (completed)
2. Manually spot-check 20-30% and correct final_label
3. Re-run evaluation for more accurate results

### Data ID Description

**Why add job_id and resume_id?**
- Only used for evaluation alignment, does not affect recommendation logic
- job_id: job_001, job_002, ..., job_022
- resume_id: resume_001, resume_002, ..., resume_007
- JobRecommendation returned by `/recommend_jobs` endpoint includes job_id

## M6: One-Click Demo (Streamlit Interactive Interface)

### Feature Overview

M6 provides a Streamlit-based interactive web interface, allowing you to experience complete job matching features without manually writing code:
- 📄 Multiple resume input methods (text box input or upload TXT file)
- 💼 Job selection (choose from jobs.jsonl database)
- 🎯 Top-K parameter configuration (recommended job count)
- 🚀 One-click matching and result display (including match scores, matched skills, skill gaps)
- 💡 Detailed explanations (click button to view RAG-generated match explanations, gap analysis, improvement suggestions)

### One-Click Run Steps

#### Prerequisites

Ensure environment configuration and dependency installation are completed (refer to "How to Run" section above).

#### Install Streamlit

```bash
# Method 1: Using requirements.txt (recommended)
pip install -r frontend/requirements.txt

# Method 2: Manual installation
pip install streamlit requests
```

#### Start Backend Service

In the **first terminal**, start FastAPI backend:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

After backend starts, API will run at http://localhost:8000

#### Start Frontend Interface

In the **second terminal**, start Streamlit frontend:

```bash
# Ensure in project root directory
streamlit run frontend/streamlit_app.py
```

After frontend starts, it will automatically open the browser, visiting: http://localhost:8501

**If browser doesn't automatically open**, manually visit http://localhost:8501

### Usage Guide

#### 1. Input Resume

**Method 1: Manual Input**
- Select "Manual Text Input"
- Enter resume content in text box
- Suggest organizing in the following format (system will auto-parse):
  ```
  Education
  Bachelor of Science in Computer Science, MIT, 2020

  Projects
  Built a recommendation system using collaborative filtering and deep learning

  Skills
  Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, NLP

  Experience
  Software Engineer at Tech Corp (2020-2023)
  - Developed ML models for user personalization
  - Improved recommendation accuracy by 25%
  ```

**Method 2: Upload File**
- Select "Upload TXT File"
- Click "Browse files" to upload TXT format resume file

#### 2. Select Job (Optional)

- Select job from dropdown list
  - List displays format: `job_id: Job Title`
  - Select "-- None (match all jobs) --" to match all jobs
  - Click "View Job Details" to view job details

#### 3. Set Matching Parameters

- Use slider to adjust **Top-K** (recommended job count)
- Range: 1-20, default value: 5

#### 4. Run Matching

- Click **"🚀 Run Match"** button
- System will:
  1. Parse resume content
  2. Call backend `/recommend_jobs` endpoint
  3. Display Top-K matching jobs

#### 5. View Results

Matching results will display for each job:
- **Job Information**: Title, company, location, level
- **Match Score**: Semantic similarity score (percentage)
- **Matched Skills**: Skill intersection between resume and job requirements
- **Skill Gaps**: Skills required by job but missing from resume

#### 6. View Detailed Explanations

- Click **"💡 Explain Match"** button under any job
- System will call `/explain` endpoint to generate detailed explanations
- Expanded explanations include:
  - **Why this job matches**: Evidence-based matching reasons
  - **Gap Analysis**: Detailed skill gap analysis
  - **Improvement Suggestions**: Actionable improvement suggestions

### Interface Feature Description

#### Sidebar

- **About**: System introduction and usage instructions
- **Backend Status**: Real-time backend service status check
  - Green: Backend running normally
  - Red: Backend not started (please start backend service first)

#### Main Interface Layout

- **Left Column**: Resume input area
- **Right Column**: Job selection area (optional)
- **Bottom**: Matching parameters and run button
- **Results Area**: Top-K job cards (sorted by match score)

### Sample Data

You can use the following sample data for quick testing:

**Sample Resume (NLP Direction)**:
```
Education
Master of Science in Natural Language Processing, Carnegie Mellon University, 2019-2021

Projects
Built conversational AI system using GPT-4 and RAG, serving 500K+ users
Developed multilingual NER system supporting 15 languages using BERT

Skills
NLP, LLM, Transformers, BERT, GPT, Claude, Prompt Engineering, RAG, Fine-tuning, Python, PyTorch, FastAPI

Experience
NLP Engineer at AI Startup (2021-2024): Built LLM-powered products, implemented RAG systems, fine-tuned models for domain adaptation
```

Then:
1. Set Top-K = 5
2. Click "Run Match"
3. View recommended NLP-related jobs (e.g., "NLP Engineer - Conversational AI", "LLM Engineer", etc.)
4. Click "Explain Match" to view detailed match explanations

### Technology Stack

- **Frontend Framework**: Streamlit (lightweight Python web framework)
- **HTTP Client**: requests
- **Backend API**: FastAPI (see M1-M5)

### Troubleshooting

**Issue: Prompt "Backend is not running" after clicking "Run Match"**
- Solution: Ensure backend service is started (`uvicorn main:app --reload`)
- Check if backend is running at http://localhost:8000
- View sidebar "Backend Status" status

**Issue: Explanation generation failure**
- Reason: Possibly OpenAI API Key not configured or RAG service exception
- Solution: Check `OPENAI_API_KEY` configuration in `.env` file (refer to M4 configuration description)
- Note: Even if RAG fails, matching feature can still work normally

**Issue: Resume parsing inaccurate**
- Solution: Suggest explicitly using section headers like "Education", "Projects", "Skills", "Experience" in resume
- Suggest separating skills with commas (e.g., "Python, Machine Learning, NLP")

**Issue: Cannot find jobs.jsonl file**
- Solution: Ensure `backend/data/jobs.jsonl` file exists
- Check if Streamlit is run from project root directory (`streamlit run frontend/streamlit_app.py`)

## M7: Learning to Rank (LTR) Complete Pipeline

### Overview

M7 introduces a complete Learning to Rank (LTR) system. Compared to M3's heuristic ranking, LTR optimizes ranking effect through **learning**.

**Core Improvements:**
1. **Full Weak Labels (1-5 scale)**: Covers all resume×job combinations (15×50=750 pairs), replacing old version that only annotated top-15 with 0-3 labels
2. **Pairwise Learning to Rank**: Uses Logistic Regression to learn ranking, not fixed weights
3. **LOOCV + Ablation**: Strict small-data evaluation, comparing three ranking methods (embedding_only / heuristic / LTR)
4. **Frontend One-Click Toggle**: Streamlit UI supports enabling/disabling LTR, real-time effect comparison

### Three-Step Complete Process

#### Step 1: Generate Full 1-5 Weak Labels

```bash
# Set environment variable (requires OpenAI API Key)
export OPENAI_API_KEY=sk-your-actual-api-key-here

# Generate labels (covering all resume×job combinations)
cd backend/eval
python generate_labels.py
```

**Feature Description:**
- Traverses all 15×50=750 resume-job combinations
- LLM independently scores (1-5), **does not leak** system ranking information
- Automatically backs up old labels to `labels_suggested_OLD_<timestamp>.jsonl`
- Supports breakpoint resumption (existing (resume_id, job_id) will be skipped)
- Validates coverage (missing pairs will report error)

**Output Files:**
- `backend/eval/labels_suggested.jsonl` - Full 750 pairs of labels (1-5 scale)

**Label Definitions (1-5 scale):**


| Label | Name | Definition |
|------|------|------|
| **1** | Not a match | Obviously irrelevant or inconsistent direction |
| **2** | Weak match | Some relevant points but missing key skills |
| **3** | Partial match | Consistent direction, some skills met, with gaps |
| **4** | Good match | Well-aligned direction, high skill coverage, slight gaps |
| **5** | Strong match | Highly matching, excellent skill coverage, minimal gaps |

**Coverage Validation:**
Script will automatically verify if all pairs are covered:
```
✅ Coverage validation PASSED: All 750 pairs are labeled!
```
If any missing, will print missing (resume_id, job_id) and report error.

---

#### Step 2: Run LOOCV + Ablation Evaluation

```bash
# Run evaluation (train LTR model + calculate metrics)
cd backend
python scripts/eval_ablation.py
```

**Evaluation Method:**
- **LOOCV (Leave-One-Out Cross-Validation)**:
  - Each time leave 1 resume for testing, remaining 14 for training
  - Total 15 folds, ensuring each resume is tested
  - Suitable for small datasets (15 resumes), avoids overfitting
- **Test Set Evaluation Scope**:
  - Evaluates ranking for **all 50 jobs** of test resume
  - **Not just evaluating top-15** (avoids bias)

**Ablation Comparison Methods:**

| Method | Description |
|------|------|
| **embedding_only** | Only uses semantic similarity ranking (M2 baseline) |
| **heuristic** | M3 heuristic weighting (embedding + skill_overlap + keyword_bonus - gap_penalty) |
| **ltr_logreg** | M7 Pairwise Logistic Regression (2 features: embedding + keyword_bonus) |

**Evaluation Metrics:**
- **NDCG@5 / NDCG@10**: Ranking quality (considers position weight, 0-1 higher is better)
- **Precision@5 / Precision@10**: Relevant job proportion (threshold: label ≥ 4, 0-1 higher is better)

**Output Files:**
- `backend/results/ablation_results.json` - Detailed results (per-fold + aggregated)
- `backend/eval/eval_report.md` - Readable evaluation report
- Terminal output summary table

**Example Output:**
```
================================================================
Summary
================================================================

embedding_only:
  ndcg@5          0.723 ± 0.045
  ndcg@10         0.801 ± 0.032
  precision@5     0.657 ± 0.089
  precision@10    0.571 ± 0.067

heuristic:
  ndcg@5          0.756 ± 0.041
  ndcg@10         0.825 ± 0.029
  precision@5     0.714 ± 0.082
  precision@10    0.600 ± 0.061

ltr_logreg:
  ndcg@5          0.782 ± 0.038
  ndcg@10         0.845 ± 0.026
  precision@5     0.743 ± 0.075
  precision@10    0.629 ± 0.058
```

**Model Saving:**
During evaluation, each fold trains an LTR model. To use in production, need to separately train final model using **all data**:
```bash
# Train final model (full data)
cd backend
python scripts/train_ltr_model.py `
  --resumes_path data/resumes.jsonl `
  --jds_path data/jobs.jsonl `
  --labels_path eval/labels_suggested.jsonl `
  --min_rel_diff 2 `
  --random_state 42

# Default output: models/ltr_logreg.joblib
```

**Output Example:**
```
================================================================================
LTR Model Training for Production
================================================================================

[1/6] Loading data...
  Loaded: 15 resumes, 50 jobs, 750 labels

[2/6] Validating data...
  ✅ Full coverage: 750/750 pairs labeled

[3/6] Building feature cache...
  [OK] Cached 750 embedding scores
  [OK] Built 750 feature vectors
  Feature dimension: 2
  Feature names: ['embedding', 'keyword_bonus']

[4/6] Constructing pairwise training data...
  [OK] Created 5700 pairwise training samples

[5/6] Training LTR model...
  [OK] Model trained successfully

  Learned feature weights:
    embedding            +3.4061
    keyword_bonus        +2.2702

[6/6] Saving model...
  [OK] Model saved to: models/ltr_logreg.joblib

Training Complete!
```

---

#### Step 3: Enable LTR in Demo

**Backend API Support:**

`/recommend_jobs` endpoint adds parameter:
```json
{
  "resume": { ... },
  "top_k": 5,
  "use_ltr": true  // New: Enable LTR ranking
}
```

**Response Adds Field:**
```json
{
  "recommendations": [ ... ],
  "total_jobs_searched": 50,
  "explanation": "...",
  "ranker": "ltr_logreg"  // New: Ranker used
}
```

**Possible ranker field values:**
- `"heuristic"` - Uses M3 heuristic ranking (default, use_ltr=false)
- `"ltr_logreg"` - Uses LTR model ranking (use_ltr=true and model exists)
- `"heuristic_fallback"` - LTR failure falls back to heuristic (model doesn't exist or load fails)

**Streamlit Frontend Usage:**

1. Start backend:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start frontend:
```bash
streamlit run frontend/streamlit_app.py
```

3. Check **"Enable LTR re-ranking (use_ltr)"** checkbox in UI
4. Click **"Run Match"** to run matching
5. View ranker identifier at top of results (🤖 LTR or 🔧 Heuristic)

**Effect Comparison:**
- Unchecked: Uses M3 heuristic ranking (fixed weights)
- Checked: Uses LTR learned ranking (if model exists)

---

### Key Design Constraints

**Prevent Label Leakage:**
- ✅ LLM generating labels **does not receive** any system ranking information (matched_skills, gap_skills, scores, topK)
- ✅ LLM only scores based on original resume and job text
- ✅ Prompt explicitly tells LLM its role is "independent evaluator"

**LTR Features (Multicollinearity-Aware):**
- ✅ LTR uses 2 features: **embedding** and **keyword_bonus** (to avoid multicollinearity)
- ✅ Removed features: skill_overlap and gap_penalty (correlation r>0.95, causing unstable weight learning)
- ✅ L2 regularization (C=0.1) stabilizes training despite remaining correlation (r=0.89)
- ✅ Difference from M3: M3 uses fixed weights (all 4 features), LTR learns weights from data (2 features)

**Pairwise Training and Mirrored Pairs:**
- Default `min_rel_diff=2`: Only constructs training pair when `label_i ≥ label_j + 2`
- For example: (label=5, label=3) → construct training pair; (label=4, label=3) → don't construct
- If a resume's labels variance is too small (all jobs' labels are close), may not construct enough pairs

**Why Need Mirrored Pairs?**

Pairwise LTR uses Logistic Regression for binary classification:
- `y=1` means "first job is better than second job"
- `y=0` means "first job is not better than second job"

**Key Constraint**: sklearn's LogisticRegression **requires training data to contain at least 2 classes**. If `y_pairs` only contains one class (all 1s), training will fail.

**Solution**: Generate mirrored negative samples for each positive pair:
```
Original pair:   (winner - loser, y=1)  # Means winner is better than loser
Mirrored pair:   (loser - winner, y=0)  # Means loser is not better than winner
```

Since `loser - winner = -(winner - loser)`, mirrored pair uses opposite feature difference vector, ensuring model learns symmetric ranking relationship.

**Implementation Details:**
- `construct_pairwise_data()` function's `add_mirror` parameter **defaults to True**
- Training script automatically checks number of classes in `y_pairs`:
  - If only 1 class → automatically reconstructs with `add_mirror=True`
  - If still fails → report error and exit
- This ensures LogisticRegression always receives valid training data

**Why Enable add_mirror by Default?**
- Ensures training stability (avoids single-class error)
- Increases training sample count (about 2x)
- Provides more balanced class distribution (usually close to 50%-50%)
- Especially important for small datasets (like this project's 15 resumes)

**Fallback Mechanism:**
- If a fold's pairwise pairs < 10, LTR training fails, automatically falls back to heuristic
- If FastAPI can't find `models/ltr_logreg.joblib`, automatically falls back to heuristic, ranker returns `"heuristic_fallback"`

---

### File Description

**New/Modified File List:**

| File Path | Description | Type |
|----------|------|------|
| `backend/eval/generate_labels.py` | Full 1-5 weak labels generation (overwrites old 0-3 top-15) | Modified |
| `backend/src/ranking/features.py` | Feature extraction and vectorization (FEATURE_NAMES fixed order) | New |
| `backend/src/ranking/pairwise.py` | Pairwise training data construction (with mirror pairs support) | New |
| `backend/src/ranking/ltr_logreg.py` | Pairwise Logistic Regression model (with save/load) | New |
| `backend/scripts/eval_ablation.py` | LOOCV + Ablation evaluation script | New |
| `backend/scripts/train_ltr_model.py` | Production environment LTR model training script (with automatic mirror pairs fallback) | New |
| `backend/main.py` | FastAPI: Added use_ltr parameter, ranker field | Modified |
| `frontend/streamlit_app.py` | Streamlit: Added LTR toggle checkbox, ranker display | Modified |
| `backend/data/resumes.jsonl` | Expanded to 15 resumes | Modified |
| `backend/data/jobs.jsonl` | Expanded to 50 jobs | Modified |
| `backend/eval/labels_suggested.jsonl` | Full 750 pairs of labels (1-5 scale) | Overwritten |
| `backend/models/ltr_logreg.joblib` | Trained LTR model | New (requires running training script) |
| `backend/results/ablation_results.json` | Ablation study results | New |

---

### Quick Command Summary

```bash
# ====== Step 1: Generate full 1-5 weak labels ======
export OPENAI_API_KEY=sk-your-key
cd backend/eval
python generate_labels.py

# ====== Step 2: Run LOOCV + Ablation evaluation ======
cd backend
python scripts/eval_ablation.py

# ====== Step 3: Train final LTR model (for production) ======
cd backend
python scripts/train_ltr_model.py \
    --resumes_path data/resumes.jsonl \
    --jds_path data/jobs.jsonl \
    --labels_path eval/labels_suggested.jsonl

# ====== Step 4: Start Demo ======
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run frontend/streamlit_app.py

# ====== Coverage validation (optional) ======
# Training script automatically validates label coverage, no separate run needed
# View validation results: Run training script to see [2/6] Validating data step output
```

---

### FAQ

**Q1: Why overwrite old labels_suggested.jsonl?**
- Old version only annotated top-15 (105 pairs: 7×15), and used 0-3 scale
- New version covers full (750 pairs: 15×50), uses 1-5 scale
- Old file is automatically backed up to `archive/` directory, won't be lost

**Q2: Where is the LTR model saved?**
- Evaluation script (`scripts/eval_ablation.py`) trains model in each fold, but doesn't save
- Need to separately train full model and save to `models/ltr_logreg.joblib` (see code snippet in step 2)
- Can also modify evaluation script to save model after last fold ends

**Q3: How does FastAPI use LTR model?**
- If `use_ltr=true` and `models/ltr_logreg.joblib` exists, load model and rank
- If model doesn't exist or load fails, automatically falls back to heuristic, ranker returns `"heuristic_fallback"`

**Q4: Is training data sufficient for each LOOCV fold?**
- 15 resumes, each fold uses 14 for training
- Each resume has 50 jobs, theoretically can construct many pairwise pairs
- But if a resume's labels variance is too small, pairs may be insufficient, will fall back to heuristic

**Q5: How to view LTR learned feature weights?**
```python
from src.ranking.ltr_logreg import PairwiseLTRModel
model = PairwiseLTRModel.load('models/ltr_logreg.joblib')
weights = model.get_feature_weights()
print(weights)
# Output example: {'embedding': 3.41, 'keyword_bonus': 2.27}
```

Or use the provided script:
```bash
cd backend
python view_ltr_weights.py
```

**Q6: How to add new features?**
1. Add new feature name at end of `FEATURE_NAMES` list in `src/ranking/features.py`
2. Calculate new feature value in `build_features()` function
3. Regenerate labels and train model (feature order change will make old model incompatible)

---

## Next Steps

Future Milestones will implement:
- ✅ ~~Semantic matching based on vector embeddings~~ (M2 completed)
- ✅ ~~Batch matching and ranking features~~ (M2 completed)
- ✅ ~~Explainable lightweight ranking layer~~ (M3 completed)
- ✅ ~~Integrate LLM for smarter matching analysis and personalized suggestions~~ (M4 completed)
- ✅ ~~Evaluation system and weak supervision label generation~~ (M5 completed)
- ✅ ~~Streamlit interactive interface Demo~~ (M6 completed)
- ✅ ~~Learning to Rank complete pipeline~~ (M7 completed)
- Database integration to store job and resume data
- User authentication and authorization system
- Cache optimization (Redis)
- Logging and monitoring
- More recommendation algorithms (hybrid recommendation, collaborative filtering, etc.)

## License

TBD
