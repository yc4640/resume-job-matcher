"""
M6 Demo - Streamlit Interface for Job-Resume Matching
Provides interactive UI for resume input, job selection, matching, and explanations.
"""
import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Job-Resume Matching System",
    page_icon="💼",
    layout="wide"
)


def load_jobs_from_jsonl(filepath: str = "backend/data/jobs.jsonl") -> List[Dict[str, Any]]:
    """Load jobs from JSONL file for dropdown selection."""
    jobs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                job_data = json.loads(line.strip())
                jobs.append(job_data)
    except FileNotFoundError:
        st.error(f"Jobs file not found at {filepath}. Please check the path.")
    return jobs


def parse_resume_text(resume_text: str) -> Dict[str, Any]:
    """
    Parse resume text into structured format.
    Simple parser that looks for section headers.
    """
    lines = resume_text.strip().split('\n')

    resume = {
        "education": "",
        "projects": "",
        "skills": [],
        "experience": ""
    }

    current_section = None
    section_content = []

    for line in lines:
        line_lower = line.lower().strip()

        # Detect section headers
        if 'education' in line_lower and len(line_lower) < 30:
            if current_section and section_content:
                resume[current_section] = '\n'.join(section_content).strip()
            current_section = 'education'
            section_content = []
        elif 'project' in line_lower and len(line_lower) < 30:
            if current_section and section_content:
                resume[current_section] = '\n'.join(section_content).strip()
            current_section = 'projects'
            section_content = []
        elif 'skill' in line_lower and len(line_lower) < 30:
            if current_section and section_content:
                resume[current_section] = '\n'.join(section_content).strip()
            current_section = 'skills'
            section_content = []
        elif 'experience' in line_lower and len(line_lower) < 30:
            if current_section and section_content:
                resume[current_section] = '\n'.join(section_content).strip()
            current_section = 'experience'
            section_content = []
        elif line.strip():
            section_content.append(line)

    # Add last section
    if current_section and section_content:
        resume[current_section] = '\n'.join(section_content).strip()

    # Parse skills into list if they're comma-separated
    if isinstance(resume['skills'], str):
        skills_text = resume['skills']
        # Try to parse as comma-separated list
        if ',' in skills_text:
            resume['skills'] = [s.strip() for s in skills_text.split(',') if s.strip()]
        else:
            # Split by newlines or spaces
            resume['skills'] = [s.strip() for s in skills_text.split() if s.strip()]

    # Ensure all fields have default values
    if not resume['education']:
        resume['education'] = "Not specified"
    if not resume['projects']:
        resume['projects'] = "Not specified"
    if not resume['experience']:
        resume['experience'] = "Not specified"
    if not resume['skills']:
        resume['skills'] = []

    return resume


def call_recommend_jobs(resume: Dict[str, Any], top_k: int) -> Optional[Dict[str, Any]]:
    """Call /recommend_jobs endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/recommend_jobs",
            json={"resume": resume, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling /recommend_jobs: {str(e)}")
        return None


def call_explain(resume: Dict[str, Any], job_id: str) -> Optional[Dict[str, Any]]:
    """Call /explain endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/explain",
            json={"resume": resume, "job_id": job_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling /explain: {str(e)}")
        return None


# Main App
st.title("💼 Job-Resume Matching System")
st.markdown("---")

# Load jobs for dropdown
all_jobs = load_jobs_from_jsonl()

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}

# Layout: Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Resume Input")

    # Resume input method selection
    resume_input_method = st.radio(
        "Choose input method:",
        ["Manual Text Input", "Upload TXT File"],
        horizontal=True
    )

    resume_text = ""

    if resume_input_method == "Manual Text Input":
        resume_text = st.text_area(
            "Enter your resume (use sections: Education, Projects, Skills, Experience):",
            height=300,
            placeholder="""Example:
Education
Bachelor of Science in Computer Science, MIT, 2020

Projects
Built a recommendation system using collaborative filtering and deep learning

Skills
Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, NLP

Experience
Software Engineer at Tech Corp (2020-2023)
- Developed ML models for user personalization
- Improved recommendation accuracy by 25%"""
        )
    else:
        uploaded_file = st.file_uploader("Upload resume TXT file", type=["txt"])
        if uploaded_file is not None:
            resume_text = uploaded_file.read().decode("utf-8")
            st.text_area("Resume content:", resume_text, height=300, disabled=True)

with col2:
    st.header("💼 Job Selection (Optional)")

    st.info("Job selection is optional. Leave unselected to match against all jobs.")

    selected_job = None

    if all_jobs:
        job_options = {f"{job['job_id']}: {job['title']}": job for job in all_jobs}
        selected_job_key = st.selectbox(
            "Select a job:",
            options=["-- None (match all jobs) --"] + list(job_options.keys())
        )
        if selected_job_key != "-- None (match all jobs) --":
            selected_job = job_options[selected_job_key]
            with st.expander("View Job Details"):
                st.write(f"**Company:** {selected_job.get('company', 'N/A')}")
                st.write(f"**Location:** {selected_job.get('location', 'N/A')}")
                st.write(f"**Level:** {selected_job.get('level', 'N/A')}")
                st.write(f"**Skills Required:** {', '.join(selected_job.get('skills', []))}")
                st.write(f"**Responsibilities:** {selected_job.get('responsibilities', 'N/A')}")
    else:
        st.warning("No jobs loaded from database.")

# Matching parameters
st.markdown("---")
st.header("🔧 Matching Parameters")

top_k = st.slider("Number of top jobs to recommend (Top-K):", min_value=1, max_value=20, value=5)

# Run Match button
if st.button("🚀 Run Match", type="primary", use_container_width=True):
    if not resume_text.strip():
        st.error("Please enter or upload a resume first!")
    else:
        with st.spinner("Analyzing resume and matching jobs..."):
            # Parse resume
            resume_data = parse_resume_text(resume_text)
            st.session_state.resume_data = resume_data

            # Call recommend_jobs
            result = call_recommend_jobs(resume_data, top_k)

            if result:
                st.session_state.recommendations = result
                st.session_state.explanations = {}  # Reset explanations
                st.success(f"Found {len(result.get('recommendations', []))} matching jobs!")
            else:
                st.error("Failed to get recommendations. Please check if the backend is running.")

# Display Results
if st.session_state.recommendations:
    st.markdown("---")
    st.header("🎯 Top Matching Jobs")

    recommendations = st.session_state.recommendations.get('recommendations', [])

    if not recommendations:
        st.info("No job recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                # Job card header
                col_title, col_score = st.columns([3, 1])
                with col_title:
                    st.subheader(f"{i}. {rec.get('title', 'N/A')}")
                    company = rec.get('company') or 'N/A'
                    location = rec.get('location') or 'N/A'
                    level = rec.get('level') or 'N/A'
                    st.caption(f"🏢 {company} | 📍 {location} | 📊 {level}")
                with col_score:
                    score = rec.get('similarity_score', 0)
                    st.metric("Match Score", f"{score:.2%}")

                # Matched skills
                matched_skills = rec.get('matched_skills', [])
                if matched_skills:
                    st.write(f"**✅ Matched Skills:** {', '.join(matched_skills)}")

                gap_skills = rec.get('gap_skills', [])
                if gap_skills:
                    st.write(f"**⚠️ Gap Skills:** {', '.join(gap_skills)}")

                # Explain button
                job_id = rec.get('job_id')
                explain_key = f"explain_{job_id}_{i}"

                if st.button(f"💡 Explain Match", key=f"btn_{explain_key}"):
                    with st.spinner("Generating explanation..."):
                        explanation_result = call_explain(st.session_state.resume_data, job_id)
                        if explanation_result:
                            st.session_state.explanations[job_id] = explanation_result

                # Display explanation if available
                if job_id in st.session_state.explanations:
                    with st.expander("📝 Detailed Explanation", expanded=True):
                        exp_data = st.session_state.explanations[job_id]

                        explanation = exp_data.get('explanation', '')
                        if explanation:
                            st.markdown("**Why this job matches:**")
                            st.write(explanation)

                        gap_analysis = exp_data.get('gap_analysis', '')
                        if gap_analysis:
                            st.markdown("**Gap Analysis:**")
                            st.write(gap_analysis)

                        improvement_suggestions = exp_data.get('improvement_suggestions', '')
                        if improvement_suggestions:
                            st.markdown("**Improvement Suggestions:**")
                            st.write(improvement_suggestions)

                st.markdown("---")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is a job-resume matching system that uses:
    - **Semantic embeddings** for similarity matching
    - **Explainable ranking** with weighted features
    - **RAG-based explanations** for match insights

    **How to use:**
    1. Enter or upload your resume
    2. (Optional) Select a specific job or match against all
    3. Set the number of top jobs to show
    4. Click "Run Match" to get recommendations
    5. Click "Explain Match" for detailed insights
    """)

    st.header("🔧 Backend Status")
    try:
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("Backend is running")
        else:
            st.error("Backend not responding properly")
    except:
        st.error("Backend is not running")
        st.info("Start backend with: `uvicorn backend.main:app --reload`")
