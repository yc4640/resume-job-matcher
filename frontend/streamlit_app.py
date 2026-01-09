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


def call_recommend_jobs(resume: Dict[str, Any], top_k: int, use_ltr: bool = False) -> Optional[Dict[str, Any]]:
    """Call /recommend_jobs endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/recommend_jobs",
            json={"resume": resume, "top_k": top_k, "use_ltr": use_ltr},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling /recommend_jobs: {str(e)}")
        return None


def call_match(resume: Dict[str, Any], job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Call /match endpoint for single job analysis."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/match",
            json={"resume": resume, "job": job},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling /match: {str(e)}")
        return None


def call_explain(resume: Dict[str, Any], job_id: str) -> Optional[Dict[str, Any]]:
    """Call /explain endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/explain",
            json={"resume": resume, "job_id": job_id},
            timeout=60
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

# Show different parameters based on mode
if selected_job:
    st.info(f"**Single Job Analysis Mode:** Analyzing only the selected job: {selected_job['title']}")
    top_k = 5  # Not used, but needs to be defined
    use_ltr = False  # Not used for single job analysis
else:
    col_param1, col_param2 = st.columns([1, 1])

    with col_param1:
        top_k = st.slider("Number of top jobs to recommend (Top-K):", min_value=1, max_value=20, value=5)

    with col_param2:
        use_ltr = st.checkbox(
            "Enable LTR re-ranking (use_ltr)",
            value=False,
            help="Use Learning to Rank model for re-ranking. Falls back to heuristic if model not available."
        )

# Run Match button
if st.button("🚀 Run Match", type="primary", use_container_width=True):
    if not resume_text.strip():
        st.error("Please enter or upload a resume first!")
    else:
        # Parse resume
        resume_data = parse_resume_text(resume_text)
        st.session_state.resume_data = resume_data

        # Check if user selected a specific job
        if selected_job:
            # Single job analysis mode
            with st.spinner(f"Analyzing match for: {selected_job['title']}..."):
                # Call /match endpoint for basic matching
                match_result = call_match(resume_data, selected_job)

                # Call /explain endpoint for detailed explanation
                explain_result = call_explain(resume_data, selected_job['job_id'])

                if match_result and explain_result:
                    # Store single job analysis result
                    st.session_state.recommendations = {
                        'ranker': 'single_job_analysis',
                        'recommendations': [{
                            'job_id': selected_job['job_id'],
                            'title': selected_job['title'],
                            'company': selected_job.get('company', 'N/A'),
                            'location': selected_job.get('location', 'N/A'),
                            'level': selected_job.get('level', 'N/A'),
                            'matched_skills': match_result.get('matched_skills', []),
                            'gap_skills': match_result.get('gaps', []),
                            'match_score': match_result.get('match_score', 0)
                        }]
                    }
                    st.session_state.explanations = {selected_job['job_id']: explain_result}
                    st.success(f"Analysis complete for: {selected_job['title']}!")
                else:
                    st.error("Failed to analyze the selected job. Please check if the backend is running.")
        else:
            # Multi-job recommendation mode
            with st.spinner("Analyzing resume and matching jobs..."):
                # Call recommend_jobs with use_ltr parameter
                result = call_recommend_jobs(resume_data, top_k, use_ltr)

                if result:
                    st.session_state.recommendations = result
                    st.session_state.explanations = {}  # Reset explanations

                    # Display which ranker was used
                    ranker = result.get('ranker', 'unknown')
                    ranker_display = {
                        'heuristic': '🔧 Heuristic',
                        'ltr_logreg': '🤖 LTR (Logistic Regression)',
                        'heuristic_fallback': '🔧 Heuristic (LTR unavailable)'
                    }.get(ranker, ranker)

                    st.success(f"Found {len(result.get('recommendations', []))} matching jobs! (Ranker: {ranker_display})")
                else:
                    st.error("Failed to get recommendations. Please check if the backend is running.")

# Display Results
if st.session_state.recommendations:
    st.markdown("---")

    # Display ranker info
    ranker = st.session_state.recommendations.get('ranker', 'unknown')
    ranker_badge = {
        'heuristic': '🔧 **Ranker:** Heuristic (Weighted Features)',
        'ltr_logreg': '🤖 **Ranker:** LTR Logistic Regression',
        'heuristic_fallback': '⚠️ **Ranker:** Heuristic (LTR model not found)',
        'single_job_analysis': '🎯 **Mode:** Single Job Analysis'
    }.get(ranker, f'**Ranker:** {ranker}')

    st.info(ranker_badge)

    st.header("🎯 Top Matching Jobs")

    recommendations = st.session_state.recommendations.get('recommendations', [])

    if not recommendations:
        st.info("No job recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                # Job card header
                st.subheader(f"{i}. {rec.get('title', 'N/A')}")
                company = rec.get('company') or 'N/A'
                location = rec.get('location') or 'N/A'
                level = rec.get('level') or 'N/A'
                st.caption(f"🏢 {company} | 📍 {location} | 📊 {level}")

                # Match score (only for single job analysis)
                match_score = rec.get('match_score')
                if match_score is not None:
                    st.metric("Match Score", f"{match_score}%")

                # Matched skills
                matched_skills = rec.get('matched_skills', [])
                if matched_skills:
                    st.write(f"**✅ Matched Skills:** {', '.join(matched_skills)}")

                gap_skills = rec.get('gap_skills', [])
                if gap_skills:
                    st.write(f"**⚠️ Gap Skills:** {', '.join(gap_skills)}")

                # Explain button (only show if not already explained)
                job_id = rec.get('job_id')
                explain_key = f"explain_{job_id}_{i}"

                # For single job analysis, explanation is already loaded
                is_single_job = st.session_state.recommendations.get('ranker') == 'single_job_analysis'

                if not is_single_job and st.button(f"💡 Explain Match", key=f"btn_{explain_key}"):
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
    - **Learning to Rank (LTR)** for improved ranking (optional)

    **How to use:**

    **Mode 1: Single Job Analysis**
    1. Enter or upload your resume
    2. Select a specific job from dropdown
    3. Click "Run Match" → Get detailed analysis for that job

    **Mode 2: Multi-Job Recommendations**
    1. Enter or upload your resume
    2. Leave job selection as "None"
    3. Set the number of top jobs (Top-K)
    4. (Optional) Enable LTR re-ranking
    5. Click "Run Match" → Get top-K recommendations
    6. Click "Explain Match" for detailed insights

    **Rankers (Mode 2 only):**
    - **Heuristic:** Weighted combination of features (default)
    - **LTR:** Pairwise logistic regression (requires trained model)
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
