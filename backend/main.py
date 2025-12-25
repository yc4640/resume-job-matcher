"""
Main FastAPI application entry point.
Provides health check endpoint, job-resume matching endpoint for M1 milestone,
job recommendation endpoint for M2 milestone,
and explainable ranking for M3 milestone.
"""
from dotenv import load_dotenv
load_dotenv()

import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from schemas import JobPosting, Resume, MatchResponse
from services.retrieval import rank_jobs
from services.ranking import rank_jobs_with_features, explain_ranking, load_config
from services.rag import generate_rag_explanation

# Initialize FastAPI application
app = FastAPI(
    title="LM Match Service",
    description="Language Model Matching Service API",
    version="0.1.0"
)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str


class MatchRequest(BaseModel):
    """Request model for match endpoint"""
    job: JobPosting
    resume: Resume


class RecommendJobsRequest(BaseModel):
    """Request model for job recommendation endpoint"""
    resume: Resume
    top_k: int = 5


class RankingFeatures(BaseModel):
    """Ranking features for explainable scoring (M3)"""
    embedding_score: float
    skill_overlap: float
    keyword_bonus: float
    gap_penalty: float
    final_score: float


class JobRecommendation(BaseModel):
    """Single job recommendation with similarity score and matched skills"""
    rank: int
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    level: Optional[str] = None
    similarity_score: float
    matched_skills: List[str]
    gap_skills: Optional[List[str]] = None  # M3: gap skills
    features: Optional[RankingFeatures] = None  # M3: explainable features
    # M4: RAG-generated explanations
    explanation: Optional[str] = None
    gap_analysis: Optional[str] = None
    improvement_suggestions: Optional[str] = None


class RecommendJobsResponse(BaseModel):
    """Response model for job recommendation endpoint"""
    recommendations: List[JobRecommendation]
    total_jobs_searched: int
    explanation: Optional[str] = None  # M3: explanation for top result


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service is running.
    Returns status and message indicating service health.
    """
    return HealthResponse(
        status="ok",
        message="Service is healthy and running"
    )


@app.post("/match", response_model=MatchResponse)
async def match(request: MatchRequest):
    """
    Match endpoint that performs job-resume matching based on skills.
    Returns match score, matched skills, skill gaps, and suggestions.

    Args:
        request: MatchRequest containing job posting and resume data

    Returns:
        MatchResponse with match score, matched skills, gaps, and suggestions
    """
    job_skills_set = set(request.job.skills)
    resume_skills_set = set(request.resume.skills)

    # Calculate matched skills and gaps
    matched_skills = list(job_skills_set & resume_skills_set)
    gaps = list(job_skills_set - resume_skills_set)

    # Calculate match score
    if len(job_skills_set) > 0:
        match_score = int(len(matched_skills) / len(job_skills_set) * 100)
    else:
        match_score = 0

    # Generate suggestions for skill gaps
    suggestions = []
    for skill in gaps:
        suggestions.append(f"Consider learning {skill} to better match this position")

    return MatchResponse(
        match_score=match_score,
        matched_skills=matched_skills,
        gaps=gaps,
        suggestions=suggestions
    )


def load_jobs_from_jsonl(filepath: str = "data/jobs.jsonl") -> List[JobPosting]:
    """
    Load all jobs from JSONL file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of JobPosting objects
    """
    jobs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            job_data = json.loads(line.strip())
            jobs.append(JobPosting(**job_data))
    return jobs


@app.post("/recommend_jobs", response_model=RecommendJobsResponse)
async def recommend_jobs(request: RecommendJobsRequest):
    """
    Recommend top-k jobs for a given resume using explainable ranking (M3) and RAG explanations (M4).

    This endpoint combines:
    - M2: Semantic similarity using embeddings
    - M3: Explainable ranking features (skill overlap, keyword bonus, gap penalty)
    - M4: RAG-based explanations (evidence retrieval + LLM generation)

    The final ranking uses a weighted combination of features configured in YAML.
    For each recommended job, RAG generates:
    - Explanation: Why the job is a good fit (evidence-based)
    - Gap Analysis: What skills/qualifications are missing
    - Improvement Suggestions: Actionable steps to improve fit

    Args:
        request: RecommendJobsRequest containing resume and top_k parameter

    Returns:
        RecommendJobsResponse with ranked job recommendations, features, and RAG explanations
    """
    # Load all jobs from JSONL file
    jobs = load_jobs_from_jsonl()

    # Step 1: Get embedding-based similarity scores (M2)
    embedding_results = rank_jobs(request.resume, jobs, top_k=len(jobs))

    # Step 2: Re-rank using explainable features (M3)
    ranked_results = rank_jobs_with_features(request.resume, embedding_results)

    # Step 3: Take top-k results
    top_k_results = ranked_results[:request.top_k]

    # Step 4: Build recommendations with features and RAG explanations (M4)
    recommendations = []
    for result in top_k_results:
        job = result["job"]

        # Create features object
        features = RankingFeatures(
            embedding_score=result["embedding_score"],
            skill_overlap=result["skill_overlap"],
            keyword_bonus=result["keyword_bonus"],
            gap_penalty=result["gap_penalty"],
            final_score=result["final_score"]
        )

        # M4: Generate RAG-based explanation for this job
        try:
            rag_result = generate_rag_explanation(
                job=job,
                resume=request.resume,
                matched_skills=result["matched_skills"],
                gap_skills=result["gap_skills"],
                final_score=result["final_score"]
            )
            explanation = rag_result.get("explanation", "")
            gap_analysis = rag_result.get("gap_analysis", "")
            improvement_suggestions = rag_result.get("improvement_suggestions", "")
        except Exception as e:
            # Fallback if RAG fails (e.g., no API key set)
            print(f"RAG explanation failed: {e}")
            explanation = None
            gap_analysis = None
            improvement_suggestions = None

        recommendation = JobRecommendation(
            rank=result["rank"],
            title=job.title,
            company=job.company,
            location=job.location,
            level=job.level,
            similarity_score=result["embedding_score"],  # Keep for backward compatibility
            matched_skills=result["matched_skills"],
            gap_skills=result["gap_skills"],
            features=features,
            # M4: Add RAG explanations
            explanation=explanation,
            gap_analysis=gap_analysis,
            improvement_suggestions=improvement_suggestions
        )
        recommendations.append(recommendation)

    # Step 5: Generate explanation for top result
    explanation = None
    if top_k_results:
        config = load_config()
        explanation = explain_ranking(top_k_results[0], config)

    return RecommendJobsResponse(
        recommendations=recommendations,
        total_jobs_searched=len(jobs),
        explanation=explanation
    )


if __name__ == "__main__":
    import uvicorn
    # Run the application with uvicorn when executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
