"""
Main FastAPI application entry point.
Provides health check endpoint, job-resume matching endpoint for M1 milestone,
and job recommendation endpoint for M2 milestone.
"""

import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from schemas import JobPosting, Resume, MatchResponse
from services.retrieval import rank_jobs

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


class JobRecommendation(BaseModel):
    """Single job recommendation with similarity score and matched skills"""
    rank: int
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    level: Optional[str] = None
    similarity_score: float
    matched_skills: List[str]


class RecommendJobsResponse(BaseModel):
    """Response model for job recommendation endpoint"""
    recommendations: List[JobRecommendation]
    total_jobs_searched: int


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
    Recommend top-k jobs for a given resume using semantic similarity.
    Loads all jobs from jobs.jsonl and ranks them using embedding-based similarity.

    Args:
        request: RecommendJobsRequest containing resume and top_k parameter

    Returns:
        RecommendJobsResponse with ranked job recommendations
    """
    # Load all jobs from JSONL file
    jobs = load_jobs_from_jsonl()

    # Rank jobs using semantic similarity
    ranked_results = rank_jobs(request.resume, jobs, request.top_k)

    # Build recommendations with matched skills
    recommendations = []
    for result in ranked_results:
        job = result["job"]
        similarity_score = result["score"]
        rank = result["rank"]

        # Calculate matched skills using M1 logic
        job_skills_set = set(job.skills)
        resume_skills_set = set(request.resume.skills)
        matched_skills = list(job_skills_set & resume_skills_set)

        recommendation = JobRecommendation(
            rank=rank,
            title=job.title,
            company=job.company,
            location=job.location,
            level=job.level,
            similarity_score=similarity_score,
            matched_skills=matched_skills
        )
        recommendations.append(recommendation)

    return RecommendJobsResponse(
        recommendations=recommendations,
        total_jobs_searched=len(jobs)
    )


if __name__ == "__main__":
    import uvicorn
    # Run the application with uvicorn when executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
