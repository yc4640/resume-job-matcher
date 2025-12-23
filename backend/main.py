"""
Main FastAPI application entry point.
Provides health check endpoint and job-resume matching endpoint for M1 milestone.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from schemas import JobPosting, Resume, MatchResponse

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


if __name__ == "__main__":
    import uvicorn
    # Run the application with uvicorn when executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
