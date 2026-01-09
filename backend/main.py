"""
Main FastAPI application entry point.
Provides health check endpoint, job-resume matching endpoint for M1 milestone,
job recommendation endpoint for M2 milestone,
and explainable ranking for M3 milestone.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from schemas import JobPosting, Resume, MatchResponse
from services.retrieval import rank_jobs
from services.ranking import rank_jobs_with_features, load_config
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
    use_ltr: bool = False  # New: Enable LTR re-ranking


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
    job_id: Optional[str] = None  # M5: for evaluation alignment
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
    ranker: str = "heuristic"  # New: which ranker was used ("heuristic", "ltr_logreg", "heuristic_fallback")


class ExplainRequest(BaseModel):
    """Request model for explain endpoint"""
    resume: Resume
    job_id: str


class ExplainResponse(BaseModel):
    """Response model for explain endpoint"""
    explanation: str
    gap_analysis: Optional[str] = None
    improvement_suggestions: Optional[str] = None


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
    - M6: Optional LTR re-ranking (use_ltr parameter)

    The final ranking uses either:
    - Heuristic: weighted combination of features (default)
    - LTR: pairwise logistic regression model (if use_ltr=True and model exists)

    For each recommended job, RAG generates:
    - Explanation: Why the job is a good fit (evidence-based)
    - Gap Analysis: What skills/qualifications are missing
    - Improvement Suggestions: Actionable steps to improve fit

    Args:
        request: RecommendJobsRequest containing resume, top_k, and use_ltr parameters

    Returns:
        RecommendJobsResponse with ranked job recommendations, features, RAG explanations, and ranker info
    """
    # Load all jobs from JSONL file
    jobs = load_jobs_from_jsonl()

    # Step 1: Get embedding-based similarity scores (M2)
    embedding_results = rank_jobs(request.resume, jobs, top_k=len(jobs))

    # Step 2: Re-rank using explainable features (M3) OR LTR (M6)
    ranker_used = "heuristic"  # Default

    if request.use_ltr:
        # Try to load and use LTR model
        try:
            from src.ranking.ltr_logreg import PairwiseLTRModel

            model_path = "models/ltr_logreg.joblib"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"LTR model not found at {model_path}")

            # Load LTR model
            ltr_model = PairwiseLTRModel.load(model_path)

            # Build embedding cache
            embedding_cache = {(request.resume.resume_id, r['job'].job_id): r['score'] for r in embedding_results}

            # Rank using LTR
            ltr_results = ltr_model.rank_jobs(request.resume, jobs, embedding_cache)

            # Convert LTR results to heuristic-compatible format
            # (LTR doesn't compute all heuristic features, so we recompute them for display)
            ranked_results = []
            for ltr_result in ltr_results:
                job = ltr_result['job']
                # Find embedding score for this job
                emb_score = next((r['score'] for r in embedding_results if r['job'].job_id == job.job_id), 0.0)

                # Recompute heuristic features for display (not used for ranking)
                heuristic_result = rank_jobs_with_features(request.resume, [{'job': job, 'score': emb_score}])

                # Merge: use LTR score but keep heuristic features for display
                result = heuristic_result[0].copy()
                result['final_score'] = ltr_result['score']  # Override with LTR score
                result['rank'] = ltr_result['rank']

                ranked_results.append(result)

            ranker_used = "ltr_logreg"

        except Exception as e:
            # Fallback to heuristic if LTR fails
            print(f"LTR ranking failed: {e}. Falling back to heuristic.")
            ranked_results = rank_jobs_with_features(request.resume, embedding_results)
            ranker_used = "heuristic_fallback"
    else:
        # Use heuristic ranking (default)
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
            job_id=job.job_id,  # M5: include job_id for evaluation alignment
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

    return RecommendJobsResponse(
        recommendations=recommendations,
        total_jobs_searched=len(jobs),
        ranker=ranker_used
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Generate detailed explanation for a specific job-resume match.

    This endpoint provides RAG-based explanations for why a job matches
    (or doesn't match) a given resume, including gap analysis and improvement suggestions.

    Args:
        request: ExplainRequest containing resume and job_id

    Returns:
        ExplainResponse with explanation, gap_analysis, and improvement_suggestions
    """
    # Load all jobs and find the requested job
    jobs = load_jobs_from_jsonl()
    target_job = None
    for job in jobs:
        if job.job_id == request.job_id:
            target_job = job
            break

    if not target_job:
        return ExplainResponse(
            explanation=f"Job with ID {request.job_id} not found.",
            gap_analysis=None,
            improvement_suggestions=None
        )

    # Calculate matched and gap skills using same logic as /recommend_jobs
    # This ensures consistency between list page and explain page
    from services.ranking import load_skills_vocabulary, expand_vocab_with_job_skills, normalize_skills
    from services.utils import merge_resume_skills

    # Load and expand vocab (same as rank_jobs_with_features)
    vocab = load_skills_vocabulary()
    vocab = expand_vocab_with_job_skills([target_job], vocab)

    # Merge resume skills (same as rank_jobs_with_features)
    vocab_list = list(vocab)
    merged_resume_skills = merge_resume_skills(request.resume, vocab_list)

    # Normalize skills
    job_skills_normalized = normalize_skills(target_job.skills, vocab)
    resume_skills_normalized = normalize_skills(merged_resume_skills, vocab)

    # Calculate matched and gap skills
    matched_skills = list(resume_skills_normalized & job_skills_normalized)
    gap_skills = list(job_skills_normalized - resume_skills_normalized)

    # Calculate a basic score for the RAG explanation
    if len(job_skills_normalized) > 0:
        basic_score = len(matched_skills) / len(job_skills_normalized)
    else:
        basic_score = 0.0

    # Generate RAG-based explanation
    try:
        rag_result = generate_rag_explanation(
            job=target_job,
            resume=request.resume,
            matched_skills=matched_skills,
            gap_skills=gap_skills,
            final_score=basic_score
        )
        explanation = rag_result.get("explanation", "No explanation available.")
        gap_analysis = rag_result.get("gap_analysis", "")
        improvement_suggestions = rag_result.get("improvement_suggestions", "")
    except Exception as e:
        # Fallback if RAG fails
        explanation = f"Unable to generate explanation: {str(e)}"
        gap_analysis = None
        improvement_suggestions = None

    return ExplainResponse(
        explanation=explanation,
        gap_analysis=gap_analysis,
        improvement_suggestions=improvement_suggestions
    )


if __name__ == "__main__":
    import uvicorn
    # Run the application with uvicorn when executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
