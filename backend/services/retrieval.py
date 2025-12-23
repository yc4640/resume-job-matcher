"""
Retrieval service for job-resume matching using semantic embeddings.
Implements text conversion and ranking based on cosine similarity.
"""

import numpy as np
from typing import List, Dict, Any
# retrieval.py
from schemas import JobPosting, Resume
from services.embedding import embed_texts



def job_to_text(job: JobPosting) -> str:
    """
    Convert JobPosting to a single text string for embedding.
    Concatenates title, responsibilities, requirements, and skills.

    Args:
        job: JobPosting object

    Returns:
        str: Combined text representation of the job
    """
    skills_text = ", ".join(job.skills) if job.skills else ""

    text_parts = [
        f"Title: {job.title}",
        f"Responsibilities: {job.responsibilities}",
        f"Requirements: {job.requirements_text}",
        f"Skills: {skills_text}"
    ]

    return " ".join(text_parts)


def resume_to_text(resume: Resume) -> str:
    """
    Convert Resume to a single text string for embedding.
    Concatenates education, projects, experience, and skills.

    Args:
        resume: Resume object

    Returns:
        str: Combined text representation of the resume
    """
    skills_text = ", ".join(resume.skills) if resume.skills else ""

    text_parts = [
        f"Education: {resume.education}",
        f"Projects: {resume.projects}",
        f"Experience: {resume.experience}",
        f"Skills: {skills_text}"
    ]

    return " ".join(text_parts)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        float: Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def rank_jobs(resume: Resume, jobs: List[JobPosting], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rank jobs by semantic similarity to resume using embeddings.

    Args:
        resume: Resume object to match against
        jobs: List of JobPosting objects to rank
        top_k: Number of top matches to return

    Returns:
        List[Dict]: Top-k job matches, each containing:
            - job: The JobPosting object
            - score: Cosine similarity score (0-1)
            - rank: Ranking position (1-based)
    """
    if not jobs:
        return []

    # Convert resume to text and get embedding
    resume_text = resume_to_text(resume)
    resume_embedding = embed_texts(resume_text)[0]  # Get first (only) embedding

    # Convert all jobs to text
    job_texts = [job_to_text(job) for job in jobs]

    # Get embeddings for all jobs
    job_embeddings = embed_texts(job_texts)

    # Calculate cosine similarities
    similarities = []
    for i, job_emb in enumerate(job_embeddings):
        similarity = cosine_similarity(resume_embedding, job_emb)
        similarities.append({
            "job": jobs[i],
            "score": float(similarity),  # Convert to Python float
            "index": i
        })

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x["score"], reverse=True)

    # Return top-k with rank
    top_k_results = []
    for rank, item in enumerate(similarities[:top_k], start=1):
        top_k_results.append({
            "job": item["job"],
            "score": item["score"],
            "rank": rank
        })

    return top_k_results
