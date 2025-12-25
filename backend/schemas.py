"""
Pydantic schemas for job-resume matching service.
Defines data models for job postings, resumes, and match responses.
"""

from typing import List, Optional
from pydantic import BaseModel


class JobPosting(BaseModel):
    """Job posting data model"""
    title: str
    responsibilities: str
    requirements_text: str
    skills: List[str]
    company: Optional[str] = None
    location: Optional[str] = None
    level: Optional[str] = None
    job_id: Optional[str] = None  # M5: for evaluation alignment


class Resume(BaseModel):
    """Resume data model"""
    education: str
    projects: str
    skills: List[str]
    experience: str
    resume_id: Optional[str] = None  # M5: for evaluation alignment


class MatchResponse(BaseModel):
    """Match result response model"""
    match_score: int
    matched_skills: List[str]
    gaps: List[str]
    suggestions: List[str]
