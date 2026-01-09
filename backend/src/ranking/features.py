"""
Feature extraction for Learning to Rank (LTR).

This module defines the feature set used for ranking and provides functions to:
1. Build feature vectors from resume-job pairs
2. Normalize/vectorize features for ML models
"""

import sys
import os
from typing import List, Set, Dict, Any
import numpy as np

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schemas import Resume, JobPosting
from services.retrieval import rank_jobs
from services.ranking import (
    load_skills_vocabulary,
    normalize_skills,
    calculate_skill_overlap,
    calculate_keyword_bonus,
    calculate_gap_penalty,
    load_config
)
from services.utils import merge_resume_skills


# Fixed feature names in order (DO NOT CHANGE ORDER - breaks model compatibility)
# IMPORTANT: Must match ranking_config.yaml weights exactly
# NOTE: Using 2 features to balance multicollinearity vs model richness
#   - embedding: Semantic similarity (r=0.89 with keyword_bonus, but acceptable)
#   - keyword_bonus: Priority keyword matching (highest correlation with labels, r=0.68)
#   - gap_penalty removed: r=0.95 with skill_overlap (too high)
#   - skill_overlap removed: r=0.93 with keyword_bonus (redundant)
FEATURE_NAMES = [
    "embedding",
    "keyword_bonus"
]


def build_features(resume: Resume, job: JobPosting, embedding_score: float = None) -> Dict[str, float]:
    """
    Build feature dictionary for a resume-job pair.

    Args:
        resume: Resume object
        job: JobPosting object
        embedding_score: Pre-computed embedding score (optional, will compute if None)

    Returns:
        Dict mapping feature names to values
    """
    # Load config and vocabulary
    config = load_config()
    vocab = load_skills_vocabulary()

    # Compute embedding score if not provided
    if embedding_score is None:
        embedding_results = rank_jobs(resume, [job], top_k=1)
        embedding_score = embedding_results[0]['score'] if embedding_results else 0.0

    # Merge resume skills (auto-extract from text)
    vocab_list = list(vocab)
    merged_skills = merge_resume_skills(resume, vocab_list)
    resume_skills_normalized = normalize_skills(merged_skills, vocab)

    # Normalize job skills
    job_skills_normalized = normalize_skills(job.skills, vocab)

    # Calculate skill-based features
    skill_overlap = calculate_skill_overlap(resume_skills_normalized, job_skills_normalized)

    keyword_bonus = calculate_keyword_bonus(
        resume_skills_normalized,
        job_skills_normalized,
        config['keywords']['high_priority'],
        config['keywords']['high_priority_multiplier'],
        config['normalization']['max_keywords']
    )

    # NOTE: Using embedding + keyword_bonus for LTR
    # These 2 features balance model richness vs multicollinearity:
    #   - embedding: Captures semantic similarity (correlation with keyword_bonus: r=0.89)
    #   - keyword_bonus: Captures priority keyword matching (r=0.68 with labels)
    # Removed features (too high multicollinearity):
    #   - skill_overlap: r=0.95 with gap_penalty, r=0.93 with keyword_bonus
    #   - gap_penalty: r=0.95 with skill_overlap (nearly identical)
    # L2 regularization (C=0.1) helps stabilize weights with remaining correlation.
    features = {
        "embedding": embedding_score,
        "keyword_bonus": keyword_bonus
    }

    return features


def vectorize(features: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dict to numpy vector using fixed feature order.

    Args:
        features: Dict mapping feature names to values

    Returns:
        Numpy array of feature values in FEATURE_NAMES order
    """
    vector = np.array([features.get(name, 0.0) for name in FEATURE_NAMES])
    return vector


def build_feature_matrix(resumes: List[Resume], jobs: List[JobPosting],
                          embedding_cache: Dict[tuple, float] = None) -> np.ndarray:
    """
    Build feature matrix for all resume-job pairs.

    Args:
        resumes: List of Resume objects
        jobs: List of JobPosting objects
        embedding_cache: Optional dict mapping (resume_id, job_id) to embedding_score

    Returns:
        Numpy array of shape (n_pairs, n_features)
    """
    feature_vectors = []

    for resume in resumes:
        for job in jobs:
            # Get embedding score from cache if available
            embedding_score = None
            if embedding_cache is not None:
                key = (resume.resume_id, job.job_id)
                embedding_score = embedding_cache.get(key)

            # Build features
            features = build_features(resume, job, embedding_score)
            vector = vectorize(features)
            feature_vectors.append(vector)

    return np.array(feature_vectors)
