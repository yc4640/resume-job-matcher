"""
Pairwise training data construction for Learning to Rank (LTR).

This module constructs pairwise preference data from weak labels:
- For each resume, find pairs of jobs where one has higher label than the other
- Create training examples: (features_winner - features_loser, label=1)
"""

import sys
import os
from typing import List, Dict, Tuple, Any
import numpy as np

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schemas import Resume, JobPosting


def construct_pairwise_data(
    labels: List[Dict[str, Any]],
    resumes_dict: Dict[str, Resume],
    jobs_dict: Dict[str, JobPosting],
    features_dict: Dict[Tuple[str, str], np.ndarray],
    min_rel_diff: int = 2,
    add_mirror: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct pairwise training data from weak labels.

    For each resume:
    1. Group jobs by label
    2. For pairs where label_i > label_j + min_rel_diff:
       - Create training example: (features_i - features_j, label=1)
       - If add_mirror=True, also create mirrored negative example:
         (features_j - features_i, label=0)

    Args:
        labels: List of label dicts with keys: resume_id, job_id, label
        resumes_dict: Dict mapping resume_id to Resume object
        jobs_dict: Dict mapping job_id to JobPosting object
        features_dict: Dict mapping (resume_id, job_id) to feature vector
        min_rel_diff: Minimum relevance difference to create a pair (default=2)
        add_mirror: If True, generate mirrored pairs (default=True)
                    - When False: only generate (winner - loser, y=1)
                    - When True: generate both (winner - loser, y=1) and (loser - winner, y=0)
                    NOTE: LogisticRegression requires at least 2 classes, so add_mirror=True is recommended.

    Returns:
        X_pairs: np.ndarray of shape (n_pairs, n_features) - feature differences
        y_pairs: np.ndarray of shape (n_pairs,) - labels where:
                 - y=1 means "the first job is preferred over the second job"
                 - y=0 means "the first job is not preferred over the second job"
                 When add_mirror=False, all labels are 1.
                 When add_mirror=True, sample count is approximately 2x with mixed labels.
    """
    # Group labels by resume
    resume_labels = {}
    for label_record in labels:
        resume_id = label_record['resume_id']
        if resume_id not in resume_labels:
            resume_labels[resume_id] = []
        resume_labels[resume_id].append(label_record)

    pairs_X = []
    pairs_y = []

    for resume_id, label_list in resume_labels.items():
        # For each resume, create pairs
        for i, label_i in enumerate(label_list):
            for j, label_j in enumerate(label_list):
                if i == j:
                    continue

                # Only create pair if label_i is significantly better than label_j
                if label_i['label'] >= label_j['label'] + min_rel_diff:
                    key_i = (label_i['resume_id'], label_i['job_id'])
                    key_j = (label_j['resume_id'], label_j['job_id'])

                    # Check if features exist for both
                    if key_i not in features_dict or key_j not in features_dict:
                        continue

                    # Create feature difference: winner - loser
                    features_winner = features_dict[key_i]
                    features_loser = features_dict[key_j]
                    feature_diff = features_winner - features_loser

                    # Add positive pair (winner - loser, y=1)
                    pairs_X.append(feature_diff)
                    pairs_y.append(1)  # Winner is preferred

                    # If add_mirror=True, also add mirrored negative pair
                    if add_mirror:
                        # Add negative pair (loser - winner, y=0)
                        pairs_X.append(-feature_diff)  # Same as features_loser - features_winner
                        pairs_y.append(0)  # Loser is not preferred

    if not pairs_X:
        # No pairs created (insufficient label variance)
        return np.array([]), np.array([])

    X_pairs = np.array(pairs_X)
    y_pairs = np.array(pairs_y)

    return X_pairs, y_pairs


def check_sufficient_pairs(X_pairs: np.ndarray, min_pairs: int = 10) -> bool:
    """
    Check if we have sufficient pairwise training data.

    Args:
        X_pairs: Pairwise feature matrix
        min_pairs: Minimum number of pairs required

    Returns:
        True if sufficient, False otherwise
    """
    return len(X_pairs) >= min_pairs
