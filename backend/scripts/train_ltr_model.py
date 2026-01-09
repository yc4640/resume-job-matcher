"""
Production LTR Model Training Script

This script trains the final LTR (Learning to Rank) model for production use.
It uses all available data to train a single model saved to disk.

Usage:
    python scripts/train_ltr_model.py \
        --resumes_path data/resumes.jsonl \
        --jds_path data/jobs.jsonl \
        --labels_path eval/labels_suggested.jsonl \
        --out models/ltr_logreg_1to5.joblib

Features:
    - CLI arguments for flexibility
    - Validates feature consistency with ranking_config.yaml
    - Prints training statistics
    - Saves model with joblib for production use
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Resume, JobPosting
from services.retrieval import rank_jobs
from src.ranking.features import build_features, vectorize, FEATURE_NAMES
from src.ranking.pairwise import construct_pairwise_data
from src.ranking.ltr_logreg import PairwiseLTRModel


def load_resumes(filepath: str) -> List[Resume]:
    """Load resumes from JSONL file."""
    resumes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                resumes.append(Resume(**json.loads(line.strip())))
    return resumes


def load_jobs(filepath: str) -> List[JobPosting]:
    """Load jobs from JSONL file."""
    jobs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                jobs.append(JobPosting(**json.loads(line.strip())))
    return jobs


def load_labels(filepath: str) -> List[Dict[str, Any]]:
    """Load weak labels from JSONL file."""
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line.strip()))
    return labels


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train LTR model for production use',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--resumes_path',
        type=str,
        required=True,
        help='Path to resumes JSONL file'
    )
    parser.add_argument(
        '--jds_path',
        type=str,
        required=True,
        help='Path to jobs JSONL file'
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        required=True,
        help='Path to labels JSONL file (1-5 scale)'
    )
    parser.add_argument(
        '--min_rel_diff',
        type=int,
        default=2,
        help='Minimum relevance difference for pairwise construction'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='models/ltr_logreg.joblib',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LTR Model Training for Production")
    print("=" * 80)

    # Step 1: Load data
    print(f"\n[1/6] Loading data...")
    print(f"  Resumes:   {args.resumes_path}")
    print(f"  Jobs:      {args.jds_path}")
    print(f"  Labels:    {args.labels_path}")

    resumes = load_resumes(args.resumes_path)
    jobs = load_jobs(args.jds_path)
    labels = load_labels(args.labels_path)

    print(f"  Loaded: {len(resumes)} resumes, {len(jobs)} jobs, {len(labels)} labels")

    # Step 2: Validate data
    print(f"\n[2/6] Validating data...")
    expected_pairs = len(resumes) * len(jobs)
    if len(labels) != expected_pairs:
        print(f"  [WARN]  Warning: Expected {expected_pairs} labels, found {len(labels)}")
    else:
        print(f"  [OK] Full coverage: {len(labels)}/{expected_pairs} pairs labeled")

    # Step 3: Build feature cache
    print(f"\n[3/6] Building feature cache...")
    print(f"  Computing embeddings for all pairs...")

    # Create lookup dicts
    resumes_dict = {r.resume_id: r for r in resumes}
    jobs_dict = {j.job_id: j for j in jobs}

    # Pre-compute embedding scores
    embedding_cache = {}
    for i, resume in enumerate(resumes, 1):
        print(f"    Resume {i}/{len(resumes)}: {resume.resume_id}", end="\r")
        embedding_results = rank_jobs(resume, jobs, top_k=len(jobs))
        for result in embedding_results:
            key = (resume.resume_id, result['job'].job_id)
            embedding_cache[key] = result['score']

    print(f"\n  [OK] Cached {len(embedding_cache)} embedding scores")

    # Build feature vectors
    print(f"  Computing features for all pairs...")
    features_dict = {}
    for i, resume in enumerate(resumes, 1):
        print(f"    Resume {i}/{len(resumes)}: {resume.resume_id}", end="\r")
        for job in jobs:
            key = (resume.resume_id, job.job_id)
            emb_score = embedding_cache.get(key, 0.0)
            features = build_features(resume, job, emb_score)
            features_dict[key] = vectorize(features)

    print(f"\n  [OK] Built {len(features_dict)} feature vectors")
    print(f"  Feature dimension: {len(FEATURE_NAMES)}")
    print(f"  Feature names: {FEATURE_NAMES}")

    # Step 4: Construct pairwise training data
    print(f"\n[4/6] Constructing pairwise training data...")
    print(f"  min_rel_diff: {args.min_rel_diff}")

    X_pairs, y_pairs = construct_pairwise_data(
        labels,
        resumes_dict,
        jobs_dict,
        features_dict,
        min_rel_diff=args.min_rel_diff
    )

    if len(X_pairs) == 0:
        print("  [FAIL] ERROR: No pairwise samples created!")
        print("  Possible reasons:")
        print("    - Labels have insufficient variance (all similar)")
        print("    - min_rel_diff is too high")
        sys.exit(1)

    # Check if y_pairs has at least 2 classes
    unique_labels = np.unique(y_pairs)
    if len(unique_labels) < 2:
        print(f"  [WARN]  WARNING: y_pairs only has {len(unique_labels)} class(es): {unique_labels}")
        print("  LogisticRegression requires at least 2 classes for training.")
        print("  Reconstructing with add_mirror=True to ensure both classes (0 and 1)...")

        X_pairs, y_pairs = construct_pairwise_data(
            labels,
            resumes_dict,
            jobs_dict,
            features_dict,
            min_rel_diff=args.min_rel_diff,
            add_mirror=True  # Force mirror pairs
        )

        unique_labels = np.unique(y_pairs)
        if len(unique_labels) < 2:
            print(f"  [FAIL] ERROR: Still only {len(unique_labels)} class after forcing add_mirror=True")
            print("  This should not happen. Please check the implementation.")
            sys.exit(1)

        print(f"  [OK] Reconstructed with {len(X_pairs)} samples, classes: {unique_labels}")

    print(f"  [OK] Created {len(X_pairs)} pairwise training samples")
    print(f"  Feature dimension: {X_pairs.shape[1]}")
    print(f"  Label distribution: {dict(zip(*np.unique(y_pairs, return_counts=True)))}")

    # Step 5: Train model
    print(f"\n[5/6] Training LTR model...")
    print(f"  Model: Pairwise Logistic Regression")
    print(f"  Random state: {args.random_state}")

    model = PairwiseLTRModel(random_state=args.random_state)
    model.train(X_pairs, y_pairs)

    print(f"  [OK] Model trained successfully")

    # Print feature weights
    weights = model.get_feature_weights()
    print(f"\n  Learned feature weights:")
    for name, weight in weights.items():
        print(f"    {name:20s} {weight:+.4f}")

    # Step 6: Save model
    print(f"\n[6/6] Saving model...")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Created directory: {output_dir}")

    model.save(args.out)
    print(f"  [OK] Model saved to: {args.out}")

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Data:")
    print(f"    Resumes:         {len(resumes)}")
    print(f"    Jobs:            {len(jobs)}")
    print(f"    Labels:          {len(labels)}")
    print(f"  Training:")
    print(f"    Pairwise samples: {len(X_pairs)}")
    print(f"    Features:         {len(FEATURE_NAMES)} {FEATURE_NAMES}")
    print(f"  Output:")
    print(f"    Model saved to:   {args.out}")
    print(f"\nNext steps:")
    print(f"  1. Start backend: uvicorn main:app --reload")
    print(f"  2. Enable LTR in Streamlit UI (checkbox)")
    print(f"  3. Model will be loaded automatically when use_ltr=True")
    print("")


if __name__ == "__main__":
    main()
