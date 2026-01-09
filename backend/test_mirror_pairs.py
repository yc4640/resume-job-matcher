"""
Test script to verify mirror pairs logic without Unicode issues.
"""
import json
import numpy as np
from typing import List, Dict, Any

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
    """Test mirror pairs logic."""
    print("=" * 80)
    print("Testing Mirror Pairs Logic")
    print("=" * 80)

    # Load small subset of data for quick test
    print("\n[1/5] Loading data...")
    resumes = load_resumes('data/resumes.jsonl')[:3]  # Only first 3 resumes
    jobs = load_jobs('data/jobs.jsonl')[:10]  # Only first 10 jobs
    labels = load_labels('eval/labels_suggested.jsonl')

    # Filter labels to match subset
    resume_ids = {r.resume_id for r in resumes}
    job_ids = {j.job_id for j in jobs}
    labels = [
        lbl for lbl in labels
        if lbl['resume_id'] in resume_ids and lbl['job_id'] in job_ids
    ]

    print(f"  Loaded: {len(resumes)} resumes, {len(jobs)} jobs, {len(labels)} labels")

    # Create lookup dicts
    resumes_dict = {r.resume_id: r for r in resumes}
    jobs_dict = {j.job_id: j for j in jobs}

    # Build feature cache (simplified - only compute for labeled pairs)
    print("\n[2/5] Building feature cache...")
    features_dict = {}
    embedding_cache = {}

    for resume in resumes:
        embedding_results = rank_jobs(resume, jobs, top_k=len(jobs))
        for result in embedding_results:
            key = (resume.resume_id, result['job'].job_id)
            embedding_cache[key] = result['score']

    for label_record in labels:
        key = (label_record['resume_id'], label_record['job_id'])
        if key in embedding_cache:
            resume = resumes_dict[key[0]]
            job = jobs_dict[key[1]]
            emb_score = embedding_cache[key]
            features = build_features(resume, job, emb_score)
            features_dict[key] = vectorize(features)

    print(f"  Built {len(features_dict)} feature vectors")

    # Test 1: Construct pairwise data with add_mirror=True (default)
    print("\n[3/5] Testing add_mirror=True (default)...")
    X_pairs_mirror, y_pairs_mirror = construct_pairwise_data(
        labels,
        resumes_dict,
        jobs_dict,
        features_dict,
        min_rel_diff=2
        # add_mirror defaults to True
    )

    print(f"  Created {len(X_pairs_mirror)} pairwise samples")
    unique_labels_mirror = np.unique(y_pairs_mirror)
    print(f"  Unique classes: {unique_labels_mirror}")
    print(f"  Label distribution: {dict(zip(*np.unique(y_pairs_mirror, return_counts=True)))}")

    if len(unique_labels_mirror) >= 2:
        print("  [PASS] add_mirror=True produces both classes (0 and 1)")
    else:
        print("  [FAIL] add_mirror=True only produces single class")
        return False

    # Test 2: Construct pairwise data with add_mirror=False
    print("\n[4/5] Testing add_mirror=False...")
    X_pairs_no_mirror, y_pairs_no_mirror = construct_pairwise_data(
        labels,
        resumes_dict,
        jobs_dict,
        features_dict,
        min_rel_diff=2,
        add_mirror=False
    )

    print(f"  Created {len(X_pairs_no_mirror)} pairwise samples")
    unique_labels_no_mirror = np.unique(y_pairs_no_mirror)
    print(f"  Unique classes: {unique_labels_no_mirror}")
    print(f"  Label distribution: {dict(zip(*np.unique(y_pairs_no_mirror, return_counts=True)))}")

    # Verify sample count relationship
    if len(X_pairs_mirror) == 2 * len(X_pairs_no_mirror):
        print("  [PASS] add_mirror=True doubles sample count")
    else:
        print(f"  [WARNING] Expected 2x samples, got {len(X_pairs_mirror)} vs {len(X_pairs_no_mirror)}")

    # Test 3: Try training with mirrored data
    print("\n[5/5] Testing model training with mirrored data...")
    try:
        model = PairwiseLTRModel(random_state=42)
        model.train(X_pairs_mirror, y_pairs_mirror)
        print("  [PASS] Model training succeeded with mirrored data")

        weights = model.get_feature_weights()
        print("\n  Learned feature weights:")
        for name, weight in weights.items():
            print(f"    {name:20s} {weight:+.4f}")

    except Exception as e:
        print(f"  [FAIL] Model training failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("All Tests PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - add_mirror=True (default) creates {len(X_pairs_mirror)} samples with classes {unique_labels_mirror}")
    print(f"  - add_mirror=False creates {len(X_pairs_no_mirror)} samples with classes {unique_labels_no_mirror}")
    print(f"  - LogisticRegression training succeeds with mirrored pairs")
    print(f"  - Feature dimension: {X_pairs_mirror.shape[1]}")
    print(f"  - Feature names: {FEATURE_NAMES}")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
