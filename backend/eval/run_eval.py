"""
Evaluation script for job recommendation system.

This script:
1. Loads resumes and jobs
2. For each resume, gets Top-15 job recommendations
3. Loads labels from labels_final.csv (uses final_label if available, else suggested_label)
4. Computes Precision@K and NDCG@K for K=[5, 10, 15]
5. Outputs aggregated metrics
"""

import json
import csv
import os
import sys
from typing import List, Dict, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import JobPosting, Resume
from services.retrieval import rank_jobs
from services.ranking import rank_jobs_with_features
from metrics import calculate_metrics_for_resume, aggregate_metrics


def get_base_dir():
    """Get the backend directory path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    return backend_dir


def load_resumes() -> List[Resume]:
    """Load all resumes from JSONL file."""
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, "data", "resumes.jsonl")
    resumes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                resume_data = json.loads(line.strip())
                resumes.append(Resume(**resume_data))
    return resumes


def load_jobs() -> List[JobPosting]:
    """Load all jobs from JSONL file."""
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, "data", "jobs.jsonl")
    jobs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                job_data = json.loads(line.strip())
                jobs.append(JobPosting(**job_data))
    return jobs


def load_labels(filepath: str) -> Dict[str, Dict[str, int]]:
    """
    Load labels from CSV file.

    Returns dict mapping resume_id -> {job_id -> label}
    Uses final_label if available, otherwise falls back to suggested_label
    """
    labels_by_resume = defaultdict(dict)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            resume_id = row['resume_id']
            job_id = row['job_id']

            # Use final_label if available, otherwise use suggested_label
            if row['final_label'] and row['final_label'].strip():
                label = int(row['final_label'])
            else:
                label = int(row['suggested_label'])

            labels_by_resume[resume_id][job_id] = label

    return labels_by_resume


def get_top_k_jobs(resume: Resume, jobs: List[JobPosting], k: int = 15) -> List[str]:
    """
    Get top-k job IDs for a resume using the current ranking system.

    Returns list of job_ids in ranked order
    """
    # Step 1: Get embedding-based similarity scores
    embedding_results = rank_jobs(resume, jobs, top_k=len(jobs))

    # Step 2: Re-rank using explainable features
    ranked_results = rank_jobs_with_features(resume, embedding_results)

    # Step 3: Extract job IDs from top-k results
    top_k_results = ranked_results[:k]
    return [result["job"].job_id for result in top_k_results]


def main():
    """Main evaluation function."""
    # Change to backend directory so relative paths work correctly
    backend_dir = get_base_dir()
    original_dir = os.getcwd()
    os.chdir(backend_dir)

    try:
        print("=" * 80)
        print("Job Recommendation System Evaluation")
        print("=" * 80)

        # Check if labels file exists
        labels_file = os.path.join("eval", "labels_final.csv")
        if not os.path.exists(labels_file):
            print(f"\nError: {labels_file} not found!")
            print("Please run generate_labels.py first to create the labels file.")
            sys.exit(1)

        # Load data
        print("\n[1/4] Loading data...")
        resumes = load_resumes()
        jobs = load_jobs()
        labels_by_resume = load_labels(labels_file)
        print(f"  Loaded {len(resumes)} resumes")
        print(f"  Loaded {len(jobs)} jobs")
        print(f"  Loaded labels for {len(labels_by_resume)} resumes")

        # Determine if using final labels or suggested labels
        using_final = False
        with open(labels_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['final_label'] and row['final_label'].strip():
                    using_final = True
                    break

        if using_final:
            print(f"  Using final_label (human-corrected) for evaluation")
        else:
            print(f"  Using suggested_label (LLM-generated) for evaluation")
            print(f"  Note: Fill in 'final_label' column in {labels_file} for human-corrected evaluation")

        # Evaluate each resume
        print("\n[2/4] Running recommendations and computing metrics...")
        k_values = [5, 10, 15]
        all_metrics = []
        per_resume_metrics = {}

        for i, resume in enumerate(resumes, 1):
            print(f"\n  Resume {i}/{len(resumes)} (ID: {resume.resume_id})")

            # Get Top-15 recommendations
            recommended_ids = get_top_k_jobs(resume, jobs, k=15)
            print(f"    Generated {len(recommended_ids)} recommendations")

            # Get labels for this resume
            if resume.resume_id not in labels_by_resume:
                print(f"    Warning: No labels found for {resume.resume_id}, skipping...")
                continue

            labels = labels_by_resume[resume.resume_id]

            # Calculate metrics
            metrics = calculate_metrics_for_resume(recommended_ids, labels, k_values)
            all_metrics.append(metrics)
            per_resume_metrics[resume.resume_id] = metrics

            # Print metrics for this resume
            print(f"    Precision@5:  {metrics['precision'][5]:.3f}")
            print(f"    Precision@10: {metrics['precision'][10]:.3f}")
            print(f"    Precision@15: {metrics['precision'][15]:.3f}")
            print(f"    NDCG@5:       {metrics['ndcg'][5]:.3f}")
            print(f"    NDCG@10:      {metrics['ndcg'][10]:.3f}")
            print(f"    NDCG@15:      {metrics['ndcg'][15]:.3f}")

        # Aggregate metrics
        print("\n[3/4] Aggregating metrics across all resumes...")
        aggregated = aggregate_metrics(all_metrics)

        # Print aggregated results
        print("\n[4/4] Final Results:")
        print("=" * 80)
        print("AGGREGATED METRICS (Mean across all resumes)")
        print("=" * 80)

        print("\nPrecision@K (proportion of relevant items in top-K):")
        for k in k_values:
            print(f"  Precision@{k:2d}: {aggregated['precision'][k]:.4f}")

        print("\nNDCG@K (ranking quality with position discount):")
        for k in k_values:
            print(f"  NDCG@{k:2d}:      {aggregated['ndcg'][k]:.4f}")

        print("\n" + "=" * 80)
        print("Evaluation complete!")
        print("=" * 80)

        # Print interpretation guide
        print("\nInterpretation:")
        print("  - Precision@K: Higher is better (0.0 - 1.0)")
        print("  - NDCG@K: Higher is better (0.0 - 1.0)")
        print("  - Relevance threshold: Label >= 2 (medium/strong match)")
        print(f"\nLabel source: {'final_label (human-corrected)' if using_final else 'suggested_label (LLM-generated)'}")

        # Save results to JSON (save to eval directory)
        results = {
            "data_stats": {
                "num_resumes": len(resumes),
                "num_jobs": len(jobs),
                "num_labels": sum(len(labels) for labels in labels_by_resume.values()),
                "label_source": "final_label" if using_final else "suggested_label"
            },
            "aggregated_metrics": aggregated,
            "per_resume_metrics": {
                resume_id: {
                    "precision": metrics["precision"],
                    "ndcg": metrics["ndcg"]
                }
                for resume_id, metrics in per_resume_metrics.items()
            }
        }

        output_file = os.path.join("eval", "eval_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_file}")

    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
