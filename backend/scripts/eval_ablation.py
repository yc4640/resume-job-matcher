"""
Evaluation script with Leave-One-Out Cross-Validation (LOOCV) and Ablation Study.

This script:
1. Loads weak labels (1-5 scale) from labels_suggested.jsonl
2. Performs LOOCV: leave one resume out for testing, train on the rest
3. Evaluates multiple ranking variants (ablation):
   - V1: embedding_only (semantic similarity baseline)
   - V2: heuristic (current weighted feature combination)
   - V3: LTR_logreg (pairwise logistic regression)
4. Computes metrics: NDCG@5/10, Precision@5/10
5. Outputs results to results/ablation_results.json and eval_report.md
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Resume, JobPosting
from services.retrieval import rank_jobs
from services.ranking import rank_jobs_with_features
from src.ranking.features import build_features, vectorize
from src.ranking.pairwise import construct_pairwise_data, check_sufficient_pairs
from src.ranking.ltr_logreg import PairwiseLTRModel


def load_labels(filepath: str = "eval/labels_suggested.jsonl") -> List[Dict[str, Any]]:
    """Load weak labels from JSONL file."""
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                labels.append(json.loads(line.strip()))
    return labels


def load_resumes(filepath: str = "data/resumes.jsonl") -> List[Resume]:
    """Load resumes from JSONL file."""
    resumes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                resumes.append(Resume(**json.loads(line.strip())))
    return resumes


def load_jobs(filepath: str = "data/jobs.jsonl") -> List[JobPosting]:
    """Load jobs from JSONL file."""
    jobs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                jobs.append(JobPosting(**json.loads(line.strip())))
    return jobs


def compute_dcg(relevances: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.

    DCG@k = sum_{i=1}^k (2^rel_i - 1) / log2(i + 1)
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2 ** rel - 1) / np.log2(i + 1)
    return dcg


def compute_ndcg(predicted_order: List[str], labels_dict: Dict[str, float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Args:
        predicted_order: List of job_ids in predicted ranking order
        labels_dict: Dict mapping job_id to relevance label
        k: Cutoff position

    Returns:
        NDCG@k score (0-1)
    """
    # Get relevances in predicted order
    predicted_relevances = [labels_dict.get(job_id, 0) for job_id in predicted_order]

    # Compute DCG@k
    dcg_k = compute_dcg(predicted_relevances, k)

    # Compute ideal DCG@k (sort by relevance descending)
    ideal_relevances = sorted(labels_dict.values(), reverse=True)
    idcg_k = compute_dcg(ideal_relevances, k)

    # Avoid division by zero
    if idcg_k == 0:
        return 0.0

    return dcg_k / idcg_k


def compute_precision(predicted_order: List[str], labels_dict: Dict[str, float], k: int, threshold: int = 4) -> float:
    """
    Compute Precision at k.

    Precision@k = (number of relevant items in top-k) / k

    Args:
        predicted_order: List of job_ids in predicted ranking order
        labels_dict: Dict mapping job_id to relevance label
        k: Cutoff position
        threshold: Minimum label to be considered relevant (default=4)

    Returns:
        Precision@k score (0-1)
    """
    top_k = predicted_order[:k]
    relevant_count = sum(1 for job_id in top_k if labels_dict.get(job_id, 0) >= threshold)
    return relevant_count / k if k > 0 else 0.0


def rank_with_variant(variant: str, resume: Resume, jobs: List[JobPosting],
                       ltr_model: PairwiseLTRModel = None,
                       embedding_cache: Dict[Tuple[str, str], float] = None) -> List[str]:
    """
    Rank jobs using a specific variant.

    Args:
        variant: "embedding_only", "heuristic", or "ltr_logreg"
        resume: Resume object
        jobs: List of JobPosting objects
        ltr_model: Trained LTR model (required for "ltr_logreg" variant)
        embedding_cache: Optional cache of embedding scores

    Returns:
        List of job_ids in ranked order
    """
    if variant == "embedding_only":
        # Baseline: use only embedding similarity
        embedding_results = rank_jobs(resume, jobs, top_k=len(jobs))
        ranked_job_ids = [result['job'].job_id for result in embedding_results]

    elif variant == "heuristic":
        # Current system: weighted feature combination
        embedding_results = rank_jobs(resume, jobs, top_k=len(jobs))
        ranked_results = rank_jobs_with_features(resume, embedding_results)
        ranked_job_ids = [result['job'].job_id for result in ranked_results]

    elif variant == "ltr_logreg":
        # LTR: pairwise logistic regression
        if ltr_model is None:
            raise ValueError("LTR model is required for ltr_logreg variant")
        ranked_results = ltr_model.rank_jobs(resume, jobs, embedding_cache)
        ranked_job_ids = [result['job'].job_id for result in ranked_results]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ranked_job_ids


def evaluate_fold(test_resume: Resume, train_resumes: List[Resume], jobs: List[JobPosting],
                   labels: List[Dict], embedding_cache: Dict) -> Dict[str, Dict[str, float]]:
    """
    Evaluate one LOOCV fold.

    Args:
        test_resume: Resume to test on
        train_resumes: Resumes to train on
        jobs: All jobs
        labels: All labels
        embedding_cache: Embedding score cache

    Returns:
        Dict mapping variant name to metrics dict
    """
    # Create dicts for easy lookup
    resumes_dict = {r.resume_id: r for r in (train_resumes + [test_resume])}
    jobs_dict = {j.job_id: j for j in jobs}

    # Get labels for test resume
    test_labels = [l for l in labels if l['resume_id'] == test_resume.resume_id]
    test_labels_dict = {l['job_id']: l['label'] for l in test_labels}

    # Build features for training data
    train_features = {}
    for resume in train_resumes:
        for job in jobs:
            key = (resume.resume_id, job.job_id)
            emb_score = embedding_cache.get(key, 0.0)
            features = build_features(resume, job, emb_score)
            train_features[key] = vectorize(features)

    # Construct pairwise training data
    train_labels = [l for l in labels if l['resume_id'] != test_resume.resume_id]
    X_pairs, y_pairs = construct_pairwise_data(
        train_labels,
        resumes_dict,
        jobs_dict,
        train_features,
        min_rel_diff=2
    )

    # Check if we have sufficient pairs for LTR
    can_train_ltr = check_sufficient_pairs(X_pairs, min_pairs=10)

    # Train LTR model if possible
    ltr_model = None
    if can_train_ltr:
        try:
            ltr_model = PairwiseLTRModel(random_state=42)
            ltr_model.train(X_pairs, y_pairs)
        except Exception as e:
            print(f"  Warning: LTR training failed for {test_resume.resume_id}: {e}")
            can_train_ltr = False

    # Evaluate each variant
    results = {}

    for variant in ["embedding_only", "heuristic", "ltr_logreg"]:
        if variant == "ltr_logreg" and not can_train_ltr:
            # Skip LTR if not enough training data, fallback to heuristic
            variant_key = "ltr_logreg_fallback"
            ranked_job_ids = rank_with_variant("heuristic", test_resume, jobs, None, embedding_cache)
        else:
            variant_key = variant
            ranked_job_ids = rank_with_variant(variant, test_resume, jobs, ltr_model, embedding_cache)

        # Compute metrics
        ndcg_5 = compute_ndcg(ranked_job_ids, test_labels_dict, k=5)
        ndcg_10 = compute_ndcg(ranked_job_ids, test_labels_dict, k=10)
        prec_5 = compute_precision(ranked_job_ids, test_labels_dict, k=5, threshold=4)
        prec_10 = compute_precision(ranked_job_ids, test_labels_dict, k=10, threshold=4)

        results[variant_key] = {
            "ndcg@5": ndcg_5,
            "ndcg@10": ndcg_10,
            "precision@5": prec_5,
            "precision@10": prec_10
        }
    return results


def run_loocv_ablation(resumes: List[Resume], jobs: List[JobPosting], labels: List[Dict]) -> Dict[str, Any]:
    """
    Run Leave-One-Out Cross-Validation with ablation study.

    Args:
        resumes: All resumes
        jobs: All jobs
        labels: All weak labels

    Returns:
        Results dict with per-fold and aggregated metrics
    """
    print("\n" + "=" * 80)
    print("Running LOOCV + Ablation Study")
    print("=" * 80)

    # Pre-compute embedding scores (cache to avoid recomputation)
    print("\n[1/3] Pre-computing embedding scores...")
    embedding_cache = {}
    for resume in resumes:
        embedding_results = rank_jobs(resume, jobs, top_k=len(jobs))
        for result in embedding_results:
            key = (resume.resume_id, result['job'].job_id)
            embedding_cache[key] = result['score']
    print(f"  Cached {len(embedding_cache)} embedding scores")

    # Run LOOCV
    print(f"\n[2/3] Running LOOCV ({len(resumes)} folds)...")
    fold_results = []

    for i, test_resume in enumerate(resumes, 1):
        print(f"\n  Fold {i}/{len(resumes)}: Testing on {test_resume.resume_id}")

        # Split data
        train_resumes = [r for r in resumes if r.resume_id != test_resume.resume_id]

        # Evaluate this fold
        fold_metrics = evaluate_fold(test_resume, train_resumes, jobs, labels, embedding_cache)

        fold_results.append({
            "fold": i,
            "test_resume_id": test_resume.resume_id,
            "metrics": fold_metrics
        })

        # Print fold results
        for variant, metrics in fold_metrics.items():
            print(f"    {variant:25s} NDCG@5={metrics['ndcg@5']:.3f}, P@5={metrics['precision@5']:.3f}")

    # Aggregate results
    print("\n[3/3] Aggregating results...")
    aggregated = {}

    # Get all variant names
    all_variants = set()
    for fold in fold_results:
        all_variants.update(fold["metrics"].keys())

    # Compute mean and std for each variant
    for variant in all_variants:
        metrics_list = defaultdict(list)

        for fold in fold_results:
            if variant in fold["metrics"]:
                for metric_name, value in fold["metrics"][variant].items():
                    metrics_list[metric_name].append(value)

        # Compute statistics
        aggregated[variant] = {}
        for metric_name, values in metrics_list.items():
            aggregated[variant][metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values
            }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_folds": len(resumes),
        "n_jobs": len(jobs),
        "per_fold_results": fold_results,
        "aggregated_results": aggregated
    }

    return results


def save_results(results: Dict[str, Any], output_path: str = "results/ablation_results.json"):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to {output_path}")


def generate_report(results: Dict[str, Any], output_path: str = "eval/eval_report.md"):
    """Generate evaluation report in Markdown format."""
    report_lines = [
        "# Evaluation Report: LOOCV + Ablation Study",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Evaluation Method:** Leave-One-Out Cross-Validation (LOOCV)",
        f"**Number of Folds:** {results['n_folds']}",
        f"**Number of Jobs:** {results['n_jobs']}",
        "",
        "## Weak Labels",
        "",
        "This evaluation uses weak labels (1-5 scale) generated by LLM:",
        "- **Label 1:** Not a match",
        "- **Label 2:** Weak match",
        "- **Label 3:** Partial match",
        "- **Label 4:** Good match (relevant threshold)",
        "- **Label 5:** Strong match",
        "",
        "Labels are generated ONLY from resume and job content (no system bias).",
        "",
        "## Evaluation Methodology",
        "",
        "**LOOCV (Leave-One-Out Cross-Validation):**",
        "- For each resume:",
        "  - Use it as test set",
        "  - Train on all other resumes",
        "  - Evaluate ranking on ALL jobs for this resume (not just top-15)",
        "",
        "**Metrics:**",
        "- **NDCG@5/10:** Normalized Discounted Cumulative Gain (ranking quality)",
        "- **Precision@5/10:** Fraction of relevant jobs (label ≥ 4) in top-K",
        "",
        "## Ablation Study Results",
        "",
        "| Variant | NDCG@5 | NDCG@10 | Precision@5 | Precision@10 |",
        "|---------|--------|---------|-------------|--------------|"
    ]

    # Add results table
    agg = results['aggregated_results']
    for variant in sorted(agg.keys()):
        metrics = agg[variant]
        ndcg5 = metrics['ndcg@5']['mean']
        ndcg5_std = metrics['ndcg@5']['std']
        ndcg10 = metrics['ndcg@10']['mean']
        ndcg10_std = metrics['ndcg@10']['std']
        prec5 = metrics['precision@5']['mean']
        prec5_std = metrics['precision@5']['std']
        prec10 = metrics['precision@10']['mean']
        prec10_std = metrics['precision@10']['std']

        row = f"| {variant:20s} | {ndcg5:.3f}±{ndcg5_std:.3f} | {ndcg10:.3f}±{ndcg10_std:.3f} | {prec5:.3f}±{prec5_std:.3f} | {prec10:.3f}±{prec10_std:.3f} |"
        report_lines.append(row)

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "**Variants:**",
        "- **embedding_only:** Baseline using only semantic similarity",
        "- **heuristic:** Current system with weighted features (embedding + skill overlap + keyword bonus - gap penalty)",
        "- **ltr_logreg:** Pairwise Learning to Rank with Logistic Regression",
        "- **ltr_logreg_fallback:** LTR fell back to heuristic (insufficient training pairs)",
        "",
        "**Key Findings:**",
        "- Compare NDCG@5 across variants to see which ranking method performs best",
        "- Higher NDCG indicates better ranking quality (relevant jobs ranked higher)",
        "- Higher Precision indicates more relevant jobs in top positions",
        "",
        "## Failure Cases & Next Steps",
        "",
        "**Potential Issues:**",
        "- Small dataset (7 resumes) may lead to high variance in LOOCV",
        "- LTR may fall back to heuristic if insufficient label variance per fold",
        "- Weak labels are noisy and may not reflect true user preferences",
        "",
        "**Next Steps:**",
        "1. Collect more resumes and jobs to improve statistical power",
        "2. Consider manual label correction for high-confidence cases",
        "3. Experiment with different LTR models (e.g., LambdaMART, RankNet)",
        "4. Tune hyperparameters (feature weights, min_rel_diff, threshold)",
        ""
    ])

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"[OK] Report saved to {output_path}")


def main():
    """Main evaluation function."""
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(backend_dir)

    print("=" * 80)
    print("Evaluation: LOOCV + Ablation Study")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    resumes = load_resumes()
    jobs = load_jobs()
    labels = load_labels()

    print(f"  Resumes: {len(resumes)}")
    print(f"  Jobs: {len(jobs)}")
    print(f"  Labels: {len(labels)}")

    # Validate labels
    expected_pairs = len(resumes) * len(jobs)
    if len(labels) != expected_pairs:
        print(f"⚠️  Warning: Expected {expected_pairs} labels, but found {len(labels)}")

    # Run LOOCV + Ablation
    results = run_loocv_ablation(resumes, jobs, labels)

    # Save results
    save_results(results)

    # Generate report
    generate_report(results)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    agg = results['aggregated_results']
    for variant in sorted(agg.keys()):
        metrics = agg[variant]
        print(f"\n{variant}:")
        for metric_name in ['ndcg@5', 'ndcg@10', 'precision@5', 'precision@10']:
            mean = metrics[metric_name]['mean']
            std = metrics[metric_name]['std']
            print(f"  {metric_name:15s} {mean:.3f} ± {std:.3f}")

    print("\n" + "=" * 80)
    print("[OK] Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
