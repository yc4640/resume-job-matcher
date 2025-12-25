"""
Evaluation metrics for job recommendation system.

Implements:
- Precision@K: Measures the proportion of relevant items in top-K recommendations
- NDCG@K: Normalized Discounted Cumulative Gain - measures ranking quality with position discount
"""

import numpy as np
from typing import List, Dict, Any


def precision_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Precision@K.

    Precision@K = (# of relevant items in top-K) / K

    Args:
        recommended_ids: List of recommended job IDs (in ranked order)
        relevant_ids: List of relevant job IDs (ground truth)
        k: Number of top recommendations to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k <= 0 or not recommended_ids:
        return 0.0

    # Take top-k recommendations
    top_k = recommended_ids[:k]

    # Count how many are relevant
    relevant_count = sum(1 for job_id in top_k if job_id in relevant_ids)

    return relevant_count / k


def dcg_at_k(recommended_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K

    Args:
        recommended_ids: List of recommended job IDs (in ranked order)
        relevance_scores: Dict mapping job_id to relevance score (0-3)
        k: Number of top recommendations to consider

    Returns:
        DCG@K score
    """
    if k <= 0 or not recommended_ids:
        return 0.0

    dcg = 0.0
    for i, job_id in enumerate(recommended_ids[:k], start=1):
        relevance = relevance_scores.get(job_id, 0.0)
        # DCG formula: rel / log2(position + 1)
        # Position starts at 1, so i + 1 for log denominator
        dcg += relevance / np.log2(i + 1)

    return dcg


def ideal_dcg_at_k(relevance_scores: Dict[str, float], k: int) -> float:
    """
    Calculate Ideal DCG at K (best possible DCG).

    IDCG@K is the DCG@K when items are sorted by relevance in descending order.

    Args:
        relevance_scores: Dict mapping job_id to relevance score (0-3)
        k: Number of top recommendations to consider

    Returns:
        IDCG@K score
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    # Sort relevance scores in descending order
    sorted_relevances = sorted(relevance_scores.values(), reverse=True)

    # Calculate DCG for ideal ranking
    idcg = 0.0
    for i, relevance in enumerate(sorted_relevances[:k], start=1):
        idcg += relevance / np.log2(i + 1)

    return idcg


def ndcg_at_k(recommended_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    NDCG normalizes DCG by the ideal DCG (best possible ranking), giving a score between 0 and 1.
    - 1.0 means perfect ranking
    - 0.0 means no relevant items in top-K

    Args:
        recommended_ids: List of recommended job IDs (in ranked order)
        relevance_scores: Dict mapping job_id to relevance score (0-3)
        k: Number of top recommendations to consider

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if k <= 0 or not recommended_ids or not relevance_scores:
        return 0.0

    dcg = dcg_at_k(recommended_ids, relevance_scores, k)
    idcg = ideal_dcg_at_k(relevance_scores, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_metrics_for_resume(
    recommended_ids: List[str],
    labels: Dict[str, int],
    k_values: List[int] = [5, 10, 15]
) -> Dict[str, Dict[int, float]]:
    """
    Calculate Precision@K and NDCG@K for multiple K values for a single resume.

    Args:
        recommended_ids: List of recommended job IDs (in ranked order)
        labels: Dict mapping job_id to label (0-3)
        k_values: List of K values to evaluate

    Returns:
        Dict with keys "precision" and "ndcg", each containing a dict of K -> score
        Example: {"precision": {5: 0.8, 10: 0.7, 15: 0.6}, "ndcg": {5: 0.85, 10: 0.78, 15: 0.72}}
    """
    # Determine relevant jobs (label >= 2 is considered relevant)
    relevant_ids = [job_id for job_id, label in labels.items() if label >= 2]

    # Convert labels to relevance scores (for NDCG)
    relevance_scores = {job_id: float(label) for job_id, label in labels.items()}

    metrics = {
        "precision": {},
        "ndcg": {}
    }

    for k in k_values:
        metrics["precision"][k] = precision_at_k(recommended_ids, relevant_ids, k)
        metrics["ndcg"][k] = ndcg_at_k(recommended_ids, relevance_scores, k)

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, Dict[int, float]]]) -> Dict[str, Dict[int, float]]:
    """
    Aggregate metrics across multiple resumes by taking the mean.

    Args:
        all_metrics: List of metric dicts (one per resume)

    Returns:
        Aggregated metrics with mean values
        Example: {"precision": {5: 0.75, 10: 0.68, 15: 0.62}, "ndcg": {5: 0.80, 10: 0.73, 15: 0.68}}
    """
    if not all_metrics:
        return {"precision": {}, "ndcg": {}}

    # Determine all K values present
    k_values = set()
    for metrics in all_metrics:
        k_values.update(metrics["precision"].keys())
    k_values = sorted(k_values)

    aggregated = {
        "precision": {},
        "ndcg": {}
    }

    for k in k_values:
        # Collect all precision and ndcg values for this K
        precision_values = [m["precision"][k] for m in all_metrics if k in m["precision"]]
        ndcg_values = [m["ndcg"][k] for m in all_metrics if k in m["ndcg"]]

        # Calculate means
        aggregated["precision"][k] = np.mean(precision_values) if precision_values else 0.0
        aggregated["ndcg"][k] = np.mean(ndcg_values) if ndcg_values else 0.0

    return aggregated


# Example usage and test
if __name__ == "__main__":
    print("Testing evaluation metrics...")

    # Example: recommended jobs (in ranked order)
    recommended = ["job_001", "job_002", "job_003", "job_004", "job_005"]

    # Example: labels (0-3)
    labels = {
        "job_001": 3,  # Strong match
        "job_002": 2,  # Medium match
        "job_003": 1,  # Weak match
        "job_004": 3,  # Strong match
        "job_005": 0,  # No match
        "job_006": 2,  # Medium match (not recommended)
        "job_007": 3,  # Strong match (not recommended)
    }

    # Calculate metrics
    metrics = calculate_metrics_for_resume(recommended, labels, k_values=[3, 5])

    print(f"\nRecommended jobs: {recommended}")
    print(f"Labels: {labels}")
    print(f"\nMetrics:")
    print(f"  Precision@3: {metrics['precision'][3]:.3f}")
    print(f"  Precision@5: {metrics['precision'][5]:.3f}")
    print(f"  NDCG@3: {metrics['ndcg'][3]:.3f}")
    print(f"  NDCG@5: {metrics['ndcg'][5]:.3f}")

    # Relevant jobs: job_001 (3), job_002 (2), job_004 (3), job_006 (2), job_007 (3)
    # Top-3: job_001 (3), job_002 (2), job_003 (1) -> 2 relevant out of 3 -> P@3 = 2/3
    # Top-5: job_001 (3), job_002 (2), job_003 (1), job_004 (3), job_005 (0) -> 3 relevant out of 5 -> P@5 = 3/5

    print("\nExpected:")
    print(f"  Precision@3: {2/3:.3f} (2 relevant in top-3)")
    print(f"  Precision@5: {3/5:.3f} (3 relevant in top-5)")
