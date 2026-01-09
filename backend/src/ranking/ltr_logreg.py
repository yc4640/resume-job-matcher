"""
Learning to Rank (LTR) using Pairwise Logistic Regression.

This module implements:
1. Pairwise LTR training with Logistic Regression
2. Model persistence (save/load with joblib)
3. Scoring function for ranking jobs
"""

import sys
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schemas import Resume, JobPosting
from src.ranking.features import build_features, vectorize, FEATURE_NAMES


class PairwiseLTRModel:
    """
    Pairwise Learning to Rank model using Logistic Regression.

    The model learns to predict which job in a pair is preferred based on
    feature differences (winner_features - loser_features).

    For scoring, we use: score(job) = w · x + b
    where w are the learned weights and x are the job features.
    """

    def __init__(self, random_state: int = 42, C: float = 0.1):
        """
        Initialize the LTR model.

        Args:
            random_state: Random seed for reproducibility
            C: Inverse of regularization strength (lower = stronger regularization)
               Default 0.1 to handle multicollinearity between features
        """
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs',
            C=C,  # Strong L2 regularization to handle multicollinearity
            penalty='l2'
        )
        self.is_fitted = False

    def train(self, X_pairs: np.ndarray, y_pairs: np.ndarray) -> 'PairwiseLTRModel':
        """
        Train the pairwise LTR model.

        Args:
            X_pairs: Feature differences (winner - loser) of shape (n_pairs, n_features)
            y_pairs: Labels of shape (n_pairs,)

        Returns:
            Self (for method chaining)
        """
        if len(X_pairs) == 0:
            raise ValueError("Cannot train with empty pairwise data")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_pairs)

        # Train logistic regression
        self.model.fit(X_scaled, y_pairs)
        self.is_fitted = True

        return self

    def score(self, features: Dict[str, float]) -> float:
        """
        Compute ranking score for a job given its features.

        Score = w · x + b (linear scoring function)

        Args:
            features: Feature dict for a resume-job pair

        Returns:
            Ranking score (higher is better)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before scoring")

        # Convert features to vector
        x = vectorize(features).reshape(1, -1)

        # Scale features
        x_scaled = self.scaler.transform(x)

        # Compute score using decision_function (w · x + b)
        score = self.model.decision_function(x_scaled)[0]

        return score

    def rank_jobs(self, resume: Resume, jobs: List[JobPosting],
                  embedding_cache: Dict[Tuple[str, str], float] = None) -> List[Dict[str, Any]]:
        """
        Rank jobs for a given resume using the learned model.

        Args:
            resume: Resume object
            jobs: List of JobPosting objects
            embedding_cache: Optional cache of embedding scores

        Returns:
            List of dicts with keys: job, score, rank (sorted by score descending)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before ranking")

        results = []

        for job in jobs:
            # Get embedding score from cache if available
            embedding_score = None
            if embedding_cache is not None:
                key = (resume.resume_id, job.job_id)
                embedding_score = embedding_cache.get(key)

            # Build features
            features = build_features(resume, job, embedding_score)

            # Compute LTR score
            ltr_score = self.score(features)

            results.append({
                'job': job,
                'score': ltr_score,
                'features': features
            })

        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Add rank
        for rank, result in enumerate(results, start=1):
            result['rank'] = rank

        return results

    def save(self, filepath: str):
        """
        Save model to disk using joblib.

        Args:
            filepath: Path to save the model (e.g., 'models/ltr_logreg.joblib')
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save both scaler and model
        model_data = {
            'scaler': self.scaler,
            'model': self.model,
            'feature_names': FEATURE_NAMES
        }

        joblib.dump(model_data, filepath)

    @staticmethod
    def load(filepath: str) -> 'PairwiseLTRModel':
        """
        Load model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded PairwiseLTRModel instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model data
        model_data = joblib.load(filepath)

        # Create new instance
        ltr_model = PairwiseLTRModel()
        ltr_model.scaler = model_data['scaler']
        ltr_model.model = model_data['model']
        ltr_model.is_fitted = True

        # Validate feature names match
        loaded_features = model_data.get('feature_names', [])
        if loaded_features != FEATURE_NAMES:
            print(f"Warning: Feature names mismatch. Expected {FEATURE_NAMES}, got {loaded_features}")

        return ltr_model

    def get_feature_weights(self) -> Dict[str, float]:
        """
        Get the learned feature weights.

        Returns:
            Dict mapping feature names to weights
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        weights = self.model.coef_[0]
        return {name: weight for name, weight in zip(FEATURE_NAMES, weights)}
