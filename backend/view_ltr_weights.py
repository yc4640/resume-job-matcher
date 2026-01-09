"""
View LTR model weights.

This script loads the trained LTR model and displays the learned feature weights.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ranking.ltr_logreg import PairwiseLTRModel


def view_weights(model_path: str = "models/ltr_logreg.joblib"):
    """
    Load and display LTR model weights.

    Args:
        model_path: Path to the saved model
    """
    print("\n" + "="*80)
    print("LTR Model Weights Viewer")
    print("="*80)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found at: {model_path}")
        print("\nTo train the model, run:")
        print("  python scripts/train_ltr_model.py")
        return

    print(f"\nLoading model from: {model_path}")

    # Load model
    try:
        ltr_model = PairwiseLTRModel.load(model_path)
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Get feature weights
    weights = ltr_model.get_feature_weights()

    print("\n" + "-"*80)
    print("Learned Feature Weights")
    print("-"*80)
    print("\nFormula: score = w1*embedding + w2*skill_overlap + w3*keyword_bonus + w4*gap_penalty + bias\n")

    # Sort by absolute weight (most influential first)
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)

    for feature_name, weight in sorted_weights:
        sign = "+" if weight >= 0 else ""
        bar_length = int(abs(weight) * 50)  # Scale for visualization
        bar = "#" * bar_length
        print(f"  {feature_name:20s} {sign}{weight:>8.4f}  {bar}")

    # Get bias term
    bias = ltr_model.model.intercept_[0]
    print(f"\n  {'Bias (intercept)':20s} {bias:>+8.4f}")

    print("\n" + "-"*80)
    print("Interpretation")
    print("-"*80)
    print("""
Positive weights: Feature increases ranking score (job ranks higher)
Negative weights: Feature decreases ranking score (job ranks lower)

Typical interpretation:
- embedding: Higher semantic similarity → better match
- skill_overlap: More skills matched → better match
- keyword_bonus: High-priority keywords matched → better match
- gap_penalty: Missing critical skills → worse match (negative weight)

The model learns these weights from training data (weak labels).
Compare with heuristic weights in config/ranking_config.yaml:
  - Heuristic: Fixed weights (e.g., 0.4, 0.3, 0.2, -0.1)
  - LTR: Learned weights (optimized for training data)
    """)

    print("\n" + "="*80 + "\n")


def compare_with_heuristic():
    """Compare LTR weights with heuristic weights."""
    import yaml

    config_path = "config/ranking_config.yaml"

    if not os.path.exists(config_path):
        print(f"[WARNING] Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    heuristic_weights = config.get('weights', {})

    print("\n" + "="*80)
    print("LTR vs Heuristic Weights Comparison")
    print("="*80)
    print(f"\n{'Feature':<20} {'Heuristic':>12} {'LTR':>12} {'Difference':>12}")
    print("-"*60)

    # Load LTR model
    try:
        ltr_model = PairwiseLTRModel.load("models/ltr_logreg.joblib")
        ltr_weights = ltr_model.get_feature_weights()

        for feature in ['embedding', 'skill_overlap', 'keyword_bonus', 'gap_penalty']:
            heur_w = heuristic_weights.get(feature, 0.0)
            ltr_w = ltr_weights.get(feature, 0.0)
            diff = ltr_w - heur_w

            print(f"{feature:<20} {heur_w:>12.4f} {ltr_w:>12.4f} {diff:>+12.4f}")

        print("\nNote: Heuristic uses fixed weights, LTR learns from data.")
        print("Large differences suggest LTR found better weight combinations.\n")

    except Exception as e:
        print(f"[ERROR] Could not load LTR model: {e}")


if __name__ == "__main__":
    # View weights
    view_weights()

    # Compare with heuristic
    compare_with_heuristic()
