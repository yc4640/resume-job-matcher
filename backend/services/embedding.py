"""
Embedding service using sentence-transformers for text vectorization.
Uses all-MiniLM-L6-v2 model for efficient local embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

# Global model instance - loaded once on first use
_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """
    Get or initialize the sentence transformer model.
    Model is loaded once and cached for subsequent calls.

    Returns:
        SentenceTransformer: The loaded model instance
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {_model_name}")
        _model = SentenceTransformer(_model_name)
        print(f"Model loaded successfully")
    return _model


def embed_texts(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Generate embeddings for given text(s) using sentence-transformers.

    Args:
        texts: Single text string or list of text strings to embed

    Returns:
        np.ndarray: Embeddings array of shape (n_texts, embedding_dim)
                   For all-MiniLM-L6-v2, embedding_dim = 384
    """
    model = _get_model()

    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)

    return embeddings
