#!/usr/bin/env python3
"""
TOROIDAL OS - Embedding Utilities
==================================
Utility functions for embedding operations.
"""

import numpy as np
from typing import Union, List, Tuple


def normalize(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize an embedding to unit length.

    Args:
        embedding: Single embedding (1D) or batch of embeddings (2D)

    Returns:
        Normalized embedding(s) with unit L2 norm
    """
    if embedding.ndim == 1:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    else:
        # Batch normalization
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return embedding / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        a: First embedding (1D array)
        b: Second embedding (1D array)

    Returns:
        Similarity score between -1 and 1
    """
    if a is None or b is None:
        return 0.0

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes must match: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of embeddings.

    Args:
        embeddings_a: First set of embeddings (N x D)
        embeddings_b: Second set of embeddings (M x D)

    Returns:
        Similarity matrix (N x M)
    """
    embeddings_a = np.asarray(embeddings_a, dtype=np.float32)
    embeddings_b = np.asarray(embeddings_b, dtype=np.float32)

    # Normalize both sets
    embeddings_a = normalize(embeddings_a)
    embeddings_b = normalize(embeddings_b)

    # Compute similarity matrix
    return np.dot(embeddings_a, embeddings_b.T)


def semantic_search(
    query_embedding: np.ndarray,
    embeddings: Union[List[np.ndarray], np.ndarray],
    top_k: int = 10,
    threshold: float = 0.0
) -> List[Tuple[int, float]]:
    """
    Search for most similar embeddings to a query.

    Args:
        query_embedding: Query embedding (1D array)
        embeddings: List or array of embeddings to search
        top_k: Maximum number of results
        threshold: Minimum similarity threshold

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    if len(embeddings) == 0:
        return []

    # Ensure 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Normalize query
    query_embedding = normalize(np.asarray(query_embedding, dtype=np.float32))

    # Normalize embeddings
    embeddings = normalize(embeddings)

    # Compute similarities
    similarities = np.dot(embeddings, query_embedding)

    # Filter by threshold and sort
    results = [
        (i, float(sim))
        for i, sim in enumerate(similarities)
        if sim >= threshold
    ]

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]


def embedding_distance(
    a: np.ndarray,
    b: np.ndarray,
    metric: str = "cosine"
) -> float:
    """
    Compute distance between two embeddings.

    Args:
        a: First embedding
        b: Second embedding
        metric: Distance metric ("cosine", "euclidean", "manhattan")

    Returns:
        Distance value (lower = more similar)
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if metric == "cosine":
        # Convert similarity to distance
        return 1.0 - cosine_similarity(a, b)
    elif metric == "euclidean":
        return float(np.linalg.norm(a - b))
    elif metric == "manhattan":
        return float(np.sum(np.abs(a - b)))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def average_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute average of multiple embeddings.

    Args:
        embeddings: List of embeddings to average

    Returns:
        Averaged and normalized embedding
    """
    if not embeddings:
        raise ValueError("Cannot average empty list of embeddings")

    avg = np.mean(embeddings, axis=0)
    return normalize(avg)


def weighted_average_embedding(
    embeddings: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Compute weighted average of embeddings.

    Args:
        embeddings: List of embeddings
        weights: Weight for each embedding

    Returns:
        Weighted average embedding
    """
    if len(embeddings) != len(weights):
        raise ValueError("Number of embeddings must match number of weights")

    if not embeddings:
        raise ValueError("Cannot average empty list of embeddings")

    embeddings = np.array(embeddings)
    weights = np.array(weights)

    # Normalize weights
    weights = weights / np.sum(weights)

    weighted = np.sum(embeddings * weights[:, np.newaxis], axis=0)
    return normalize(weighted)