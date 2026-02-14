#!/usr/bin/env python3
"""
TOROIDAL OS - Embeddings Module
================================
Semantic embeddings using Octen-Embedding-0.6B.

This module provides:
- OctenEmbeddingService: Main embedding service
- EmbeddingCache: LRU cache with memory limits
- EmbeddingToTorusMapper: Map embeddings to 4D torus positions
- Utility functions for similarity and search

Example:
    >>> from toroidal.embeddings import OctenEmbeddingService, EmbeddingConfig
    >>>
    >>> # Initialize service
    >>> config = EmbeddingConfig(lazy_load=True)
    >>> service = OctenEmbeddingService(config)
    >>>
    >>> # Encode text
    >>> embedding = service.encode("Hello, world!")
    >>> print(f"Embedding shape: {embedding.shape}")  # (1024,)
    >>>
    >>> # Compute similarity
    >>> sim = service.similarity("cat", "feline")
    >>> print(f"Similarity: {sim:.3f}")
    >>>
    >>> # Map to torus
    >>> from toroidal.embeddings import EmbeddingToTorusMapper
    >>> mapper = EmbeddingToTorusMapper(config)
    >>> position = mapper.map_embedding(embedding)
    >>> print(f"Torus position: ({position.th1}, {position.th2}, {position.th3}, {position.th4})")
"""

# Configuration
from .config import (
    EmbeddingConfig,
    DEFAULT_CONFIG,
    LOW_MEMORY_CONFIG,
)

# Core service
from .service import (
    OctenEmbeddingService,
    get_embedding_service,
)

# Caching
from .cache import (
    EmbeddingCache,
    get_global_cache,
)

# Torus mapping
from .torus_mapper import (
    TorusPosition,
    EmbeddingToTorusMapper,
)

# Utilities
from .utils import (
    normalize,
    cosine_similarity,
    cosine_similarity_matrix,
    semantic_search,
    embedding_distance,
    average_embedding,
    weighted_average_embedding,
)

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    # Configuration
    "EmbeddingConfig",
    "DEFAULT_CONFIG",
    "LOW_MEMORY_CONFIG",

    # Service
    "OctenEmbeddingService",
    "get_embedding_service",

    # Cache
    "EmbeddingCache",
    "get_global_cache",

    # Torus mapping
    "TorusPosition",
    "EmbeddingToTorusMapper",

    # Utilities
    "normalize",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "semantic_search",
    "embedding_distance",
    "average_embedding",
    "weighted_average_embedding",
]