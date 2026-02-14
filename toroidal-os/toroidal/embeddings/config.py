#!/usr/bin/env python3
"""
TOROIDAL OS - Embedding Configuration
======================================
Configuration for Octen-Embedding-0.6B integration.

Model: Octen-Embedding-0.6B (Apache 2.0)
- 1024-dimensional embeddings
- Multilingual support
- Sentence-transformers compatible
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""

    # Model settings
    model_name: str = "Octen/Octen-Embedding-0.6B"
    model_dim: int = 1024  # Embedding dimension

    # Memory management
    cache_max_items: int = 500  # Max items in LRU cache
    cache_max_memory_mb: int = 50  # Max memory for cache (MB)

    # Loading behavior
    lazy_load: bool = True  # Load model on first use
    device: str = "cpu"  # Device to run on (cpu/cuda)

    # Similarity threshold
    semantic_threshold: float = 0.7  # Threshold for "similar" items

    # Torus mapping
    torus_dims: int = 4  # Number of torus angle dimensions

    # Optional model path for offline use
    local_model_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.cache_max_items <= 0:
            raise ValueError("cache_max_items must be positive")
        if self.cache_max_memory_mb <= 0:
            raise ValueError("cache_max_memory_mb must be positive")
        if not 0 <= self.semantic_threshold <= 1:
            raise ValueError("semantic_threshold must be between 0 and 1")


# Default configuration for ToroidalOS on Mi Mix (6GB RAM)
DEFAULT_CONFIG = EmbeddingConfig(
    model_name="Octen/Octen-Embedding-0.6B",
    model_dim=1024,
    cache_max_items=500,
    cache_max_memory_mb=50,
    lazy_load=True,
    device="cpu",
    semantic_threshold=0.7,
    torus_dims=4,
    local_model_path=None
)


# Configuration for low-memory environments
LOW_MEMORY_CONFIG = EmbeddingConfig(
    model_name="Octen/Octen-Embedding-0.6B",
    model_dim=1024,
    cache_max_items=200,
    cache_max_memory_mb=20,
    lazy_load=True,
    device="cpu",
    semantic_threshold=0.75,
    torus_dims=4,
    local_model_path=None
)