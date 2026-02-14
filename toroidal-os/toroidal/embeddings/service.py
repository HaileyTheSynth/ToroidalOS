#!/usr/bin/env python3
"""
TOROIDAL OS - Octen Embedding Service
=====================================
Main embedding service using Octen-Embedding-0.6B.

Model: Octen-Embedding-0.6B
- 1024 dimensions
- Multilingual
- Apache 2.0 license
- sentence-transformers compatible
"""

import numpy as np
from typing import Union, List, Optional
import threading
import logging

from .config import EmbeddingConfig, DEFAULT_CONFIG
from .cache import EmbeddingCache
from .utils import cosine_similarity, normalize, semantic_search

# Configure logging
logger = logging.getLogger(__name__)


class OctenEmbeddingService:
    """
    Embedding service using Octen-Embedding-0.6B.

    Features:
    - Lazy loading (load model on first use)
    - LRU caching with memory limits
    - Thread-safe operations
    - Batch encoding support
    - Memory management (unload when not needed)

    Example:
        >>> service = OctenEmbeddingService()
        >>> embedding = service.encode("Hello world")
        >>> similarity = service.similarity(embedding, service.encode("Hi there"))
    """

    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the embedding service.

        Args:
            config: Configuration for the service
        """
        self.config = config or DEFAULT_CONFIG

        # Model (lazy loaded)
        self._model = None
        self._model_lock = threading.Lock()

        # Cache
        self._cache = EmbeddingCache(self.config)

        # Statistics
        self._encode_count = 0
        self._batch_count = 0

    def _load_model(self):
        """Load the embedding model (lazy loading)."""
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer

                if self.config.local_model_path:
                    logger.info(f"Loading embedding model from {self.config.local_model_path}")
                    self._model = SentenceTransformer(
                        self.config.local_model_path,
                        device=self.config.device
                    )
                else:
                    logger.info(f"Loading embedding model {self.config.model_name}")
                    self._model = SentenceTransformer(
                        self.config.model_name,
                        device=self.config.device
                    )

                logger.info(f"Model loaded successfully (dim={self.config.model_dim})")

            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def encode(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True,
        normalize_embedding: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts
            use_cache: Whether to use cached embeddings
            normalize_embedding: Whether to normalize embeddings to unit length

        Returns:
            Embedding(s) as numpy array
            - Single text: shape (1024,)
            - Multiple texts: shape (N, 1024)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        if use_cache:
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Encode uncached texts
        if uncached_texts:
            self._load_model()

            new_embeddings = self._model.encode(
                uncached_texts,
                normalize_embeddings=normalize_embedding,
                convert_to_numpy=True
            )

            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                if use_cache:
                    self._cache.put(text, embedding)

            for i, embedding in zip(uncached_indices, new_embeddings):
                results.append((i, embedding))

        # Sort by original index and extract embeddings
        results.sort(key=lambda x: x[0])
        embeddings = np.array([e for _, e in results])

        self._encode_count += len(texts)

        if single_input:
            return embeddings[0]
        return embeddings

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode a large batch of texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embeddings array (N, 1024)
        """
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        self._batch_count += 1
        self._encode_count += len(texts)

        return embeddings

    def similarity(
        self,
        a: Union[str, np.ndarray],
        b: Union[str, np.ndarray]
    ) -> float:
        """
        Compute similarity between two texts or embeddings.

        Args:
            a: Text or embedding
            b: Text or embedding

        Returns:
            Similarity score (-1 to 1)
        """
        if isinstance(a, str):
            a = self.encode(a)
        if isinstance(b, str):
            b = self.encode(b)

        return cosine_similarity(a, b)

    def find_similar(
        self,
        query: Union[str, np.ndarray],
        candidates: List[Union[str, np.ndarray]],
        top_k: int = 10,
        threshold: float = None
    ) -> List[tuple]:
        """
        Find most similar items to query.

        Args:
            query: Query text or embedding
            candidates: List of candidate texts or embeddings
            top_k: Maximum results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity) tuples
        """
        threshold = threshold or self.config.semantic_threshold

        # Encode query if needed
        if isinstance(query, str):
            query_embedding = self.encode(query)
        else:
            query_embedding = np.asarray(query)

        # Encode candidates if needed
        candidate_embeddings = []
        for c in candidates:
            if isinstance(c, str):
                candidate_embeddings.append(self.encode(c))
            else:
                candidate_embeddings.append(np.asarray(c))

        return semantic_search(
            query_embedding,
            candidate_embeddings,
            top_k=top_k,
            threshold=threshold
        )

    def unload(self) -> None:
        """
        Unload the model to free memory.

        The model will be reloaded on next encode() call.
        """
        with self._model_lock:
            if self._model is not None:
                logger.info("Unloading embedding model")
                del self._model
                self._model = None

                # Force garbage collection
                import gc
                gc.collect()

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "model_loaded": self.is_loaded(),
            "model_name": self.config.model_name,
            "model_dim": self.config.model_dim,
            "encode_count": self._encode_count,
            "batch_count": self._batch_count,
            "cache_stats": self._cache.get_stats()
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"OctenEmbeddingService("
            f"model={self.config.model_name}, "
            f"loaded={self.is_loaded()}, "
            f"cached={len(self._cache)})"
            f")"
        )


# Convenience function for quick access
_default_service: Optional[OctenEmbeddingService] = None
_default_lock = threading.Lock()


def get_embedding_service(config: EmbeddingConfig = None) -> OctenEmbeddingService:
    """
    Get the default embedding service singleton.

    Args:
        config: Optional configuration (used on first call)

    Returns:
        OctenEmbeddingService instance
    """
    global _default_service

    with _default_lock:
        if _default_service is None:
            _default_service = OctenEmbeddingService(config)
        return _default_service