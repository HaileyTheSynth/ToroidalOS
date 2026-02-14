#!/usr/bin/env python3
"""
TOROIDAL OS - Embedding Cache
=============================
LRU cache for embeddings with memory limits.

Optimized for 6GB Mi Mix:
- Max 500 items (default)
- Max 50MB memory
- ~50MB for 500 x 1024 x 4 bytes (float32)
"""

import threading
import time
from collections import OrderedDict
from typing import Callable, Optional, Tuple
import hashlib
import numpy as np

from .config import EmbeddingConfig


class EmbeddingCache:
    """
    LRU cache for text embeddings with memory budget.

    Features:
    - LRU eviction when item limit reached
    - Memory-based eviction when memory budget exceeded
    - Thread-safe operations
    - Content-addressable keys (hash of text)
    """

    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the embedding cache.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.max_items = self.config.cache_max_items
        self.max_memory = self.config.cache_max_memory_mb * 1024 * 1024  # Convert to bytes

        # Cache storage: key -> (embedding, timestamp)
        self._cache: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()

        # Current memory usage estimate
        self._memory_usage = 0

        # Lock for thread safety
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

    def _compute_key(self, text: str) -> str:
        """Compute content-addressable key for text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _estimate_size(self, embedding: np.ndarray) -> int:
        """Estimate memory size of an embedding."""
        # embedding is float32 (4 bytes per element)
        return embedding.nbytes

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache if available.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None
        """
        key = self._compute_key(text)

        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                embedding, _ = self._cache.pop(key)
                self._cache[key] = (embedding, time.time())
                self._hits += 1
                return embedding.copy()

            self._misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        key = self._compute_key(text)
        size = self._estimate_size(embedding)

        with self._lock:
            # If already exists, remove old entry
            if key in self._cache:
                old_embedding, _ = self._cache.pop(key)
                self._memory_usage -= self._estimate_size(old_embedding)

            # Evict items if necessary
            while (len(self._cache) >= self.max_items or
                   self._memory_usage + size > self.max_memory):
                if not self._cache:
                    break
                # Remove least recently used (first item)
                oldest_key, (oldest_embedding, _) = self._cache.popitem(last=False)
                self._memory_usage -= self._estimate_size(oldest_embedding)

            # Add new entry
            self._cache[key] = (embedding.copy(), time.time())
            self._memory_usage += size

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], np.ndarray]
    ) -> np.ndarray:
        """
        Get embedding from cache or compute if not present.

        Args:
            text: Text to embed
            compute_fn: Function to compute embedding if not cached

        Returns:
            Embedding vector
        """
        # Try cache first
        cached = self.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = compute_fn(text)

        # Cache it
        self.put(text, embedding)

        return embedding

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "items": len(self._cache),
                "max_items": self.max_items,
                "memory_mb": self._memory_usage / (1024 * 1024),
                "max_memory_mb": self.config.cache_max_memory_mb,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate
            }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, text: str) -> bool:
        key = self._compute_key(text)
        return key in self._cache


# Global singleton cache (optional)
_global_cache: Optional[EmbeddingCache] = None
_global_cache_lock = threading.Lock()


def get_global_cache(config: EmbeddingConfig = None) -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _global_cache

    with _global_cache_lock:
        if _global_cache is None:
            _global_cache = EmbeddingCache(config)
        return _global_cache