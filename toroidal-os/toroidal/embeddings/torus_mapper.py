#!/usr/bin/env python3
"""
TOROIDAL OS - Embedding to Torus Mapper
=======================================
Maps 1024-dim embeddings to 4D torus positions.

This creates a semantic topology where similar concepts
are positioned near each other on the torus manifold.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import EmbeddingConfig


@dataclass
class TorusPosition:
    """Position on a 4D torus (4 angle coordinates)."""
    th1: float  # Angle 1 (degrees, 0-360)
    th2: float  # Angle 2
    th3: float  # Angle 3
    th4: float  # Angle 4 (major winding)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.th1, self.th2, self.th3, self.th4])

    def to_radians(self) -> np.ndarray:
        """Convert to radians."""
        return np.radians(self.to_array())

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TorusPosition':
        """Create from numpy array."""
        arr = np.asarray(arr)
        return cls(
            th1=float(arr[0]) % 360,
            th2=float(arr[1]) % 360,
            th3=float(arr[2]) % 360,
            th4=float(arr[3]) % 360
        )

    def distance_to(self, other: 'TorusPosition') -> float:
        """
        Compute toroidal distance to another position.

        Uses the shortest path on the torus (wrapping around).

        Args:
            other: Another position

        Returns:
            Distance in degrees (0 to ~540)
        """
        diff = np.abs(self.to_array() - other.to_array())
        # Handle wraparound (shortest path)
        diff = np.where(diff > 180, 360 - diff, diff)
        return float(np.sqrt(np.sum(diff ** 2)))

    def angular_separation(self, other: 'TorusPosition') -> np.ndarray:
        """
        Compute angular separation for each dimension.

        Args:
            other: Another position

        Returns:
            Array of angular differences (0-180 degrees)
        """
        diff = np.abs(self.to_array() - other.to_array())
        return np.where(diff > 180, 360 - diff, diff)


class EmbeddingToTorusMapper:
    """
    Maps embeddings to positions on a 4D torus.

    Strategy:
    1. Normalize embedding to unit vector
    2. Apply learned projection to 4D
    3. Map each dimension to angle (0-360 degrees)

    The projection can be:
    - Random (but deterministic via seed)
    - Learned via training on semantic data
    - PCA-based for dimensionality reduction
    """

    def __init__(self, config: EmbeddingConfig = None, seed: int = 42):
        """
        Initialize the mapper.

        Args:
            config: Embedding configuration
            seed: Random seed for reproducibility
        """
        self.config = config or EmbeddingConfig()
        self.seed = seed

        # Projection matrix (1024 -> 4)
        # Initialized deterministically
        np.random.seed(seed)
        self._projection = self._init_projection()

    def _init_projection(self) -> np.ndarray:
        """
        Initialize projection matrix.

        Uses a pseudo-orthogonal random projection that
        preserves distances reasonably well.
        """
        input_dim = self.config.model_dim  # 1024
        output_dim = self.config.torus_dims  # 4

        # Random projection with controlled variance
        # Using orthogonal-like initialization
        matrix = np.random.randn(input_dim, output_dim)

        # QR decomposition for more orthogonal projection
        q, r = np.linalg.qr(matrix)

        # Use Q (orthogonal columns)
        return q[:, :output_dim].astype(np.float32)

    def map_embedding(self, embedding: np.ndarray) -> TorusPosition:
        """
        Map an embedding to a torus position.

        Args:
            embedding: Embedding vector (1024-dim)

        Returns:
            TorusPosition with 4 angle coordinates
        """
        embedding = np.asarray(embedding, dtype=np.float32)

        # Ensure 1D
        if embedding.ndim > 1:
            embedding = embedding.flatten()

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Project to 4D
        projected = np.dot(embedding, self._projection)

        # Map to angles (0-360 degrees)
        # Use arctan2 to map from [-inf, inf] to [-pi, pi], then to [0, 360]
        angles = (np.degrees(np.arctan2(projected, np.roll(projected, 1))) + 180) % 360

        return TorusPosition(
            th1=float(angles[0]),
            th2=float(angles[1]),
            th3=float(angles[2]),
            th4=float(angles[3])
        )

    def map_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Map a batch of embeddings to torus positions.

        Args:
            embeddings: Batch of embeddings (N x 1024)

        Returns:
            Array of positions (N x 4) in degrees
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize batch
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        embeddings = embeddings / norms

        # Project to 4D
        projected = np.dot(embeddings, self._projection)

        # Map to angles
        rolled = np.roll(projected, 1, axis=1)
        angles = (np.degrees(np.arctan2(projected, rolled)) + 180) % 360

        return angles

    def compute_similarity_from_positions(
        self,
        pos1: TorusPosition,
        pos2: TorusPosition
    ) -> float:
        """
        Compute similarity from torus positions.

        Closer positions = higher similarity.

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Similarity score (0-1)
        """
        distance = pos1.distance_to(pos2)
        # Max distance is sqrt(4 * 180^2) = 360
        max_distance = np.sqrt(4 * 180**2)
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, similarity))

    def get_projection_matrix(self) -> np.ndarray:
        """Get the current projection matrix."""
        return self._projection.copy()

    def set_projection_matrix(self, matrix: np.ndarray) -> None:
        """
        Set a custom projection matrix.

        Useful for learned projections.

        Args:
            matrix: Projection matrix (1024 x 4)
        """
        if matrix.shape != (self.config.model_dim, self.config.torus_dims):
            raise ValueError(
                f"Matrix must be {self.config.model_dim} x {self.config.torus_dims}, "
                f"got {matrix.shape}"
            )
        self._projection = matrix.astype(np.float32)