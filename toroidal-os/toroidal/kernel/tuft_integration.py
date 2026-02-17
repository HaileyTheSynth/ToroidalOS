#!/usr/bin/env python3
"""
TOROIDAL OS - TUFT Integration
================================
Integrates Topo9 TUFT dynamics with the HypergraphKernel.

This module adds:
1. Torus angles (th1-th4) to HypergraphKernel nodes
2. Barnes-Hut force computation for O(N log N) dynamics
3. Entropy field from node energy/access patterns
4. Coherence metrics for self-referential convergence
5. Semantic embeddings for enhanced topology and coherence
"""

import numpy as np
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import time

# Import base kernel
from .hypergraph import HypergraphKernel, Node, NodeType, Hyperedge

# Import embedding components
if TYPE_CHECKING:
    from toroidal.embeddings import OctenEmbeddingService, EmbeddingToTorusMapper

# Import TUFT dynamics (parent directory structure)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
try:
    from topo9_tuft_dynamics import BarnesHutTree, EntropyField
    TUFT_AVAILABLE = True
except ImportError:
    TUFT_AVAILABLE = False


@dataclass
class TUFTNode(Node):
    """
    Extended node with TUFT physics properties.

    Adds:
    - Torus angles (th1-th4) for 4D position
    - Velocity for dynamics
    - Berry phase (cumulative topological charge)
    - Region mask for bridge detection
    """
    # Torus position (degrees 0-360)
    th1: float = 0.0
    th2: float = 0.0
    th3: float = 0.0
    th4: float = 0.0

    # Velocity on torus (degrees per step)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Topological charge
    berry: float = 0.0

    # Region mask (for bridge detection)
    region_mask: int = 0

    # Windings (orbit completions on th4)
    windings: int = 0


class TUFTHypergraphKernel(HypergraphKernel):
    """
    HypergraphKernel extended with TUFT physics.

    This adds:
    1. Barnes-Hut dynamics for O(N log N) force computation
    2. Entropy field from energy/access patterns
    3. Torus positions for semantic topology
    4. Coherence metrics for convergence detection
    5. Semantic embeddings for enhanced topology and coherence
    """

    def __init__(
        self,
        max_nodes: int = 10000,
        grid_size: int = 16,
        embedding_service: "OctenEmbeddingService" = None,
        torus_mapper: "EmbeddingToTorusMapper" = None
    ):
        super().__init__(max_nodes=max_nodes)

        # Semantic embedding support
        self.embedding_service = embedding_service
        self.torus_mapper = torus_mapper

        # TUFT components
        self.grid_size = grid_size
        self.bh_tree = None
        self.entropy_field = None

        # Dynamics parameters
        self.beta_S = 0.2      # Entropy-gradient coupling
        self.k_coup = 0.5      # Diffusion coupling
        self.damping = 0.95
        self.force_scale = 5.0

        # Torus angles for each node
        self._torus_positions: Dict[str, np.ndarray] = {}
        self._velocities: Dict[str, np.ndarray] = {}
        self._berry: Dict[str, float] = {}

        # Embedding storage for semantic coherence
        self._embeddings: Dict[str, np.ndarray] = {}

        # Initialize entropy field if TUFT available
        if TUFT_AVAILABLE:
            self.entropy_field = EntropyField(grid_size)

        # Initialize torus mapper if embedding service provided
        if embedding_service and not torus_mapper:
            from toroidal.embeddings import EmbeddingToTorusMapper
            self.torus_mapper = EmbeddingToTorusMapper()

    def add_node(
        self,
        id: str,
        type: NodeType,
        data: Dict[str, Any] = None,
        content: str = None,
        trust=None
    ) -> Node:
        """
        Add a node with TUFT properties.

        Args:
            id: Node identifier
            type: Node type
            data: Node data dictionary
            content: Optional content string for semantic positioning
            trust: Optional trust tier for the node

        If embedding_service is available and content is provided,
        the node's torus position is derived from the semantic embedding.
        """
        # Pass trust to parent if available
        if trust is not None:
            node = super().add_node(id, type, data, trust=trust)
        else:
            node = super().add_node(id, type, data)

        # Generate embedding and semantic torus position if available
        embedding = None
        if self.embedding_service and content:
            try:
                embedding = self.embedding_service.encode(content)
                self._embeddings[id] = embedding

                # Map embedding to torus position
                if self.torus_mapper:
                    position = self.torus_mapper.map_embedding(embedding)
                    self._torus_positions[id] = position.to_array()
                    self._velocities[id] = np.zeros(4)
                    self._berry[id] = 0.0
                    return node
            except Exception:
                pass  # Fall back to hash-based position

        # Initialize torus position from hash (fallback)
        h = hash(id) % (2**32)
        self._torus_positions[id] = np.array([
            (h % 360),
            ((h >> 8) % 360),
            ((h >> 16) % 360),
            ((h >> 24) % 360)
        ], dtype=float)

        self._velocities[id] = np.zeros(4)
        self._berry[id] = 0.0

        return node

    def get_torus_position(self, node_id: str) -> np.ndarray:
        """Get torus position (th1, th2, th3, th4)"""
        return self._torus_positions.get(node_id, np.zeros(4))

    def get_berry(self, node_id: str) -> float:
        """Get Berry phase (topological charge)"""
        return self._berry.get(node_id, 0.0)

    def compute_coherence(self, node_id: str) -> float:
        """
        Compute coherence score for a node.

        Coherence = average similarity to hyperedge-connected neighbors.
        Uses semantic embedding similarity (40%), energy (30%), and torus distance (30%).
        """
        if node_id not in self.nodes:
            return 0.0

        node = self.nodes[node_id]
        pos = self._torus_positions.get(node_id, np.zeros(4))
        node_embedding = self._embeddings.get(node_id)

        # Get all neighbors via hyperedges
        neighbors = set()
        for he in self.hyperedges.values():
            if node_id in he.nodes:
                neighbors.update(he.nodes)
        neighbors.discard(node_id)

        if not neighbors:
            return 0.5  # Neutral coherence for isolated nodes

        coherence = 0.0
        for neighbor_id in neighbors:
            if neighbor_id not in self.nodes:
                continue

            neighbor = self.nodes[neighbor_id]
            neighbor_pos = self._torus_positions.get(neighbor_id, np.zeros(4))
            neighbor_embedding = self._embeddings.get(neighbor_id)

            # Energy similarity
            energy_sim = 1.0 - abs(node.energy - neighbor.energy)

            # Torus distance (normalized)
            diff = np.abs(pos - neighbor_pos)
            diff = np.where(diff > 180, 360 - diff, diff)
            torus_dist = np.sqrt(np.sum(diff ** 2)) / 360.0
            torus_sim = 1.0 - torus_dist

            # Semantic embedding similarity (if available)
            sem_sim = 0.0
            if node_embedding is not None and neighbor_embedding is not None:
                try:
                    from toroidal.embeddings.utils import cosine_similarity
                    sem_sim = cosine_similarity(node_embedding, neighbor_embedding)
                except Exception:
                    sem_sim = 0.0

            # Combine: 40% semantic, 30% energy, 30% torus
            if node_embedding is not None and neighbor_embedding is not None:
                coherence += 0.4 * sem_sim + 0.3 * energy_sim + 0.3 * torus_sim
            else:
                # Fall back to 50/50 if no embeddings
                coherence += (energy_sim + torus_sim) / 2

        return coherence / len(neighbors)

    def compute_curvature(self) -> float:
        """
        Compute global curvature.

        Curvature measures how much the "topic" has drifted.
        High curvature = rapid context changes.
        """
        if len(self.nodes) < 3:
            return 0.0

        # Get recent nodes (by creation time)
        recent = sorted(
            self.nodes.values(),
            key=lambda n: n.created_at,
            reverse=True
        )[:10]

        if len(recent) < 3:
            return 0.0

        curvature = 0.0
        for i in range(len(recent) - 1):
            n1, n2 = recent[i], recent[i+1]
            pos1 = self._torus_positions.get(n1.id, np.zeros(4))
            pos2 = self._torus_positions.get(n2.id, np.zeros(4))

            # Hamming-like distance in semantic space
            diff = np.abs(pos1 - pos2)
            diff = np.where(diff > 180, 360 - diff, diff)
            curvature += np.mean(diff)

        return curvature / (len(recent) - 1)

    def find_bridges(self, min_berry: float = 10.0, min_regions: int = 2) -> List[str]:
        """
        Find bridge nodes that connect multiple semantic regions.

        Bridges have:
        - High Berry phase (accessed frequently)
        - Multiple region membership
        """
        bridges = []

        # Assign regions based on torus position
        for node_id, pos in self._torus_positions.items():
            if node_id not in self.nodes:
                continue

            node = self.nodes[node_id]
            berry = self._berry.get(node_id, 0.0)

            # Determine regions from hyperedges
            regions = set()
            for he in self.hyperedges.values():
                if node_id in he.nodes:
                    # Region based on hyperedge type
                    if "conv" in he.relation.lower():
                        regions.add(0)
                    elif "topic" in he.relation.lower():
                        regions.add(1)
                    elif "entity" in he.relation.lower():
                        regions.add(2)
                    else:
                        regions.add(3)

            # Check bridge criteria
            if berry >= min_berry and len(regions) >= min_regions:
                bridges.append(node_id)

        return bridges

    def find_semantic_defects(
        self,
        torus_distance_threshold: float = 60.0,
        similarity_threshold: float = 0.3,
        max_defects: int = 20
    ) -> List[tuple]:
        """
        Find semantic defects - pairs of nodes that are close on the torus
        but semantically dissimilar (low embedding similarity).

        These represent cases where the semantic topology is inconsistent
        with the embedding space, indicating potential issues or interesting
        relationships.

        Args:
            torus_distance_threshold: Max torus distance (degrees) to consider "close"
            similarity_threshold: Min embedding similarity to consider "similar"
            max_defects: Maximum number of defects to return

        Returns:
            List of (node_id_a, node_id_b, torus_distance, embedding_similarity) tuples
        """
        if not self.embedding_service or not self._embeddings:
            return []

        defects = []
        node_ids = list(self._embeddings.keys())

        for i, id_a in enumerate(node_ids):
            if id_a not in self._torus_positions:
                continue

            pos_a = self._torus_positions[id_a]
            emb_a = self._embeddings[id_a]

            for id_b in node_ids[i + 1:]:
                if id_b not in self._torus_positions:
                    continue

                pos_b = self._torus_positions[id_b]
                emb_b = self._embeddings[id_b]

                # Compute torus distance
                diff = np.abs(pos_a - pos_b)
                diff = np.where(diff > 180, 360 - diff, diff)
                torus_dist = np.sqrt(np.sum(diff ** 2))

                # Only consider close pairs
                if torus_dist > torus_distance_threshold:
                    continue

                # Compute embedding similarity
                from toroidal.embeddings.utils import cosine_similarity
                similarity = cosine_similarity(emb_a, emb_b)

                # Check if this is a defect (close but dissimilar)
                if similarity < similarity_threshold:
                    defects.append((id_a, id_b, torus_dist, similarity))

                    if len(defects) >= max_defects:
                        return defects

        # Sort by severity (low similarity, low distance)
        defects.sort(key=lambda x: x[2] / (x[3] + 0.01))

        return defects

    def tuft_step(self):
        """
        One step of TUFT dynamics.

        Updates node positions on torus using Barnes-Hut forces.
        """
        if not TUFT_AVAILABLE or len(self.nodes) < 2:
            return

        # Build entropy field from node energies
        self._update_entropy_field()

        # Build Barnes-Hut tree
        self._build_bh_tree()

        # Compute forces and update positions
        for node_id, node in self.nodes.items():
            if node_id not in self._torus_positions:
                continue

            pos = self._torus_positions[node_id]
            vel = self._velocities[node_id]
            berry = self._berry.get(node_id, 1.0)

            # Get force from tree
            if self.bh_tree and self.bh_tree.root:
                # Create a temporary node-like object for force computation
                class TempNode:
                    def __init__(self, th1, th2, th3, th4, tau, windings, berry_milli):
                        self.th1, self.th2, self.th3, self.th4 = th1, th2, th3, th4
                        self.tau = tau
                        self.windings = windings
                        self.berry_milli = berry_milli

                temp = TempNode(pos[0], pos[1], pos[2], pos[3],
                               self.tau % 256, 0, int(berry * 1000))
                force = self.bh_tree.compute_force(temp, self.beta_S, self.k_coup)
            else:
                force = np.zeros(4)

            # Update velocity with semantic coherence term
            coh = self.compute_coherence(node_id)
            coherence_boost = 1.0 + 0.2 * coh  # Coherent nodes move more confidently
            berry_factor = np.log1p(berry) / 10.0 + 0.1
            vel = self.damping * vel + force * self.force_scale * berry_factor * coherence_boost
            self._velocities[node_id] = vel

            # Update position (toroidal)
            old_th4 = pos[3]
            pos = (pos + vel) % 360
            self._torus_positions[node_id] = pos

            # Track windings
            if pos[3] < old_th4 and vel[3] > 0:
                if node_id in self._berry:
                    self._berry[node_id] += 1.0

    def compute_wilson_loops(self) -> Dict[str, Any]:
        """
        Compute Wilson loop observables for topological analysis.

        Wilson loops measure the holonomy (phase change) when parallel
        transporting a vector around closed loops on the torus.

        Returns:
            Dictionary with Wilson loop statistics
        """
        if not self._torus_positions or len(self._torus_positions) < 3:
            return {"wilson_loops": 0, "avg_holonomy": 0.0}

        loops = []
        total_holonomy = 0.0

        # Find triangular loops via hyperedges
        for he in self.hyperedges.values():
            if not he.nodes or len(he.nodes) < 3:
                continue

            node_ids = list(he.nodes)[:4]  # Consider up to 4 nodes per hyperedge

            # Compute holonomy for each triangular path
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    for k in range(j + 1, len(node_ids)):
                        ids = [node_ids[i], node_ids[j], node_ids[k]]
                        if all(nid in self._torus_positions for nid in ids):
                            holonomy = self._compute_loop_holonomy(ids)
                            loops.append({
                                "nodes": ids,
                                "holonomy": holonomy,
                            })
                            total_holonomy += holonomy

        return {
            "wilson_loops": len(loops),
            "avg_holonomy": total_holonomy / len(loops) if loops else 0.0,
            "max_holonomy": max((l["holonomy"] for l in loops), default=0.0),
            "min_holonomy": min((l["holonomy"] for l in loops), default=0.0),
        }

    def _compute_loop_holonomy(self, node_ids: List[str]) -> float:
        """
        Compute holonomy for a closed loop of nodes.

        Holonomy measures the accumulated phase change around the loop.
        Large holonomy indicates topological tension/defect.

        Args:
            node_ids: List of node IDs forming the loop

        Returns:
            Holonomy value (0 = no tension, higher = more tension)
        """
        if len(node_ids) < 3:
            return 0.0

        positions = [self._torus_positions[nid] for nid in node_ids if nid in self._torus_positions]
        if len(positions) < 3:
            return 0.0

        # Compute angular differences around the loop
        total_angle = 0.0
        for i in range(len(positions)):
            pos1 = positions[i]
            pos2 = positions[(i + 1) % len(positions)]

            # Angular difference with torus wrapping
            diff = np.abs(pos1 - pos2)
            diff = np.where(diff > 180, 360 - diff, diff)
            total_angle += np.sqrt(np.sum(diff ** 2)) / 360.0 * 2 * np.pi

        # Holonomy is deviation from expected 2*pi for flat space
        holonomy = abs(total_angle - 2 * np.pi)
        return holonomy

    def _update_entropy_field(self):
        """Update entropy field from node states"""
        if not self.entropy_field:
            return

        # Reset field
        self.entropy_field.S.fill(0)

        N = self.entropy_field.N
        for node_id, node in self.nodes.items():
            pos = self._torus_positions.get(node_id, np.zeros(4))
            berry = self._berry.get(node_id, 0.0)

            # Grid position
            i = int(pos[0] / 360 * N) % N
            j = int(pos[1] / 360 * N) % N
            k = int(pos[2] / 360 * N) % N
            l = int(pos[3] / 360 * N) % N

            # Entropy contribution
            entropy = node.energy + node.access_count * 0.1 + berry * 0.5
            self.entropy_field.S[i, j, k, l] += entropy

        # Normalize
        max_val = np.max(self.entropy_field.S)
        if max_val > 0:
            self.entropy_field.S /= max_val

    def _build_bh_tree(self):
        """Build Barnes-Hut tree from node positions"""
        if not TUFT_AVAILABLE:
            return

        # Create pseudo-nodes for tree building
        class TempNode:
            def __init__(self, node_id, pos, berry, access_count):
                self.th1, self.th2, self.th3, self.th4 = pos
                self.berry_milli = int(berry * 1000)
                self.tau = 0
                self.windings = 0
                self.access_count = access_count
                self.used = True

        temp_nodes = []
        for node_id, node in self.nodes.items():
            pos = self._torus_positions.get(node_id, np.zeros(4))
            berry = self._berry.get(node_id, 1.0)
            temp_nodes.append(TempNode(node_id, pos, berry, node.access_count))

        # Build tree
        self.bh_tree = BarnesHutTree(theta=0.5)
        if self.entropy_field:
            self.bh_tree.build(temp_nodes, self.entropy_field.get_field(), self.grid_size)

    def step(self):
        """
        Override step to include TUFT dynamics.
        """
        # Call parent step (decay, process execution)
        super().step()

        # Add TUFT dynamics
        self.tuft_step()

        # Update Berry phases based on access and coherence
        for node_id, node in self.nodes.items():
            if node.access_count > 0:
                coh = self.compute_coherence(node_id)
                # Berry phase accumulates faster for coherent, frequently accessed nodes
                coherence_bonus = 1.0 + 0.5 * coh
                if node_id in self._berry:
                    self._berry[node_id] += 0.1 * node.energy * coherence_bonus

    def get_wilson_loop_stats(self) -> Dict[str, Any]:
        """Get Wilson loop topological statistics."""
        return self.compute_wilson_loops()

    def get_tuft_stats(self) -> Dict[str, Any]:
        """Get TUFT-specific statistics"""
        stats = {
            "tau": self.tau,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "hyperedges": len(self.hyperedges),
            "curvature": self.compute_curvature(),
            "bridges": len(self.find_bridges()),
            "avg_berry": np.mean(list(self._berry.values())) if self._berry else 0,
            "max_berry": max(self._berry.values()) if self._berry else 0,
        }

        if self.entropy_field:
            S = self.entropy_field.get_field()
            stats["entropy_max"] = float(np.max(S))
            stats["entropy_mean"] = float(np.mean(S))

        # Enhanced metrics
        if self._embeddings:
            stats["nodes_with_embeddings"] = len(self._embeddings)
            stats["embedding_coverage"] = len(self._embeddings) / len(self.nodes) if self.nodes else 0

        # Compute average coherence
        coherence_scores = [self.compute_coherence(nid) for nid in list(self.nodes.keys())[:10]]
        if coherence_scores:
            stats["avg_coherence"] = float(np.mean(coherence_scores))
            stats["coherence_std"] = float(np.std(coherence_scores))

        # Torus position statistics
        if self._torus_positions:
            positions = np.array(list(self._torus_positions.values()))
            stats["torus_spread"] = float(np.mean(np.std(positions, axis=0)))
            stats["avg_velocity"] = float(np.mean([np.linalg.norm(v) for v in self._velocities.values()]))

        return stats

    def get_semantic_clusters(self, distance_threshold: float = 60.0) -> List[List[str]]:
        """
        Find semantic clusters based on torus proximity.

        Uses a simple distance-based clustering algorithm.

        Args:
            distance_threshold: Maximum torus distance for cluster membership

        Returns:
            List of clusters, where each cluster is a list of node IDs
        """
        if not self._torus_positions:
            return []

        # Build adjacency based on torus distance
        node_ids = list(self._torus_positions.keys())
        clusters = []
        visited = set()

        for seed_id in node_ids:
            if seed_id in visited:
                continue

            # Start new cluster
            cluster = [seed_id]
            visited.add(seed_id)

            # Find all nodes within threshold distance
            for other_id in node_ids:
                if other_id in visited:
                    continue

                pos_seed = self._torus_positions[seed_id]
                pos_other = self._torus_positions[other_id]

                # Compute torus distance
                diff = np.abs(pos_seed - pos_other)
                diff = np.where(diff > 180, 360 - diff, diff)
                dist = np.sqrt(np.sum(diff ** 2))

                if dist <= distance_threshold:
                    cluster.append(other_id)
                    visited.add(other_id)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def compute_semantic_centroid(self, node_ids: List[str]) -> Optional[np.ndarray]:
        """
        Compute the semantic centroid (average torus position) for a set of nodes.

        Args:
            node_ids: List of node IDs

        Returns:
            Average torus position (4,) or None if no valid nodes
        """
        valid_positions = []
        for nid in node_ids:
            if nid in self._torus_positions:
                valid_positions.append(self._torus_positions[nid])

        if not valid_positions:
            return None

        return np.mean(valid_positions, axis=0)

    def to_prompt_with_tuft(self) -> str:
        """Generate LLM-readable summary including TUFT metrics"""
        base = self.to_prompt()

        tuft_stats = self.get_tuft_stats()

        tuft_section = f"""
TUFT METRICS:
- Curvature: {tuft_stats['curvature']:.2f}
- Bridge nodes: {tuft_stats['bridges']}
- Avg Berry phase: {tuft_stats['avg_berry']:.2f}
- Max Berry phase: {tuft_stats['max_berry']:.2f}
"""

        return base + tuft_section


# ============================================================================
# INTEGRATION WITH SELF-REFERENTIAL ENGINE
# ============================================================================

def compute_convergence_from_tuft(kernel: TUFTHypergraphKernel,
                                   thoughts: List[str]) -> float:
    """
    Use TUFT metrics to determine convergence.

    Returns a convergence score (0-1) based on:
    - Coherence of recent nodes
    - Curvature (lower = more stable)
    - Bridge formation (bridges indicate stable structure)
    """
    coherence_scores = []
    for thought_id in [n.id for n in kernel.query({"type": NodeType.THOUGHT})[-5:]]:
        coherence_scores.append(kernel.compute_coherence(thought_id))

    avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
    curvature = kernel.compute_curvature()
    bridges = len(kernel.find_bridges())

    # Normalize curvature (higher = more drift = less convergence)
    curvature_factor = max(0, 1.0 - curvature / 180.0)

    # Bridge factor (more bridges = more stable structure)
    bridge_factor = min(1.0, bridges / 5.0)

    # Combine
    convergence = (avg_coherence * 0.4 + curvature_factor * 0.3 + bridge_factor * 0.3)

    return convergence


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing TUFTHypergraphKernel...")

    kernel = TUFTHypergraphKernel(max_nodes=100)

    # Add some nodes
    for i in range(10):
        kernel.add_node(f"node_{i}", NodeType.THOUGHT, {"content": f"Thought {i}"})

    # Add some hyperedges
    kernel.add_hyperedge({"node_0", "node_1", "node_2"}, "conv_0")
    kernel.add_hyperedge({"node_2", "node_3", "node_4"}, "topic_1")
    kernel.add_hyperedge({"node_4", "node_5", "node_6"}, "entity_2")

    # Run steps
    for i in range(5):
        kernel.step()
        print(f"Step {i+1}: tau={kernel.tau}, curvature={kernel.compute_curvature():.2f}")

    # Get stats
    stats = kernel.get_tuft_stats()
    print("\nTUFT Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Find bridges
    bridges = kernel.find_bridges()
    print(f"\nBridges: {bridges}")

    print("\nTest complete!")