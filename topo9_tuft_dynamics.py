#!/usr/bin/env python3
"""
Topo9 TUFT Dynamics with Barnes-Hut
===================================

Extends the Topo9 kernel with:
1. Continuous entropy field S(x) from Berry phase
2. Phase field θ(x) from node positions
3. Barnes-Hut tree for O(N log N) force computation
4. TUFT emergent gravity: F ~ β_S (∇S · ∇θ)

This integrates TUFT physics into the discrete Topo9 memory system.

Usage:
    python topo9_tuft_dynamics.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time

# Import the base kernel
from simulate import Kernel, Node, NMAX, SOL_H, HEDGE_CONV, HEDGE_TOPIC, HEDGE_ENTITY, HEDGE_FIBER, HEDGE_CUSTOM


# ═══════════════════════════════════════════════════════════════════════════
# BARNES-HUT TREE FOR 4D TORUS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BHNode:
    """Single node in the Barnes-Hut tree."""

    # Spatial extent
    center: np.ndarray          # Center position (th1, th2, th3, th4) in degrees
    size: float                 # Angular extent of this region

    # Aggregate quantities (like "mass")
    total_entropy: float = 0.0
    berry_sum: float = 0.0
    node_count: int = 0

    # Gradients (direction of information flow)
    entropy_gradient: np.ndarray = field(default_factory=lambda: np.zeros(4))
    phase_gradient: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Center of mass (Berry-weighted position)
    com: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Tree structure
    children: List[Optional['BHNode']] = field(default_factory=list)
    is_leaf: bool = True
    leaf_node_ids: List[int] = field(default_factory=list)


class BarnesHutTree:
    """
    Barnes-Hut tree for efficient O(N log N) force computation.

    Works on a 4D torus (θ₁, θ₂, θ₃, θ₄) each in [0, 360°).
    Uses hierarchical subdivision with 16 children per node (2^4).
    """

    def __init__(self, theta: float = 0.5):
        """
        Initialize tree.

        Args:
            theta: Opening angle criterion (smaller = more accurate, slower)
        """
        self.theta = theta
        self.root = None
        self.nodes_data = {}  # id -> (position, berry, entropy)

    def torus_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance on 4D torus with periodic boundaries.

        Each dimension wraps at 360°.
        """
        diff = np.abs(a - b)
        # Wrap around: if diff > 180, use 360 - diff
        diff = np.where(diff > 180, 360 - diff, diff)
        return np.sqrt(np.sum(diff ** 2))

    def torus_wrap(self, theta: np.ndarray) -> np.ndarray:
        """Wrap angles to [0, 360)."""
        return theta % 360

    def get_octant(self, position: np.ndarray, center: np.ndarray) -> int:
        """
        Determine which octant (child) a position falls into.

        Returns index 0-15 for the 16 children (2^4 dimensions).
        """
        index = 0
        for i in range(4):
            # Compare position to center in each dimension
            # Handle torus wrap-around
            diff = (position[i] - center[i]) % 360
            if diff > 180:
                diff -= 360
            if diff >= 0:
                index |= (1 << i)
        return index

    def get_child_center(self, parent_center: np.ndarray, octant: int,
                         parent_size: float) -> np.ndarray:
        """Get center position for a child octant."""
        child_size = parent_size / 2
        offset = child_size / 2

        center = parent_center.copy()
        for i in range(4):
            if octant & (1 << i):
                center[i] = (center[i] + offset) % 360
            else:
                center[i] = (center[i] - offset) % 360

        return center

    def build(self, topo_nodes: List[Node], entropy_field: np.ndarray,
              grid_size: int = 32):
        """
        Build Barnes-Hut tree from Topo9 nodes.

        Args:
            topo_nodes: List of Topo9 Node objects
            entropy_field: 4D entropy field from which to extract gradients
            grid_size: Grid size for gradient computation
        """
        # Collect active nodes
        active_nodes = [(i, n) for i, n in enumerate(topo_nodes) if n.used]

        if not active_nodes:
            self.root = None
            return

        # Initialize root node
        self.root = BHNode(
            center=np.array([180.0, 180.0, 180.0, 180.0]),
            size=360.0
        )
        self.root.children = [None] * 16

        # Compute entropy gradients from field
        entropy_gradients = self._compute_entropy_gradients(entropy_field, grid_size)
        phase_gradients = self._compute_phase_gradients(topo_nodes)

        # Insert each node
        for node_id, node in active_nodes:
            position = np.array([node.th1, node.th2, node.th3, node.th4])
            berry = node.berry_milli

            # Entropy from Berry phase (normalized)
            entropy = node.berry_milli / 1000.0 + node.access_count / 100.0

            # Get gradient at this position
            ent_grad = self._interpolate_gradient(entropy_gradients, position, grid_size)
            phase_grad = self._interpolate_gradient(phase_gradients, position, grid_size)

            self._insert(self.root, node_id, position, berry, entropy,
                        ent_grad, phase_grad, depth=0)

    def _compute_entropy_gradients(self, S: np.ndarray, N: int) -> np.ndarray:
        """
        Compute ∇S (entropy gradient) on the 4D grid.

        Uses central differences with periodic boundaries.
        """
        gradients = np.zeros((N, N, N, N, 4))
        dx = 360.0 / N

        for i in range(4):
            # Roll along axis i
            rolled_plus = np.roll(S, -1, axis=i)
            rolled_minus = np.roll(S, 1, axis=i)
            gradients[..., i] = (rolled_plus - rolled_minus) / (2 * dx)

        return gradients

    def _compute_phase_gradients(self, nodes: List[Node]) -> np.ndarray:
        """
        Compute phase gradient field from node positions.

        The "phase" here is derived from the angular positions th1-th4.
        """
        # Simple approximation: gradient points toward high-berry nodes
        N = 16
        gradients = np.zeros((N, N, N, N, 4))

        for node in nodes:
            if not node.used:
                continue

            # Grid position
            gi = int(node.th1 / 360 * N) % N
            gj = int(node.th2 / 360 * N) % N
            gk = int(node.th3 / 360 * N) % N
            gl = int(node.th4 / 360 * N) % N

            # Weight by Berry phase
            weight = node.berry_milli / 1000.0

            # Gradient points toward this node's position
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    for dk in range(-2, 3):
                        for dl in range(-2, 3):
                            ii = (gi + di) % N
                            jj = (gj + dj) % N
                            kk = (gk + dk) % N
                            ll = (gl + dl) % N

                            # Direction vector
                            gradients[ii, jj, kk, ll, 0] += di * weight
                            gradients[ii, jj, kk, ll, 1] += dj * weight
                            gradients[ii, jj, kk, ll, 2] += dk * weight
                            gradients[ii, jj, kk, ll, 3] += dl * weight

        return gradients

    def _interpolate_gradient(self, gradients: np.ndarray,
                              position: np.ndarray, N: int) -> np.ndarray:
        """Interpolate gradient at continuous position."""
        # Nearest neighbor for simplicity
        gi = int(position[0] / 360 * N) % N
        gj = int(position[1] / 360 * N) % N
        gk = int(position[2] / 360 * N) % N
        gl = int(position[3] / 360 * N) % N

        return gradients[gi, gj, gk, gl]

    def _insert(self, bh_node: BHNode, node_id: int, position: np.ndarray,
                berry: float, entropy: float,
                ent_grad: np.ndarray, phase_grad: np.ndarray,
                depth: int, max_depth: int = 8):
        """Insert a node into the Barnes-Hut tree."""

        # Update aggregate quantities
        old_count = bh_node.node_count
        new_count = old_count + 1

        # Update center of mass (Berry-weighted)
        if old_count > 0 and (bh_node.berry_sum + berry) > 0:
            bh_node.com = (bh_node.com * bh_node.berry_sum + position * berry) / (bh_node.berry_sum + berry)
        else:
            bh_node.com = position.copy()

        bh_node.berry_sum += berry
        bh_node.total_entropy += entropy
        bh_node.node_count = new_count

        # Update gradients (average)
        bh_node.entropy_gradient = (bh_node.entropy_gradient * old_count + ent_grad) / new_count
        bh_node.phase_gradient = (bh_node.phase_gradient * old_count + phase_grad) / new_count

        # Check if we need to subdivide
        if bh_node.is_leaf and len(bh_node.leaf_node_ids) >= 1 and depth < max_depth:
            # Subdivide
            bh_node.is_leaf = False
            bh_node.children = [None] * 16

            # Re-insert existing leaf nodes
            for existing_id in bh_node.leaf_node_ids:
                ex_pos, ex_berry, ex_entropy, ex_ent_grad, ex_phase_grad = self.nodes_data[existing_id]
                self._insert_into_child(bh_node, existing_id, ex_pos, ex_berry,
                                       ex_entropy, ex_ent_grad, ex_phase_grad, depth + 1, max_depth)

            bh_node.leaf_node_ids = []

        # Insert new node
        if bh_node.is_leaf:
            bh_node.leaf_node_ids.append(node_id)
            self.nodes_data[node_id] = (position, berry, entropy, ent_grad, phase_grad)
        else:
            self._insert_into_child(bh_node, node_id, position, berry,
                                   entropy, ent_grad, phase_grad, depth + 1, max_depth)

    def _insert_into_child(self, bh_node: BHNode, node_id: int,
                          position: np.ndarray, berry: float, entropy: float,
                          ent_grad: np.ndarray, phase_grad: np.ndarray,
                          depth: int, max_depth: int):
        """Insert node into appropriate child."""
        octant = self.get_octant(position, bh_node.center)
        child_size = bh_node.size / 2

        if bh_node.children[octant] is None:
            child_center = self.get_child_center(bh_node.center, octant, bh_node.size)
            bh_node.children[octant] = BHNode(
                center=child_center,
                size=child_size
            )

        self._insert(bh_node.children[octant], node_id, position, berry,
                    entropy, ent_grad, phase_grad, depth, max_depth)

    def compute_force(self, target_node: Node, beta_S: float = 0.2,
                      k_coup: float = 0.5) -> np.ndarray:
        """
        Compute TUFT-style force on target node using Barnes-Hut.

        The TUFT force is:
            F ~ β_S (∇S · ∇θ) + k_coup * diffusion

        Where:
            - ∇S is the entropy gradient (information flow direction)
            - ∇θ is the phase gradient (structure alignment)
            - β_S is the coupling strength
            - k_coup controls diffusion toward high-entropy regions

        Returns:
            Force vector (4D) in angular velocity units
        """
        if self.root is None:
            return np.zeros(4)

        target_pos = np.array([target_node.th1, target_node.th2,
                              target_node.th3, target_node.th4])

        # Compute phase gradient from node's own dynamics
        # Higher tau = more recent activity = stronger "phase"
        # Windings = orbit completions = accumulated topology
        phase_strength = 0.1 + target_node.tau / 255.0 + target_node.windings * 0.1
        target_phase_grad = np.array([
            np.sin(np.radians(target_node.th1 * 2)) * phase_strength,
            np.cos(np.radians(target_node.th2 * 2)) * phase_strength,
            np.sin(np.radians(target_node.th3 * 3)) * phase_strength,
            np.sin(np.radians(target_node.th4 * 5)) * phase_strength,  # th4 is special
        ])
        target_phase_grad = np.array([
            (target_node.tau * 17) % 360 / 180.0 - 1,  # Normalized phase
            (target_node.windings * 29) % 360 / 180.0 - 1,
            (target_node.berry_milli * 0.01) % 360 / 180.0 - 1,
            target_node.th4 / 180.0 - 1
        ])

        return self._compute_force_recursive(self.root, target_pos,
                                             target_phase_grad, beta_S, k_coup)

    def _compute_force_recursive(self, bh_node: BHNode, target_pos: np.ndarray,
                                 target_phase_grad: np.ndarray,
                                 beta_S: float, k_coup: float) -> np.ndarray:
        """Recursively compute force from tree node."""

        if bh_node.node_count == 0:
            return np.zeros(4)

        d = self.torus_distance(bh_node.com, target_pos)

        # Barnes-Hut criterion
        if bh_node.is_leaf or (bh_node.size / max(d, 1e-6) < self.theta):
            # Use aggregate
            return self._compute_force_from_aggregate(bh_node, target_pos,
                                                      target_phase_grad, beta_S, k_coup, d)
        else:
            # Recurse into children
            total_force = np.zeros(4)
            for child in bh_node.children:
                if child is not None:
                    force = self._compute_force_recursive(child, target_pos,
                                                         target_phase_grad, beta_S, k_coup)
                    total_force += force
            return total_force

    def _compute_force_from_aggregate(self, bh_node: BHNode, target_pos: np.ndarray,
                                      target_phase_grad: np.ndarray,
                                      beta_S: float, k_coup: float,
                                      distance: float) -> np.ndarray:
        """
        Compute force from an aggregated tree node.

        TUFT formula: F = β_S * (∇S · ∇θ) * berry_weight / d²

        Also includes semantic attraction based on Berry phase similarity.
        """
        if distance < 1e-6 or bh_node.node_count == 0:
            return np.zeros(4)

        # Direction toward center of mass (with torus wrap)
        direction = bh_node.com - target_pos
        for i in range(4):
            if direction[i] > 180:
                direction[i] -= 360
            elif direction[i] < -180:
                direction[i] += 360

        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        else:
            direction = np.zeros(4)

        # 1. TUFT entropy-gradient coupling (emergent gravity)
        coupling = np.dot(bh_node.entropy_gradient, target_phase_grad)

        # 2. Semantic attraction (Berry phase as "mass")
        # Similar to gravitational attraction but with Berry as mass
        # F_grav = G * m1 * m2 / r²
        berry_attraction = bh_node.berry_sum / (distance ** 2 + 10)

        # 3. Entropy diffusion (spreading toward high-entropy regions)
        entropy_diffusion = bh_node.total_entropy / (distance + 1)

        # Combined force magnitude
        # TUFT coupling can be positive or negative (attract/repel based on phase alignment)
        tuft_force = beta_S * coupling * bh_node.berry_sum / (distance + 1)

        # Total force
        force_mag = tuft_force + k_coup * (berry_attraction + entropy_diffusion * 0.1)

        # Apply force in direction of center of mass
        force = force_mag * direction

        return force

        # Force vector
        force = force_mag * direction

        return force


# ═══════════════════════════════════════════════════════════════════════════
# ENTROPY FIELD
# ═══════════════════════════════════════════════════════════════════════════

class EntropyField:
    """
    4D entropy field derived from node Berry phases and access patterns.

    The entropy field S(x) represents information density in the semantic space.
    High entropy = high information content = strong phase steering.
    """

    def __init__(self, grid_size: int = 16):
        self.N = grid_size
        self.S = np.zeros((grid_size, grid_size, grid_size, grid_size))
        self.dx = 360.0 / grid_size

    def update_from_nodes(self, nodes: List[Node]):
        """Update entropy field from node states."""
        self.S = np.zeros((self.N, self.N, self.N, self.N))

        for node in nodes:
            if not node.used:
                continue

            # Grid position from torus angles
            i = int(node.th1 / 360 * self.N) % self.N
            j = int(node.th2 / 360 * self.N) % self.N
            k = int(node.th3 / 360 * self.N) % self.N
            l = int(node.th4 / 360 * self.N) % self.N

            # Entropy contribution
            # Berry phase = cumulative access weight
            # Access count = recency/activity
            # Windings = orbit completions (stability)
            entropy_contrib = (
                node.berry_milli / 1000.0 +
                node.access_count / 50.0 +
                node.windings * 0.5 +
                (1 if node.region_mask & 0b111 > 1 else 0) * 2.0  # Bridge bonus
            )

            # Add to field with Gaussian spreading
            self._add_gaussian(i, j, k, l, entropy_contrib, sigma=1.5)

        # Normalize
        max_val = np.max(self.S)
        if max_val > 0:
            self.S /= max_val

    def _add_gaussian(self, i: int, j: int, k: int, l: int,
                      value: float, sigma: float = 1.0):
        """Add Gaussian contribution centered at (i,j,k,l)."""
        for di in range(-3, 4):
            for dj in range(-3, 4):
                for dk in range(-3, 4):
                    for dl in range(-3, 4):
                        r2 = di**2 + dj**2 + dk**2 + dl**2
                        weight = np.exp(-r2 / (2 * sigma**2))

                        ii = (i + di) % self.N
                        jj = (j + dj) % self.N
                        kk = (k + dk) % self.N
                        ll = (l + dl) % self.N

                        self.S[ii, jj, kk, ll] += value * weight

    def get_field(self) -> np.ndarray:
        """Return the entropy field."""
        return self.S

    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        """Get entropy gradient at a position."""
        # Central difference with periodic BC
        i = int(position[0] / 360 * self.N) % self.N
        j = int(position[1] / 360 * self.N) % self.N
        k = int(position[2] / 360 * self.N) % self.N
        l = int(position[3] / 360 * self.N) % self.N

        grad = np.zeros(4)

        # dx/dth1
        grad[0] = (self.S[(i+1)%self.N, j, k, l] -
                   self.S[(i-1)%self.N, j, k, l]) / (2 * self.dx)
        # dx/dth2
        grad[1] = (self.S[i, (j+1)%self.N, k, l] -
                   self.S[i, (j-1)%self.N, k, l]) / (2 * self.dx)
        # dx/dth3
        grad[2] = (self.S[i, j, (k+1)%self.N, l] -
                   self.S[i, j, (k-1)%self.N, l]) / (2 * self.dx)
        # dx/dth4
        grad[3] = (self.S[i, j, k, (l+1)%self.N] -
                   self.S[i, j, k, (l-1)%self.N]) / (2 * self.dx)

        return grad


# ═══════════════════════════════════════════════════════════════════════════
# TUFT DYNAMICS KERNEL
# ═══════════════════════════════════════════════════════════════════════════

class Topo9TUFT(Kernel):
    """
    Extended Topo9 kernel with TUFT physics and Barnes-Hut dynamics.

    Adds:
    - Continuous entropy field S(x)
    - Barnes-Hut tree for O(N log N) force computation
    - TUFT emergent gravity during EVOLVE/TICK
    - Wilson loop Berry phase computation
    """

    def __init__(self, grid_size: int = 16, bh_theta: float = 0.5):
        super().__init__()

        self.grid_size = grid_size
        self.bh_theta = bh_theta

        # New components
        self.entropy_field = EntropyField(grid_size)
        self.bh_tree = BarnesHutTree(theta=bh_theta)

        # Velocity for each node (angular velocity on torus)
        self.velocities = {}  # node_id -> (v1, v2, v3, v4)

        # TUFT parameters
        self.beta_S = 0.5       # Entropy-gradient coupling (increased)
        self.k_coup = 2.0       # Diffusion coupling (increased for visible motion)
        self.damping = 0.95     # Velocity damping
        self.dt = 1.0           # Time step
        self.force_scale = 5.0  # Scale factor for visible motion

        # Knot density field (Berry phase from Wilson loops)
        self.knot_density = None

    def store_node(self, st: int, region: int) -> int:
        """Store a node and initialize its velocity."""
        nid = super().store_node(st, region)
        if nid >= 0:
            self.velocities[nid] = np.zeros(4)
        return nid

    def update_entropy_field(self):
        """Update the entropy field from current node states."""
        self.entropy_field.update_from_nodes(self.nodes)

    def build_barnes_hut_tree(self):
        """Build Barnes-Hut tree for force computation."""
        self.bh_tree.build(self.nodes, self.entropy_field.get_field(),
                          self.grid_size)

    def compute_wilson_loops(self):
        """
        Compute Berry phase knot density from Wilson loops.

        This is the true TUFT computation of topological charge.
        """
        N = self.grid_size
        self.knot_density = np.zeros((N, N, N, N))

        # Build a phase field from node positions
        phase_field = np.zeros((N, N, N, N))
        for node in self.nodes:
            if not node.used:
                continue

            i = int(node.th1 / 360 * N) % N
            j = int(node.th2 / 360 * N) % N
            k = int(node.th3 / 360 * N) % N
            l = int(node.th4 / 360 * N) % N

            # Phase from tau
            phase_field[i, j, k, l] += node.tau * 2 * np.pi / 256

        # Compute Wilson loops (circulation around plaquettes)
        # For each plaquette, compute the phase circulation
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        # Circulation in each of 4 planes
                        circ = 0

                        # Plane 1-2 (th1-th2)
                        circ += phase_field[(i+1)%N, j, k, l] - phase_field[i, j, k, l]
                        circ += phase_field[(i+1)%N, (j+1)%N, k, l] - phase_field[(i+1)%N, j, k, l]
                        circ += phase_field[i, (j+1)%N, k, l] - phase_field[(i+1)%N, (j+1)%N, k, l]
                        circ += phase_field[i, j, k, l] - phase_field[i, (j+1)%N, k, l]

                        # Similar for other planes (simplified)
                        circ += phase_field[i, (j+1)%N, k, l] - phase_field[i, j, k, l]
                        circ += phase_field[i, (j+1)%N, (k+1)%N, l] - phase_field[i, (j+1)%N, k, l]

                        # Knot density = circulation / 2π
                        self.knot_density[i, j, k, l] = circ / (2 * np.pi)

        return self.knot_density

    def tuft_evolve_step(self, steps: int = 1):
        """
        One step of TUFT-style evolution using Barnes-Hut.

        This replaces/approximates the O(N²) hyperedge sync with O(N log N).
        """
        # Update entropy field
        self.update_entropy_field()

        # Build Barnes-Hut tree
        self.build_barnes_hut_tree()

        for _ in range(steps):
            # Also run the original discrete dynamics
            self.evolve_steps(1)

            # Apply TUFT forces to each node
            for i, node in enumerate(self.nodes):
                if not node.used:
                    continue

                # Compute force from Barnes-Hut tree
                force = self.bh_tree.compute_force(node, self.beta_S, self.k_coup)

                # Scale force for visible motion
                force = force * self.force_scale

                # Update velocity (inertia scaled by inverse "mass" ~ 1/berry)
                vel = self.velocities.get(i, np.zeros(4))
                mass = max(node.berry_milli / 100.0, 0.1)  # Normalize berry to reasonable mass
                vel = self.damping * vel + force * self.dt / mass
                self.velocities[i] = vel

                # Update position (toroidal)
                node.th1 = (node.th1 + vel[0] * self.dt) % 360
                node.th2 = (node.th2 + vel[1] * self.dt) % 360
                node.th3 = (node.th3 + vel[2] * self.dt) % 360

                old_th4 = node.th4
                node.th4 = (node.th4 + vel[3] * self.dt) % 360

                # Track windings (orbit completions)
                if node.th4 < old_th4 and vel[3] > 0:
                    node.windings += 1

        # Compute knot density
        self.compute_wilson_loops()

    def tuft_tick(self, n: int = 1):
        """
        TUFT-style autonomous tick with Barnes-Hut forces.
        """
        # Update entropy field
        self.update_entropy_field()

        # Build tree
        self.build_barnes_hut_tree()

        flip_bit = 0
        active_region = 0
        min_bridge_berry = 1200

        results = []

        for t in range(n):
            # Curvature from context
            curv = self.curvature_scaled()
            multi_region_mode = (t % 2) == 1

            # Find best node to activate
            best = 0
            best_s = 0
            for i in range(self.node_count):
                if not self.nodes[i].used:
                    continue
                s = self.scheduler_score(i, curv, multi_region_mode, min_bridge_berry)

                # Bonus for high knot density (topological charge)
                if self.knot_density is not None:
                    ki = int(self.nodes[i].th1 / 360 * self.grid_size) % self.grid_size
                    kj = int(self.nodes[i].th2 / 360 * self.grid_size) % self.grid_size
                    kk = int(self.nodes[i].th3 / 360 * self.grid_size) % self.grid_size
                    kl = int(self.nodes[i].th4 / 360 * self.grid_size) % self.grid_size
                    s += abs(self.knot_density[ki, kj, kk, kl]) * 100

                if s >= best_s:
                    best_s = s
                    best = i

            # Perturb state
            self.nodes[best].state ^= (1 << (flip_bit % 9))
            flip_region = self.region_of_bit(flip_bit % 9)
            flip_bit += 1

            # Determine access region
            access_region = active_region if flip_region == active_region else (active_region + 1) % 3

            # Record access (original dynamics)
            self.record_access(best, access_region)
            self.update_edges_on_access(best)
            self.ctx_push(best)

            # Apply TUFT force to this node
            force = self.bh_tree.compute_force(self.nodes[best], self.beta_S, self.k_coup)
            vel = self.velocities.get(best, np.zeros(4))
            vel = self.damping * vel + force * 2.0 / max(self.nodes[best].berry_milli, 1)
            self.velocities[best] = vel

            old_th4 = self.nodes[best].th4
            self.nodes[best].th4 = (self.nodes[best].th4 + vel[3] * self.dt) % 360

            if self.nodes[best].th4 < old_th4 and vel[3] > 0:
                self.nodes[best].windings += 1

            results.append({
                "t": t,
                "id": best,
                "state": self.nodes[best].state,
                "tau": self.nodes[best].tau,
                "windings": self.nodes[best].windings,
                "berry": self.nodes[best].berry_milli,
                "th4": self.nodes[best].th4,
                "coh": self.coherence_score(best),
                "curv": curv,
                "velocity": self.velocities[best].tolist(),
            })

        return results

    def get_tuft_stats(self) -> dict:
        """Get statistics including TUFT-specific metrics."""
        base_stats = {
            "nodes": self.node_count,
            "hedges": sum(1 for e in range(self.hedge_count) if self.hedges[e].used),
            "curvature": self.curvature_scaled(),
            "context_len": self.ctx_len,
        }

        # TUFT-specific
        tuft_stats = {
            "entropy_field_max": float(np.max(self.entropy_field.get_field())),
            "entropy_field_mean": float(np.mean(self.entropy_field.get_field())),
            "bh_tree_depth": getattr(self.bh_tree.root, 'size', 0) if self.bh_tree.root else 0,
            "active_velocities": len([v for v in self.velocities.values() if np.linalg.norm(v) > 0.01]),
        }

        if self.knot_density is not None:
            tuft_stats["knot_density_max"] = float(np.max(np.abs(self.knot_density)))
            tuft_stats["knot_density_rms"] = float(np.sqrt(np.mean(self.knot_density**2)))

        # Bridge analysis
        bridges = [i for i in range(self.node_count)
                   if self.nodes[i].used and self.is_bridge(i, 100)]
        tuft_stats["bridge_count"] = len(bridges)

        return {**base_stats, **tuft_stats}


# ═══════════════════════════════════════════════════════════════════════════
# TEST / DEMO
# ═══════════════════════════════════════════════════════════════════════════

def demo_tuft_dynamics():
    """Demonstrate TUFT dynamics with Barnes-Hut."""
    print("=" * 60)
    print("Topo9 TUFT Dynamics with Barnes-Hut")
    print("=" * 60)

    # Initialize
    kernel = Topo9TUFT(grid_size=16, bh_theta=0.5)
    kernel.init_seed_nodes()

    print(f"\nInitialized with {kernel.node_count} seed nodes")
    print(f"Entropy field: {kernel.grid_size}^4 grid")
    print(f"Barnes-Hut theta: {kernel.bh_theta}")

    # Store some additional nodes with cross-region access
    print("\n--- Storing nodes and building entropy ---")
    for i in range(10):
        state = (0b101010101 ^ (i * 17)) & 0x1FF
        region = i % 3
        nid = kernel.store_node(state, region)
        print(f"  Node {nid}: state={bin(state)[2:].zfill(9)}, region={region}")

    # Simulate cross-region access patterns
    print("\n--- Simulating cross-region access patterns ---")
    for region in [0, 1, 2, 0, 2, 1, 2, 0, 1, 2]:
        # Find a node and access it
        for nid in range(kernel.node_count):
            if kernel.nodes[nid].used:
                kernel.record_access(nid, region)
                kernel.update_edges_on_access(nid)
                kernel.ctx_push(nid)
                break

    # Update entropy field
    print("\n--- Updating entropy field ---")
    kernel.update_entropy_field()
    S = kernel.entropy_field.get_field()
    print(f"  Entropy field: max={np.max(S):.3f}, mean={np.mean(S):.3f}")

    # Build Barnes-Hut tree
    print("\n--- Building Barnes-Hut tree ---")
    kernel.build_barnes_hut_tree()
    if kernel.bh_tree.root:
        print(f"  Tree root: berry_sum={kernel.bh_tree.root.berry_sum:.1f}")
        print(f"  Tree root: entropy={kernel.bh_tree.root.total_entropy:.3f}")

    # Run TUFT evolution
    print("\n--- Running TUFT evolution (10 steps) ---")
    start_time = time.time()
    kernel.tuft_evolve_step(10)
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.3f}s")

    # Get stats
    stats = kernel.get_tuft_stats()
    print("\n--- TUFT Statistics ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Run TUFT tick
    print("\n--- Running TUFT tick (5 steps) ---")
    results = kernel.tuft_tick(5)

    print("\n  Tick results:")
    for r in results[:3]:
        print(f"    t={r['t']}: id={r['id']}, th4={r['th4']:.1f}°, "
              f"vel={np.linalg.norm(r['velocity']):.3f}")

    # Check for bridges
    print("\n--- Bridge nodes ---")
    bridges = [i for i in range(kernel.node_count)
               if kernel.nodes[i].used and kernel.is_bridge(i, 100)]
    for bid in bridges[:5]:
        n = kernel.nodes[bid]
        print(f"  Node {bid}: Berry={n.berry_milli}, regions={bin(n.region_mask)}, "
              f"th4={n.th4:.1f}°, windings={n.windings}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return kernel


if __name__ == "__main__":
    kernel = demo_tuft_dynamics()