#!/usr/bin/env python3
"""
TOROIDAL OS — Dream Cycle / Pattern Mining
=============================================
Periodic offline processing that clusters solenoid history into
reusable patterns and consolidates memory.

Inspired by biological sleep cycles:
  - The system accumulates raw experiences during "waking"
  - Periodically (or when idle), it enters a "dream" phase
  - During dream: cluster solenoid histories, find recurring patterns,
    compress redundant memories, strengthen important connections,
    and generate insight-thoughts from pattern overlap

Architecture:
    DreamCycle operates on two data sources:
    1. Solenoid histories from the Topo9 kernel (sensor pattern sequences)
    2. Solenoid memory levels (text memories at various compression levels)

    Clustering:
    - State sequences are clustered by Hamming distance between solenoid
      histories.  Recurring state patterns become "archetypes" stored as
      HULL-tier belief nodes.
    - Text memories at level 0 (raw) are clustered by word overlap.
      Dense clusters are compressed and promoted to level 1+.

    Pattern Mining:
    - Cross-reference sensor archetypes with memory clusters:
      "When sensor pattern X occurs, the system tends to think about Y"
    - These cross-modal patterns become new beliefs or goal reinforcements
      in the desire field.

    Output:
    - Consolidated memory (fewer items, higher quality)
    - Archetype nodes in the hypergraph (HULL tier)
    - Optional insight-thoughts fed to the desire field
"""

import time
import threading
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class SensorArchetype:
    """A recurring sensor state pattern found in solenoid histories."""
    id: str
    pattern: List[int]           # The representative state sequence
    frequency: int               # How many times this pattern appeared
    avg_coherence: float         # Average coherence when this pattern occurs
    regions_touched: List[int]   # Which regions are active in this pattern
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class MemoryCluster:
    """A cluster of related text memories."""
    id: str
    centroid_words: List[str]    # Most common words in the cluster
    member_ids: List[str]        # IDs of memories in this cluster
    summary: str                 # Compressed summary
    importance: float            # Average importance of members
    created_at: float = field(default_factory=time.time)


@dataclass
class CrossPattern:
    """A cross-modal pattern linking sensor archetypes to memory clusters."""
    archetype_id: str
    cluster_id: str
    co_occurrence: int           # How often they co-occur temporally
    insight: str                 # Natural language insight


@dataclass
class DreamReport:
    """Summary of a dream cycle's findings."""
    cycle_number: int
    duration_sec: float
    archetypes_found: int
    clusters_formed: int
    cross_patterns: int
    memories_consolidated: int
    insights: List[str]
    timestamp: float = field(default_factory=time.time)


class DreamCycle:
    """
    Periodic pattern mining over solenoid histories and memory.

    Call dream() during idle periods or on a timer.  The cycle:
    1. Harvest solenoid histories from all known kernel nodes
    2. Cluster sensor state sequences by Hamming distance
    3. Cluster raw text memories by word overlap
    4. Cross-reference sensor archetypes with memory clusters
    5. Generate insights and consolidate memory
    """

    def __init__(
        self,
        graph=None,
        memory=None,
        kernel_bridge=None,
        desire_field=None,
        min_cluster_size: int = 3,
        hamming_threshold: int = 2,
        word_overlap_threshold: float = 0.3,
    ):
        self.graph = graph
        self.memory = memory
        self.bridge = kernel_bridge
        self.desire = desire_field
        self.min_cluster_size = min_cluster_size
        self.hamming_threshold = hamming_threshold
        self.word_overlap_threshold = word_overlap_threshold

        self.archetypes: Dict[str, SensorArchetype] = {}
        self.memory_clusters: Dict[str, MemoryCluster] = {}
        self.cross_patterns: List[CrossPattern] = []
        self.reports: List[DreamReport] = []
        self._cycle_count = 0
        self._lock = threading.Lock()

    # ========================================================================
    # SOLENOID HISTORY CLUSTERING
    # ========================================================================

    def _harvest_solenoid_histories(self) -> List[Tuple[int, List[int]]]:
        """
        Collect solenoid histories from all kernel nodes.
        Returns list of (node_id, state_sequence) tuples.
        """
        histories = []

        if not self.bridge:
            return histories

        kernel = self.bridge.kernel
        for i in range(kernel.node_count):
            node = kernel.nodes[i]
            if node.sol_len > 0:
                seq = node.sol_hist[:node.sol_len]
                histories.append((i, list(seq)))

        return histories

    def _hamming_distance_seq(self, a: List[int], b: List[int]) -> float:
        """
        Average per-element Hamming distance between two state sequences.
        Sequences may differ in length — compare the overlap.
        """
        min_len = min(len(a), len(b))
        if min_len == 0:
            return float('inf')

        total = 0
        for i in range(min_len):
            xor = a[i] ^ b[i]
            total += bin(xor).count('1')  # popcount

        return total / min_len

    def _cluster_histories(self, histories: List[Tuple[int, List[int]]]) -> List[SensorArchetype]:
        """
        Simple single-linkage clustering of solenoid histories.
        O(n²) but n is small (bounded by kernel node count).
        """
        if not histories:
            return []

        n = len(histories)
        labels = list(range(n))  # Each point starts in its own cluster

        # Merge clusters with Hamming distance below threshold
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._hamming_distance_seq(histories[i][1], histories[j][1])
                if dist <= self.hamming_threshold:
                    # Union: merge cluster j into cluster i
                    old_label = labels[j]
                    new_label = labels[i]
                    for k in range(n):
                        if labels[k] == old_label:
                            labels[k] = new_label

        # Group by label
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        # Build archetypes from clusters meeting minimum size
        archetypes = []
        for label, members in clusters.items():
            if len(members) < self.min_cluster_size:
                continue

            # Representative pattern: first member's history
            rep_seq = histories[members[0]][1]

            # Determine which regions are touched
            regions = set()
            for m_idx in members:
                for state in histories[m_idx][1]:
                    if state & 0x007:  # bits 0-2 (MOTION)
                        regions.add(0)
                    if state & 0x038:  # bits 3-5 (ENVIRON)
                        regions.add(1)
                    if state & 0x1C0:  # bits 6-8 (SEMANTIC)
                        regions.add(2)

            archetype = SensorArchetype(
                id=f"arch_{self._cycle_count}_{label}",
                pattern=rep_seq[:8],  # Cap sequence length
                frequency=len(members),
                avg_coherence=500.0,  # Will be refined if bridge available
                regions_touched=sorted(regions),
            )
            archetypes.append(archetype)

        return archetypes

    # ========================================================================
    # TEXT MEMORY CLUSTERING
    # ========================================================================

    def _harvest_raw_memories(self) -> List[Tuple[str, str, float]]:
        """Collect raw (level 0) memories. Returns (id, content, timestamp)."""
        if not self.memory:
            return []

        items = self.memory.levels[0].get_all()
        return [(item.id, item.content, item.timestamp) for item in items]

    def _word_set(self, text: str) -> set:
        """Extract lowercase word set from text."""
        return set(text.lower().split())

    def _word_overlap(self, a: set, b: set) -> float:
        """Jaccard similarity between word sets."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _cluster_memories(self, memories: List[Tuple[str, str, float]]) -> List[MemoryCluster]:
        """Cluster text memories by word overlap."""
        if not memories:
            return []

        n = len(memories)
        word_sets = [self._word_set(m[1]) for m in memories]
        labels = list(range(n))

        # Single-linkage clustering
        for i in range(n):
            for j in range(i + 1, n):
                overlap = self._word_overlap(word_sets[i], word_sets[j])
                if overlap >= self.word_overlap_threshold:
                    old_label = labels[j]
                    new_label = labels[i]
                    for k in range(n):
                        if labels[k] == old_label:
                            labels[k] = new_label

        # Group
        groups = defaultdict(list)
        for idx, label in enumerate(labels):
            groups[label].append(idx)

        clusters = []
        for label, members in groups.items():
            if len(members) < self.min_cluster_size:
                continue

            # Find common words
            all_words = defaultdict(int)
            for m_idx in members:
                for w in word_sets[m_idx]:
                    all_words[w] += 1

            # Top words (excluding stopwords)
            stopwords = {"the", "a", "an", "is", "was", "are", "to", "of", "in",
                         "and", "or", "for", "it", "i", "my", "me"}
            centroid = sorted(
                [(w, c) for w, c in all_words.items() if w not in stopwords],
                key=lambda x: -x[1]
            )[:8]

            # Simple summary: combine first words of each member
            summaries = [memories[m][1][:60] for m in members[:4]]
            summary = " | ".join(summaries)

            avg_importance = 1.0
            if self.memory:
                items = self.memory.levels[0].get_all()
                item_map = {item.id: item for item in items}
                importances = [item_map[memories[m][0]].importance
                               for m in members if memories[m][0] in item_map]
                if importances:
                    avg_importance = sum(importances) / len(importances)

            cluster = MemoryCluster(
                id=f"mclust_{self._cycle_count}_{label}",
                centroid_words=[w for w, _ in centroid],
                member_ids=[memories[m][0] for m in members],
                summary=summary,
                importance=avg_importance,
            )
            clusters.append(cluster)

        return clusters

    # ========================================================================
    # CROSS-MODAL PATTERN DISCOVERY
    # ========================================================================

    def _find_cross_patterns(
        self,
        archetypes: List[SensorArchetype],
        clusters: List[MemoryCluster],
        memories: List[Tuple[str, str, float]],
    ) -> List[CrossPattern]:
        """
        Find temporal co-occurrences between sensor archetypes and memory clusters.

        Basic approach: if a memory cluster's items were created during a time
        window when a particular sensor archetype was active, they co-occur.
        """
        if not archetypes or not clusters:
            return []

        patterns = []
        mem_id_to_time = {m[0]: m[2] for m in memories}

        for archetype in archetypes:
            for cluster in clusters:
                # Check temporal proximity of cluster members to archetype
                # (simplified: count members created within last cycle)
                co_count = 0
                for mid in cluster.member_ids:
                    t = mem_id_to_time.get(mid, 0)
                    if time.time() - t < self._cycle_count * 60 + 300:
                        co_count += 1

                if co_count >= 2:
                    regions = ", ".join(
                        ["MOTION", "ENVIRON", "SEMANTIC"][r]
                        for r in archetype.regions_touched
                    )
                    insight = (
                        f"When sensor regions [{regions}] are active "
                        f"(pattern freq={archetype.frequency}), "
                        f"thinking tends toward: {', '.join(cluster.centroid_words[:4])}"
                    )
                    patterns.append(CrossPattern(
                        archetype_id=archetype.id,
                        cluster_id=cluster.id,
                        co_occurrence=co_count,
                        insight=insight,
                    ))

        return patterns

    # ========================================================================
    # CONSOLIDATION
    # ========================================================================

    def _consolidate_memories(self, clusters: List[MemoryCluster]) -> int:
        """
        Consolidate clustered memories: compress cluster members into
        a single summary at a higher memory level.
        Returns number of memories consolidated.
        """
        if not self.memory or not clusters:
            return 0

        count = 0
        for cluster in clusters:
            # Promote cluster summary to level 1 (summary level)
            self.memory.wind(
                content=f"[Dream consolidated] {cluster.summary}",
                importance=cluster.importance * 1.1,
                level=min(1, self.memory.num_levels - 1),
            )
            count += len(cluster.member_ids)

        return count

    def _store_archetypes(self, archetypes: List[SensorArchetype]):
        """Store discovered archetypes in the hypergraph as HULL beliefs."""
        if not self.graph:
            return

        from kernel.hypergraph import NodeType, TrustTier

        for arch in archetypes:
            regions = ", ".join(
                ["MOTION", "ENVIRON", "SEMANTIC"][r]
                for r in arch.regions_touched
            )
            self.graph.add_node(
                arch.id,
                NodeType.BELIEF,
                {
                    "content": f"Sensor archetype: regions [{regions}], "
                               f"frequency {arch.frequency}, "
                               f"coherence {arch.avg_coherence:.0f}",
                    "pattern": arch.pattern[:8],
                    "frequency": arch.frequency,
                    "dream_cycle": self._cycle_count,
                },
                trust=TrustTier.HULL,
            )

        self.archetypes.update({a.id: a for a in archetypes})

    def _feed_insights_to_desire(self, patterns: List[CrossPattern]):
        """Feed cross-modal insights to the desire field as goal reinforcements."""
        if not self.desire or not patterns:
            return

        for pattern in patterns:
            # Curiosity pressure from novel cross-modal patterns
            self.desire.reinforce_goal("goal_curiosity", 0.05 * pattern.co_occurrence)

            # If pattern involves all 3 regions, reinforce coherence goal
            arch = self.archetypes.get(pattern.archetype_id)
            if arch and len(arch.regions_touched) == 3:
                self.desire.reinforce_goal("goal_coherence", 0.03)

    # ========================================================================
    # MAIN DREAM CYCLE
    # ========================================================================

    def dream(self) -> DreamReport:
        """
        Run one full dream cycle.

        Should be called during idle periods or on a timer.
        Typical frequency: every 5-15 minutes of waking operation.
        """
        with self._lock:
            start = time.time()
            self._cycle_count += 1

            # Phase 1: Harvest
            histories = self._harvest_solenoid_histories()
            raw_memories = self._harvest_raw_memories()

            # Phase 2: Cluster solenoid histories
            archetypes = self._cluster_histories(histories)

            # Phase 3: Cluster text memories
            mem_clusters = self._cluster_memories(raw_memories)

            # Phase 4: Cross-modal pattern discovery
            cross = self._find_cross_patterns(archetypes, mem_clusters, raw_memories)

            # Phase 5: Store and consolidate
            self._store_archetypes(archetypes)
            n_consolidated = self._consolidate_memories(mem_clusters)

            # Phase 6: Feed insights to desire field
            self._feed_insights_to_desire(cross)

            # Store clusters for later reference
            self.memory_clusters.update({c.id: c for c in mem_clusters})
            self.cross_patterns.extend(cross)

            # Build report
            insights = [p.insight for p in cross]
            report = DreamReport(
                cycle_number=self._cycle_count,
                duration_sec=time.time() - start,
                archetypes_found=len(archetypes),
                clusters_formed=len(mem_clusters),
                cross_patterns=len(cross),
                memories_consolidated=n_consolidated,
                insights=insights,
            )
            self.reports.append(report)

            # Record dream in memory
            if self.memory:
                self.memory.wind(
                    f"[Dream #{self._cycle_count}] Found {len(archetypes)} sensor archetypes, "
                    f"{len(mem_clusters)} memory clusters, {len(cross)} cross-patterns. "
                    f"Consolidated {n_consolidated} memories.",
                    importance=1.5,
                    level=min(1, self.memory.num_levels - 1),
                )

            return report

    def should_dream(self, idle_seconds: float = 60.0) -> bool:
        """Heuristic: should we enter a dream cycle?"""
        if not self.reports:
            return True  # Never dreamed yet

        last = self.reports[-1]
        elapsed = time.time() - last.timestamp

        # Dream more often if lots of raw memories are accumulating
        raw_pressure = 0
        if self.memory:
            raw_count = len(self.memory.levels[0].items)
            raw_max = self.memory.levels[0].max_items
            raw_pressure = raw_count / raw_max

        # Base interval: 5 minutes, reduced by memory pressure
        interval = max(60.0, 300.0 * (1.0 - raw_pressure))
        return elapsed >= interval

    def get_dream_summary(self) -> Dict[str, Any]:
        """Summary for status display."""
        last = self.reports[-1] if self.reports else None
        return {
            "total_dreams": self._cycle_count,
            "archetypes": len(self.archetypes),
            "memory_clusters": len(self.memory_clusters),
            "cross_patterns": len(self.cross_patterns),
            "last_dream": {
                "cycle": last.cycle_number if last else 0,
                "ago_sec": round(time.time() - last.timestamp, 1) if last else None,
                "insights": last.insights[:3] if last else [],
            },
        }
