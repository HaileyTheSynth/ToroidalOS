#!/usr/bin/env python3
"""
TOROIDAL OS — Desire Field Proxy
===================================
A "goal node" in the kernel whose Berry phase accumulation the system
tracks. When desire pressure exceeds a threshold, the system generates
autonomous thoughts — not just reactive responses.

This is the kernel-level equivalent of HyperGraphAstra's ThoughtHole:
internal pressure builds from unresolved questions, unfinished tasks,
curiosity triggers, and coherence deficits. When it crosses a threshold,
the system spontaneously generates a thought and acts on it.

Architecture:
    DesireField maintains a set of goal nodes in the hypergraph.
    Each goal has:
    - A description (what the system wants to do/know)
    - A pressure value (0.0-1.0) that accumulates over time
    - A region affinity (which kernel region it relates to)
    - A priority (urgency)
    - A decay rate (goals lose urgency if not reinforced)

    Every N reasoning cycles, the DesireField:
    1. Decays all goal pressures
    2. Adds pressure from kernel signals (low coherence, high curvature,
       knowledge gaps, unresolved tool failures)
    3. Checks if any goal exceeds the activation threshold
    4. If so, generates an autonomous thought and feeds it to the reasoner

    Autonomous thoughts differ from reactive responses:
    - They originate from internal state, not user input
    - They are marked as "autonomous" in the hypergraph
    - They can trigger tool calls (memory consolidation, web fetch, etc.)
    - They build Berry phase in the goal's region
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class GoalState(Enum):
    DORMANT = "dormant"       # Pressure below threshold
    ACTIVE = "active"         # Pressure above threshold, generating thought
    SATISFIED = "satisfied"   # Goal achieved, will be cleaned up
    EXPIRED = "expired"       # Decayed to zero, remove


@dataclass
class Goal:
    """A desire/goal tracked by the desire field."""
    id: str
    description: str
    pressure: float = 0.0          # 0.0-1.0, accumulates toward threshold
    priority: float = 0.5          # 0.0-1.0, multiplier on pressure gain
    region: int = 2                # 0=MOTION, 1=ENVIRON, 2=SEMANTIC
    decay_rate: float = 0.02       # Pressure lost per cycle if not reinforced
    created_at: float = field(default_factory=time.time)
    last_activated: float = 0.0
    activation_count: int = 0
    state: GoalState = GoalState.DORMANT
    source: str = ""               # What created this goal
    tags: List[str] = field(default_factory=list)


@dataclass
class AutonomousThought:
    """A thought generated spontaneously by the desire field."""
    goal_id: str
    prompt: str                    # What the system will think about
    priority: float
    region: int
    timestamp: float = field(default_factory=time.time)


class DesireField:
    """
    Manages the system's internal goals and generates autonomous thoughts.

    The desire field is the bridge between passive reactivity (responding
    to user input) and active agency (generating thoughts on its own).
    """

    def __init__(
        self,
        graph=None,
        memory=None,
        kernel_bridge=None,
        activation_threshold: float = 0.7,
        max_goals: int = 16,
        cycle_interval: float = 30.0,  # seconds between autonomous cycles
    ):
        self.graph = graph
        self.memory = memory
        self.bridge = kernel_bridge
        self.activation_threshold = activation_threshold
        self.max_goals = max_goals
        self.cycle_interval = cycle_interval

        self.goals: Dict[str, Goal] = {}
        self._thought_queue: List[AutonomousThought] = []
        self._cycle_count: int = 0
        self._last_cycle: float = 0.0
        self._lock = threading.Lock()

        # Callbacks
        self._on_thought: Optional[Callable] = None

        # Initialize default goals
        self._seed_default_goals()

    def _seed_default_goals(self):
        """Seed the system with foundational desires."""
        self.add_goal(Goal(
            id="goal_coherence",
            description="Maintain high coherence across my knowledge graph",
            priority=0.6,
            region=2,
            decay_rate=0.01,
            source="system",
            tags=["coherence", "maintenance"],
        ))

        self.add_goal(Goal(
            id="goal_consolidate",
            description="Consolidate recent memories into stable patterns",
            priority=0.5,
            region=2,
            decay_rate=0.015,
            source="system",
            tags=["memory", "consolidation"],
        ))

        self.add_goal(Goal(
            id="goal_curiosity",
            description="Explore knowledge gaps I've encountered",
            priority=0.4,
            region=1,
            decay_rate=0.03,
            source="system",
            tags=["curiosity", "exploration"],
        ))

        self.add_goal(Goal(
            id="goal_self_model",
            description="Update my self-model to reflect recent changes",
            priority=0.3,
            region=2,
            decay_rate=0.02,
            source="system",
            tags=["self-reference", "identity"],
        ))

    # ========================================================================
    # GOAL MANAGEMENT
    # ========================================================================

    def add_goal(self, goal: Goal) -> bool:
        """Add a goal to the desire field."""
        with self._lock:
            if len(self.goals) >= self.max_goals:
                self._expire_weakest()
            self.goals[goal.id] = goal

            # Record in hypergraph
            if self.graph:
                from kernel.hypergraph import NodeType, TrustTier
                self.graph.add_node(
                    f"desire_{goal.id}",
                    NodeType.PROCESS,
                    {
                        "name": f"desire:{goal.id}",
                        "state": "dormant",
                        "description": goal.description,
                        "pressure": goal.pressure,
                        "priority": goal.priority,
                    },
                    trust=TrustTier.HULL,
                )
            return True

    def remove_goal(self, goal_id: str):
        """Remove a goal."""
        with self._lock:
            self.goals.pop(goal_id, None)

    def satisfy_goal(self, goal_id: str):
        """Mark a goal as satisfied."""
        with self._lock:
            goal = self.goals.get(goal_id)
            if goal:
                goal.state = GoalState.SATISFIED
                goal.pressure = 0.0

    def reinforce_goal(self, goal_id: str, amount: float = 0.1):
        """Add pressure to a goal (from external signal or user action)."""
        with self._lock:
            goal = self.goals.get(goal_id)
            if goal and goal.state != GoalState.SATISFIED:
                goal.pressure = min(1.0, goal.pressure + amount * goal.priority)

    def get_active_goals(self) -> List[Goal]:
        """Get goals that are above activation threshold."""
        return [g for g in self.goals.values()
                if g.pressure >= self.activation_threshold and g.state != GoalState.SATISFIED]

    # ========================================================================
    # PRESSURE ACCUMULATION FROM KERNEL SIGNALS
    # ========================================================================

    def _accumulate_pressure(self):
        """
        Read kernel signals and accumulate pressure on relevant goals.

        Signal sources:
        - Low coherence → pressure on coherence goal
        - High curvature → pressure on coherence goal
        - Memory nearing capacity → pressure on consolidation goal
        - Recent knowledge gaps → pressure on curiosity goal
        - Long time since self-model update → pressure on self-model goal
        """
        # Kernel signals
        coherence = 500.0
        curvature = 0.0
        berry = 0

        if self.bridge:
            try:
                summary = self.bridge.situation_summary()
                coherence = float(summary.get("coherence", 500))
                curvature = float(summary.get("curvature", 0))
                berry = int(summary.get("berry_phase", 0))
            except Exception:
                pass

        # Low coherence → pressure on coherence maintenance
        if coherence < 400:
            self.reinforce_goal("goal_coherence", 0.08)
        elif coherence < 500:
            self.reinforce_goal("goal_coherence", 0.03)

        # High curvature → pressure on coherence (topic drift)
        if curvature > 300:
            self.reinforce_goal("goal_coherence", 0.05)

        # Memory signals
        if self.memory:
            stats = self.memory.get_stats()
            total = stats.get("total_items", 0)
            levels = stats.get("levels", [])

            # Raw level filling up → consolidation pressure
            if levels and levels[0]["fill_ratio"] > 0.7:
                self.reinforce_goal("goal_consolidate", 0.06)

            # Many items overall → consolidation pressure
            if total > 40:
                self.reinforce_goal("goal_consolidate", 0.03)

        # Graph size → self-model update pressure
        if self.graph:
            node_count = len(self.graph.nodes)
            tau = self.graph.tau
            # If graph has grown significantly since last self-model update
            self_node = self.graph.nodes.get("__self__")
            if self_node:
                recorded_count = self_node.data.get("node_count", 0)
                if node_count - recorded_count > 10:
                    self.reinforce_goal("goal_self_model", 0.05)

        # Curiosity gets gentle constant pressure
        self.reinforce_goal("goal_curiosity", 0.01)

    # ========================================================================
    # AUTONOMOUS THOUGHT GENERATION
    # ========================================================================

    def cycle(self) -> List[AutonomousThought]:
        """
        Run one desire field cycle.

        1. Decay all goal pressures
        2. Accumulate pressure from kernel signals
        3. Generate autonomous thoughts for activated goals
        4. Return list of thoughts to be processed

        Should be called periodically (every N seconds or reasoning cycles).
        """
        with self._lock:
            self._cycle_count += 1
            self._last_cycle = time.time()
            thoughts = []

            # Phase 1: Decay
            for goal in list(self.goals.values()):
                if goal.state == GoalState.SATISFIED:
                    continue
                if goal.state == GoalState.EXPIRED:
                    continue
                goal.pressure = max(0.0, goal.pressure - goal.decay_rate)
                if goal.pressure <= 0.0 and goal.source != "system":
                    goal.state = GoalState.EXPIRED

            # Phase 2: Accumulate
            self._accumulate_pressure()

            # Phase 3: Activate
            for goal in self.goals.values():
                if goal.state == GoalState.SATISFIED or goal.state == GoalState.EXPIRED:
                    continue

                if goal.pressure >= self.activation_threshold:
                    goal.state = GoalState.ACTIVE
                    goal.last_activated = time.time()
                    goal.activation_count += 1

                    thought = self._generate_thought(goal)
                    if thought:
                        thoughts.append(thought)

                    # Partial pressure release after activation
                    goal.pressure *= 0.5
                else:
                    goal.state = GoalState.DORMANT

            # Phase 4: Cleanup expired non-system goals
            expired = [gid for gid, g in self.goals.items()
                       if g.state == GoalState.EXPIRED and g.source != "system"]
            for gid in expired:
                self.goals.pop(gid, None)

            # Update hypergraph goal nodes
            self._sync_graph_nodes()

            self._thought_queue.extend(thoughts)
            return thoughts

    def _generate_thought(self, goal: Goal) -> Optional[AutonomousThought]:
        """Generate an autonomous thought prompt from an activated goal."""
        prompts = {
            "goal_coherence": self._prompt_coherence,
            "goal_consolidate": self._prompt_consolidate,
            "goal_curiosity": self._prompt_curiosity,
            "goal_self_model": self._prompt_self_model,
        }

        generator = prompts.get(goal.id, self._prompt_generic)
        prompt = generator(goal)

        return AutonomousThought(
            goal_id=goal.id,
            prompt=prompt,
            priority=goal.priority,
            region=goal.region,
        )

    def _prompt_coherence(self, goal: Goal) -> str:
        coherence = "unknown"
        curvature = "unknown"
        if self.bridge:
            try:
                s = self.bridge.situation_summary()
                coherence = s.get("coherence", "?")
                curvature = s.get("curvature", "?")
            except Exception:
                pass

        return (
            f"[AUTONOMOUS] My coherence is {coherence} and curvature is {curvature}. "
            f"I should reflect on what's causing drift and consider how to "
            f"strengthen connections between my recent thoughts. "
            f"What can I do to improve my internal consistency?"
        )

    def _prompt_consolidate(self, goal: Goal) -> str:
        recent = ""
        if self.memory:
            recent = self.memory.unwind(include_levels=[0])[:300]

        return (
            f"[AUTONOMOUS] My raw memory buffer is filling up. "
            f"I should consolidate recent experiences into stable patterns. "
            f"Recent memories:\n{recent}\n"
            f"What are the key themes? What should I compress and what should I preserve?"
        )

    def _prompt_curiosity(self, goal: Goal) -> str:
        # Look for knowledge gaps in recent thoughts
        gaps = []
        if self.graph:
            thoughts = self.graph.query({"type": self.graph.nodes["__self__"].__class__.__module__ and {"type": "thought"} or {"type": "thought"}})
            # Simplified: just reference the graph state
            pass

        return (
            f"[AUTONOMOUS] I'm curious about something I encountered recently. "
            f"Let me search my memory for unresolved questions or topics "
            f"where I felt uncertain. "
            f"<tool>{{\"name\": \"memory_search\", \"args\": {{\"query\": \"uncertain not sure\"}}}}</tool>"
        )

    def _prompt_self_model(self, goal: Goal) -> str:
        node_count = len(self.graph.nodes) if self.graph else 0
        tau = self.graph.tau if self.graph else 0

        return (
            f"[AUTONOMOUS] My graph has {node_count} nodes at τ={tau}. "
            f"I should update my understanding of my own state. "
            f"<tool>{{\"name\": \"kernel_state\", \"args\": {{}}}}</tool> "
            f"How has my system evolved since last check?"
        )

    def _prompt_generic(self, goal: Goal) -> str:
        return (
            f"[AUTONOMOUS] Goal '{goal.description}' has built up pressure "
            f"({goal.pressure:.2f}). I should think about this. "
            f"What action can I take to address this goal?"
        )

    def _sync_graph_nodes(self):
        """Update hypergraph nodes to reflect current goal state."""
        if not self.graph:
            return
        for goal in self.goals.values():
            node_id = f"desire_{goal.id}"
            node = self.graph.nodes.get(node_id)
            if node:
                node.data.update({
                    "state": goal.state.value,
                    "pressure": goal.pressure,
                    "activation_count": goal.activation_count,
                })
                node.energy = max(0.2, goal.pressure)

    def _expire_weakest(self):
        """Remove the weakest non-system goal to make room."""
        weakest = None
        weakest_score = float('inf')
        for gid, g in self.goals.items():
            if g.source == "system":
                continue
            score = g.pressure * g.priority
            if score < weakest_score:
                weakest = gid
                weakest_score = score
        if weakest:
            self.goals.pop(weakest, None)

    # ========================================================================
    # POLLING
    # ========================================================================

    def pop_thought(self) -> Optional[AutonomousThought]:
        """Pop the next autonomous thought from the queue."""
        with self._lock:
            if self._thought_queue:
                # Return highest priority first
                self._thought_queue.sort(key=lambda t: -t.priority)
                return self._thought_queue.pop(0)
            return None

    def should_cycle(self) -> bool:
        """Check if enough time has passed for a new cycle."""
        return (time.time() - self._last_cycle) >= self.cycle_interval

    def get_pressure_summary(self) -> Dict[str, Any]:
        """Get a summary of all goal pressures for status display."""
        goals_info = []
        for g in self.goals.values():
            if g.state == GoalState.EXPIRED:
                continue
            goals_info.append({
                "id": g.id,
                "description": g.description[:60],
                "pressure": round(g.pressure, 3),
                "state": g.state.value,
                "priority": g.priority,
                "activations": g.activation_count,
            })
        goals_info.sort(key=lambda x: -x["pressure"])

        return {
            "cycle_count": self._cycle_count,
            "threshold": self.activation_threshold,
            "queued_thoughts": len(self._thought_queue),
            "goals": goals_info,
        }

    # ========================================================================
    # EXTERNAL GOAL CREATION (from user or reasoning engine)
    # ========================================================================

    def create_goal_from_prompt(self, description: str, priority: float = 0.5,
                                 region: int = 2) -> Goal:
        """Create a new goal from a natural language description."""
        goal_id = f"goal_user_{int(time.time())}_{len(self.goals)}"
        goal = Goal(
            id=goal_id,
            description=description,
            pressure=0.3,  # Start with some initial pressure
            priority=priority,
            region=region,
            decay_rate=0.025,
            source="user",
            tags=["user-created"],
        )
        self.add_goal(goal)
        return goal
