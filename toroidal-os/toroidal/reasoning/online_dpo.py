#!/usr/bin/env python3
"""
TOROIDAL OS — Online DPO (Direct Preference Optimization)
============================================================
Closed-loop training signal derived from kernel metrics.

Standard DPO requires paired human preference data.  We can't do that
on-device.  Instead, we use the topological kernel as an *implicit
preference oracle*:

    After the system produces a response, the kernel metrics
    (coherence, Berry phase delta, curvature change, convergence speed)
    score it.  This score becomes a reward signal.

    We maintain a buffer of (prompt, response, reward) tuples.
    Periodically, we rank responses and construct preference pairs:
        chosen  = high-reward response
        rejected = low-reward response
        (for the same or similar prompts)

    These pairs are used to:
    1. Adjust system-level biases (prompt engineering, temperature,
       tool-use preferences, reasoning depth)
    2. Optionally generate LoRA training data for offline fine-tuning
    3. Reinforce or decay goals in the desire field based on outcome

This is NOT backpropagation through the LLM — that's impossible at
runtime on a phone.  It's a *behavioral* DPO that adjusts the system's
prompt construction, tool selection, and reasoning strategy based on
what the kernel metrics say worked well.

Architecture:
    OnlineDPO maintains:
    - A reward buffer of recent (prompt, response, reward) tuples
    - A preference pair archive for offline training export
    - A set of behavioral biases that modify prompt construction
    - Exponential moving averages of kernel metrics for baseline
"""

import time
import math
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class RewardSignal:
    """A reward signal computed from kernel metrics after a response."""
    coherence_delta: float       # Change in coherence (positive = good)
    berry_delta: float           # Berry phase accumulated (higher = richer)
    curvature_delta: float       # Curvature change (lower = more stable)
    convergence_speed: float     # 1/iterations to converge (higher = faster)
    tool_success_rate: float     # Fraction of tool calls that succeeded
    timestamp: float = field(default_factory=time.time)

    @property
    def composite(self) -> float:
        """
        Weighted composite reward score.

        Weights reflect what we value:
        - Coherence improvement is most important (0.35)
        - Fast convergence is valuable (0.25)
        - Rich cross-modal engagement (Berry) (0.20)
        - Stability (low curvature increase) (0.10)
        - Tool effectiveness (0.10)
        """
        score = (
            0.35 * _normalize(self.coherence_delta, -200, 200) +
            0.25 * _normalize(self.convergence_speed, 0, 1) +
            0.20 * _normalize(self.berry_delta, 0, 500) +
            0.10 * (1.0 - _normalize(abs(self.curvature_delta), 0, 300)) +
            0.10 * self.tool_success_rate
        )
        return max(0.0, min(1.0, score))


def _normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] range."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


@dataclass
class Experience:
    """A single experience: prompt + response + reward."""
    id: str
    prompt_hash: str             # Hash of the prompt (for pairing)
    prompt_summary: str          # Short summary of prompt
    response_summary: str        # Short summary of response
    reward: RewardSignal
    composite_reward: float
    iteration_count: int         # How many reasoning iterations
    tools_used: List[str]        # Which tools were invoked
    temperature: float           # Temperature used
    timestamp: float = field(default_factory=time.time)


@dataclass
class PreferencePair:
    """A DPO preference pair: chosen vs rejected."""
    prompt_summary: str
    chosen: Experience
    rejected: Experience
    reward_gap: float            # Difference in composite reward
    created_at: float = field(default_factory=time.time)


@dataclass
class BehavioralBias:
    """An adjustable bias that modifies system behavior."""
    name: str
    value: float                 # Current value (0.0-1.0 typically)
    default: float               # Default value
    min_val: float = 0.0
    max_val: float = 1.0
    description: str = ""
    last_adjusted: float = 0.0


class OnlineDPO:
    """
    Closed-loop preference optimization from kernel metrics.

    Does NOT modify LLM weights.  Instead adjusts:
    - Temperature for different query types
    - Reasoning depth (max iterations)
    - Tool-use preferences
    - Prompt construction biases
    - Memory retrieval depth
    """

    def __init__(
        self,
        graph=None,
        memory=None,
        kernel_bridge=None,
        desire_field=None,
        buffer_size: int = 128,
        pair_buffer_size: int = 64,
        learning_rate: float = 0.05,
    ):
        self.graph = graph
        self.memory = memory
        self.bridge = kernel_bridge
        self.desire = desire_field
        self.learning_rate = learning_rate

        # Experience buffer (ring buffer)
        self.experiences: deque = deque(maxlen=buffer_size)

        # Preference pairs
        self.preference_pairs: deque = deque(maxlen=pair_buffer_size)

        # Behavioral biases
        self.biases: Dict[str, BehavioralBias] = {}
        self._init_biases()

        # Exponential moving averages for baseline
        self._ema_coherence = 500.0
        self._ema_curvature = 0.0
        self._ema_berry = 0.0
        self._ema_alpha = 0.1  # Smoothing factor

        # Kernel metric snapshots (before/after response)
        self._pre_snapshot: Optional[Dict[str, float]] = None

        # Counters
        self._experience_count = 0
        self._pair_count = 0
        self._update_count = 0
        self._lock = threading.Lock()

    def _init_biases(self):
        """Initialize behavioral biases with defaults."""
        defaults = [
            ("temperature", 0.7, 0.1, 1.2,
             "LLM sampling temperature"),
            ("reasoning_depth", 0.6, 0.2, 1.0,
             "Fraction of max_iterations to use (scaled)"),
            ("tool_eagerness", 0.5, 0.0, 1.0,
             "Likelihood of using tools vs pure reasoning"),
            ("memory_depth", 0.5, 0.1, 1.0,
             "How much memory context to include"),
            ("self_ref_weight", 0.5, 0.1, 1.0,
             "Weight on self-referential previous-thought inclusion"),
            ("exploration_rate", 0.3, 0.05, 0.8,
             "How much to explore vs exploit in responses"),
        ]

        for name, default, lo, hi, desc in defaults:
            self.biases[name] = BehavioralBias(
                name=name,
                value=default,
                default=default,
                min_val=lo,
                max_val=hi,
                description=desc,
            )

    # ========================================================================
    # REWARD COMPUTATION
    # ========================================================================

    def snapshot_pre(self):
        """Take a kernel metric snapshot BEFORE generating a response."""
        self._pre_snapshot = self._take_snapshot()

    def _take_snapshot(self) -> Dict[str, float]:
        """Read current kernel metrics."""
        snap = {
            "coherence": 500.0,
            "curvature": 0.0,
            "berry": 0.0,
            "timestamp": time.time(),
        }

        if self.bridge:
            try:
                snap["coherence"] = float(self.bridge.coherence())
                snap["curvature"] = float(self.bridge.curvature())
                snap["berry"] = float(self.bridge.berry_phase())
            except Exception:
                pass

        return snap

    def compute_reward(
        self,
        iterations: int,
        max_iterations: int,
        tool_calls: int = 0,
        tool_successes: int = 0,
    ) -> RewardSignal:
        """
        Compute reward signal from kernel metric deltas.

        Call this AFTER the response is generated.
        """
        post = self._take_snapshot()
        pre = self._pre_snapshot or post

        coherence_delta = post["coherence"] - pre["coherence"]
        curvature_delta = post["curvature"] - pre["curvature"]
        berry_delta = post["berry"] - pre["berry"]
        convergence_speed = 1.0 / max(1, iterations) if iterations <= max_iterations else 0.0
        tool_rate = tool_successes / max(1, tool_calls)

        reward = RewardSignal(
            coherence_delta=coherence_delta,
            berry_delta=berry_delta,
            curvature_delta=curvature_delta,
            convergence_speed=convergence_speed,
            tool_success_rate=tool_rate,
        )

        # Update EMAs
        self._ema_coherence = (self._ema_alpha * post["coherence"] +
                               (1 - self._ema_alpha) * self._ema_coherence)
        self._ema_curvature = (self._ema_alpha * post["curvature"] +
                               (1 - self._ema_alpha) * self._ema_curvature)
        self._ema_berry = (self._ema_alpha * post["berry"] +
                           (1 - self._ema_alpha) * self._ema_berry)

        return reward

    # ========================================================================
    # EXPERIENCE RECORDING
    # ========================================================================

    def record_experience(
        self,
        prompt: str,
        response: str,
        reward: RewardSignal,
        iterations: int,
        tools_used: List[str] = None,
    ) -> Experience:
        """Record a complete experience (prompt → response → reward)."""
        with self._lock:
            self._experience_count += 1

            # Hash prompt for pairing similar prompts later
            import hashlib
            prompt_hash = hashlib.md5(prompt[:200].encode()).hexdigest()[:8]

            exp = Experience(
                id=f"exp_{self._experience_count}",
                prompt_hash=prompt_hash,
                prompt_summary=prompt[:100],
                response_summary=response[:100],
                reward=reward,
                composite_reward=reward.composite,
                iteration_count=iterations,
                tools_used=tools_used or [],
                temperature=self.biases["temperature"].value,
            )

            self.experiences.append(exp)
            return exp

    # ========================================================================
    # PREFERENCE PAIR CONSTRUCTION
    # ========================================================================

    def construct_pairs(self) -> List[PreferencePair]:
        """
        Construct DPO preference pairs from the experience buffer.

        Strategy: for each prompt hash that has 2+ experiences,
        pair the highest-reward with the lowest-reward.
        """
        with self._lock:
            # Group by prompt hash
            groups: Dict[str, List[Experience]] = {}
            for exp in self.experiences:
                groups.setdefault(exp.prompt_hash, []).append(exp)

            pairs = []
            for prompt_hash, exps in groups.items():
                if len(exps) < 2:
                    continue

                # Sort by composite reward
                sorted_exps = sorted(exps, key=lambda e: e.composite_reward, reverse=True)
                chosen = sorted_exps[0]
                rejected = sorted_exps[-1]

                gap = chosen.composite_reward - rejected.composite_reward
                if gap < 0.05:
                    continue  # Not enough signal

                pair = PreferencePair(
                    prompt_summary=chosen.prompt_summary,
                    chosen=chosen,
                    rejected=rejected,
                    reward_gap=gap,
                )
                pairs.append(pair)
                self.preference_pairs.append(pair)
                self._pair_count += 1

            return pairs

    # ========================================================================
    # BEHAVIORAL BIAS UPDATES
    # ========================================================================

    def update_biases(self):
        """
        Adjust behavioral biases based on recent experience statistics.

        This is the "training" step — not weight updates, but behavioral
        parameter adjustment based on what reward signals tell us.
        """
        with self._lock:
            self._update_count += 1
            recent = list(self.experiences)[-20:]
            if len(recent) < 5:
                return

            avg_reward = sum(e.composite_reward for e in recent) / len(recent)
            avg_iters = sum(e.iteration_count for e in recent) / len(recent)
            avg_tools = sum(len(e.tools_used) for e in recent) / len(recent)

            # Correlation analysis: which biases correlate with high reward?
            high_reward = [e for e in recent if e.composite_reward > avg_reward]
            low_reward = [e for e in recent if e.composite_reward <= avg_reward]

            # Temperature adjustment
            if high_reward and low_reward:
                high_temp = sum(e.temperature for e in high_reward) / len(high_reward)
                low_temp = sum(e.temperature for e in low_reward) / len(low_reward)
                temp_gradient = (high_temp - low_temp) * self.learning_rate
                self._adjust_bias("temperature", temp_gradient)

            # Reasoning depth: if more iterations → higher reward, increase depth
            if avg_iters > 3 and avg_reward > 0.5:
                self._adjust_bias("reasoning_depth", 0.02 * self.learning_rate)
            elif avg_iters < 2 and avg_reward > 0.6:
                self._adjust_bias("reasoning_depth", -0.02 * self.learning_rate)

            # Tool eagerness: if tool use correlates with reward
            if avg_tools > 0.5 and avg_reward > 0.5:
                self._adjust_bias("tool_eagerness", 0.02 * self.learning_rate)
            elif avg_tools < 0.2 and avg_reward > 0.5:
                self._adjust_bias("tool_eagerness", -0.01 * self.learning_rate)

            # Exploration rate: decay over time (exploit more as we learn)
            exp_rate = self.biases["exploration_rate"]
            decay = 0.001 * math.log(1 + self._update_count)
            self._adjust_bias("exploration_rate", -decay * self.learning_rate)

            # Feed reward signals to desire field
            if self.desire and avg_reward < 0.4:
                # Low reward → reinforce coherence and self-model goals
                self.desire.reinforce_goal("goal_coherence", 0.05)
                self.desire.reinforce_goal("goal_self_model", 0.03)

    def _adjust_bias(self, name: str, delta: float):
        """Adjust a bias value, clamping to valid range."""
        bias = self.biases.get(name)
        if not bias:
            return
        bias.value = max(bias.min_val, min(bias.max_val, bias.value + delta))
        bias.last_adjusted = time.time()

    # ========================================================================
    # BIAS APPLICATION (called by the reasoning engine)
    # ========================================================================

    def get_temperature(self) -> float:
        """Get current adjusted temperature."""
        return self.biases["temperature"].value

    def get_max_iterations(self, base_max: int) -> int:
        """Get adjusted max iterations."""
        depth = self.biases["reasoning_depth"].value
        return max(2, int(base_max * depth))

    def get_memory_depth(self) -> float:
        """Get memory inclusion depth (0-1)."""
        return self.biases["memory_depth"].value

    def should_use_tools(self) -> bool:
        """Probabilistic: should we include tool prompts?"""
        import random
        return random.random() < self.biases["tool_eagerness"].value

    def should_explore(self) -> bool:
        """Probabilistic: should we try something different?"""
        import random
        return random.random() < self.biases["exploration_rate"].value

    # ========================================================================
    # EXPORT FOR OFFLINE TRAINING
    # ========================================================================

    def export_training_pairs(self) -> List[Dict[str, Any]]:
        """
        Export preference pairs in a format suitable for offline
        DPO fine-tuning (e.g., with TRL or similar framework).

        Format:
        [
            {
                "prompt": "...",
                "chosen": "...",
                "rejected": "...",
                "reward_gap": 0.15,
            },
            ...
        ]
        """
        pairs = []
        for pair in self.preference_pairs:
            pairs.append({
                "prompt": pair.prompt_summary,
                "chosen": pair.chosen.response_summary,
                "rejected": pair.rejected.response_summary,
                "reward_gap": pair.reward_gap,
                "chosen_reward": pair.chosen.composite_reward,
                "rejected_reward": pair.rejected.composite_reward,
            })
        return pairs

    def export_to_jsonl(self, path: str):
        """Export training pairs to JSONL file."""
        pairs = self.export_training_pairs()
        with open(path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')

    # ========================================================================
    # STATUS
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Status summary for display."""
        recent = list(self.experiences)[-10:]
        avg_reward = (sum(e.composite_reward for e in recent) / len(recent)
                      if recent else 0.0)

        return {
            "experiences": len(self.experiences),
            "preference_pairs": len(self.preference_pairs),
            "updates": self._update_count,
            "avg_recent_reward": round(avg_reward, 3),
            "ema_coherence": round(self._ema_coherence, 1),
            "ema_curvature": round(self._ema_curvature, 1),
            "ema_berry": round(self._ema_berry, 1),
            "biases": {
                name: round(bias.value, 3)
                for name, bias in self.biases.items()
            },
        }
