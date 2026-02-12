#!/usr/bin/env python3
"""
TOROIDAL OS — Epistemic State Detector
========================================
5-state model that classifies the system's current knowledge condition
and gates access to external tools (especially web fetch).

Ported from HyperGraphAstra's epistemic authority model.

The 5 states:

  KNOWLEDGE_GAP       — The system recognizes it doesn't know something.
                         Web fetch is ALLOWED. This is the only state
                         where reaching out for new information is justified.

  EXTERNAL_AUTHORITY   — The system has fetched external info and is
                         currently relying on it. Web fetch is BLOCKED
                         to prevent dependency cascades.

  INTERNAL_GROUNDED    — The system's response is grounded in its own
                         memory/beliefs. Web fetch is BLOCKED — it knows
                         enough. Fetching would dilute grounded confidence.

  MIXED_AUTHORITY      — Response combines internal knowledge with
                         external data. Web fetch is CAUTIONED — allowed
                         only if confidence is below threshold.

  SELF_REFERENTIAL     — The system is reasoning about itself.
                         Web fetch is BLOCKED — no external source can
                         answer "what am I thinking?"

The detector uses signals from:
  - Memory search results (found relevant memories → INTERNAL_GROUNDED)
  - Recent tool history (used web recently → EXTERNAL_AUTHORITY)
  - Prompt analysis (questions about self → SELF_REFERENTIAL)
  - Confidence from convergence (high confidence → INTERNAL_GROUNDED)
  - Kernel coherence (low coherence → possible KNOWLEDGE_GAP)
"""

import time
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


class EpistemicState(Enum):
    KNOWLEDGE_GAP = "knowledge_gap"
    EXTERNAL_AUTHORITY = "external_authority"
    INTERNAL_GROUNDED = "internal_grounded"
    MIXED_AUTHORITY = "mixed_authority"
    SELF_REFERENTIAL = "self_referential"


# Web access policy per state
_WEB_POLICY = {
    EpistemicState.KNOWLEDGE_GAP:     "allowed",
    EpistemicState.EXTERNAL_AUTHORITY: "blocked",
    EpistemicState.INTERNAL_GROUNDED:  "blocked",
    EpistemicState.MIXED_AUTHORITY:    "cautioned",
    EpistemicState.SELF_REFERENTIAL:   "blocked",
}


@dataclass
class EpistemicSignals:
    """Raw signals fed into the epistemic state detector."""
    # From memory system
    memory_hits: int = 0          # How many memories matched the query
    memory_relevance: float = 0.0  # Best match relevance (0-1)

    # From tool history
    web_fetches_recent: int = 0   # Web fetches in last N iterations
    tools_used_recent: List[str] = field(default_factory=list)

    # From prompt analysis
    is_self_question: bool = False  # "What are you?" / "How do you work?"
    is_factual_question: bool = False  # "What is X?" / "When did Y happen?"
    uncertainty_markers: int = 0   # "I think", "maybe", "not sure"

    # From reasoning engine
    convergence_confidence: float = 0.5  # From ReasoningResult
    iteration_count: int = 0

    # From kernel
    coherence: float = 500.0
    curvature: float = 0.0


@dataclass
class EpistemicDetection:
    """Result of epistemic state detection."""
    state: EpistemicState
    confidence: float          # 0.0-1.0, how certain the detection is
    web_policy: str            # "allowed", "blocked", "cautioned"
    reason: str                # Human-readable explanation
    signals: EpistemicSignals  # The raw signals used


# ============================================================================
# PROMPT ANALYSIS PATTERNS
# ============================================================================

SELF_PATTERNS = [
    r"\bwhat are you\b",
    r"\bwho are you\b",
    r"\bhow do you (work|think|reason|feel)\b",
    r"\byour (memory|state|beliefs?|thoughts?)\b",
    r"\babout yourself\b",
    r"\byour own\b",
    r"\bself-referent",
    r"\bintrospect",
]

FACTUAL_PATTERNS = [
    r"\bwhat (is|are|was|were) \b",
    r"\bwhen (did|was|were|is)\b",
    r"\bhow (many|much|far|long|old)\b",
    r"\bwhere (is|are|was|were)\b",
    r"\bdefine\b",
    r"\bexplain\b",
    r"\btell me about\b",
    r"\bwhat.*happen",
]

UNCERTAINTY_PATTERNS = [
    r"\bi('m| am) not sure\b",
    r"\bi think\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bI don'?t know\b",
    r"\bnot certain\b",
    r"\bunclear\b",
    r"\bmight be\b",
]

_self_re = [re.compile(p, re.IGNORECASE) for p in SELF_PATTERNS]
_fact_re = [re.compile(p, re.IGNORECASE) for p in FACTUAL_PATTERNS]
_unc_re = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS]


def analyze_prompt(text: str) -> Tuple[bool, bool, int]:
    """Analyze a prompt for epistemic signals.

    Returns: (is_self_question, is_factual_question, uncertainty_count)
    """
    is_self = any(p.search(text) for p in _self_re)
    is_fact = any(p.search(text) for p in _fact_re)
    unc_count = sum(1 for p in _unc_re if p.search(text))
    return is_self, is_fact, unc_count


# ============================================================================
# DETECTOR
# ============================================================================

class EpistemicDetector:
    """
    Detects the system's current epistemic state.

    Used as a prerequisite check by the topo:// protocol
    to gate web fetch and other external tools.
    """

    def __init__(
        self,
        memory=None,
        tool_dispatcher=None,
        kernel_bridge=None,
        confidence_threshold: float = 0.6,
    ):
        self.memory = memory
        self.tools = tool_dispatcher
        self.bridge = kernel_bridge
        self.confidence_threshold = confidence_threshold
        self._last_detection: Optional[EpistemicDetection] = None
        self._detection_cache_ttl: float = 2.0  # seconds
        self._last_detection_time: float = 0.0

    def detect(
        self,
        prompt: str = "",
        prev_output: str = "",
        convergence_confidence: float = 0.5,
        iteration: int = 0,
    ) -> EpistemicDetection:
        """
        Detect the current epistemic state from available signals.
        """
        # Gather signals
        signals = self._gather_signals(prompt, prev_output, convergence_confidence, iteration)

        # Score each state
        scores = self._score_states(signals)

        # Pick the highest-scoring state
        best_state = max(scores, key=scores.get)
        best_score = scores[best_state]

        # Normalize confidence
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.5

        detection = EpistemicDetection(
            state=best_state,
            confidence=confidence,
            web_policy=_WEB_POLICY[best_state],
            reason=self._explain(best_state, signals),
            signals=signals,
        )

        self._last_detection = detection
        self._last_detection_time = time.time()

        return detection

    def get_cached(self) -> Optional[EpistemicDetection]:
        """Get the most recent detection if still within TTL."""
        if self._last_detection and (time.time() - self._last_detection_time) < self._detection_cache_ttl:
            return self._last_detection
        return None

    def web_allowed(self, prompt: str = "", prev_output: str = "") -> Tuple[bool, str]:
        """
        Quick check: is web fetch currently allowed?

        Returns (allowed: bool, reason: str).
        Used as a prerequisite in the topo:// protocol.
        """
        cached = self.get_cached()
        if cached:
            detection = cached
        else:
            detection = self.detect(prompt=prompt, prev_output=prev_output)

        if detection.web_policy == "allowed":
            return True, f"Epistemic state: {detection.state.value} — web access permitted"
        elif detection.web_policy == "cautioned":
            # Allow if confidence is low enough
            if detection.confidence < self.confidence_threshold:
                return True, f"Epistemic state: {detection.state.value} (low confidence) — web access permitted with caution"
            return False, f"Epistemic state: {detection.state.value} — web access cautioned (confidence {detection.confidence:.2f} >= threshold {self.confidence_threshold})"
        else:
            return False, f"Epistemic state: {detection.state.value} — web access blocked: {detection.reason}"

    def as_prerequisite(self) -> Callable:
        """Return a prerequisite check function for the topo:// protocol."""
        def check():
            allowed, reason = self.web_allowed()
            return allowed, reason
        return check

    # ========================================================================
    # INTERNAL
    # ========================================================================

    def _gather_signals(
        self,
        prompt: str,
        prev_output: str,
        convergence_confidence: float,
        iteration: int,
    ) -> EpistemicSignals:
        """Gather all available signals."""
        is_self, is_fact, unc_count = analyze_prompt(prompt)

        # Check previous output for uncertainty too
        if prev_output:
            _, _, prev_unc = analyze_prompt(prev_output)
            unc_count += prev_unc

        # Memory signals
        memory_hits = 0
        memory_relevance = 0.0
        if self.memory and prompt:
            results = self.memory.search(prompt[:100])
            memory_hits = len(results)
            if results:
                # Approximate relevance from importance and recency
                best = results[0]
                memory_relevance = min(1.0, best.importance / 2.0)

        # Tool history signals
        web_fetches = 0
        recent_tools = []
        if self.tools:
            recent = self.tools._call_history[-10:]  # last 10 calls
            recent_tools = [r.tool_name for r in recent]
            web_fetches = sum(1 for t in recent_tools if t in ("web_fetch", "web_search"))

        # Kernel signals
        coherence = 500.0
        curvature = 0.0
        if self.bridge:
            try:
                summary = self.bridge.situation_summary()
                coherence = float(summary.get("coherence", 500))
                curvature = float(summary.get("curvature", 0))
            except Exception:
                pass

        return EpistemicSignals(
            memory_hits=memory_hits,
            memory_relevance=memory_relevance,
            web_fetches_recent=web_fetches,
            tools_used_recent=recent_tools,
            is_self_question=is_self,
            is_factual_question=is_fact,
            uncertainty_markers=unc_count,
            convergence_confidence=convergence_confidence,
            iteration_count=iteration,
            coherence=coherence,
            curvature=curvature,
        )

    def _score_states(self, s: EpistemicSignals) -> Dict[EpistemicState, float]:
        """Score each epistemic state based on signals.

        Higher score = more likely to be in that state.
        """
        scores = {state: 0.1 for state in EpistemicState}  # Small base for all

        # --- SELF_REFERENTIAL ---
        if s.is_self_question:
            scores[EpistemicState.SELF_REFERENTIAL] += 5.0

        # --- KNOWLEDGE_GAP ---
        if s.is_factual_question and s.memory_hits == 0:
            scores[EpistemicState.KNOWLEDGE_GAP] += 4.0
        if s.uncertainty_markers >= 2:
            scores[EpistemicState.KNOWLEDGE_GAP] += 2.0
        if s.convergence_confidence < 0.4:
            scores[EpistemicState.KNOWLEDGE_GAP] += 1.5
        if s.coherence < 350:
            scores[EpistemicState.KNOWLEDGE_GAP] += 1.0

        # --- EXTERNAL_AUTHORITY ---
        if s.web_fetches_recent >= 1:
            scores[EpistemicState.EXTERNAL_AUTHORITY] += 3.0
        if s.web_fetches_recent >= 3:
            scores[EpistemicState.EXTERNAL_AUTHORITY] += 2.0

        # --- INTERNAL_GROUNDED ---
        if s.memory_hits >= 3:
            scores[EpistemicState.INTERNAL_GROUNDED] += 3.0
        if s.memory_relevance > 0.6:
            scores[EpistemicState.INTERNAL_GROUNDED] += 2.0
        if s.convergence_confidence > 0.8:
            scores[EpistemicState.INTERNAL_GROUNDED] += 2.0
        if s.coherence > 600:
            scores[EpistemicState.INTERNAL_GROUNDED] += 1.0

        # --- MIXED_AUTHORITY ---
        if s.memory_hits > 0 and s.web_fetches_recent > 0:
            scores[EpistemicState.MIXED_AUTHORITY] += 3.0
        if s.memory_relevance > 0.3 and s.uncertainty_markers >= 1:
            scores[EpistemicState.MIXED_AUTHORITY] += 1.5

        return scores

    def _explain(self, state: EpistemicState, s: EpistemicSignals) -> str:
        """Generate a human-readable explanation for the detection."""
        if state == EpistemicState.SELF_REFERENTIAL:
            return "Prompt asks about the system's own state or identity"
        elif state == EpistemicState.KNOWLEDGE_GAP:
            parts = []
            if s.memory_hits == 0:
                parts.append("no relevant memories found")
            if s.uncertainty_markers >= 2:
                parts.append(f"{s.uncertainty_markers} uncertainty markers in output")
            if s.convergence_confidence < 0.4:
                parts.append(f"low convergence confidence ({s.convergence_confidence:.2f})")
            return "Knowledge gap detected: " + "; ".join(parts) if parts else "Factual question with insufficient internal knowledge"
        elif state == EpistemicState.EXTERNAL_AUTHORITY:
            return f"Relying on external data ({s.web_fetches_recent} recent web fetches)"
        elif state == EpistemicState.INTERNAL_GROUNDED:
            return f"Grounded in internal knowledge ({s.memory_hits} memory hits, confidence {s.convergence_confidence:.2f})"
        elif state == EpistemicState.MIXED_AUTHORITY:
            return f"Mixed internal ({s.memory_hits} memories) and external ({s.web_fetches_recent} fetches) authority"
        return "Unknown"


# ============================================================================
# CONVENIENCE
# ============================================================================

def create_epistemic_prerequisite(detector: EpistemicDetector) -> Callable:
    """Create a prerequisite check function for the topo:// protocol."""
    def epistemic_check():
        allowed, reason = detector.web_allowed()
        return allowed, reason
    return epistemic_check
