from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

@dataclass(frozen=True)
class RewardWeights:
    w_coh: float = 1.0
    w_curv: float = 0.8
    w_bridge: float = 0.25
    w_stability: float = 0.35
    curv_scale: float = 300.0  # curvature_scaled is ~0..900

def compute_reward(
    *,
    coh_before: float,
    coh_after: float,
    curvature_scaled: float,
    bridge_count: int,
    weights: RewardWeights = RewardWeights(),
) -> Tuple[float, Dict[str, Any]]:
    curv_n = max(0.0, min(1.0, curvature_scaled / weights.curv_scale))
    smooth = 1.0 - curv_n
    stability = max(0.0, coh_after - coh_before)

    reward = (
        weights.w_coh * coh_after
        + weights.w_curv * (smooth * 1000.0)
        + weights.w_bridge * (bridge_count * 100.0)
        + weights.w_stability * stability
    )

    diag = {
        "coh_before": coh_before,
        "coh_after": coh_after,
        "curvature_scaled": curvature_scaled,
        "smooth": smooth,
        "bridge_count": bridge_count,
        "stability": stability,
        "reward": reward,
    }
    return reward, diag
