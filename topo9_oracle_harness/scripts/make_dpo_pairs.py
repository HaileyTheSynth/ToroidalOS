from __future__ import annotations
import argparse
import json
from typing import Any, Dict, List

from scripts.oracle import OracleConfig, Topo9Oracle
from scripts.reward import compute_reward, RewardWeights

def run_trace(o: Topo9Oracle, trace: List[Dict[str, Any]]) -> None:
    for step in trace:
        cmd = step["cmd"].upper()
        args = step.get("args", [])
        line = " ".join([cmd] + [str(a) for a in args]).strip()
        o.send_cmd(line)

def score_candidate(o: Topo9Oracle, cand: Dict[str, Any]) -> Dict[str, Any]:
    trace = cand.get("trace", [])
    focus_id = int(cand.get("focus_id", 0))

    coh_before = o.coherent(focus_id, 5).get("coherence") or 0
    run_trace(o, trace)

    # consolidation step: measure whether the candidate leaves the system healthier
    o.send_cmd("EVOLVE 10")

    coh_after = o.coherent(focus_id, 5).get("coherence") or 0
    curv = o.curvature()
    bridges = o.bridges(1200)

    reward, diag = compute_reward(
        coh_before=float(coh_before),
        coh_after=float(coh_after),
        curvature_scaled=float(curv if curv >= 0 else 0),
        bridge_count=len(bridges),
        weights=RewardWeights(),
    )
    return {"reward": reward, "diag": diag}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", required=True)
    ap.add_argument("--candidates", required=True, help="JSONL with {prompt, candidates:[{final_answer, trace, focus_id}...]}")
    ap.add_argument("--out", required=True, help="output JSONL DPO pairs")
    args = ap.parse_args()

    cfg = OracleConfig(iso_path=args.iso)

    with open(args.candidates, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            prompt = rec["prompt"]
            candidates = rec["candidates"]

            scored = []
            for i, c in enumerate(candidates):
                # boot fresh per candidate for comparability
                with Topo9Oracle(cfg) as o:
                    s = score_candidate(o, c)
                scored.append((i, s["reward"], s["diag"]))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_i, _, best_diag = scored[0]
            worst_i, _, worst_diag = scored[-1]

            pair = {
                "prompt": prompt,
                "chosen": candidates[best_i]["final_answer"],
                "rejected": candidates[worst_i]["final_answer"],
                "meta": {"chosen_diag": best_diag, "rejected_diag": worst_diag, "ranking": scored},
            }
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
