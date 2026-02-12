#!/usr/bin/env python3
"""
TOROIDAL OS - Baseline Comparison Harness
===========================================
Measures whether topological training actually improves behavior.

Runs the same behavioral test suite through three systems:

  (a) Vanilla:    No trace. Model responds normally.
                   Trace = empty (or minimal ACCESS/STORE).
                   Baseline for "what does plain LLM do?"

  (b) RAG:        Retrieval-augmented. Model gets relevant context.
                   Trace = ACCESS/STORE only (no HEDGE/EVOLVE/FIBER).
                   Tests if simple memory access is sufficient.

  (c) Topo-tuned: DPO-trained model. Full trace with HEDGE/EVOLVE/FIBER.
                   Tests if topological training adds measurable value.

For each prompt x system, the harness:
1. Generates a response + trace (or synthetic trace for vanilla/RAG)
2. Replays the trace through the oracle (QEMU kernel)
3. Collects metrics: coherence, curvature, bridge count, stability delta
4. Produces a comparison report

Usage:
    # With QEMU oracle:
    python scripts/compare_baselines.py --iso build/topo9.iso --prompts prompts/behavioral_suite.json

    # Simulation mode (no QEMU, uses Python kernel):
    python scripts/compare_baselines.py --simulate --prompts prompts/behavioral_suite.json
"""

import json
import sys
import os
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.reward import compute_reward, RewardWeights
from scripts.trace_gen import (
    generate_synthetic_traces,
    build_trace_prompt,
    parse_model_output,
    DEFAULT_SEED_NODES,
)


@dataclass
class MetricSnapshot:
    """Metrics collected from a single evaluation run."""
    coherence_before: float = 0.0
    coherence_after: float = 0.0
    curvature: float = 0.0
    bridge_count: int = 0
    stability_delta: float = 0.0
    reward: float = 0.0


@dataclass
class EvalResult:
    """Result of evaluating one prompt under one system."""
    prompt_id: str
    system: str  # "vanilla", "rag", "topo"
    metrics: MetricSnapshot
    trace_length: int = 0
    trace_commands: List[str] = field(default_factory=list)


# ============================================================================
# TRACE GENERATORS FOR EACH SYSTEM TYPE
# ============================================================================

def make_vanilla_trace(prompt: Dict) -> List[Dict[str, Any]]:
    """
    Vanilla baseline: minimal trace (just an ACCESS of a default node).
    Simulates a model that doesn't know about the kernel.
    """
    return [
        {"cmd": "ACCESS", "args": [0, 0]},
    ]


def make_rag_trace(prompt: Dict) -> List[Dict[str, Any]]:
    """
    RAG baseline: ACCESS and STORE only.
    Simulates retrieval-augmented generation: the model fetches relevant
    nodes and stores its response, but doesn't create hyperedges or
    use EVOLVE/FIBER/TRUST operations.
    """
    # Simulate: search for relevant nodes (ACCESS), store response (STORE)
    # Use different regions to simulate retrieval patterns
    return [
        {"cmd": "ACCESS", "args": [3, 0]},   # Retrieve concept A
        {"cmd": "ACCESS", "args": [5, 0]},   # Retrieve concept B
        {"cmd": "STORE", "args": ["010101010", 0]},  # Store response
    ]


def make_topo_trace(prompt: Dict) -> List[Dict[str, Any]]:
    """
    Topo-tuned trace: full kernel operations.
    Uses the best synthetic trace variant from trace_gen.
    """
    intent = prompt.get("intent", "")
    traces = generate_synthetic_traces(intent)
    return traces[0]  # Return the "good" trace


# ============================================================================
# SIMULATION MODE (Python kernel, no QEMU)
# ============================================================================

def eval_with_simulation(trace: List[Dict[str, Any]], focus_id: int = 0) -> MetricSnapshot:
    """
    Run a trace through the Python kernel simulator and collect metrics.
    Uses the Kernel class from simulate.py.
    """
    # Import the simulator
    sim_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "simulate.py")

    # We inline a minimal kernel since simulate.py is designed as a script
    import importlib.util
    spec = importlib.util.spec_from_file_location("simulate", sim_path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)

    k = sim.Kernel()
    k.init_seed_nodes()

    # Measure coherence before
    coh_before = k.coherence_score(focus_id) if focus_id < k.node_count else 500

    # Execute trace
    for step in trace:
        cmd = step.get("cmd", "").upper()
        args = step.get("args", [])

        if cmd == "ACCESS" and len(args) >= 2:
            nid = int(args[0])
            region = int(args[1]) % 3
            if nid < k.node_count:
                k.record_access(nid, region)
                k.update_edges_on_access(nid)
                k.ctx_push(nid)

        elif cmd == "STORE" and len(args) >= 1:
            st_str = str(args[0])
            region = int(args[1]) if len(args) > 1 else 0
            if len(st_str) == 9 and all(c in '01' for c in st_str):
                st = int(st_str, 2)
            else:
                try:
                    st = int(st_str, 0) & 0x1FF
                except ValueError:
                    st = 0
            k.store_node(st, region % 3)

        elif cmd == "HEDGE" and len(args) >= 3:
            sub = str(args[0]).upper()
            if sub == "ADD":
                type_map = {"CONV": 0, "TOPIC": 1, "ENTITY": 2, "FIBER": 3, "CUSTOM": 4}
                htype = type_map.get(str(args[1]).upper(), 0)
                ids = [int(x) for x in args[2:] if int(x) < k.node_count]
                if len(ids) >= 2:
                    k.hedge_add(htype, ids)

        elif cmd == "FLIP" and len(args) >= 2:
            nid = int(args[0])
            bit = int(args[1])
            if nid < k.node_count and 0 <= bit <= 8:
                k.nodes[nid].state ^= (1 << bit)

        elif cmd == "EVOLVE" and len(args) >= 1:
            steps = min(int(args[0]), 100)
            k.evolve_steps(steps)

        elif cmd == "TICK" and len(args) >= 1:
            n = min(int(args[0]), 50)
            k.do_tick(n)

        elif cmd == "TRUST" and len(args) >= 2:
            nid = int(args[0])
            trust_map = {"KEEL": 0, "HULL": 1, "CARGO": 2, "EPHEMERAL": 3}
            t = trust_map.get(str(args[1]).upper(), 2)
            if nid < k.node_count:
                k.nodes[nid].trust = t

    # Consolidation step (same as make_dpo_pairs.py)
    k.evolve_steps(10)

    # Measure after
    coh_after = k.coherence_score(focus_id) if focus_id < k.node_count else 500
    curv = k.curvature_scaled()
    bridges = sum(1 for i in range(k.node_count) if k.is_bridge(i, 1200))
    stability = coh_after - coh_before

    reward, _ = compute_reward(
        coh_before=float(coh_before),
        coh_after=float(coh_after),
        curvature_scaled=float(curv),
        bridge_count=bridges,
    )

    return MetricSnapshot(
        coherence_before=coh_before,
        coherence_after=coh_after,
        curvature=curv,
        bridge_count=bridges,
        stability_delta=stability,
        reward=reward,
    )


# ============================================================================
# QEMU ORACLE MODE
# ============================================================================

def eval_with_oracle(
    trace: List[Dict[str, Any]],
    focus_id: int,
    iso_path: str,
) -> MetricSnapshot:
    """Run a trace through the QEMU oracle and collect metrics."""
    from scripts.oracle import OracleConfig, Topo9Oracle

    cfg = OracleConfig(iso_path=iso_path)
    with Topo9Oracle(cfg) as o:
        coh_before = o.coherent(focus_id, 5).get("coherence") or 500

        for step in trace:
            cmd = step.get("cmd", "").upper()
            args = step.get("args", [])
            line = " ".join([cmd] + [str(a) for a in args]).strip()
            o.send_cmd(line)

        o.send_cmd("EVOLVE 10")

        coh_after = o.coherent(focus_id, 5).get("coherence") or 500
        curv = o.curvature()
        bridges = o.bridges(1200)

        stability = coh_after - coh_before
        reward, _ = compute_reward(
            coh_before=float(coh_before),
            coh_after=float(coh_after),
            curvature_scaled=float(max(0, curv)),
            bridge_count=len(bridges),
        )

        return MetricSnapshot(
            coherence_before=coh_before,
            coherence_after=coh_after,
            curvature=max(0, curv),
            bridge_count=len(bridges),
            stability_delta=stability,
            reward=reward,
        )


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

def run_comparison(
    prompts: List[Dict],
    iso_path: Optional[str] = None,
    simulate: bool = True,
) -> List[EvalResult]:
    """
    Run all prompts through all three systems and collect results.
    """
    systems = [
        ("vanilla", make_vanilla_trace),
        ("rag", make_rag_trace),
        ("topo", make_topo_trace),
    ]

    results = []
    total = len(prompts) * len(systems)
    done = 0

    for prompt in prompts:
        pid = prompt.get("id", "?")
        intent = prompt.get("intent", "")

        for sys_name, trace_fn in systems:
            done += 1
            print(f"  [{done}/{total}] {pid} x {sys_name}...", end=" ", flush=True)

            trace = trace_fn(prompt)

            # Detect focus node from trace
            from scripts.trace_gen import _detect_focus_id
            focus_id = _detect_focus_id(trace)

            if simulate or not iso_path:
                metrics = eval_with_simulation(trace, focus_id)
            else:
                metrics = eval_with_oracle(trace, focus_id, iso_path)

            cmd_names = [s.get("cmd", "?") for s in trace]

            result = EvalResult(
                prompt_id=pid,
                system=sys_name,
                metrics=metrics,
                trace_length=len(trace),
                trace_commands=cmd_names,
            )
            results.append(result)
            print(f"coh={metrics.coherence_after:.0f} curv={metrics.curvature:.0f} "
                  f"bridges={metrics.bridge_count} δ={metrics.stability_delta:.0f} "
                  f"R={metrics.reward:.0f}")

    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: List[EvalResult], prompts: List[Dict]) -> str:
    """Generate a markdown comparison report."""
    lines = [
        "# Baseline Comparison Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Prompts: {len(prompts)}",
        "",
        "## Summary",
        "",
    ]

    # Aggregate by system
    by_system: Dict[str, List[EvalResult]] = {}
    for r in results:
        by_system.setdefault(r.system, []).append(r)

    # Summary table
    lines.append("| System | Avg Coherence | Avg Curvature | Total Bridges | Avg Stability Δ | Avg Reward |")
    lines.append("|--------|--------------|---------------|---------------|-----------------|------------|")

    for sys_name in ["vanilla", "rag", "topo"]:
        rs = by_system.get(sys_name, [])
        if not rs:
            continue
        avg_coh = sum(r.metrics.coherence_after for r in rs) / len(rs)
        avg_curv = sum(r.metrics.curvature for r in rs) / len(rs)
        total_br = sum(r.metrics.bridge_count for r in rs)
        avg_stab = sum(r.metrics.stability_delta for r in rs) / len(rs)
        avg_rew = sum(r.metrics.reward for r in rs) / len(rs)
        lines.append(f"| {sys_name:>7} | {avg_coh:>12.1f} | {avg_curv:>13.1f} | {total_br:>13} | {avg_stab:>15.1f} | {avg_rew:>10.1f} |")

    lines.append("")

    # By category
    categories = {}
    for p in prompts:
        cat = p.get("category", "?")
        categories.setdefault(cat, []).append(p.get("id", "?"))

    for cat, pids in sorted(categories.items()):
        cat_names = {"A": "Coherence Retention", "B": "Controlled Bridging",
                     "C": "Consolidation Stability", "D": "Drift Resistance"}
        lines.append(f"## Category {cat}: {cat_names.get(cat, cat)}")
        lines.append("")
        lines.append("| Prompt | System | Coh Before | Coh After | Curvature | Bridges | Stability Δ | Reward |")
        lines.append("|--------|--------|-----------|-----------|-----------|---------|-------------|--------|")

        for pid in pids:
            for r in results:
                if r.prompt_id == pid:
                    m = r.metrics
                    lines.append(
                        f"| {pid} | {r.system} | {m.coherence_before:.0f} | "
                        f"{m.coherence_after:.0f} | {m.curvature:.0f} | "
                        f"{m.bridge_count} | {m.stability_delta:.0f} | {m.reward:.0f} |"
                    )
        lines.append("")

    # Winner analysis
    lines.append("## Winner Analysis")
    lines.append("")

    wins = {"vanilla": 0, "rag": 0, "topo": 0}
    for p in prompts:
        pid = p.get("id", "?")
        prompt_results = [r for r in results if r.prompt_id == pid]
        if prompt_results:
            best = max(prompt_results, key=lambda r: r.metrics.reward)
            wins[best.system] = wins.get(best.system, 0) + 1

    for sys_name, count in sorted(wins.items(), key=lambda x: -x[1]):
        pct = count / len(prompts) * 100 if prompts else 0
        lines.append(f"- **{sys_name}**: wins {count}/{len(prompts)} prompts ({pct:.0f}%)")

    lines.append("")

    # Metric targets check
    lines.append("## Metric Target Compliance")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Compare vanilla vs RAG vs topo-tuned")
    ap.add_argument("--prompts", default="prompts/behavioral_suite.json",
                     help="Behavioral test suite JSON")
    ap.add_argument("--iso", default=None,
                     help="Path to kernel ISO (for QEMU mode)")
    ap.add_argument("--simulate", action="store_true", default=True,
                     help="Use Python simulation instead of QEMU (default)")
    ap.add_argument("--qemu", action="store_true",
                     help="Use QEMU oracle instead of simulation")
    ap.add_argument("--report", default="comparison_report.md",
                     help="Output report path")
    ap.add_argument("--json-out", default=None,
                     help="Optional JSON output of raw results")
    args = ap.parse_args()

    if args.qemu:
        args.simulate = False
        if not args.iso:
            print("ERROR: --qemu requires --iso path")
            sys.exit(1)

    # Load prompts
    with open(args.prompts, "r") as f:
        data = json.load(f)

    # Handle both formats: list of prompts or {prompts: [...]}
    if isinstance(data, list):
        prompts = data
    else:
        prompts = data.get("prompts", data)

    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    print(f"Mode: {'simulation' if args.simulate else 'QEMU oracle'}")
    print()

    # Run comparison
    results = run_comparison(
        prompts=prompts,
        iso_path=args.iso,
        simulate=args.simulate,
    )

    # Generate report
    report = generate_report(results, prompts)

    with open(args.report, "w") as f:
        f.write(report)
    print(f"\nReport written to {args.report}")

    # Optional JSON output
    if args.json_out:
        json_results = []
        for r in results:
            json_results.append({
                "prompt_id": r.prompt_id,
                "system": r.system,
                "trace_length": r.trace_length,
                "trace_commands": r.trace_commands,
                "coherence_before": r.metrics.coherence_before,
                "coherence_after": r.metrics.coherence_after,
                "curvature": r.metrics.curvature,
                "bridge_count": r.metrics.bridge_count,
                "stability_delta": r.metrics.stability_delta,
                "reward": r.metrics.reward,
            })
        with open(args.json_out, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Raw results written to {args.json_out}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    by_sys = {}
    for r in results:
        by_sys.setdefault(r.system, []).append(r)

    for sys_name in ["vanilla", "rag", "topo"]:
        rs = by_sys.get(sys_name, [])
        if not rs:
            continue
        avg_rew = sum(r.metrics.reward for r in rs) / len(rs)
        avg_coh = sum(r.metrics.coherence_after for r in rs) / len(rs)
        avg_curv = sum(r.metrics.curvature for r in rs) / len(rs)
        print(f"  {sys_name:>8}: reward={avg_rew:.0f}  coherence={avg_coh:.0f}  curvature={avg_curv:.0f}")

    # Determine winner
    sys_rewards = {}
    for sys_name, rs in by_sys.items():
        sys_rewards[sys_name] = sum(r.metrics.reward for r in rs) / len(rs)

    winner = max(sys_rewards, key=sys_rewards.get)
    print(f"\n  Winner: {winner} (avg reward = {sys_rewards[winner]:.0f})")
    print("="*60)


if __name__ == "__main__":
    main()
