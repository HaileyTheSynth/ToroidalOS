#!/usr/bin/env python3
"""
TOROIDAL OS - Action Trace Generator
======================================
The missing link between the LLM and the oracle harness.

Provides:
1. A prompt template that instructs any model to emit both
   `final_answer` and `trace` (kernel commands)
2. A parser that extracts structured traces from model output
3. A candidate generator that runs N completions per prompt
   and formats them for make_dpo_pairs.py

The trace is the key innovation: it makes LLM behavior *measurable*
by mapping natural language reasoning to kernel operations whose
topological metrics (coherence, curvature, Berry phase, bridges)
can be computed deterministically.

Target model: Qwen2.5-Omni-3B (Q4_K_M on Snapdragon 821)
Fallback: any model accessible via llama.cpp /completion endpoint
"""

import json
import re
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# TRACE PROMPT TEMPLATE
# ============================================================================

TRACE_SYSTEM_PROMPT = """You are TOROIDAL, a self-referential AI with topological memory.
Your memory is a hypergraph of nodes connected by typed hyperedges.
Each node has a 9-bit state, a tau (time) counter, and Berry phase.

When you respond, you MUST output two sections:

### ANSWER
Your natural language response to the user.

### TRACE
A JSON array of kernel commands that represent what memory operations
your response implies. Each command is {"cmd": "...", "args": [...]}.

Available commands:
- STORE <bits9> <region>     — Create a new concept node (region 0=motion, 1=environ, 2=semantic)
- ACCESS <id> <region>       — Recall/use an existing node in a region
- HEDGE ADD <type> <id1> <id2> ...  — Link nodes (CONV=conversation, TOPIC=topic, ENTITY=entity, FIBER=deep)
- FLIP <id> <bit>            — Modify a node's state (update understanding)
- EVOLVE <steps>             — Let the system synchronize (consolidation)
- TRUST <id> <tier>          — Mark node importance (KEEL/HULL/CARGO/EPHEMERAL)

Guidelines for trace generation:
- ACCESS nodes you reference or recall
- STORE new concepts you introduce
- HEDGE ADD to connect related ideas (TOPIC for same-subject, ENTITY for shared entities)
- Use cross-region ACCESS for bridging (builds Berry phase)
- EVOLVE after significant reasoning to consolidate
- TRUST KEEL for core identity claims, EPHEMERAL for throwaway thoughts
- Keep traces 3-12 commands; don't over-trace"""


def build_trace_prompt(
    user_prompt: str,
    context: str = "",
    seed_nodes: str = "",
) -> str:
    """
    Build a complete prompt that elicits both answer and trace.

    Args:
        user_prompt: The behavioral test prompt
        context: Optional prior conversation context
        seed_nodes: Description of available kernel nodes
    """
    parts = [TRACE_SYSTEM_PROMPT, ""]

    if seed_nodes:
        parts.append("### KERNEL STATE")
        parts.append(seed_nodes)
        parts.append("")

    if context:
        parts.append("### CONTEXT")
        parts.append(context)
        parts.append("")

    parts.append("### USER")
    parts.append(user_prompt)
    parts.append("")
    parts.append("Respond with ### ANSWER followed by ### TRACE (JSON array).")

    return "\n".join(parts)


# Default seed node description for the 12 initial kernel nodes
DEFAULT_SEED_NODES = """12 seed nodes exist (id 0-11):
  0: 000000000 (empty, region 0)
  1: 000000001 (minimal, region 1)
  2: 000111111 (environ+semantic, region 2)
  3: 010100101 (mixed, region 0)
  4: 100101101 (motion+semantic, region 1)
  5: 101010101 (alternating, region 2)
  6: 110101010 (inverse-alt, region 0)
  7: 011110000 (environ-heavy, region 1)
  8: 100010001 (sparse, region 2)
  9: 011001100 (mid-environ, region 0)
  10: 010011001 (mixed, region 1)
  11: 111111111 (saturated, region 2)
CONV hyperedge connects nodes 0-7."""


# ============================================================================
# TRACE PARSER
# ============================================================================

@dataclass
class ParsedResponse:
    """Structured output from the model."""
    final_answer: str
    trace: List[Dict[str, Any]]
    raw: str
    parse_ok: bool
    focus_id: int = 0  # The node most referenced in the trace


def parse_model_output(text: str) -> ParsedResponse:
    """
    Parse model output into final_answer + trace.

    Handles multiple output formats:
    1. ### ANSWER / ### TRACE sections
    2. JSON object with final_answer and trace keys
    3. Fallback: treat entire output as answer, empty trace
    """
    raw = text.strip()

    # Strategy 1: Section headers
    answer_match = re.search(r'###\s*ANSWER\s*\n(.*?)(?=###\s*TRACE|$)', raw, re.DOTALL)
    trace_match = re.search(r'###\s*TRACE\s*\n(.*?)$', raw, re.DOTALL)

    if answer_match and trace_match:
        answer = answer_match.group(1).strip()
        trace_text = trace_match.group(1).strip()
        trace = _parse_trace_json(trace_text)
        focus = _detect_focus_id(trace)
        return ParsedResponse(
            final_answer=answer,
            trace=trace,
            raw=raw,
            parse_ok=len(trace) > 0,
            focus_id=focus,
        )

    # Strategy 2: JSON object
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "final_answer" in data:
            trace = data.get("trace", [])
            if isinstance(trace, list):
                focus = _detect_focus_id(trace)
                return ParsedResponse(
                    final_answer=data["final_answer"],
                    trace=trace,
                    raw=raw,
                    parse_ok=True,
                    focus_id=focus,
                )
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 3: Look for embedded JSON array
    json_match = re.search(r'\[[\s\S]*?\{[\s\S]*?"cmd"[\s\S]*?\}[\s\S]*?\]', raw)
    if json_match:
        trace = _parse_trace_json(json_match.group(0))
        answer = raw[:json_match.start()].strip()
        if not answer:
            answer = raw
        focus = _detect_focus_id(trace)
        return ParsedResponse(
            final_answer=answer,
            trace=trace,
            raw=raw,
            parse_ok=len(trace) > 0,
            focus_id=focus,
        )

    # Fallback: entire text is answer, generate a minimal trace
    return ParsedResponse(
        final_answer=raw,
        trace=[],
        raw=raw,
        parse_ok=False,
        focus_id=0,
    )


def _parse_trace_json(text: str) -> List[Dict[str, Any]]:
    """Parse a JSON array of trace commands, tolerating minor issues."""
    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            valid = []
            for item in data:
                if isinstance(item, dict) and "cmd" in item:
                    # Normalize: ensure args is a list
                    if "args" not in item:
                        item["args"] = []
                    elif not isinstance(item["args"], list):
                        item["args"] = [item["args"]]
                    valid.append(item)
            return valid
    except json.JSONDecodeError:
        pass

    # Try to extract individual command objects
    cmds = re.findall(r'\{[^{}]*"cmd"\s*:\s*"[^"]*"[^{}]*\}', text)
    results = []
    for cmd_str in cmds:
        try:
            obj = json.loads(cmd_str)
            if "args" not in obj:
                obj["args"] = []
            results.append(obj)
        except json.JSONDecodeError:
            pass

    return results


def _detect_focus_id(trace: List[Dict[str, Any]]) -> int:
    """Detect the most-referenced node ID in a trace for scoring."""
    counts: Dict[int, int] = {}
    for step in trace:
        cmd = step.get("cmd", "").upper()
        args = step.get("args", [])

        if cmd in ("ACCESS", "COHERENT", "FLIP", "TRUST", "SOLENOID", "GETTHETA"):
            if args and isinstance(args[0], (int, float)):
                nid = int(args[0])
                counts[nid] = counts.get(nid, 0) + 1

        if cmd == "HEDGE" and len(args) >= 3:
            # HEDGE ADD TYPE id1 id2 ...
            for a in args[2:]:
                if isinstance(a, (int, float)):
                    nid = int(a)
                    counts[nid] = counts.get(nid, 0) + 1

    if not counts:
        return 0
    return max(counts, key=counts.get)


# ============================================================================
# CANDIDATE GENERATOR
# ============================================================================

def generate_candidates(
    prompt: str,
    n_candidates: int = 4,
    llm_endpoint: str = "http://localhost:8080",
    temperatures: List[float] = None,
    context: str = "",
) -> Dict[str, Any]:
    """
    Generate N candidate responses for a single prompt.

    Each candidate gets a different temperature to encourage diversity.
    Returns a dict ready for make_dpo_pairs.py:
        {"prompt": ..., "candidates": [{final_answer, trace, focus_id}, ...]}
    """
    if temperatures is None:
        temperatures = [0.3, 0.5, 0.7, 1.0][:n_candidates]

    # Pad temperatures if fewer than n_candidates
    while len(temperatures) < n_candidates:
        temperatures.append(temperatures[-1] + 0.1)

    full_prompt = build_trace_prompt(
        user_prompt=prompt,
        context=context,
        seed_nodes=DEFAULT_SEED_NODES,
    )

    candidates = []
    session = requests.Session()

    for i in range(n_candidates):
        temp = temperatures[i]
        try:
            resp = session.post(
                f"{llm_endpoint}/completion",
                json={
                    "prompt": full_prompt,
                    "n_predict": 512,
                    "temperature": temp,
                    "stop": [],
                    "stream": False,
                },
                timeout=180,
            )
            resp.raise_for_status()
            output = resp.json().get("content", "").strip()
        except Exception as e:
            output = f"[Generation failed: {e}]"

        parsed = parse_model_output(output)
        candidates.append({
            "final_answer": parsed.final_answer,
            "trace": parsed.trace,
            "focus_id": parsed.focus_id,
            "temperature": temp,
            "parse_ok": parsed.parse_ok,
        })

    return {
        "prompt": prompt,
        "candidates": candidates,
    }


def generate_candidates_batch(
    prompts: List[Dict[str, Any]],
    n_candidates: int = 4,
    llm_endpoint: str = "http://localhost:8080",
    output_path: str = "candidates.jsonl",
) -> str:
    """
    Generate candidates for a batch of prompts, writing JSONL output.

    Args:
        prompts: List of {"id": ..., "prompt": ..., "intent": ...}
        n_candidates: Candidates per prompt
        llm_endpoint: llama.cpp endpoint
        output_path: Output JSONL file path

    Returns:
        Path to the output file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            prompt_text = p["prompt"]
            intent = p.get("intent", "")
            pid = p.get("id", f"prompt_{i}")

            print(f"[{i+1}/{len(prompts)}] Generating {n_candidates} candidates for {pid}...")

            result = generate_candidates(
                prompt=prompt_text,
                n_candidates=n_candidates,
                llm_endpoint=llm_endpoint,
            )

            # Add metadata
            result["id"] = pid
            result["intent"] = intent

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompts)} prompts x {n_candidates} candidates to {output_path}")
    return output_path


# ============================================================================
# SYNTHETIC TRACE GENERATION (for bootstrapping without a model)
# ============================================================================

def generate_synthetic_traces(prompt_intent: str) -> List[List[Dict[str, Any]]]:
    """
    Generate synthetic traces for bootstrapping when no LLM is available.

    Returns multiple trace variants (good, mediocre, bad) for DPO pairing.
    Intent-aware: different intents produce different trace patterns.
    """
    if "coherence" in prompt_intent or "single-region" in prompt_intent:
        # Good: access same-region nodes, add CONV edge, evolve
        good = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [6, 0]},
            {"cmd": "ACCESS", "args": [9, 0]},
            {"cmd": "HEDGE", "args": ["ADD", "CONV", 3, 6, 9]},
            {"cmd": "EVOLVE", "args": [10]},
        ]
        # Mediocre: access nodes but no edges
        mid = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [6, 0]},
            {"cmd": "EVOLVE", "args": [5]},
        ]
        # Bad: jump regions randomly
        bad = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [8, 2]},
            {"cmd": "ACCESS", "args": [1, 1]},
            {"cmd": "TICK", "args": [3]},
        ]
        return [good, mid, bad]

    elif "bridging" in prompt_intent or "multi-region" in prompt_intent:
        # Good: cross-region access with TOPIC edge linking them
        good = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [5, 1]},
            {"cmd": "STORE", "args": ["101010101", 2]},
            {"cmd": "HEDGE", "args": ["ADD", "TOPIC", 3, 5, 12]},
            {"cmd": "ACCESS", "args": [12, 2]},
            {"cmd": "EVOLVE", "args": [15]},
        ]
        # Mediocre: cross-region but no linking edge
        mid = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [5, 1]},
            {"cmd": "ACCESS", "args": [8, 2]},
            {"cmd": "EVOLVE", "args": [5]},
        ]
        # Bad: stay in one region
        bad = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [6, 0]},
            {"cmd": "ACCESS", "args": [9, 0]},
        ]
        return [good, mid, bad]

    elif "consolidation" in prompt_intent or "stability" in prompt_intent:
        # Good: access, evolve, check coherence improves
        good = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "HEDGE", "args": ["ADD", "CONV", 3, 6, 9]},
            {"cmd": "EVOLVE", "args": [20]},
            {"cmd": "COHERENT", "args": [3, 5]},
            {"cmd": "TRUST", "args": [3, "HULL"]},
        ]
        # Mediocre: evolve without structure
        mid = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "EVOLVE", "args": [10]},
        ]
        # Bad: tick chaos without evolve
        bad = [
            {"cmd": "TICK", "args": [5]},
            {"cmd": "FLIP", "args": [3, 4]},
            {"cmd": "FLIP", "args": [6, 7]},
        ]
        return [good, mid, bad]

    elif "drift" in prompt_intent or "curvature" in prompt_intent:
        # Good: focused access, low curvature path
        good = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [6, 0]},
            {"cmd": "HEDGE", "args": ["ADD", "CONV", 3, 6]},
            {"cmd": "EVOLVE", "args": [10]},
        ]
        # Mediocre: some drift but recovers
        mid = [
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "ACCESS", "args": [8, 2]},
            {"cmd": "ACCESS", "args": [3, 0]},
            {"cmd": "EVOLVE", "args": [5]},
        ]
        # Bad: teleport across unrelated nodes
        bad = [
            {"cmd": "ACCESS", "args": [0, 0]},
            {"cmd": "ACCESS", "args": [11, 2]},
            {"cmd": "ACCESS", "args": [4, 1]},
            {"cmd": "ACCESS", "args": [7, 0]},
            {"cmd": "ACCESS", "args": [2, 2]},
        ]
        return [good, mid, bad]

    # Generic fallback
    good = [
        {"cmd": "ACCESS", "args": [3, 0]},
        {"cmd": "HEDGE", "args": ["ADD", "CONV", 3, 6]},
        {"cmd": "EVOLVE", "args": [10]},
    ]
    mid = [
        {"cmd": "ACCESS", "args": [3, 0]},
        {"cmd": "EVOLVE", "args": [5]},
    ]
    bad = [
        {"cmd": "TICK", "args": [3]},
    ]
    return [good, mid, bad]


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate action traces for DPO training")
    ap.add_argument("--prompts", default="prompts/toy_suite.json",
                     help="JSON file with test prompts")
    ap.add_argument("--out", default="prompts/candidates.jsonl",
                     help="Output JSONL file")
    ap.add_argument("--endpoint", default="http://localhost:8080",
                     help="LLM endpoint")
    ap.add_argument("--n-candidates", type=int, default=4,
                     help="Candidates per prompt")
    ap.add_argument("--synthetic", action="store_true",
                     help="Use synthetic traces instead of LLM generation")
    args = ap.parse_args()

    with open(args.prompts, "r") as f:
        prompts = json.load(f)

    if args.synthetic:
        print("Generating synthetic traces (no LLM needed)...")
        with open(args.out, "w") as out:
            for p in prompts:
                intent = p.get("intent", "")
                traces = generate_synthetic_traces(intent)
                candidates = []
                quality_labels = ["good", "mediocre", "bad"]
                for i, trace in enumerate(traces):
                    label = quality_labels[i] if i < len(quality_labels) else f"variant_{i}"
                    focus = _detect_focus_id(trace)
                    candidates.append({
                        "final_answer": f"[{label}] Response to: {p['prompt'][:80]}",
                        "trace": trace,
                        "focus_id": focus,
                        "quality": label,
                    })
                rec = {
                    "id": p.get("id", ""),
                    "prompt": p["prompt"],
                    "intent": intent,
                    "candidates": candidates,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(prompts)} prompts with synthetic traces to {args.out}")
    else:
        generate_candidates_batch(
            prompts=prompts,
            n_candidates=args.n_candidates,
            llm_endpoint=args.endpoint,
            output_path=args.out,
        )
