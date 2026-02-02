# Topo9 Oracle Harness (next move for goal #1)

This folder is a *host-side* harness that turns your bootable Topo9 Hopf/Solenoid runtime
into a **training/evaluation oracle**.

It does 3 jobs:
1) Boots QEMU with a TCP serial port (machine-friendly).
2) Sends a candidate's "action trace" (STORE/ACCESS/HEDGE/EVOLVE/TICK...).
3) Reads back metrics (STATS/COHERENT/CURVATURE/BRIDGES) and computes a reward.

You can then generate **preference pairs** for DPO/IPO:
best candidate = chosen, worst candidate = rejected.

---

## Prereqs

- Build your kernel ISO from the earlier project:
  `topo9_hopf_solenoid_runtime.zip`
- Install tools:
  - qemu-system-i386
  - python 3.10+

---

## Run oracle smoke test

```bash
python scripts/oracle_smoke_test.py --iso /path/to/topo9_hopf_solenoid_runtime/build/topo9_hopf_solenoid.iso
```

(If your ISO name differs, pass the right path.)

---

## Recommended "interface contract" (important)

To score *text* answers, you need the model to also output a **structured action trace**.
Otherwise you're guessing what it "used" internally.

So, for each prompt, have your model generate:

- `final_answer`: the user-facing text
- `trace`: a JSON list of oracle commands

Example trace:

```json
[
  {"cmd":"STORE","args":["010101010","0"]},
  {"cmd":"STORE","args":["0x12D","2"]},
  {"cmd":"HEDGE","args":["ADD","TOPIC","0","1"]},
  {"cmd":"ACCESS","args":["0","0"]},
  {"cmd":"ACCESS","args":["1","2"]},
  {"cmd":"EVOLVE","args":["20"]},
  {"cmd":"CURVATURE","args":[]},
  {"cmd":"STATS","args":[]}
]
```

Then the oracle runs the trace and produces a reward + diagnostics.

---

## Dataset generation (DPO)

1) For each prompt, sample N candidates (N=4 is a good start).
2) Each candidate includes (final_answer, trace).
3) Oracle scores each candidate.
4) Write DPO JSONL records:
   {prompt, chosen, rejected, meta:{chosen_diag, rejected_diag}}

Use `scripts/make_dpo_pairs.py` to do steps 3-4 if you already have candidates.

---

## Evaluation (prove it's "epic")

Run a held-out suite of prompts and report:
- mean coherence score
- mean curvature (lower better)
- bridge rate (how often bridges appear when multi-region prompts)
- stability delta (coherence after EVOLVE - before)
- hyperedge health: #hedges, avg hedge size (from STATS + HEDGE LIST)

The scripts here output these as JSON so you can plot them.
