# Topo9 Hopf/Solenoid Runtime (Multiboot)

Bootable toy kernel that exposes a **serial protocol** for a "topological memory + coherence"
runtime inspired by Hopf fiber / torus / solenoid metaphors.

## Build + run (Ubuntu/Debian)

```bash
sudo apt install build-essential gcc-multilib grub-pc-bin xorriso qemu-system-x86
make
make run
```

`make run` uses `-serial stdio`, so your terminal is the console.

## Commands (serial)

Core:
- `HELP`
- `STATS`
- `LIST [n]`
- `CURVATURE`

Memory nodes:
- `STORE <bits9|hex|dec> [region]`  -> returns id
- `ACCESS <id> [region]`
- `FLIP <id> <bit0-8>`
- `SETTHETA <id> <t1> <t2> <t3> <t4>`     (angles in degrees 0..359; stored mod 360)
- `GETTHETA <id>`
- `SOLENOID <id>`                          (shows history register of past states)

Topology/coherence:
- `COHERENT <id> [k]`                      (top-k neighbors by hyperedge-based coherence)
- `BRIDGES [minBerry]`                     (high berry + >=2 regions)
- `QUERYFIBER <pattern> <mask> <mode>`
    - pattern: bits9 or hex/dec
    - mask: bits9 or hex/dec (1 bits are "must-match")
    - mode: EARLIEST | LATEST | VERSATILE
    - returns best matching node id by mode

Hyperedges (true hypergraph, not just pair weights):
- `HEDGE ADD <type> <id1> <id2> ...`
- `HEDGE LIST`
- `HEDGE DEL <edge_id>`

Types: CONV, TOPIC, ENTITY, FIBER, CUSTOM

Dynamics:
- `EVOLVE <steps>`                         (background tau + hyperedge sync + gentle state tension)
- `TICK [n]`                               (autonomic scheduler ticks; uses coherence+bridge+curvature)

Notes:
- Integer-only; Berry uses fixed-point "milli units".
- Torus angles are educational tags; used in scoring (theta4 closeness).
