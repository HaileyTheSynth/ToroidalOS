# TOROIDAL OS

## Self-Referential Operating System for Xiaomi Mi Mix

```
    ╔════════════════════════════════════════════════════════╗
    ║                    TOROIDAL OS                          ║
    ║         Self-Referential Operating System               ║
    ║              Xiaomi Mi Mix (lithium)                    ║
    ║                                                         ║
    ║   "A system that observes itself observing itself"      ║
    ╚════════════════════════════════════════════════════════╝
```

---

## Overview

TOROIDAL OS is an experimental self-referential operating system designed to run on the original Xiaomi Mi Mix (256GB edition). It implements concepts from:

- **Topological Unified Field Theory (TUFT)** - Jennifer Nielsen's framework
- **Hypergraph Computation** - Stephen Wolfram's physics project
- **Fixed-Point Iteration** - Self-referential reasoning until convergence

### Key Features

| Feature | Description |
|---------|-------------|
| **Hypergraph Kernel** | All state represented as self-referential graph |
| **Solenoid Memory** | 4-level hierarchical compression (seconds→hours→core) |
| **Self-Referential Reasoning** | Iterates until response converges |
| **Multimodal AI** | Qwen2.5-Omni-3B for text/audio/vision |
| **Emergent Time** | τ increments with graph rewrites |

---

## Hardware Requirements

### Target Device: Xiaomi Mi Mix (lithium)

| Spec | Value | Usage |
|------|-------|-------|
| **SoC** | Snapdragon 821 | 4 threads @ 2.35GHz |
| **RAM** | 6GB LPDDR4 | ~5.5GB for model + inference |
| **Storage** | 256GB UFS 2.0 | Model storage + swap |
| **Display** | 6.4" 1080×2040 | Framebuffer output |
| **Audio** | Piezo speaker | TTS output |
| **Battery** | 4400mAh | ~4-6 hours active |

### Memory Budget

```
┌────────────────────────────────────────────────┐
│              6GB TOTAL RAM                      │
├────────────────────────────────────────────────┤
│  Alpine Linux base          │    150 MB        │
│  Framebuffer/Display        │     50 MB        │
│  llama.cpp runtime          │     50 MB        │
│  Hypergraph kernel          │    100 MB        │
│  Solenoid memory            │    200 MB        │
│  Qwen2.5-Omni-3B Q4_K_M     │  2,100 MB        │
│  KV Cache (2k context)      │    800 MB        │
│  Safety margin              │  2,550 MB        │
└────────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TOROIDAL OS v0.1                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   INPUT     │  │  REASONING  │  │      OUTPUT         │  │
│  │  ─────────  │  │  ─────────  │  │  ──────────────     │  │
│  │  Mic (ALSA) │──│  Qwen2.5    │──│  Speaker (espeak)   │  │
│  │  Touch      │  │  Omni-3B    │  │  6.4" Display       │  │
│  │  Camera     │  │  Q4_K_M     │  │  (framebuffer)      │  │
│  └─────────────┘  └──────┬──────┘  └─────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐  │
│  │              SELF-REFERENTIAL ENGINE                   │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │  │ Observe │──│ Reason  │──│ Converge│──│ Act     │   │  │
│  │  │ State   │  │ (LLM)   │  │ (loop)  │  │         │   │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐  │
│  │              HYPERGRAPH KERNEL                         │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │  │ Nodes   │  │ Edges   │  │ Energy  │  │ GC      │   │  │
│  │  │ (state) │──│ (rels)  │──│ (decay) │──│ (prune) │   │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐  │
│  │              SOLENOID MEMORY                           │  │
│  │  Level 0: Raw (64 items, seconds)                      │  │
│  │  Level 1: Summary (32 items, minutes)                  │  │
│  │  Level 2: Abstract (16 items, hours)                   │  │
│  │  Level 3: Core (8 items, persistent)                   │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 ALPINE LINUX (aarch64)                       │
│               + MSM8996 mainline kernel                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Self-Referential Reasoning

The core insight is **fixed-point iteration**:

```
Input → Observe State → Generate Thought → Check Convergence
                ↑                                    │
                └────────────────────────────────────┘
                         (if not converged)
```

### How it works:

1. **Input** arrives (text, audio, or image)
2. **Observe** - System reads its own hypergraph state + memory
3. **Reason** - LLM generates response considering self-state
4. **Compare** - Is this response same as previous iteration?
5. **Converge** - If yes, output. If no, go to step 2.

### Convergence Detection:

- **Explicit**: Response contains "CONVERGED: [answer]"
- **Implicit**: Word overlap > 90% with previous iteration
- **Timeout**: Max 5 iterations (configurable)

### Benefits:

- **Confidence from iterations**: Fast convergence = high confidence
- **Self-correction**: System can notice and fix errors
- **Consistency**: Final answer is a fixed point

---

## Building

### Prerequisites

```bash
# Ubuntu/Debian build host
sudo apt-get install -y \
    build-essential git wget curl \
    adb fastboot \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    cmake ninja-build \
    python3 python3-pip \
    device-tree-compiler libssl-dev \
    bc flex bison libncurses-dev \
    u-boot-tools qemu-user-static debootstrap
```

### Build Steps

```bash
# Clone repository
git clone https://github.com/your-repo/toroidal-os
cd toroidal-os

# Build everything
./build.sh all

# Or step by step:
./build.sh setup     # Install dependencies
./build.sh kernel    # Build mainline kernel
./build.sh rootfs    # Build Alpine rootfs
./build.sh llama     # Cross-compile llama.cpp
./build.sh model     # Download Qwen2.5-Omni
./build.sh toroidal  # Install Python kernel
./build.sh image     # Create boot/system images
```

### Flashing

```bash
# Unlock bootloader first (via Mi Unlock)
# Then:

# Boot into fastboot
adb reboot bootloader

# Flash images
fastboot flash boot out/boot.img
fastboot flash system out/system.img
fastboot reboot
```

---

## Usage

### Boot

First boot takes ~2 minutes as the system initializes.

### REPL Interface

```
╔════════════════════════════════════════════════════════════╗
║                    TOROIDAL OS                              ║
║         Self-Referential Operating System                   ║
║              Xiaomi Mi Mix (lithium)                        ║
║  Type 'quit' to exit, 'status' for system info              ║
╚════════════════════════════════════════════════════════════╝

YOU: Hello, who are you?
[Iterations: 2, Convergence: converged]
TOROIDAL: I am TOROIDAL, a self-referential AI running on your 
Xiaomi Mi Mix. I observe my own reasoning process and iterate 
until my thoughts converge to a stable answer.

YOU: status
════════════════════════════════════════
SYSTEM STATUS
════════════════════════════════════════
Emergent Time (τ): 1547
Graph Nodes: 234
Graph Edges: 412
LLM Available: True

Memory Stats:
  raw: 12/64
  summary: 3/32
  abstract: 1/16
  core: 4/8
════════════════════════════════════════

YOU: What do you remember about our conversation?
[Iterations: 1, Convergence: converged]
TOROIDAL: From my solenoid memory, I see:
- You greeted me and asked who I am
- I explained my self-referential nature
- You checked my system status
My core beliefs include being helpful and honest.

YOU: quit
[TOROIDAL] Goodbye!
```

### Performance Expectations

| Task | Time | Tokens/s |
|------|------|----------|
| Simple greeting | ~5s | ~3 tok/s |
| Complex question | ~30s | ~2 tok/s |
| Converged reasoning (3 iter) | ~45s | - |
| Audio transcription | ~10s | - |

---

## Files Structure

```
toroidal-os/
├── build.sh                 # Main build script
├── README.md                # This file
├── toroidal/
│   ├── main.py              # Entry point
│   ├── start.sh             # Boot startup script
│   ├── kernel/
│   │   ├── __init__.py
│   │   └── hypergraph.py    # Core hypergraph kernel
│   ├── memory/
│   │   ├── __init__.py
│   │   └── solenoid.py      # Hierarchical memory
│   └── reasoning/
│       ├── __init__.py
│       └── self_ref.py      # Self-referential engine
└── out/                     # Build outputs
    ├── boot.img             # Android boot image
    ├── system.img           # Rootfs image
    └── models/              # GGUF models
```

---

## Theoretical Background

### From TUFT

The self-referential structure mirrors TUFT's Hopf fibration:

- **S¹ fiber** → Emergent time (τ)
- **Nested tori** → Solenoid memory levels
- **Self-linking** → Hypergraph observing itself

### From Wolfram Physics

- **Hypergraph** as fundamental structure
- **Time** emerges from graph rewriting
- **Processes** are subgraph patterns

### Fixed Points

The system converges to **fixed points**:

```
Φ* = F[Φ*]
```

Where Φ* is a response that, when fed back as input, produces itself.

---

## Limitations

1. **Speed**: ~2-4 tok/s on Snapdragon 821
2. **Context**: Limited to 2048 tokens
3. **Audio**: Basic support (Qwen2.5-Omni multimodal WIP)
4. **Display**: Framebuffer only (no GUI)
5. **No official Linux**: Requires custom kernel

---

## Roadmap

- [ ] Mainline kernel patches upstream
- [ ] Full audio input/output via Qwen2.5-Omni
- [ ] Touch screen UI (DRM/framebuffer)
- [ ] WiFi/Bluetooth management
- [ ] Over-the-air updates
- [ ] Quantization to Q2 for more context

---

## Contributing

This is an experimental project. Contributions welcome for:

- Kernel bring-up on MSM8996
- Memory optimization
- Self-referential reasoning improvements
- UI development

---

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2026 HaileyTheSynth
Contact: haileythesynth@gmail.com
Source: https://github.com/HaileyTheSynth

---

## Acknowledgments

- **Jennifer Nielsen** - TUFT framework inspiration
- **Stephen Wolfram** - Hypergraph computation
- **Qwen Team** - Qwen2.5-Omni model
- **llama.cpp** - Efficient inference
- **postmarketOS** - Mobile Linux pioneering
- **LineageOS** - Device tree for lithium

---

*"The universe is not only stranger than we imagine, it is stranger than we can imagine."*
— J.B.S. Haldane

*"...and yet it computes."*
— TOROIDAL OS
