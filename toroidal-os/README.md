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
| **Voice I/O** | Speech-to-text and text-to-speech via Qwen2.5-Omni |
| **Connectivity Tools** | WiFi/Bluetooth via native Android/Linux stack |
| **Emergent Time** | τ increments with graph rewrites |
| **Embedding Layer** | Octen-Embedding-0.6B for semantic similarity |

---

## Hardware Requirements

### Target Device: Xiaomi Mi Mix (lithium)

| Spec | Value | Usage |
|------|-------|-------|
| **SoC** | Snapdragon 821 | 4 threads @ 2.35GHz |
| **RAM** | 6GB LPDDR4 | ~5.5GB for model + inference |
| **Storage** | 256GB UFS 2.0 | Model storage + swap |
| **Display** | 6.4" 1080×2040 | Framebuffer output |
| **Audio** | Piezo speaker + mic | Voice I/O |
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
│  Embedding cache            │    200 MB        │
│  Qwen2.5-Omni-3B Q4_K_M     │  2,100 MB        │
│  KV Cache (2k context)      │    800 MB        │
│  Safety margin              │  2,350 MB        │
└────────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOROIDAL OS v0.2                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌─────────────┐  ┌───────────────────────┐  │
│  │     INPUT     │  │  REASONING  │  │       OUTPUT          │  │
│  │  ───────────  │  │  ─────────  │  │  ─────────────────    │  │
│  │  Mic (ALSA)   │  │  Qwen2.5    │  │  Speaker (Qwen TTS)   │  │
│  │  Touch        │──│  Omni-3B    │──│  6.4" Display         │  │
│  │  Camera       │  │  + mmproj   │  │  (framebuffer)        │  │
│  └───────────────┘  └──────┬──────┘  └───────────────────────┘  │
│          │                 │              ▲                      │
│          ▼                 │              │                      │
│  ┌─────────────────────────┴──────────────┴───────────────────┐ │
│  │                 MULTIMODAL CLIENT                           │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │
│  │  │ STT (audio) │  │ Chat + TTS  │  │ Vision (future)     │ │ │
│  │  │ 16kHz WAV   │  │ multimodal  │  │ image understanding │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              SELF-REFERENTIAL ENGINE                       │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │  │
│  │  │ Observe │──│ Reason  │──│ Converge│──│ Act     │       │  │
│  │  │ State   │  │ (LLM)   │  │ (loop)  │  │         │       │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │  │
│  │                                                            │  │
│  │  [Tier 4: Desire Field → Dream Cycle → Online DPO]       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              HYPERGRAPH KERNEL                             │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │  │
│  │  │ Nodes   │  │ Edges   │  │ Energy  │  │ GC      │       │  │
│  │  │ (state) │──│ (rels)  │──│ (decay) │──│ (prune) │       │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │  │
│  │                                                            │  │
│  │  [Trust Tiers: KEEL > HULL > DECK > RIGGING]             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              SOLENOID MEMORY                               │  │
│  │  Level 0: Raw (64 items, seconds)                         │  │
│  │  Level 1: Summary (32 items, minutes)                     │  │
│  │  Level 2: Abstract (16 items, hours)                      │  │
│  │  Level 3: Core (8 items, persistent)                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              EMBEDDING LAYER (Octen-Embedding-0.6B)        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │  │
│  │  │ Encode text │  │ Similarity  │  │ Torus Mapping       │ │  │
│  │  │ 1024-dim    │──│ cosine sim  │──│ θ₁,θ₂,θ₃,θ₄ → node  │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              CONNECTIVITY (Native Android/Linux)           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │  │
│  │  │ WiFi        │  │ Bluetooth   │  │ Sensors (IIO)       │ │  │
│  │  │ wpa_supplicant│ │ BlueZ       │  │ accelerometer, etc  │ │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                 ALPINE LINUX (aarch64)                           │
│               + MSM8996 mainline kernel                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Voice I/O (Qwen2.5-Omni)

ToroidalOS supports voice interaction through Qwen2.5-Omni's multimodal capabilities.

### Audio Input Flow

```
Microphone (3s chunks, 16kHz)
        │
        ▼
┌───────────────────┐
│  AudioHandler     │  PyAudio capture
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  AudioProcessor   │  VAD (energy-based)
└─────────┬─────────┘
          │ speech detected
          ▼
┌───────────────────┐
│  MultimodalClient │  POST /completion
│  transcribe_audio │  multimodal_data: [base64]
└─────────┬─────────┘
          │
          ▼
    Transcribed text
          │
          ▼
┌───────────────────┐
│  ToroidalOS       │  Self-referential reasoning
│  .process(text)   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  ActionEngine     │  TTS output
│  .speak(response) │  (espeak fallback)
└───────────────────┘
```

### Usage

```bash
# With voice input enabled
python main.py --mmproj /opt/models/mmproj.gguf

# With voice input AND output
python main.py --mmproj /opt/models/mmproj.gguf --voice-output

# Text only (no audio)
python main.py --no-voice

# Test mode (no LLM server)
python main.py --test
```

---

## Tool System

ToroidalOS includes a structured tool system organized by region (Berry phase cost):

### Semantic Region (bits 6-8) - Cost: 30 milli-units

| Tool | Description |
|------|-------------|
| `memory_search` | Search solenoid memory for past experiences |
| `memory_store` | Store important observations in memory |
| `kernel_state` | Query topological invariants (coherence, curvature, Berry phase) |
| `kernel_coherence` | Get coherence score for a specific node |
| `graph_query` | Query hypergraph for nodes by type/pattern |

### Environ Region (bits 3-5) - Cost: 50 milli-units

| Tool | Description |
|------|-------------|
| `sensors` | Read current sensor state (9-bit fused) |
| `sensor_request` | Active query: battery, location, orientation, light, pressure |

### Motion Region (bits 0-2) - Cost: 110 milli-units

| Tool | Description |
|------|-------------|
| `shell` | Execute shell commands (sandboxed) |
| `notes_write` | Write persistent note (bypasses compression) |
| `notes_read` | Read persistent note by key |
| `wifi_status` | Get WiFi connection status |
| `wifi_scan` | Scan for available networks |
| `wifi_connect` | Connect to a WiFi network |
| `bluetooth_status` | Get Bluetooth adapter status |
| `bluetooth_scan` | Scan for Bluetooth devices |
| `bluetooth_connect` | Pair and connect to a device |
| `bluetooth_disconnect` | Disconnect from a device |

### Tool Calling Format

The LLM emits tool calls as JSON blocks:

```
<tool>{"name": "wifi_status", "args": {}}</tool>
<tool>{"name": "memory_store", "args": {"content": "User prefers dark mode", "importance": 1.5}}</tool>
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
- **Implicit**: Embedding cosine similarity > 0.9 (uses Octen-Embedding-0.6B)
- **Timeout**: Max 5 iterations (configurable via Online DPO)

### Benefits:

- **Confidence from iterations**: Fast convergence = high confidence
- **Self-correction**: System can notice and fix errors
- **Consistency**: Final answer is a fixed point

---

## Tier 4: Autonomy & Self-Improvement

ToroidalOS v0.2 includes advanced autonomy features:

### Desire Field

Tracks internal goals with Berry phase pressure:

```
DesireField:
  goals:
    - "Learn user preferences" (pressure: 0.75)
    - "Maintain coherence" (pressure: 0.60)
    - "Answer accurately" (pressure: 0.85)
  cycle_count: 42
  queued_thoughts: 3
```

### Dream Cycle

Periodic solenoid history clustering during idle:

```
DreamCycle:
  total_dreams: 15
  archetypes: 7
  cross_patterns: 23
  last_insight: "User often asks about technical details"
```

### Online DPO

Closed-loop training signal from kernel metrics:

```
OnlineDPO:
  experiences: 127
  preference_pairs: 23
  avg_reward: 0.72
  biases:
    coherence_weight: 1.05
    curvature_penalty: 0.82
    max_iterations_bias: 1.1
```

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
    python3 python3-pip python3-venv \
    device-tree-compiler libssl-dev \
    bc flex bison libncurses-dev \
    u-boot-tools qemu-user-static debootstrap \
    portaudio19-dev  # For PyAudio
```

### Python Dependencies

```bash
pip install requests pyaudio numpy
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
./build.sh model     # Download Qwen2.5-Omni + mmproj
./build.sh toroidal  # Install Python kernel
./build.sh image     # Create boot/system images
```

### Model Setup

```bash
# Download Qwen2.5-Omni-3B (quantized)
mkdir -p /opt/models
cd /opt/models

# GGUF model (Q4_K_M recommended for 6GB RAM)
wget https://huggingface.co/Qwen/Qwen2.5-Omni-3B-GGUF/resolve/main/qwen2.5-omni-3b-q4_k_m.gguf
mv qwen2.5-omni-3b-q4_k_m.gguf qwen2.5-omni-3b.gguf

# Multimodal projector (required for audio/vision)
wget https://huggingface.co/Qwen/Qwen2.5-Omni-3B-GGUF/resolve/main/mmproj.gguf
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

[TOROIDAL] Initializing...
[TOROIDAL] Embedding service initialized (lazy load)
[TOROIDAL] Ready. (Tier 4: desire field, dream cycle, online DPO active)

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
Embeddings: Available (Octen-Embedding-0.6B)

Topological State:
  Sensor bits: 0b000101010
  Coherence: 847
  Curvature: 23
  Berry phase: 330
  Is bridge: False
  Windings: 15

Epistemic State: KNOWLEDGE_GAP
  Web policy: ALLOW_WITH_CAUTION
  Confidence: 0.78

Tools: 15 registered (10 via topo://), 23 calls made

Memory Stats:
  raw: 12/64
  summary: 3/32
  abstract: 1/16
  core: 4/8

Desire Field (42 cycles, 3 queued):
  learn_user_prefs     [████████████████░░░░] 0.75 (active)
  maintain_coherence   [████████████░░░░░░░░] 0.60 (idle)
  answer_accurately    [█████████████████░░░] 0.85 (active)

Dream Cycle (15 dreams, 7 archetypes, 23 cross-patterns)

Online DPO (127 exp, 23 pairs, avg reward: 0.720):
  coherence_weight: 1.05
  curvature_penalty: 0.82
  max_iterations_bias: 1.10
════════════════════════════════════════

YOU: <tool>{"name": "wifi_status", "args": {}}</tool>
[TOOL RESULT] wifi_status:
  operstate: up
  ESSID: "HomeNetwork"
  Signal level: -45 dBm (excellent)

YOU: What do you remember about our conversation?
[Iterations: 1, Convergence: converged]
TOROIDAL: From my solenoid memory, I see:
- You greeted me and asked who I am
- I explained my self-referential nature
- You checked my system status
- You queried WiFi status (connected to HomeNetwork)
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
| Audio transcription (3s) | ~10s | - |
| Embedding similarity | ~50ms | - |

---

## Files Structure

```
toroidal-os/
├── build.sh                 # Main build script
├── README.md                # This file
├── toroidal/
│   ├── main.py              # Entry point + voice handling
│   ├── start.sh             # Boot startup script
│   ├── kernel/
│   │   ├── __init__.py
│   │   ├── hypergraph.py    # Core hypergraph kernel
│   │   └── tuft_integration.py  # TUFT invariants + KernelBridge
│   ├── memory/
│   │   ├── __init__.py
│   │   └── solenoid.py      # Hierarchical memory + LLM compression
│   ├── embeddings/
│   │   ├── __init__.py      # Public API
│   │   ├── service.py       # Octen-Embedding-0.6B service
│   │   ├── cache.py         # LRU cache with memory limits
│   │   ├── torus_mapper.py  # 4D torus coordinate mapping
│   │   └── utils.py         # Similarity, search utilities
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── self_ref.py      # Self-referential engine + ToroidalOS
│   │   ├── tools.py         # Tool dispatcher + built-in tools
│   │   ├── tools_ext.py     # WiFi/Bluetooth + extended tools
│   │   ├── topo_protocol.py # topo:// tool protocol
│   │   ├── epistemic.py     # Epistemic state detector
│   │   ├── desire.py        # Desire field (Tier 4)
│   │   ├── dream.py         # Dream cycle (Tier 4)
│   │   ├── online_dpo.py    # Online DPO (Tier 4)
│   │   └── multimodal.py    # Qwen2.5-Omni audio I/O
│   ├── hal/
│   │   └── lithium.py       # Hardware abstraction for Mi Mix
│   └── manifests/
│       └── tools.json       # Tool manifests for topo:// protocol
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
- **Berry phase** → Memory pressure metric

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
3. **Display**: Framebuffer only (no GUI)
4. **No official Linux**: Requires custom kernel
5. **Audio quality**: Basic 16kHz capture, espeak TTS fallback

---

## Roadmap

- [x] Mainline kernel patches upstream
- [x] Full audio input/output via Qwen2.5-Omni
- [ ] Touch screen UI (DRM/framebuffer)
- [x] WiFi/Bluetooth management (via native stack)
- [ ] Over-the-air updates
- [ ] Quantization to Q2 for more context
- [ ] Vision input (camera)

---

## Contributing

This is an experimental project. Contributions welcome for:

- Kernel bring-up on MSM8996
- Memory optimization
- Self-referential reasoning improvements
- UI development
- Vision multimodal integration

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
- **Octen AI** - Octen-Embedding-0.6B

---

*"The universe is not only stranger than we imagine, it is stranger than we can imagine."*
— J.B.S. Haldane

*"...and yet it computes."*
— TOROIDAL OS