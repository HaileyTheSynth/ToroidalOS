#!/usr/bin/env python3
"""
ToroidalOS Interactive Simulation
===================================
A terminal-based visual simulation of the Topo9 Hopf/Solenoid Runtime.
Faithfully implements the kernel's topological memory system with live
visualization of nodes, hyperedges, metrics, and the serial console.

Usage:
    python3 simulate.py
"""

import os
import sys
import time
import random
import shutil
import threading

# ═══════════════════════════════════════════════════════════════════════════
# ANSI HELPERS
# ═══════════════════════════════════════════════════════════════════════════

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
BLINK = "\033[5m"

BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

BRIGHT_BLACK = "\033[90m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def move_cursor(row, col):
    sys.stdout.write(f"\033[{row};{col}H")
    sys.stdout.flush()


def hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════
# KERNEL DATA STRUCTURES (faithful to kernel.c)
# ═══════════════════════════════════════════════════════════════════════════

SOL_H = 16
NMAX = 64
HEDGE_MAX = 48
HEDGE_MEM_MAX = 16
CTXK = 12

HEDGE_CONV = 0
HEDGE_TOPIC = 1
HEDGE_ENTITY = 2
HEDGE_FIBER = 3
HEDGE_CUSTOM = 4

HEDGE_NAMES = ["CONV", "TOPIC", "ENTITY", "FIBER", "CUSTOM"]

# Trust tiers
TRUST_KEEL = 0       # Core identity — immune to decay/GC
TRUST_HULL = 1       # Important context — slow decay
TRUST_CARGO = 2      # Normal working memory
TRUST_EPHEMERAL = 3  # Temporary — fast decay, GC priority
TRUST_NAMES = ["KEEL", "HULL", "CARGO", "EPHEMERAL"]


class Node:
    def __init__(self):
        self.used = False
        self.state = 0       # 9-bit
        self.tau = 0
        self.windings = 0
        self.berry_milli = 0
        self.last_region = 0
        self.region_mask = 0
        self.access_count = 0
        self.trust = TRUST_CARGO  # default trust tier
        self.th1 = 0
        self.th2 = 0
        self.th3 = 0
        self.th4 = 0
        self.sol_hist = [0] * SOL_H
        self.sol_len = 0


class Hyperedge:
    def __init__(self):
        self.used = False
        self.type = 0
        self.m = 0
        self.ids = [0] * HEDGE_MEM_MAX


class Kernel:
    """Faithful Python port of the Topo9 kernel.c runtime."""

    def __init__(self):
        self.nodes = [Node() for _ in range(NMAX)]
        self.node_count = 0
        self.edge_w = [[0] * NMAX for _ in range(NMAX)]
        self.hedges = [Hyperedge() for _ in range(HEDGE_MAX)]
        self.hedge_count = 0
        self.ctx_ring = [0] * CTXK
        self.ctx_len = 0
        self.ctx_head = 0

    # --- bit helpers ---
    @staticmethod
    def popcount9(s):
        c = 0
        for i in range(9):
            c += (s >> i) & 1
        return c

    @staticmethod
    def hamming9(a, b):
        return Kernel.popcount9((a ^ b) & 0x1FF)

    @staticmethod
    def region_of_bit(bit_index):
        if bit_index < 3:
            return 0
        if bit_index < 6:
            return 1
        return 2

    @staticmethod
    def mod360(x):
        return x % 360

    @staticmethod
    def ang_dist(a, b):
        d = abs(a - b)
        return 360 - d if d > 180 else d

    @staticmethod
    def bits9_str(s):
        return ''.join(str((s >> (8 - i)) & 1) for i in range(9))

    @staticmethod
    def region_count(mask):
        return ((mask & 1) != 0) + ((mask & 2) != 0) + ((mask & 4) != 0)

    # --- context ring ---
    def ctx_push(self, nid):
        self.ctx_ring[self.ctx_head] = nid
        self.ctx_head = (self.ctx_head + 1) % CTXK
        if self.ctx_len < CTXK:
            self.ctx_len += 1

    # --- solenoid ---
    def sol_push(self, n, st):
        for i in range(SOL_H - 1, 0, -1):
            n.sol_hist[i] = n.sol_hist[i - 1]
        n.sol_hist[0] = st & 0x1FF
        if n.sol_len < SOL_H:
            n.sol_len += 1

    # --- curvature ---
    def curvature_scaled(self):
        if self.ctx_len < 3:
            return 0
        acc = 0
        edges = 0
        for i in range(1, self.ctx_len):
            idxA = (self.ctx_head + CTXK - 1 - i) % CTXK
            idxB = (self.ctx_head + CTXK - 1 - (i - 1)) % CTXK
            a = self.nodes[self.ctx_ring[idxA]]
            b = self.nodes[self.ctx_ring[idxB]]
            acc += self.hamming9(a.state, b.state)
            edges += 1
        return (acc * 100) // edges if edges else 0

    # --- bridge ---
    def is_bridge(self, nid, min_berry):
        n = self.nodes[nid]
        return n.berry_milli >= min_berry and self.region_count(n.region_mask) >= 2

    # --- access ---
    def record_access(self, nid, region):
        n = self.nodes[nid]
        old = n.tau
        n.tau = (n.tau + 17) & 0xFF
        if n.tau < old:
            n.windings += 1
        add = 30 if region == n.last_region else 110
        n.berry_milli += add
        n.last_region = region
        n.region_mask |= 1 << region
        n.access_count += 1
        self.sol_push(n, n.state)
        n.th4 = (n.th4 + (3 if region == n.last_region else 9)) % 360

    def update_edges_on_access(self, nid):
        for i in range(self.ctx_len):
            idx = (self.ctx_head + CTXK - 1 - i) % CTXK
            other = self.ctx_ring[idx]
            if other == nid:
                continue
            if self.edge_w[nid][other] < 0xFFFF:
                self.edge_w[nid][other] += 1
            if self.edge_w[other][nid] < 0xFFFF:
                self.edge_w[other][nid] += 1

    # --- coherence ---
    def coherence_score(self, nid):
        a = self.nodes[nid]
        acc = 0
        cnt = 0
        for e in range(self.hedge_count):
            if not self.hedges[e].used:
                continue
            in_edge = any(self.hedges[e].ids[k] == nid for k in range(self.hedges[e].m))
            if not in_edge:
                continue
            w_type = {HEDGE_CONV: 3, HEDGE_TOPIC: 2, HEDGE_ENTITY: 2, HEDGE_FIBER: 2}.get(self.hedges[e].type, 1)
            for k in range(self.hedges[e].m):
                j = self.hedges[e].ids[k]
                if j == nid or not self.nodes[j].used:
                    continue
                b = self.nodes[j]
                dtau = abs(a.tau - b.tau)
                dh = self.hamming9(a.state, b.state)
                dth = self.ang_dist(a.th4, b.th4)
                local = (255 - dtau) + (9 - dh) * 20 + (180 - dth)
                acc += w_type * local
                cnt += w_type
        return acc // cnt if cnt else 500

    # --- store ---
    def store_node(self, st, region, trust=TRUST_CARGO):
        if self.node_count >= NMAX:
            return -1
        nid = self.node_count
        self.node_count += 1
        n = self.nodes[nid]
        n.used = True
        n.state = st & 0x1FF
        n.tau = (nid * 19) & 0xFF
        n.windings = 0
        n.berry_milli = 0
        n.last_region = region % 3
        n.region_mask = 1 << (region % 3)
        n.access_count = 0
        n.trust = trust & 3
        n.th1 = self.mod360(nid * 17)
        n.th2 = self.mod360(nid * 29)
        n.th3 = self.mod360(nid * 41)
        n.th4 = self.mod360(nid * 53)
        n.sol_len = 0
        n.sol_hist = [0] * SOL_H
        self.sol_push(n, n.state)
        return nid

    # --- seed ---
    def init_seed_nodes(self):
        seeds = [0x000, 0x001, 0x03F, 0x0A5, 0x12D, 0x155, 0x1AA, 0x0F0, 0x111, 0x0CC, 0x099, 0x1FF]
        for i, s in enumerate(seeds):
            self.store_node(s, i % 3)
        ids = list(range(min(self.node_count, 8)))
        self.hedge_add(HEDGE_CONV, ids)

    # --- hedge ---
    def hedge_add(self, htype, ids):
        m = len(ids)
        if m < 2 or m > HEDGE_MEM_MAX:
            return -1
        for i in range(HEDGE_MAX):
            if not self.hedges[i].used:
                self.hedges[i].used = True
                self.hedges[i].type = htype
                self.hedges[i].m = m
                for k in range(m):
                    self.hedges[i].ids[k] = ids[k]
                if i >= self.hedge_count:
                    self.hedge_count = i + 1
                return i
        return -1

    def hedge_del(self, eid):
        if eid >= HEDGE_MAX or not self.hedges[eid].used:
            return False
        self.hedges[eid].used = False
        return True

    # --- evolve ---
    def apply_hyperedge_sync(self):
        for e in range(self.hedge_count):
            if not self.hedges[e].used:
                continue
            total = 0
            m = 0
            for k in range(self.hedges[e].m):
                nid = self.hedges[e].ids[k]
                if not self.nodes[nid].used:
                    continue
                total += self.nodes[nid].tau
                m += 1
            if m < 2:
                continue
            mean = total // m
            for k in range(self.hedges[e].m):
                nid = self.hedges[e].ids[k]
                if not self.nodes[nid].used:
                    continue
                t = self.nodes[nid].tau
                if t < mean:
                    self.nodes[nid].tau = t + 1
                elif t > mean:
                    self.nodes[nid].tau = t - 1
            for i in range(self.hedges[e].m):
                for j in range(i + 1, self.hedges[e].m):
                    a = self.hedges[e].ids[i]
                    b = self.hedges[e].ids[j]
                    if not self.nodes[a].used or not self.nodes[b].used:
                        continue
                    dh = self.hamming9(self.nodes[a].state, self.nodes[b].state)
                    if dh <= 1:
                        continue
                    diff = (self.nodes[a].state ^ self.nodes[b].state) & 0x1FF
                    bit = 0
                    while bit < 9 and ((diff >> bit) & 1) == 0:
                        bit += 1
                    if bit >= 9:
                        continue
                    if self.nodes[a].berry_milli >= self.nodes[b].berry_milli:
                        self.nodes[a].state ^= 1 << bit
                    else:
                        self.nodes[b].state ^= 1 << bit

    def evolve_steps(self, steps):
        for _ in range(steps):
            for i in range(self.node_count):
                if not self.nodes[i].used:
                    continue
                old = self.nodes[i].tau
                self.nodes[i].tau = (self.nodes[i].tau + 3) & 0xFF
                if self.nodes[i].tau < old:
                    self.nodes[i].windings += 1
            self.apply_hyperedge_sync()

    # --- tick ---
    def scheduler_score(self, nid, curv_scaled, multi_region_mode, min_bridge_berry):
        coh = self.coherence_score(nid)
        w = self.popcount9(self.nodes[nid].state)
        activity = 200 if (w <= 1 or w >= 8) else 420
        bridge = 350 if (multi_region_mode and self.is_bridge(nid, min_bridge_berry)) else 0
        curv_pen = curv_scaled * 2
        return coh + activity + bridge - curv_pen

    def do_tick(self, n):
        results = []
        flip_bit = 0
        active_region = 0
        min_bridge_berry = 1200
        for t in range(n):
            curv = self.curvature_scaled()
            multi_region_mode = (t % 2) == 1
            best = 0
            best_s = 0
            for i in range(self.node_count):
                s = self.scheduler_score(i, curv, multi_region_mode, min_bridge_berry)
                if s >= best_s:
                    best_s = s
                    best = i
            self.nodes[best].state ^= (1 << (flip_bit % 9))
            flip_region = self.region_of_bit(flip_bit % 9)
            flip_bit += 1
            access_region = active_region if flip_region == active_region else (active_region + 1) % 3
            self.record_access(best, access_region)
            self.update_edges_on_access(best)
            self.ctx_push(best)
            results.append({
                "t": t, "id": best,
                "state": self.nodes[best].state,
                "tau": self.nodes[best].tau,
                "windings": self.nodes[best].windings,
                "berry": self.nodes[best].berry_milli,
                "th4": self.nodes[best].th4,
                "coh": self.coherence_score(best),
                "curv": curv,
            })
        return results

    # --- top neighbors ---
    def top_neighbors(self, nid, k):
        out = []
        for j in range(self.node_count):
            if not self.nodes[j].used or j == nid:
                continue
            connected = False
            for e in range(self.hedge_count):
                if not self.hedges[e].used:
                    continue
                has_i = any(self.hedges[e].ids[kk] == nid for kk in range(self.hedges[e].m))
                has_j = any(self.hedges[e].ids[kk] == j for kk in range(self.hedges[e].m))
                if has_i and has_j:
                    connected = True
                    break
            if not connected:
                continue
            a = self.nodes[nid]
            b = self.nodes[j]
            dtau = abs(a.tau - b.tau)
            dh = self.hamming9(a.state, b.state)
            dth = self.ang_dist(a.th4, b.th4)
            s = (255 - dtau) + (9 - dh) * 20 + (180 - dth) + self.edge_w[nid][j] * 2
            out.append((j, s, dh, dtau, dth))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    # --- fiber query ---
    def query_fiber(self, pattern, mask, mode):
        best = -1
        for i in range(self.node_count):
            if not self.nodes[i].used:
                continue
            if (self.nodes[i].state & mask) != (pattern & mask):
                continue
            if best < 0:
                best = i
                continue
            if mode == 0:  # EARLIEST
                if self.nodes[i].tau < self.nodes[best].tau:
                    best = i
            elif mode == 1:  # LATEST
                if self.nodes[i].tau > self.nodes[best].tau:
                    best = i
            else:  # VERSATILE
                if self.nodes[i].berry_milli > self.nodes[best].berry_milli:
                    best = i
        return best


# ═══════════════════════════════════════════════════════════════════════════
# VISUAL RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def bar(value, max_val, width, color=GREEN):
    """Render a colored progress bar."""
    if max_val == 0:
        filled = 0
    else:
        filled = min(width, int(value / max_val * width))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def spark(values, width=20):
    """Tiny sparkline."""
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    chars = "▁▂▃▄▅▆▇█"
    rng = mx - mn if mx != mn else 1
    out = ""
    for v in values[-width:]:
        idx = int((v - mn) / rng * (len(chars) - 1))
        out += chars[idx]
    return out.ljust(width)


def region_badge(mask):
    """Show region membership as colored badges."""
    r0 = f"{BG_BLUE} R0 {RESET}" if mask & 1 else f"{DIM} ·· {RESET}"
    r1 = f"{BG_MAGENTA} R1 {RESET}" if mask & 2 else f"{DIM} ·· {RESET}"
    r2 = f"{BG_CYAN} R2 {RESET}" if mask & 4 else f"{DIM} ·· {RESET}"
    return r0 + r1 + r2


def state_visual(state):
    """Render 9-bit state as colored blocks grouped by region."""
    out = ""
    colors = [BLUE, BLUE, BLUE, MAGENTA, MAGENTA, MAGENTA, CYAN, CYAN, CYAN]
    for i in range(9):
        bit = (state >> (8 - i)) & 1
        if i == 3 or i == 6:
            out += " "
        if bit:
            out += f"{colors[i]}█{RESET}"
        else:
            out += f"{BRIGHT_BLACK}·{RESET}"
    return out


# ═══════════════════════════════════════════════════════════════════════════
# BOOT SEQUENCE ANIMATION
# ═══════════════════════════════════════════════════════════════════════════

def boot_sequence():
    """Simulate the GRUB + kernel boot sequence."""
    clear_screen()

    # GRUB stage
    grub_lines = [
        f"{WHITE}{BG_BLACK}",
        f"                    GNU GRUB  version 2.06                     ",
        f"",
        f"  ┌──────────────────────────────────────────────────────────┐ ",
        f"  │  {BRIGHT_WHITE}*Topo9 Hopf/Solenoid Runtime{WHITE}                            │ ",
        f"  │                                                          │ ",
        f"  │                                                          │ ",
        f"  └──────────────────────────────────────────────────────────┘ ",
        f"",
        f"       Use the ▲ and ▼ keys to select which entry is        ",
        f"       highlighted. Press enter to boot the selected OS.     ",
    ]
    for line in grub_lines:
        print(line)
        time.sleep(0.05)

    time.sleep(1.0)
    clear_screen()

    # Kernel boot messages
    boot_msgs = [
        (BRIGHT_BLACK, "[    0.000000] ", WHITE, "Booting Topo9 Hopf/Solenoid Runtime..."),
        (BRIGHT_BLACK, "[    0.000012] ", WHITE, "Multiboot header validated"),
        (BRIGHT_BLACK, "[    0.000031] ", WHITE, "Serial port COM1 initialized at 38400 baud"),
        (BRIGHT_BLACK, "[    0.000045] ", GREEN, "VGA framebuffer: 80x25 @ 0xB8000"),
        (BRIGHT_BLACK, "[    0.000089] ", WHITE, "Memory: 64 node slots available"),
        (BRIGHT_BLACK, "[    0.000102] ", WHITE, "Hyperedge table: 48 slots, 16 members max"),
        (BRIGHT_BLACK, "[    0.000134] ", CYAN,  "Topology engine: 9-bit Hopf fibration"),
        (BRIGHT_BLACK, "[    0.000156] ", CYAN,  "Solenoid depth: 16-level history register"),
        (BRIGHT_BLACK, "[    0.000178] ", CYAN,  "Torus angles: θ₁ θ₂ θ₃ θ₄ (360° each)"),
        (BRIGHT_BLACK, "[    0.000201] ", YELLOW, "Seeding 12 initial nodes..."),
        (BRIGHT_BLACK, "[    0.000234] ", WHITE, "  node[0]  = 000000000  region=0  τ=0"),
        (BRIGHT_BLACK, "[    0.000256] ", WHITE, "  node[1]  = 000000001  region=1  τ=19"),
        (BRIGHT_BLACK, "[    0.000278] ", WHITE, "  node[2]  = 000111111  region=2  τ=38"),
        (BRIGHT_BLACK, "[    0.000301] ", WHITE, "  node[3]  = 010100101  region=0  τ=57"),
        (BRIGHT_BLACK, "[    0.000323] ", WHITE, "  node[4]  = 100101101  region=1  τ=76"),
        (BRIGHT_BLACK, "[    0.000345] ", WHITE, "  node[5]  = 101010101  region=2  τ=95"),
        (BRIGHT_BLACK, "[    0.000367] ", WHITE, "  node[6]  = 110101010  region=0  τ=114"),
        (BRIGHT_BLACK, "[    0.000389] ", WHITE, "  node[7]  = 011110000  region=1  τ=133"),
        (BRIGHT_BLACK, "[    0.000412] ", WHITE, "  node[8]  = 100010001  region=2  τ=152"),
        (BRIGHT_BLACK, "[    0.000434] ", WHITE, "  node[9]  = 011001100  region=0  τ=171"),
        (BRIGHT_BLACK, "[    0.000456] ", WHITE, "  node[10] = 010011001  region=1  τ=190"),
        (BRIGHT_BLACK, "[    0.000478] ", WHITE, "  node[11] = 111111111  region=2  τ=209"),
        (BRIGHT_BLACK, "[    0.000501] ", GREEN, "Seed CONV hyperedge: {0,1,2,3,4,5,6,7}"),
        (BRIGHT_BLACK, "[    0.000523] ", BRIGHT_GREEN, "Kernel ready. Entering serial console."),
    ]

    for parts in boot_msgs:
        line = ""
        for i in range(0, len(parts), 2):
            line += parts[i] + parts[i + 1]
        print(line + RESET)
        time.sleep(0.06)

    print()
    time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD VIEW
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard(k):
    """Render a full-screen dashboard of current kernel state."""
    cols = shutil.get_terminal_size().columns
    rows = shutil.get_terminal_size().lines

    clear_screen()

    # Header
    title = " TOPO9 HOPF/SOLENOID RUNTIME "
    pad = (cols - len(title) - 2) // 2
    print(f"{BG_BLUE}{BRIGHT_WHITE}{'═' * pad}{title}{'═' * (cols - pad - len(title))}{RESET}")

    # Stats bar
    hedges_used = sum(1 for e in range(k.hedge_count) if k.hedges[e].used)
    edges_nz = 0
    edge_sum = 0
    for i in range(k.node_count):
        for j in range(i + 1, k.node_count):
            if k.edge_w[i][j]:
                edges_nz += 1
                edge_sum += k.edge_w[i][j]
    avg_w = edge_sum // edges_nz if edges_nz else 0
    curv = k.curvature_scaled()

    print(f"  {BOLD}Nodes:{RESET} {BRIGHT_CYAN}{k.node_count}{RESET}/{NMAX}"
          f"  {BOLD}Hyperedges:{RESET} {BRIGHT_MAGENTA}{hedges_used}{RESET}/{HEDGE_MAX}"
          f"  {BOLD}Pair edges:{RESET} {BRIGHT_YELLOW}{edges_nz}{RESET} (avg w={avg_w})"
          f"  {BOLD}Context:{RESET} {k.ctx_len}/{CTXK}"
          f"  {BOLD}Curvature:{RESET} {curv}")
    print(f"  {DIM}{'─' * (cols - 4)}{RESET}")

    # Node table
    print(f"  {BOLD}{UNDERLINE}ID  State          τ    Wind  Berry    Regions       θ₄   Coh   Acc  Trust{RESET}")

    display_count = min(k.node_count, rows - 12)
    max_berry = max((k.nodes[i].berry_milli for i in range(k.node_count) if k.nodes[i].used), default=1) or 1

    for i in range(display_count):
        n = k.nodes[i]
        if not n.used:
            continue

        coh = k.coherence_score(i)
        is_br = k.is_bridge(i, 1200)

        id_col = f"{BRIGHT_YELLOW}" if is_br else ""
        bridge_mark = f" {BRIGHT_YELLOW}⚡{RESET}" if is_br else "  "

        berry_bar = bar(n.berry_milli, max_berry, 8, YELLOW)

        trust_colors = {TRUST_KEEL: BRIGHT_YELLOW, TRUST_HULL: BRIGHT_CYAN,
                        TRUST_CARGO: DIM, TRUST_EPHEMERAL: BRIGHT_BLACK}
        t_col = trust_colors.get(n.trust, DIM)
        t_name = TRUST_NAMES[n.trust] if n.trust < len(TRUST_NAMES) else "?"

        print(f"  {id_col}{i:>2}{RESET}  "
              f"{state_visual(n.state)}  "
              f"{BRIGHT_WHITE}{n.tau:>3}{RESET}  "
              f"{DIM}{n.windings:>4}{RESET}  "
              f"{n.berry_milli:>5} {berry_bar}  "
              f"{region_badge(n.region_mask)}  "
              f"{n.th4:>3}°  "
              f"{BRIGHT_GREEN if coh > 500 else BRIGHT_RED}{coh:>4}{RESET}  "
              f"{n.access_count:>3}  "
              f"{t_col}{t_name}{RESET}"
              f"{bridge_mark}")

    print(f"\n  {DIM}{'─' * (cols - 4)}{RESET}")

    # Hyperedge list
    print(f"  {BOLD}Hyperedges:{RESET}")
    for e in range(k.hedge_count):
        if not k.hedges[e].used:
            continue
        htype = HEDGE_NAMES[k.hedges[e].type] if k.hedges[e].type < len(HEDGE_NAMES) else "?"
        ids = [k.hedges[e].ids[kk] for kk in range(k.hedges[e].m)]
        type_color = {0: BLUE, 1: MAGENTA, 2: CYAN, 3: GREEN, 4: YELLOW}.get(k.hedges[e].type, WHITE)
        id_strs = ",".join(str(x) for x in ids)
        print(f"    {type_color}[{htype:>6}]{RESET} eid={e}  nodes={{ {id_strs} }}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# TORUS VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def render_torus_map(k):
    """Render nodes mapped onto a toroidal surface (θ₃ x θ₄)."""
    W, H = 60, 20
    grid = [[' '] * W for _ in range(H)]

    # Place nodes on the torus
    for i in range(k.node_count):
        n = k.nodes[i]
        if not n.used:
            continue
        x = int(n.th4 / 360 * (W - 1))
        y = int(n.th3 / 360 * (H - 1))
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))

        # Color by region
        rc = k.region_count(n.region_mask)
        if rc >= 3:
            ch = f"{BRIGHT_YELLOW}{i if i < 10 else chr(55 + i)}{RESET}"
        elif n.region_mask & 1:
            ch = f"{BLUE}{i if i < 10 else chr(55 + i)}{RESET}"
        elif n.region_mask & 2:
            ch = f"{MAGENTA}{i if i < 10 else chr(55 + i)}{RESET}"
        else:
            ch = f"{CYAN}{i if i < 10 else chr(55 + i)}{RESET}"
        grid[y][x] = ch

    print(f"  {BOLD}Torus Map (θ₃ × θ₄){RESET}  {DIM}Nodes projected onto toroidal surface{RESET}")
    print(f"  {DIM}θ₄ → 0°{'─' * (W - 16)}360°{RESET}")
    for row in range(H):
        label = f"{int(row / (H - 1) * 360):>3}°" if row % 5 == 0 else "    "
        cells = ''.join(grid[row])
        wrap = f"{DIM}│{RESET}"
        print(f"  {DIM}{label}{RESET}{wrap}{cells}{wrap}")
    print(f"  {DIM}{'↑ θ₃':<6}{'(wraps toroidally)':^{W}}{'':>{6}}{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SOLENOID VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def render_solenoid(k, nid):
    """Visualize a node's solenoid history as a state trace."""
    n = k.nodes[nid]
    print(f"  {BOLD}Solenoid History for Node {nid}{RESET}  {DIM}(most recent first){RESET}")
    print(f"  {DIM}depth  state       region bits{RESET}")
    for i in range(n.sol_len):
        st = n.sol_hist[i]
        depth_bar = f"{CYAN}{'▓' * (n.sol_len - i)}{DIM}{'░' * i}{RESET}"
        age = f"{BRIGHT_WHITE}now{RESET}" if i == 0 else f"{DIM}t-{i}{RESET}"
        print(f"  [{i:>2}]  {state_visual(st)}  {depth_bar}  {age}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE CONSOLE
# ═══════════════════════════════════════════════════════════════════════════

def format_output(text, color=WHITE):
    """Format kernel output."""
    return f"  {color}{text}{RESET}"


def run_console(k):
    """Run the interactive topo> console with visual feedback."""
    print(f"\n{BRIGHT_GREEN}Topo9 Hopf/Solenoid Runtime booted.{RESET}")
    print(f"{GREEN}Type HELP for commands, DASHBOARD for visual overview, QUIT to exit.{RESET}\n")

    history = []

    while True:
        try:
            raw = input(f"{BRIGHT_CYAN}topo> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = raw.strip()
        if not line:
            continue

        history.append(line)
        parts = line.split()
        cmd = parts[0].upper()
        args = parts[1:]

        # Extended visual commands
        if cmd == "QUIT" or cmd == "EXIT":
            print(f"\n{DIM}Shutting down Topo9 runtime...{RESET}")
            break

        elif cmd == "DASHBOARD" or cmd == "DASH":
            render_dashboard(k)

        elif cmd == "TORUS":
            render_torus_map(k)

        elif cmd == "VISUAL":
            render_dashboard(k)
            render_torus_map(k)

        elif cmd == "HELP":
            print(format_output("Commands:", BOLD))
            print(format_output("  HELP                                      Show this help", DIM))
            print(format_output("  STATS                                     System statistics", DIM))
            print(format_output("  LIST [n]                                  List first n nodes", DIM))
            print(format_output("  STORE <bits9|hex|dec> [region]             Store a new node", DIM))
            print(format_output("  ACCESS <id> [region]                      Access a node", DIM))
            print(format_output("  FLIP <id> <bit0-8>                        Flip a bit in node state", DIM))
            print(format_output("  SETTHETA <id> <t1> <t2> <t3> <t4>        Set torus angles", DIM))
            print(format_output("  GETTHETA <id>                             Get torus angles", DIM))
            print(format_output("  SOLENOID <id>                             Show solenoid history", DIM))
            print(format_output("  COHERENT <id> [k]                         Coherence + neighbors", DIM))
            print(format_output("  BRIDGES [minBerry]                        List bridge nodes", DIM))
            print(format_output("  QUERYFIBER <pat> <mask> <mode>            Fiber query", DIM))
            print(format_output("  HEDGE ADD <type> <id1> <id2> ...          Add hyperedge", DIM))
            print(format_output("  HEDGE LIST                                List hyperedges", DIM))
            print(format_output("  HEDGE DEL <eid>                           Delete hyperedge", DIM))
            print(format_output("  CURVATURE                                 Show curvature metric", DIM))
            print(format_output("  EVOLVE <steps> [TUFT]                     Evolve (sync) steps, TUFT uses Barnes-Hut", DIM))
            print(format_output("  TICK [n]                                  Scheduler ticks", DIM))
            print(format_output("  TRUST <id> [KEEL|HULL|CARGO|EPHEMERAL]     Get/set trust tier", DIM))
            print(format_output("  TUFTTICK [n]                              TUFT dynamics tick with Barnes-Hut forces", DIM))
            print(format_output("  TUFTSTATS                                 Show TUFT-specific statistics", DIM))
            print(format_output("  ─── Visual Commands ───", YELLOW))
            print(format_output("  DASHBOARD / DASH                          Full system dashboard", DIM))
            print(format_output("  TORUS                                     Torus map projection", DIM))
            print(format_output("  VISUAL                                    Dashboard + torus map", DIM))
            print(format_output("  QUIT / EXIT                               Exit simulation", DIM))

        elif cmd == "STATS":
            hedges_used = sum(1 for e in range(k.hedge_count) if k.hedges[e].used)
            edges_nz = 0
            edge_sum = 0
            for i in range(k.node_count):
                for j in range(i + 1, k.node_count):
                    if k.edge_w[i][j]:
                        edges_nz += 1
                        edge_sum += k.edge_w[i][j]
            avg_w = edge_sum // edges_nz if edges_nz else 0
            curv = k.curvature_scaled()
            print(format_output(f"STATS nodes={k.node_count} ctx_len={k.ctx_len} curvature={curv} "
                                f"pair_edges={edges_nz} avg_pair_w={avg_w} hedges={hedges_used}", BRIGHT_WHITE))

        elif cmd == "LIST":
            n = int(args[0]) if args else 16
            n = min(n, k.node_count, 32)
            for i in range(n):
                nd = k.nodes[i]
                if not nd.used:
                    continue
                t_name = TRUST_NAMES[nd.trust] if nd.trust < len(TRUST_NAMES) else "?"
                print(format_output(
                    f"id={i} s={k.bits9_str(nd.state)} tau={nd.tau} w={nd.windings} "
                    f"B={nd.berry_milli} R=0x{nd.region_mask:08X} th4={nd.th4} acc={nd.access_count} T={t_name}"))

        elif cmd == "STORE":
            if not args:
                print(format_output("ERR missing state", RED))
                continue
            st_str = args[0]
            region = int(args[1]) if len(args) > 1 else 0
            if len(st_str) == 9 and all(c in '01' for c in st_str):
                st = int(st_str, 2)
            else:
                try:
                    st = int(st_str, 0) & 0x1FF
                except ValueError:
                    print(format_output("ERR state must be bits9 or hex/dec", RED))
                    continue
            nid = k.store_node(st, region)
            if nid < 0:
                print(format_output("ERR node table full", RED))
            else:
                print(format_output(
                    f"OK id={nid} s={k.bits9_str(k.nodes[nid].state)} region={region}", GREEN))

        elif cmd == "ACCESS":
            if not args:
                print(format_output("ERR missing id", RED))
                continue
            try:
                nid = int(args[0])
            except ValueError:
                print(format_output("ERR invalid id", RED))
                continue
            if nid >= k.node_count:
                print(format_output("ERR invalid id", RED))
                continue
            region = int(args[1]) if len(args) > 1 else k.nodes[nid].last_region
            k.record_access(nid, region % 3)
            k.update_edges_on_access(nid)
            k.ctx_push(nid)
            n = k.nodes[nid]
            coh = k.coherence_score(nid)
            print(format_output(
                f"OK id={nid} tau={n.tau} w={n.windings} B={n.berry_milli} "
                f"R=0x{n.region_mask:08X} th4={n.th4} coh={coh}", GREEN))

        elif cmd == "FLIP":
            if len(args) < 2:
                print(format_output("ERR usage: FLIP <id> <bit0-8>", RED))
                continue
            try:
                nid = int(args[0])
                bit = int(args[1])
            except ValueError:
                print(format_output("ERR invalid args", RED))
                continue
            if nid >= k.node_count or bit > 8:
                print(format_output("ERR invalid id or bit", RED))
                continue
            k.nodes[nid].state ^= 1 << bit
            print(format_output(f"OK id={nid} s={k.bits9_str(k.nodes[nid].state)}", GREEN))

        elif cmd == "SETTHETA":
            if len(args) < 5:
                print(format_output("ERR usage: SETTHETA <id> <t1> <t2> <t3> <t4>", RED))
                continue
            try:
                nid = int(args[0])
                t1, t2, t3, t4 = int(args[1]), int(args[2]), int(args[3]), int(args[4])
            except ValueError:
                print(format_output("ERR invalid args", RED))
                continue
            if nid >= k.node_count:
                print(format_output("ERR invalid id", RED))
                continue
            k.nodes[nid].th1 = t1 % 360
            k.nodes[nid].th2 = t2 % 360
            k.nodes[nid].th3 = t3 % 360
            k.nodes[nid].th4 = t4 % 360
            n = k.nodes[nid]
            print(format_output(f"OK id={nid} th={n.th1},{n.th2},{n.th3},{n.th4}", GREEN))

        elif cmd == "GETTHETA":
            if not args:
                print(format_output("ERR missing id", RED))
                continue
            nid = int(args[0])
            if nid >= k.node_count:
                print(format_output("ERR invalid id", RED))
                continue
            n = k.nodes[nid]
            print(format_output(f"THETA id={nid} th={n.th1},{n.th2},{n.th3},{n.th4}"))

        elif cmd == "SOLENOID":
            if not args:
                print(format_output("ERR missing id", RED))
                continue
            nid = int(args[0])
            if nid >= k.node_count:
                print(format_output("ERR invalid id", RED))
                continue
            render_solenoid(k, nid)

        elif cmd == "COHERENT":
            if not args:
                print(format_output("ERR missing id", RED))
                continue
            nid = int(args[0])
            if nid >= k.node_count:
                print(format_output("ERR invalid id", RED))
                continue
            kk = int(args[1]) if len(args) > 1 else 5
            coh = k.coherence_score(nid)
            print(format_output(f"COHERENT id={nid} coh={coh} topN:", BRIGHT_WHITE))
            neighbors = k.top_neighbors(nid, kk)
            for j, s, dh, dtau, dth in neighbors:
                print(format_output(f"  n={j} s={s} dh={dh} dtau={dtau} dth4={dth}"))

        elif cmd == "BRIDGES":
            minB = int(args[0]) if args else 1200
            print(format_output(f"BRIDGES minB={minB}", BRIGHT_WHITE))
            count = 0
            for i in range(k.node_count):
                if k.is_bridge(i, minB):
                    count += 1
                    n = k.nodes[i]
                    print(format_output(
                        f"  {BRIGHT_YELLOW}⚡{RESET} id={i} s={k.bits9_str(n.state)} "
                        f"B={n.berry_milli} R=0x{n.region_mask:08X} th4={n.th4} w={n.windings}"))
            if not count:
                print(format_output("  (none)", DIM))

        elif cmd == "QUERYFIBER":
            if len(args) < 3:
                print(format_output("ERR usage: QUERYFIBER <pattern> <mask> <EARLIEST|LATEST|VERSATILE>", RED))
                continue
            ptn_s, msk_s, mode_s = args[0], args[1], args[2].upper()
            try:
                if len(ptn_s) == 9 and all(c in '01' for c in ptn_s):
                    ptn = int(ptn_s, 2)
                else:
                    ptn = int(ptn_s, 0) & 0x1FF
                if len(msk_s) == 9 and all(c in '01' for c in msk_s):
                    msk = int(msk_s, 2)
                else:
                    msk = int(msk_s, 0) & 0x1FF
            except ValueError:
                print(format_output("ERR invalid pattern/mask", RED))
                continue
            mode = {"EARLIEST": 0, "LATEST": 1, "VERSATILE": 2}.get(mode_s, -1)
            if mode < 0:
                print(format_output("ERR mode must be EARLIEST|LATEST|VERSATILE", RED))
                continue
            best = k.query_fiber(ptn, msk, mode)
            if best < 0:
                print(format_output("OK none", YELLOW))
            else:
                n = k.nodes[best]
                print(format_output(
                    f"OK id={best} s={k.bits9_str(n.state)} tau={n.tau} B={n.berry_milli}", GREEN))

        elif cmd == "HEDGE":
            if not args:
                print(format_output("ERR usage: HEDGE LIST | ADD | DEL", RED))
                continue
            sub = args[0].upper()
            if sub == "LIST":
                print(format_output("HEDGES:", BRIGHT_WHITE))
                for e in range(k.hedge_count):
                    if not k.hedges[e].used:
                        continue
                    htype = HEDGE_NAMES[k.hedges[e].type] if k.hedges[e].type < len(HEDGE_NAMES) else "?"
                    ids = [k.hedges[e].ids[kk] for kk in range(k.hedges[e].m)]
                    type_color = {0: BLUE, 1: MAGENTA, 2: CYAN, 3: GREEN, 4: YELLOW}.get(k.hedges[e].type, WHITE)
                    print(format_output(f"  eid={e} {type_color}type={htype}{RESET} m={k.hedges[e].m} ids={','.join(str(x) for x in ids)}"))
            elif sub == "ADD":
                if len(args) < 4:
                    print(format_output("ERR usage: HEDGE ADD <type> <id1> <id2> ...", RED))
                    continue
                type_map = {"CONV": 0, "TOPIC": 1, "ENTITY": 2, "FIBER": 3, "CUSTOM": 4}
                htype = type_map.get(args[1].upper(), -1)
                if htype < 0:
                    print(format_output("ERR type must be CONV|TOPIC|ENTITY|FIBER|CUSTOM", RED))
                    continue
                try:
                    ids = [int(x) for x in args[2:]]
                except ValueError:
                    print(format_output("ERR invalid node ids", RED))
                    continue
                eid = k.hedge_add(htype, ids)
                if eid < 0:
                    print(format_output("ERR could not add edge (full?)", RED))
                else:
                    print(format_output(f"OK eid={eid}", GREEN))
            elif sub == "DEL":
                if len(args) < 2:
                    print(format_output("ERR usage: HEDGE DEL <eid>", RED))
                    continue
                eid = int(args[1])
                if k.hedge_del(eid):
                    print(format_output("OK", GREEN))
                else:
                    print(format_output("ERR no such edge", RED))
            else:
                print(format_output("ERR usage: HEDGE LIST | ADD | DEL", RED))

        elif cmd == "CURVATURE":
            curv = k.curvature_scaled()
            print(format_output(f"CURVATURE scaled={curv}", BRIGHT_WHITE))
            # Visual bar
            print(format_output(f"  {bar(curv, 900, 40, YELLOW)}  {curv}/900"))

        elif cmd == "EVOLVE":
            if not args:
                print(format_output("ERR usage: EVOLVE <steps> [TUFT]", RED))
                continue
            steps = int(args[0])
            if steps <= 0:
                print(format_output("ERR steps must be > 0", RED))
                continue
            use_tuft = len(args) > 1 and args[1].upper() == "TUFT"
            print(format_output(f"Evolving {steps} steps{' (TUFT mode)' if use_tuft else ''}...", DIM))
            if use_tuft:
                # Lazy import to avoid circular dependency
                try:
                    from topo9_tuft_dynamics import Topo9TUFT
                    # Convert kernel to TUFT kernel
                    tuft_k = Topo9TUFT()
                    tuft_k.nodes = k.nodes
                    tuft_k.node_count = k.node_count
                    tuft_k.hedges = k.hedges
                    tuft_k.hedge_count = k.hedge_count
                    tuft_k.edge_w = k.edge_w
                    tuft_k.ctx_ring = k.ctx_ring
                    tuft_k.ctx_len = k.ctx_len
                    tuft_k.ctx_head = k.ctx_head
                    tuft_k.tuft_evolve_step(steps)
                    k.nodes = tuft_k.nodes
                    print(format_output(f"OK TUFT evolved steps={steps}", GREEN))
                except ImportError as e:
                    print(format_output(f"ERR TUFT mode unavailable: {e}", RED))
                    k.evolve_steps(steps)
                    print(format_output(f"OK evolved steps={steps}", GREEN))
            else:
                k.evolve_steps(steps)
                print(format_output(f"OK evolved steps={steps}", GREEN))

        elif cmd == "TRUST":
            if not args:
                print(format_output("ERR usage: TRUST <id> [KEEL|HULL|CARGO|EPHEMERAL]", RED))
                continue
            nid = int(args[0])
            if nid >= k.node_count:
                print(format_output("ERR invalid id", RED))
                continue
            if len(args) > 1:
                trust_map = {"KEEL": TRUST_KEEL, "HULL": TRUST_HULL,
                             "CARGO": TRUST_CARGO, "EPHEMERAL": TRUST_EPHEMERAL}
                t = trust_map.get(args[1].upper(), -1)
                if t < 0:
                    print(format_output("ERR trust must be KEEL|HULL|CARGO|EPHEMERAL", RED))
                    continue
                k.nodes[nid].trust = t
            t_name = TRUST_NAMES[k.nodes[nid].trust]
            print(format_output(f"OK id={nid} trust={t_name}", GREEN))

        elif cmd == "TICK":
            n = int(args[0]) if args else 1
            n = max(1, min(n, 1000))
            print(format_output(f"Running {n} scheduler tick(s)...", DIM))
            results = k.do_tick(n)
            for r in results:
                coh_color = BRIGHT_GREEN if r["coh"] > 500 else BRIGHT_RED
                print(format_output(
                    f"TICK t={r['t']} id={r['id']} "
                    f"s={k.bits9_str(r['state'])} "
                    f"tau={r['tau']} w={r['windings']} "
                    f"B={r['berry']} th4={r['th4']} "
                    f"{coh_color}coh={r['coh']}{RESET} curv={r['curv']}"))

        elif cmd == "TUFTTICK":
            n = int(args[0]) if args else 1
            n = max(1, min(n, 100))
            print(format_output(f"Running {n} TUFT tick(s) with Barnes-Hut forces...", DIM))
            try:
                from topo9_tuft_dynamics import Topo9TUFT
                tuft_k = Topo9TUFT()
                tuft_k.nodes = k.nodes
                tuft_k.node_count = k.node_count
                tuft_k.hedges = k.hedges
                tuft_k.hedge_count = k.hedge_count
                tuft_k.edge_w = k.edge_w
                tuft_k.ctx_ring = k.ctx_ring
                tuft_k.ctx_len = k.ctx_len
                tuft_k.ctx_head = k.ctx_head
                results = tuft_k.tuft_tick(n)
                k.nodes = tuft_k.nodes
                for r in results:
                    coh_color = BRIGHT_GREEN if r["coh"] > 500 else BRIGHT_RED
                    vel_mag = sum(v**2 for v in r["velocity"])**0.5
                    print(format_output(
                        f"TUFTTICK t={r['t']} id={r['id']} "
                        f"th4={r['th4']:.1f}° w={r['windings']} "
                        f"B={r['berry']} vel={vel_mag:.3f} "
                        f"{coh_color}coh={r['coh']}{RESET}"))
                print(format_output(f"OK TUFT tick completed", GREEN))
            except ImportError as e:
                print(format_output(f"ERR TUFT unavailable: {e}", RED))

        elif cmd == "TUFTSTATS":
            try:
                from topo9_tuft_dynamics import Topo9TUFT
                tuft_k = Topo9TUFT()
                tuft_k.nodes = k.nodes
                tuft_k.node_count = k.node_count
                tuft_k.hedges = k.hedges
                tuft_k.hedge_count = k.hedge_count
                tuft_k.edge_w = k.edge_w
                tuft_k.ctx_ring = k.ctx_ring
                tuft_k.ctx_len = k.ctx_len
                tuft_k.ctx_head = k.ctx_head
                tuft_k.update_entropy_field()
                tuft_k.build_barnes_hut_tree()
                tuft_k.compute_wilson_loops()
                stats = tuft_k.get_tuft_stats()
                print(format_output("TUFT STATISTICS:", BRIGHT_CYAN))
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(format_output(f"  {key}: {value:.4f}", DIM))
                    else:
                        print(format_output(f"  {key}: {value}", DIM))
            except ImportError as e:
                print(format_output(f"ERR TUFT unavailable: {e}", RED))

        else:
            print(format_output(f"ERR unknown command '{cmd}' (try HELP)", RED))


# ═══════════════════════════════════════════════════════════════════════════
# DEMO SCENARIO
# ═══════════════════════════════════════════════════════════════════════════

def run_demo(k):
    """Run a scripted demo showing the system in action."""
    cols = shutil.get_terminal_size().columns

    def narrate(text):
        print(f"\n  {BRIGHT_YELLOW}▸ {text}{RESET}")
        time.sleep(0.8)

    def cmd_echo(cmd_str):
        print(f"  {BRIGHT_CYAN}topo> {RESET}{cmd_str}")
        time.sleep(0.3)

    print(f"\n{'═' * cols}")
    print(f"{BOLD}{BRIGHT_WHITE}  TOROIDAL OS — INTERACTIVE DEMO{RESET}")
    print(f"{DIM}  Demonstrating topological memory with measurable metrics{RESET}")
    print(f"{'═' * cols}\n")
    time.sleep(1)

    # Show initial state
    narrate("System booted with 12 seed nodes and 1 CONV hyperedge.")
    narrate("Let's see the initial state...")
    render_dashboard(k)
    time.sleep(1.5)

    # Store a new node
    narrate("Storing a new node with state 101010101 (alternating bits) in region 1...")
    cmd_echo("STORE 101010101 1")
    nid = k.store_node(0b101010101, 1)
    print(format_output(f"OK id={nid} s={k.bits9_str(k.nodes[nid].state)} region=1", GREEN))
    time.sleep(0.5)

    # Create topic hyperedge
    narrate("Creating a TOPIC hyperedge connecting nodes 0, 3, 5, and our new node...")
    cmd_echo(f"HEDGE ADD TOPIC 0 3 5 {nid}")
    eid = k.hedge_add(HEDGE_TOPIC, [0, 3, 5, nid])
    print(format_output(f"OK eid={eid}", GREEN))
    time.sleep(0.5)

    # Access pattern
    narrate("Simulating cross-region access pattern to build Berry phase...")
    for region in [0, 1, 2, 0, 2, 1]:
        cmd_echo(f"ACCESS {nid} {region}")
        k.record_access(nid, region)
        k.update_edges_on_access(nid)
        k.ctx_push(nid)
        n = k.nodes[nid]
        coh = k.coherence_score(nid)
        print(format_output(
            f"OK id={nid} tau={n.tau} w={n.windings} B={n.berry_milli} "
            f"R=0x{n.region_mask:08X} th4={n.th4} coh={coh}", GREEN))
        time.sleep(0.15)

    time.sleep(0.5)

    # Check coherence
    narrate(f"Checking coherence neighborhood of node {nid}...")
    cmd_echo(f"COHERENT {nid}")
    coh = k.coherence_score(nid)
    print(format_output(f"COHERENT id={nid} coh={coh} topN:", BRIGHT_WHITE))
    neighbors = k.top_neighbors(nid, 5)
    for j, s, dh, dtau, dth in neighbors:
        print(format_output(f"  n={j} s={s} dh={dh} dtau={dtau} dth4={dth}"))
    time.sleep(0.5)

    # Evolve
    narrate("Running 20 evolution steps (tau sync + state tension resolution)...")
    cmd_echo("EVOLVE 20")
    k.evolve_steps(20)
    print(format_output("OK evolved steps=20", GREEN))
    time.sleep(0.5)

    # Tick
    narrate("Running 5 scheduler ticks (autonomous activity selection)...")
    cmd_echo("TICK 5")
    results = k.do_tick(5)
    for r in results:
        coh_color = BRIGHT_GREEN if r["coh"] > 500 else BRIGHT_RED
        print(format_output(
            f"TICK t={r['t']} id={r['id']} "
            f"s={k.bits9_str(r['state'])} "
            f"tau={r['tau']} w={r['windings']} "
            f"B={r['berry']} th4={r['th4']} "
            f"{coh_color}coh={r['coh']}{RESET} curv={r['curv']}"))
        time.sleep(0.1)
    time.sleep(0.5)

    # Show solenoid
    narrate(f"Viewing solenoid (state history) of node {nid}...")
    cmd_echo(f"SOLENOID {nid}")
    render_solenoid(k, nid)
    time.sleep(0.5)

    # Check bridges
    narrate("Looking for bridge nodes (high Berry phase, multi-region)...")
    cmd_echo("BRIDGES 100")
    count = 0
    for i in range(k.node_count):
        if k.is_bridge(i, 100):
            count += 1
            n = k.nodes[i]
            print(format_output(
                f"  {BRIGHT_YELLOW}⚡{RESET} id={i} s={k.bits9_str(n.state)} "
                f"B={n.berry_milli} R=0x{n.region_mask:08X}"))
    if not count:
        print(format_output("  (none yet — need more cross-region access)", DIM))
    time.sleep(0.5)

    # Torus map
    narrate("Projecting nodes onto toroidal surface...")
    render_torus_map(k)
    time.sleep(0.5)

    # Final dashboard
    narrate("Final system state after all operations:")
    render_dashboard(k)

    print(f"\n{'═' * cols}")
    print(f"{BOLD}{BRIGHT_GREEN}  Demo complete!{RESET} Dropping into interactive console...")
    print(f"{DIM}  Type HELP for commands, DASHBOARD for overview, QUIT to exit.{RESET}")
    print(f"{'═' * cols}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Initialize kernel
    k = Kernel()
    k.init_seed_nodes()

    # Parse args
    skip_boot = "--no-boot" in sys.argv
    skip_demo = "--no-demo" in sys.argv
    demo_only = "--demo" in sys.argv

    if not skip_boot:
        boot_sequence()

    if not skip_demo:
        run_demo(k)

    if not demo_only:
        run_console(k)

    show_cursor()
    print(f"\n{DIM}Topo9 runtime terminated.{RESET}\n")


if __name__ == "__main__":
    main()
