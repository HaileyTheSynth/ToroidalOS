#include <stdint.h>
#include <stddef.h>

/* ===========================
   Port I/O (x86)
   =========================== */
static inline void outb(uint16_t port, uint8_t val) {
    __asm__ __volatile__ ("outb %0, %1" : : "a"(val), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ __volatile__ ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

/* ===========================
   Serial (COM1) — polled
   =========================== */
#define COM1 0x3F8
static void serial_init(void) {
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x80);
    outb(COM1 + 0, 0x03);  /* 38400 baud */
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x03);
    outb(COM1 + 2, 0xC7);
    outb(COM1 + 4, 0x0B);
}
static int serial_can_read(void) { return inb(COM1 + 5) & 1; }
static int serial_can_write(void) { return inb(COM1 + 5) & 0x20; }
static void serial_write_char(char c) { while (!serial_can_write()) {} outb(COM1, (uint8_t)c); }
static void serial_write_str(const char* s) { while (*s) serial_write_char(*s++); }
static void serial_write_u32(uint32_t x) {
    char buf[11];
    int i = 0;
    if (x == 0) { serial_write_char('0'); return; }
    while (x && i < 10) { buf[i++] = (char)('0' + (x % 10)); x /= 10; }
    while (i--) serial_write_char(buf[i]);
}
static void serial_write_hex32(uint32_t x) {
    const char* h = "0123456789ABCDEF";
    for (int i = 7; i >= 0; --i) serial_write_char(h[(x >> (i*4)) & 0xF]);
}
static int serial_try_read_char(void) { if (!serial_can_read()) return -1; return (int)inb(COM1); }

/* ===========================
   VGA (minimal)
   =========================== */
static volatile uint16_t* const VGA = (volatile uint16_t*)0xB8000;
static uint8_t vga_row = 0, vga_col = 0, vga_attr = 0x07;
static void vga_putc(char c) {
    if (c == '\n') { vga_col = 0; if (++vga_row >= 25) vga_row = 0; return; }
    VGA[vga_row * 80 + vga_col] = (uint16_t)c | ((uint16_t)vga_attr << 8);
    if (++vga_col >= 80) { vga_col = 0; if (++vga_row >= 25) vga_row = 0; }
}
static void vga_print(const char* s) { while (*s) vga_putc(*s++); }

/* ============================================================
   Topo9 Hopf/Torus/Solenoid runtime
   ============================================================ */
typedef uint16_t BitState; /* lower 9 bits used */

static inline uint8_t popcount9(BitState s) {
    uint8_t c = 0;
    for (int i = 0; i < 9; ++i) c += (s >> i) & 1;
    return c;
}
static inline uint8_t hamming9(BitState a, BitState b) { return popcount9((a ^ b) & 0x1FF); }
static inline uint8_t region_of_bit(uint8_t bit_index) {
    if (bit_index < 3) return 0;
    if (bit_index < 6) return 1;
    return 2;
}
static inline uint16_t mod360(uint32_t x) { return (uint16_t)(x % 360U); }
static inline uint16_t ang_dist(uint16_t a, uint16_t b) {
    uint16_t d = (a > b) ? (a - b) : (b - a);
    return (d > 180) ? (uint16_t)(360 - d) : d;
}

/* Solenoid history: store last H states (like a shift-register of "experienced" states) */
#define SOL_H 16

/* Trust tiers: 2-bit field encoding node resilience.
   KEEL nodes are core identity — immune to decay and GC.
   HULL nodes are important context — reduced decay rate.
   CARGO nodes are normal working memory.
   EPHEMERAL nodes decay fast and are GC'd first. */
typedef enum {
    TRUST_KEEL      = 0,  /* Core beliefs, identity — never decayed */
    TRUST_HULL      = 1,  /* Important context — slow decay */
    TRUST_CARGO     = 2,  /* Normal working memory */
    TRUST_EPHEMERAL = 3   /* Temporary — fast decay, GC priority */
} TrustTier;

typedef struct {
    uint8_t used;
    BitState state;
    uint8_t tau;
    uint16_t windings;
    uint32_t berry_milli;

    uint8_t last_region;
    uint8_t region_mask; /* bits 0..2 */
    uint16_t access_count;
    uint8_t trust;       /* TrustTier: 0=KEEL, 1=HULL, 2=CARGO, 3=EPHEMERAL */

    /* Torus angles (educational tags): theta1..theta4 in degrees */
    uint16_t th1, th2, th3, th4;

    /* Solenoid: history of past states (most recent at index 0) */
    BitState sol_hist[SOL_H];
    uint8_t sol_len;
} Node;

#define NMAX 64
static Node nodes[NMAX];
static uint8_t node_count = 0;

/* Pair weights still exist (cheap co-access memory), but coherence uses hyperedges */
static uint16_t edge_w[NMAX][NMAX];

/* True hyperedges */
typedef enum {
    HEDGE_CONV = 0,
    HEDGE_TOPIC = 1,
    HEDGE_ENTITY = 2,
    HEDGE_FIBER = 3,
    HEDGE_CUSTOM = 4
} HedgeType;

#define HEDGE_MAX 48
#define HEDGE_MEM_MAX 16

typedef struct {
    uint8_t used;
    HedgeType type;
    uint8_t m;
    uint8_t ids[HEDGE_MEM_MAX];
} Hyperedge;

static Hyperedge hedges[HEDGE_MAX];
static uint8_t hedge_count = 0;

/* Recent context path for curvature proxy (conversation trace) */
#define CTXK 12
static uint8_t ctx_ring[CTXK];
static uint8_t ctx_len = 0;
static uint8_t ctx_head = 0;
static void ctx_push(uint8_t id) { ctx_ring[ctx_head] = id; ctx_head = (uint8_t)((ctx_head + 1) % CTXK); if (ctx_len < CTXK) ctx_len++; }

static uint32_t curvature_scaled(void) {
    if (ctx_len < 3) return 0;
    uint32_t acc = 0;
    uint8_t edges = 0;
    for (uint8_t i = 1; i < ctx_len; ++i) {
        uint8_t idxA = (uint8_t)((ctx_head + CTXK - 1 - i) % CTXK);
        uint8_t idxB = (uint8_t)((ctx_head + CTXK - 1 - (i-1)) % CTXK);
        Node* a = &nodes[ctx_ring[idxA]];
        Node* b = &nodes[ctx_ring[idxB]];
        acc += hamming9(a->state, b->state);
        edges++;
    }
    return edges ? (acc * 100U) / edges : 0U;
}

static uint8_t region_count(uint8_t mask) { return (uint8_t)(((mask & 1) != 0) + ((mask & 2) != 0) + ((mask & 4) != 0)); }
static uint8_t is_bridge(uint8_t id, uint32_t min_berry) { return (nodes[id].berry_milli >= min_berry) && (region_count(nodes[id].region_mask) >= 2); }

/* Solenoid push */
static void sol_push(Node* n, BitState st) {
    for (int i = SOL_H - 1; i >= 1; --i) n->sol_hist[i] = n->sol_hist[i-1];
    n->sol_hist[0] = (BitState)(st & 0x1FF);
    if (n->sol_len < SOL_H) n->sol_len++;
}

/* Access update (tau + windings + berry + solenoid) */
static void record_access(uint8_t id, uint8_t region) {
    Node* n = &nodes[id];

    uint8_t old = n->tau;
    n->tau = (uint8_t)(n->tau + 17);
    if (n->tau < old) n->windings++;

    uint32_t add = (region == n->last_region) ? 30U : 110U;
    n->berry_milli += add;

    n->last_region = region;
    n->region_mask |= (uint8_t)(1U << region);
    n->access_count++;

    /* store experience trace */
    sol_push(n, n->state);

    /* small semantic drift on theta4: cross-region access rotates semantic angle */
    n->th4 = (uint16_t)((n->th4 + (region == n->last_region ? 3 : 9)) % 360);
}

/* Co-access pair weights */
static void update_edges_on_access(uint8_t id) {
    for (uint8_t i = 0; i < ctx_len; ++i) {
        uint8_t idx = (uint8_t)((ctx_head + CTXK - 1 - i) % CTXK);
        uint8_t other = ctx_ring[idx];
        if (other == id) continue;
        if (edge_w[id][other] < 0xFFFF) edge_w[id][other]++;
        if (edge_w[other][id] < 0xFFFF) edge_w[other][id]++;
    }
}

/* Hyperedge helpers */
static int hedge_add(HedgeType type, const uint8_t* ids, uint8_t m) {
    if (m < 2 || m > HEDGE_MEM_MAX) return -1;
    for (uint8_t i = 0; i < HEDGE_MAX; ++i) {
        if (!hedges[i].used) {
            hedges[i].used = 1;
            hedges[i].type = type;
            hedges[i].m = m;
            for (uint8_t k = 0; k < m; ++k) hedges[i].ids[k] = ids[k];
            if (i >= hedge_count) hedge_count = (uint8_t)(i + 1);
            return (int)i;
        }
    }
    return -1;
}
static int hedge_del(uint32_t eid) {
    if (eid >= HEDGE_MAX || !hedges[eid].used) return 0;
    hedges[eid].used = 0;
    return 1;
}

/* Hyperedge-based coherence: for node i, gather all nodes connected via any hedge containing i.
   Score each neighbor by tau sync + bit proximity + theta4 proximity; weighted by hedge-type. */
static uint32_t coherence_score(uint8_t id) {
    Node* a = &nodes[id];
    uint32_t acc = 0;
    uint32_t cnt = 0;

    for (uint8_t e = 0; e < hedge_count; ++e) {
        if (!hedges[e].used) continue;
        /* is id in this edge? */
        uint8_t in = 0;
        for (uint8_t k = 0; k < hedges[e].m; ++k) if (hedges[e].ids[k] == id) { in = 1; break; }
        if (!in) continue;

        uint32_t w_type = 1;
        switch (hedges[e].type) {
            case HEDGE_CONV:   w_type = 3; break;
            case HEDGE_TOPIC:  w_type = 2; break;
            case HEDGE_ENTITY: w_type = 2; break;
            case HEDGE_FIBER:  w_type = 2; break;
            default:           w_type = 1; break;
        }

        for (uint8_t k = 0; k < hedges[e].m; ++k) {
            uint8_t j = hedges[e].ids[k];
            if (j == id || !nodes[j].used) continue;
            Node* b = &nodes[j];

            uint8_t dtau = (a->tau > b->tau) ? (a->tau - b->tau) : (b->tau - a->tau);
            uint8_t dh   = hamming9(a->state, b->state);
            uint16_t dth = ang_dist(a->th4, b->th4);

            uint32_t local = 0;
            local += (uint32_t)(255 - dtau);
            local += (uint32_t)(9 - dh) * 20U;
            local += (uint32_t)(180 - dth); /* 0..180 */

            acc += w_type * local;
            cnt += w_type;
        }
    }

    if (cnt == 0) return 500U;
    return acc / cnt;
}

/* Top-k neighbors for COHERENT: compute neighbor score within hyperedge neighborhood */
static void top_neighbors(uint8_t id, uint8_t k, uint8_t* out_ids, uint32_t* out_s) {
    for (uint8_t i = 0; i < k; ++i) { out_ids[i] = 0xFF; out_s[i] = 0; }

    /* build a candidate set: any node that shares a hyperedge with id */
    for (uint8_t j = 0; j < node_count; ++j) {
        if (!nodes[j].used || j == id) continue;

        /* check if connected via any edge */
        uint8_t connected = 0;
        for (uint8_t e = 0; e < hedge_count && !connected; ++e) {
            if (!hedges[e].used) continue;
            uint8_t has_i = 0, has_j = 0;
            for (uint8_t k2 = 0; k2 < hedges[e].m; ++k2) {
                uint8_t x = hedges[e].ids[k2];
                if (x == id) has_i = 1;
                if (x == j)  has_j = 1;
            }
            if (has_i && has_j) connected = 1;
        }
        if (!connected) continue;

        /* neighbor score: tau sync + bit proximity + theta4 proximity + edge weight bonus */
        Node* a = &nodes[id];
        Node* b = &nodes[j];
        uint8_t dtau = (a->tau > b->tau) ? (a->tau - b->tau) : (b->tau - a->tau);
        uint8_t dh   = hamming9(a->state, b->state);
        uint16_t dth = ang_dist(a->th4, b->th4);
        uint32_t s = 0;
        s += (uint32_t)(255 - dtau);
        s += (uint32_t)(9 - dh) * 20U;
        s += (uint32_t)(180 - dth);
        s += (uint32_t)edge_w[id][j] * 2U; /* optional memory */

        for (uint8_t slot = 0; slot < k; ++slot) {
            if (s > out_s[slot]) {
                for (uint8_t m = (uint8_t)(k - 1); m > slot; --m) { out_s[m] = out_s[m-1]; out_ids[m] = out_ids[m-1]; }
                out_s[slot] = s; out_ids[slot] = j;
                break;
            }
        }
    }
}

/* Gentle synchronization during evolve:
   - tau nudged toward hyperedge mean
   - states nudged by flipping one differing bit between strongly related nodes
*/
static void apply_hyperedge_sync(void) {
    for (uint8_t e = 0; e < hedge_count; ++e) {
        if (!hedges[e].used) continue;

        /* tau mean */
        uint32_t sum = 0;
        uint8_t m = 0;
        for (uint8_t k = 0; k < hedges[e].m; ++k) {
            uint8_t id = hedges[e].ids[k];
            if (!nodes[id].used) continue;
            sum += nodes[id].tau;
            m++;
        }
        if (m < 2) continue;
        uint8_t mean = (uint8_t)(sum / m);

        /* nudge tau toward mean */
        for (uint8_t k = 0; k < hedges[e].m; ++k) {
            uint8_t id = hedges[e].ids[k];
            if (!nodes[id].used) continue;
            uint8_t t = nodes[id].tau;
            if (t < mean) nodes[id].tau = (uint8_t)(t + 1);
            else if (t > mean) nodes[id].tau = (uint8_t)(t - 1);
        }

        /* state tension: choose pairs in edge and reduce hamming by one bit occasionally */
        for (uint8_t i = 0; i < hedges[e].m; ++i) {
            for (uint8_t j = (uint8_t)(i + 1); j < hedges[e].m; ++j) {
                uint8_t a = hedges[e].ids[i], b = hedges[e].ids[j];
                if (!nodes[a].used || !nodes[b].used) continue;

                uint8_t dh = hamming9(nodes[a].state, nodes[b].state);
                if (dh <= 1) continue;

                BitState diff = (BitState)((nodes[a].state ^ nodes[b].state) & 0x1FF);
                uint8_t bit = 0;
                while (bit < 9 && (((diff >> bit) & 1) == 0)) bit++;
                if (bit >= 9) continue;

                /* flip in the higher-berry node (encourages bridges to 'align' communities) */
                if (nodes[a].berry_milli >= nodes[b].berry_milli) nodes[a].state ^= (BitState)(1U << bit);
                else nodes[b].state ^= (BitState)(1U << bit);
            }
        }
    }
}

static void evolve_steps(uint32_t steps) {
    for (uint32_t s = 0; s < steps; ++s) {
        for (uint8_t i = 0; i < node_count; ++i) {
            if (!nodes[i].used) continue;
            uint8_t old = nodes[i].tau;
            nodes[i].tau = (uint8_t)(nodes[i].tau + 3);
            if (nodes[i].tau < old) nodes[i].windings++;
        }
        apply_hyperedge_sync();
    }
}

/* Fiber query: select nodes matching (state & mask) == (pattern & mask) */
static int query_fiber(BitState pattern, BitState mask, uint8_t mode) {
    int best = -1;
    for (uint8_t i = 0; i < node_count; ++i) {
        if (!nodes[i].used) continue;
        if ( (nodes[i].state & mask) != (pattern & mask) ) continue;

        if (best < 0) { best = (int)i; continue; }

        if (mode == 0) { /* EARLIEST: smallest tau (older) */
            if (nodes[i].tau < nodes[best].tau) best = (int)i;
        } else if (mode == 1) { /* LATEST: largest tau */
            if (nodes[i].tau > nodes[best].tau) best = (int)i;
        } else { /* VERSATILE: largest berry */
            if (nodes[i].berry_milli > nodes[best].berry_milli) best = (int)i;
        }
    }
    return best;
}

/* ===========================
   Parsing helpers
   =========================== */
static void print_prompt(void) { serial_write_str("topo> "); }
static int is_space(char c) { return c == ' ' || c == '\t'; }

static void str_trim(char* s) {
    char* p = s;
    while (*p && is_space(*p)) p++;
    if (p != s) { char* d = s; while (*p) *d++ = *p++; *d = 0; }
    int n = 0; while (s[n]) n++;
    while (n > 0 && is_space(s[n-1])) s[--n] = 0;
}

static int str_ieq(const char* a, const char* b) {
    while (*a && *b) {
        char ca = *a, cb = *b;
        if (ca >= 'a' && ca <= 'z') ca = (char)(ca - 32);
        if (cb >= 'a' && cb <= 'z') cb = (char)(cb - 32);
        if (ca != cb) return 0;
        a++; b++;
    }
    return (*a == 0 && *b == 0);
}

static uint32_t parse_u32(const char* s, int* ok) {
    *ok = 0;
    if (!s || !*s) return 0;
    uint32_t base = 10;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
    uint32_t val = 0;
    while (*s) {
        char c = *s++;
        uint8_t d;
        if (c >= '0' && c <= '9') d = (uint8_t)(c - '0');
        else if (base == 16 && c >= 'A' && c <= 'F') d = (uint8_t)(10 + (c - 'A'));
        else if (base == 16 && c >= 'a' && c <= 'f') d = (uint8_t)(10 + (c - 'a'));
        else return 0;
        if (d >= base) return 0;
        val = val * base + d;
    }
    *ok = 1;
    return val;
}

static int parse_bits9(const char* s, BitState* out) {
    if (!s) return 0;
    int n = 0; while (s[n]) n++;
    if (n != 9) return 0;
    BitState v = 0;
    for (int i = 0; i < 9; ++i) {
        char c = s[i];
        if (c != '0' && c != '1') return 0;
        v = (BitState)((v << 1) | (BitState)(c == '1'));
    }
    *out = (BitState)(v & 0x1FF);
    return 1;
}

static void serial_write_bits9(BitState s) { for (int i = 8; i >= 0; --i) serial_write_char(((s >> i) & 1) ? '1' : '0'); }

/* ===========================
   Node ops + init
   =========================== */
static int store_node_trust(BitState st, uint8_t region, uint8_t trust) {
    if (node_count >= NMAX) return -1;
    uint8_t id = node_count++;
    nodes[id].used = 1;
    nodes[id].state = (BitState)(st & 0x1FF);
    nodes[id].tau = (uint8_t)(id * 19);
    nodes[id].windings = 0;
    nodes[id].berry_milli = 0;
    nodes[id].last_region = (uint8_t)(region % 3);
    nodes[id].region_mask = (uint8_t)(1U << (region % 3));
    nodes[id].access_count = 0;
    nodes[id].trust = (uint8_t)(trust & 3);

    nodes[id].th1 = mod360(id * 17U);
    nodes[id].th2 = mod360(id * 29U);
    nodes[id].th3 = mod360(id * 41U);
    nodes[id].th4 = mod360(id * 53U);

    nodes[id].sol_len = 0;
    for (int i = 0; i < SOL_H; ++i) nodes[id].sol_hist[i] = 0;
    sol_push(&nodes[id], nodes[id].state);
    return (int)id;
}

static int store_node(BitState st, uint8_t region) {
    return store_node_trust(st, region, TRUST_CARGO);
}

static void init_seed_nodes(void) {
    static const uint16_t seeds[] = { 0x000, 0x001, 0x03F, 0x0A5, 0x12D, 0x155, 0x1AA, 0x0F0, 0x111, 0x0CC, 0x099, 0x1FF };
    for (uint8_t i = 0; i < (uint8_t)(sizeof(seeds)/sizeof(seeds[0])); ++i) {
        (void)store_node((BitState)seeds[i], (uint8_t)(i % 3));
    }
    /* Seed a CONV hyperedge for the initial nodes */
    uint8_t ids[8];
    uint8_t m = (node_count < 8) ? node_count : 8;
    for (uint8_t k = 0; k < m; ++k) ids[k] = k;
    (void)hedge_add(HEDGE_CONV, ids, m);
}

/* ===========================
   Scheduler tick
   =========================== */
static uint32_t scheduler_score(uint8_t id, uint32_t curv_scaled, uint8_t multi_region_mode, uint32_t min_bridge_berry) {
    uint32_t coh = coherence_score(id);
    uint8_t w = popcount9(nodes[id].state);
    uint32_t activity = (w <= 1 || w >= 8) ? 200U : 420U;
    uint32_t bridge = (multi_region_mode && is_bridge(id, min_bridge_berry)) ? 350U : 0U;
    uint32_t curv_pen = curv_scaled * 2U;
    return coh + activity + bridge - curv_pen;
}

static void do_tick(uint32_t n) {
    uint8_t flip_bit = 0;
    uint8_t active_region = 0;
    const uint32_t min_bridge_berry = 1200U;

    for (uint32_t t = 0; t < n; ++t) {
        uint32_t curv = curvature_scaled();
        uint8_t multi_region_mode = (uint8_t)((t % 2) == 1);

        uint8_t best = 0;
        uint32_t best_s = 0;
        for (uint8_t i = 0; i < node_count; ++i) {
            uint32_t s = scheduler_score(i, curv, multi_region_mode, min_bridge_berry);
            if (s >= best_s) { best_s = s; best = i; }
        }

        nodes[best].state ^= (BitState)(1U << (flip_bit % 9));
        uint8_t flip_region = region_of_bit((uint8_t)(flip_bit % 9));
        flip_bit++;

        uint8_t access_region = (flip_region == active_region) ? active_region : (uint8_t)((active_region + 1) % 3);
        record_access(best, access_region);

        update_edges_on_access(best);
        ctx_push(best);

        serial_write_str("TICK t=");
        serial_write_u32(t);
        serial_write_str(" id=");
        serial_write_u32(best);
        serial_write_str(" s=");
        serial_write_bits9(nodes[best].state);
        serial_write_str(" tau=");
        serial_write_u32(nodes[best].tau);
        serial_write_str(" w=");
        serial_write_u32(nodes[best].windings);
        serial_write_str(" B=");
        serial_write_u32(nodes[best].berry_milli);
        serial_write_str(" th4=");
        serial_write_u32(nodes[best].th4);
        serial_write_str(" coh=");
        serial_write_u32(coherence_score(best));
        serial_write_str(" curv=");
        serial_write_u32(curv);
        serial_write_str("\n");
    }
}

/* ===========================
   Commands
   =========================== */
#define CMDBUF 220
static char cmd_buf[CMDBUF];
static uint32_t cmd_len = 0;

static void print_help(void) {
    serial_write_str(
        "Commands:\n"
        "  HELP\n"
        "  STATS\n"
        "  LIST [n]\n"
        "  STORE <bits9|hex|dec> [region]\n"
        "  ACCESS <id> [region]\n"
        "  FLIP <id> <bit0-8>\n"
        "  SETTHETA <id> <t1> <t2> <t3> <t4>\n"
        "  GETTHETA <id>\n"
        "  SOLENOID <id>\n"
        "  COHERENT <id> [k]\n"
        "  BRIDGES [minBerry]\n"
        "  QUERYFIBER <pattern> <mask> <EARLIEST|LATEST|VERSATILE>\n"
        "  HEDGE ADD <CONV|TOPIC|ENTITY|FIBER|CUSTOM> <id1> <id2> ...\n"
        "  HEDGE LIST\n"
        "  HEDGE DEL <edge_id>\n"
        "  CURVATURE\n"
        "  EVOLVE <steps>\n"
        "  TICK [n]\n"
        "  TRUST <id> [KEEL|HULL|CARGO|EPHEMERAL]  Get/set trust tier\n"
    );
}

static HedgeType parse_hedge_type(const char* s, int* ok) {
    *ok = 1;
    if (str_ieq(s, "CONV")) return HEDGE_CONV;
    if (str_ieq(s, "TOPIC")) return HEDGE_TOPIC;
    if (str_ieq(s, "ENTITY")) return HEDGE_ENTITY;
    if (str_ieq(s, "FIBER")) return HEDGE_FIBER;
    if (str_ieq(s, "CUSTOM")) return HEDGE_CUSTOM;
    *ok = 0;
    return HEDGE_CUSTOM;
}

static void cmd_stats(void) {
    uint32_t edges_nonzero = 0;
    uint32_t edge_sum = 0;
    for (uint8_t i = 0; i < node_count; ++i) {
        for (uint8_t j = (uint8_t)(i + 1); j < node_count; ++j) {
            uint16_t w = edge_w[i][j];
            if (w) { edges_nonzero++; edge_sum += w; }
        }
    }
    uint32_t avg_edge_w = edges_nonzero ? (edge_sum / edges_nonzero) : 0;

    uint32_t hedges_used = 0;
    for (uint8_t e = 0; e < hedge_count; ++e) if (hedges[e].used) hedges_used++;

    serial_write_str("STATS nodes=");
    serial_write_u32(node_count);
    serial_write_str(" ctx_len=");
    serial_write_u32(ctx_len);
    serial_write_str(" curvature=");
    serial_write_u32(curvature_scaled());
    serial_write_str(" pair_edges=");
    serial_write_u32(edges_nonzero);
    serial_write_str(" avg_pair_w=");
    serial_write_u32(avg_edge_w);
    serial_write_str(" hedges=");
    serial_write_u32(hedges_used);
    serial_write_str("\n");
}

static void cmd_list(uint32_t n) {
    if (n == 0 || n > node_count) n = node_count;
    if (n > 32) n = 32;
    for (uint32_t i = 0; i < n; ++i) {
        Node* x = &nodes[i];
        serial_write_str("id=");
        serial_write_u32(i);
        serial_write_str(" s=");
        serial_write_bits9(x->state);
        serial_write_str(" tau=");
        serial_write_u32(x->tau);
        serial_write_str(" w=");
        serial_write_u32(x->windings);
        serial_write_str(" B=");
        serial_write_u32(x->berry_milli);
        serial_write_str(" R=0x");
        serial_write_hex32(x->region_mask);
        serial_write_str(" th4=");
        serial_write_u32(x->th4);
        serial_write_str(" acc=");
        serial_write_u32(x->access_count);
        serial_write_str(" T=");
        serial_write_str(trust_name(x->trust));
        serial_write_str("\n");
    }
}

static const char* trust_name(uint8_t t) {
    switch (t) {
        case TRUST_KEEL:      return "KEEL";
        case TRUST_HULL:      return "HULL";
        case TRUST_CARGO:     return "CARGO";
        case TRUST_EPHEMERAL: return "EPHEMERAL";
        default:              return "?";
    }
}

static uint8_t parse_trust(const char* s, int* ok) {
    *ok = 1;
    if (str_ieq(s, "KEEL"))      return TRUST_KEEL;
    if (str_ieq(s, "HULL"))      return TRUST_HULL;
    if (str_ieq(s, "CARGO"))     return TRUST_CARGO;
    if (str_ieq(s, "EPHEMERAL")) return TRUST_EPHEMERAL;
    /* Also accept numeric 0-3 */
    int ok2 = 0;
    uint32_t v = parse_u32(s, &ok2);
    if (ok2 && v <= 3) return (uint8_t)v;
    *ok = 0;
    return TRUST_CARGO;
}

static void cmd_store(const char* a0, const char* a1) {
    BitState st = 0;
    uint8_t region = 0;
    int ok = 0;

    if (!a0) { serial_write_str("ERR missing state\n"); return; }

    if (parse_bits9(a0, &st)) ok = 1;
    else {
        uint32_t v = parse_u32(a0, &ok);
        if (ok) st = (BitState)(v & 0x1FF);
    }
    if (!ok) { serial_write_str("ERR state must be bits9 or hex/dec\n"); return; }

    if (a1) {
        int ok2 = 0;
        uint32_t r = parse_u32(a1, &ok2);
        if (ok2) region = (uint8_t)(r % 3);
    }

    int id = store_node(st, region);
    if (id < 0) { serial_write_str("ERR node table full\n"); return; }

    serial_write_str("OK id=");
    serial_write_u32((uint32_t)id);
    serial_write_str(" s=");
    serial_write_bits9(nodes[id].state);
    serial_write_str(" region=");
    serial_write_u32(region);
    serial_write_str(" trust=");
    serial_write_str(trust_name(nodes[id].trust));
    serial_write_str("\n");
}

static void cmd_trust(const char* a0, const char* a1) {
    int ok0 = 0;
    uint32_t id = parse_u32(a0, &ok0);
    if (!ok0 || id >= node_count) { serial_write_str("ERR invalid id\n"); return; }

    if (a1) {
        int ok1 = 0;
        uint8_t t = parse_trust(a1, &ok1);
        if (!ok1) { serial_write_str("ERR trust must be KEEL|HULL|CARGO|EPHEMERAL\n"); return; }
        nodes[id].trust = t;
    }

    serial_write_str("OK id=");
    serial_write_u32(id);
    serial_write_str(" trust=");
    serial_write_str(trust_name(nodes[id].trust));
    serial_write_str("\n");
}

static void cmd_access(const char* a0, const char* a1) {
    int ok0 = 0;
    uint32_t id = parse_u32(a0, &ok0);
    if (!ok0 || id >= node_count) { serial_write_str("ERR invalid id\n"); return; }

    uint8_t region = nodes[id].last_region;
    if (a1) {
        int ok1 = 0;
        uint32_t r = parse_u32(a1, &ok1);
        if (ok1) region = (uint8_t)(r % 3);
    }

    record_access((uint8_t)id, region);
    update_edges_on_access((uint8_t)id);
    ctx_push((uint8_t)id);

    serial_write_str("OK id=");
    serial_write_u32(id);
    serial_write_str(" tau=");
    serial_write_u32(nodes[id].tau);
    serial_write_str(" w=");
    serial_write_u32(nodes[id].windings);
    serial_write_str(" B=");
    serial_write_u32(nodes[id].berry_milli);
    serial_write_str(" R=0x");
    serial_write_hex32(nodes[id].region_mask);
    serial_write_str(" th4=");
    serial_write_u32(nodes[id].th4);
    serial_write_str(" coh=");
    serial_write_u32(coherence_score((uint8_t)id));
    serial_write_str("\n");
}

static void cmd_flip(const char* a0, const char* a1) {
    int ok0 = 0, ok1 = 0;
    uint32_t id = parse_u32(a0, &ok0);
    uint32_t bit = parse_u32(a1, &ok1);
    if (!ok0 || !ok1 || id >= node_count || bit > 8) { serial_write_str("ERR usage: FLIP <id> <bit0-8>\n"); return; }

    nodes[id].state ^= (BitState)(1U << bit);
    serial_write_str("OK id=");
    serial_write_u32(id);
    serial_write_str(" s=");
    serial_write_bits9(nodes[id].state);
    serial_write_str("\n");
}

static void cmd_settheta(const char* a0, const char* a1, const char* a2, const char* a3, const char* a4) {
    int ok0=0, ok1=0, ok2=0, ok3=0, ok4=0;
    uint32_t id = parse_u32(a0, &ok0);
    uint32_t t1 = parse_u32(a1, &ok1);
    uint32_t t2 = parse_u32(a2, &ok2);
    uint32_t t3 = parse_u32(a3, &ok3);
    uint32_t t4 = parse_u32(a4, &ok4);
    if (!ok0 || !ok1 || !ok2 || !ok3 || !ok4 || id >= node_count) { serial_write_str("ERR usage: SETTHETA <id> <t1> <t2> <t3> <t4>\n"); return; }

    nodes[id].th1 = mod360(t1);
    nodes[id].th2 = mod360(t2);
    nodes[id].th3 = mod360(t3);
    nodes[id].th4 = mod360(t4);

    serial_write_str("OK id=");
    serial_write_u32(id);
    serial_write_str(" th=");
    serial_write_u32(nodes[id].th1); serial_write_char(',');
    serial_write_u32(nodes[id].th2); serial_write_char(',');
    serial_write_u32(nodes[id].th3); serial_write_char(',');
    serial_write_u32(nodes[id].th4);
    serial_write_str("\n");
}

static void cmd_gettheta(const char* a0) {
    int ok0=0;
    uint32_t id = parse_u32(a0, &ok0);
    if (!ok0 || id >= node_count) { serial_write_str("ERR invalid id\n"); return; }
    serial_write_str("THETA id=");
    serial_write_u32(id);
    serial_write_str(" th=");
    serial_write_u32(nodes[id].th1); serial_write_char(',');
    serial_write_u32(nodes[id].th2); serial_write_char(',');
    serial_write_u32(nodes[id].th3); serial_write_char(',');
    serial_write_u32(nodes[id].th4);
    serial_write_str("\n");
}

static void cmd_solenoid(const char* a0) {
    int ok0=0;
    uint32_t id = parse_u32(a0, &ok0);
    if (!ok0 || id >= node_count) { serial_write_str("ERR invalid id\n"); return; }
    Node* n = &nodes[id];
    serial_write_str("SOLENOID id=");
    serial_write_u32(id);
    serial_write_str(" len=");
    serial_write_u32(n->sol_len);
    serial_write_str("\n");
    for (uint8_t i = 0; i < n->sol_len; ++i) {
        serial_write_str("  [");
        serial_write_u32(i);
        serial_write_str("] ");
        serial_write_bits9(n->sol_hist[i]);
        serial_write_str("\n");
    }
}

static void cmd_coherent(const char* a0, const char* a1) {
    int ok0 = 0;
    uint32_t id = parse_u32(a0, &ok0);
    if (!ok0 || id >= node_count) { serial_write_str("ERR invalid id\n"); return; }

    uint8_t k = 5;
    if (a1) { int ok1=0; uint32_t kv=parse_u32(a1, &ok1); if (ok1 && kv>=1 && kv<=10) k=(uint8_t)kv; }

    uint32_t coh = coherence_score((uint8_t)id);
    serial_write_str("COHERENT id=");
    serial_write_u32(id);
    serial_write_str(" coh=");
    serial_write_u32(coh);
    serial_write_str(" topN:\n");

    uint8_t ids[10];
    uint32_t ss[10];
    top_neighbors((uint8_t)id, k, ids, ss);

    for (uint8_t i = 0; i < k; ++i) {
        if (ids[i] == 0xFF) continue;
        uint8_t j = ids[i];
        serial_write_str("  n=");
        serial_write_u32(j);
        serial_write_str(" s=");
        serial_write_u32(ss[i]);
        serial_write_str(" dh=");
        serial_write_u32(hamming9(nodes[id].state, nodes[j].state));
        serial_write_str(" dtau=");
        uint8_t dtau = (nodes[id].tau > nodes[j].tau) ? (nodes[id].tau - nodes[j].tau) : (nodes[j].tau - nodes[id].tau);
        serial_write_u32(dtau);
        serial_write_str(" dth4=");
        serial_write_u32(ang_dist(nodes[id].th4, nodes[j].th4));
        serial_write_str("\n");
    }
}

static void cmd_bridges(const char* a0) {
    uint32_t minB = 1200U;
    if (a0) { int ok0=0; uint32_t v=parse_u32(a0, &ok0); if (ok0) minB=v; }

    serial_write_str("BRIDGES minB=");
    serial_write_u32(minB);
    serial_write_str("\n");

    uint32_t count = 0;
    for (uint8_t i = 0; i < node_count; ++i) {
        if (is_bridge(i, minB)) {
            count++;
            serial_write_str("  id=");
            serial_write_u32(i);
            serial_write_str(" s=");
            serial_write_bits9(nodes[i].state);
            serial_write_str(" B=");
            serial_write_u32(nodes[i].berry_milli);
            serial_write_str(" R=0x");
            serial_write_hex32(nodes[i].region_mask);
            serial_write_str(" th4=");
            serial_write_u32(nodes[i].th4);
            serial_write_str(" w=");
            serial_write_u32(nodes[i].windings);
            serial_write_str("\n");
        }
    }
    if (!count) serial_write_str("  (none)\n");
}

static void cmd_queryfiber(const char* ptn, const char* msk, const char* mode_s) {
    if (!ptn || !msk || !mode_s) { serial_write_str("ERR usage: QUERYFIBER <pattern> <mask> <EARLIEST|LATEST|VERSATILE>\n"); return; }
    BitState pattern = 0, mask = 0;
    int okp = 0, okm = 0;

    if (parse_bits9(ptn, &pattern)) okp = 1;
    else { uint32_t v = parse_u32(ptn, &okp); if (okp) pattern = (BitState)(v & 0x1FF); }

    if (parse_bits9(msk, &mask)) okm = 1;
    else { uint32_t v = parse_u32(msk, &okm); if (okm) mask = (BitState)(v & 0x1FF); }

    if (!okp || !okm) { serial_write_str("ERR pattern/mask must be bits9 or hex/dec\n"); return; }

    uint8_t mode = 2;
    if (str_ieq(mode_s, "EARLIEST")) mode = 0;
    else if (str_ieq(mode_s, "LATEST")) mode = 1;
    else if (str_ieq(mode_s, "VERSATILE")) mode = 2;
    else { serial_write_str("ERR mode must be EARLIEST|LATEST|VERSATILE\n"); return; }

    int best = query_fiber(pattern, mask, mode);
    if (best < 0) { serial_write_str("OK none\n"); return; }

    serial_write_str("OK id=");
    serial_write_u32((uint32_t)best);
    serial_write_str(" s=");
    serial_write_bits9(nodes[best].state);
    serial_write_str(" tau=");
    serial_write_u32(nodes[best].tau);
    serial_write_str(" B=");
    serial_write_u32(nodes[best].berry_milli);
    serial_write_str("\n");
}

static void cmd_hedge_list(void) {
    serial_write_str("HEDGES:\n");
    for (uint8_t e = 0; e < hedge_count; ++e) {
        if (!hedges[e].used) continue;
        serial_write_str("  eid=");
        serial_write_u32(e);
        serial_write_str(" type=");
        switch (hedges[e].type) {
            case HEDGE_CONV: serial_write_str("CONV"); break;
            case HEDGE_TOPIC: serial_write_str("TOPIC"); break;
            case HEDGE_ENTITY: serial_write_str("ENTITY"); break;
            case HEDGE_FIBER: serial_write_str("FIBER"); break;
            default: serial_write_str("CUSTOM"); break;
        }
        serial_write_str(" m=");
        serial_write_u32(hedges[e].m);
        serial_write_str(" ids=");
        for (uint8_t k = 0; k < hedges[e].m; ++k) {
            serial_write_u32(hedges[e].ids[k]);
            if (k + 1 < hedges[e].m) serial_write_char(',');
        }
        serial_write_str("\n");
    }
}

static void cmd_hedge_add(char** tok, uint8_t nt) {
    if (nt < 4) { serial_write_str("ERR usage: HEDGE ADD <type> <id1> <id2> ...\n"); return; }
    int okT = 0;
    HedgeType t = parse_hedge_type(tok[2], &okT);
    if (!okT) { serial_write_str("ERR type must be CONV|TOPIC|ENTITY|FIBER|CUSTOM\n"); return; }

    uint8_t ids[HEDGE_MEM_MAX];
    uint8_t m = 0;

    for (uint8_t i = 3; i < nt && m < HEDGE_MEM_MAX; ++i) {
        int ok=0;
        uint32_t v = parse_u32(tok[i], &ok);
        if (!ok || v >= node_count) { serial_write_str("ERR invalid node id in list\n"); return; }
        ids[m++] = (uint8_t)v;
    }
    if (m < 2) { serial_write_str("ERR need at least 2 ids\n"); return; }

    int eid = hedge_add(t, ids, m);
    if (eid < 0) { serial_write_str("ERR could not add edge (full?)\n"); return; }
    serial_write_str("OK eid=");
    serial_write_u32((uint32_t)eid);
    serial_write_str("\n");
}

static void cmd_hedge_del(const char* a0) {
    int ok0=0;
    uint32_t eid = parse_u32(a0, &ok0);
    if (!ok0) { serial_write_str("ERR usage: HEDGE DEL <edge_id>\n"); return; }
    if (hedge_del(eid)) serial_write_str("OK\n");
    else serial_write_str("ERR no such edge\n");
}

static void cmd_curvature(void) { serial_write_str("CURVATURE scaled="); serial_write_u32(curvature_scaled()); serial_write_str("\n"); }

static void cmd_evolve(const char* a0) {
    int ok0=0;
    uint32_t steps = parse_u32(a0, &ok0);
    if (!ok0 || steps == 0) { serial_write_str("ERR usage: EVOLVE <steps>\n"); return; }
    evolve_steps(steps);
    serial_write_str("OK evolved steps=");
    serial_write_u32(steps);
    serial_write_str("\n");
}

static void cmd_tick(const char* a0) {
    uint32_t n = 1;
    if (a0) { int ok0=0; uint32_t v=parse_u32(a0, &ok0); if (ok0 && v>=1 && v<=1000) n=v; }
    do_tick(n);
}

static void dispatch_command(char* line) {
    str_trim(line);
    if (!line[0]) return;

    char* tok[24];
    uint8_t nt = 0;

    char* p = line;
    while (*p && nt < 24) {
        while (*p && is_space(*p)) p++;
        if (!*p) break;
        tok[nt++] = p;
        while (*p && !is_space(*p)) p++;
        if (*p) { *p = 0; p++; }
    }

    if (nt == 0) return;

    if (str_ieq(tok[0], "HELP")) print_help();
    else if (str_ieq(tok[0], "STATS")) cmd_stats();
    else if (str_ieq(tok[0], "LIST")) {
        uint32_t n = 16;
        if (nt >= 2) { int ok=0; uint32_t v=parse_u32(tok[1], &ok); if (ok) n=v; }
        cmd_list(n);
    }
    else if (str_ieq(tok[0], "STORE")) cmd_store(nt >= 2 ? tok[1] : 0, nt >= 3 ? tok[2] : 0);
    else if (str_ieq(tok[0], "ACCESS")) cmd_access(nt >= 2 ? tok[1] : 0, nt >= 3 ? tok[2] : 0);
    else if (str_ieq(tok[0], "FLIP")) cmd_flip(nt >= 2 ? tok[1] : 0, nt >= 3 ? tok[2] : 0);
    else if (str_ieq(tok[0], "SETTHETA")) cmd_settheta(nt >= 2 ? tok[1] : 0, nt >= 3 ? tok[2] : 0, nt >= 4 ? tok[3] : 0, nt >= 5 ? tok[4] : 0, nt >= 6 ? tok[5] : 0);
    else if (str_ieq(tok[0], "GETTHETA")) cmd_gettheta(nt >= 2 ? tok[1] : 0);
    else if (str_ieq(tok[0], "SOLENOID")) cmd_solenoid(nt >= 2 ? tok[1] : 0);
    else if (str_ieq(tok[0], "COHERENT")) cmd_coherent(nt >= 2 ? tok[1] : 0, nt >= 3 ? tok[2] : 0);
    else if (str_ieq(tok[0], "BRIDGES")) cmd_bridges(nt >= 2 ? tok[1] : 0);
    else if (str_ieq(tok[0], "QUERYFIBER")) cmd_queryfiber(nt >= 2 ? tok[1] : 0, nt >= 3 ? tok[2] : 0, nt >= 4 ? tok[3] : 0);
    else if (str_ieq(tok[0], "HEDGE")) {
        if (nt >= 2 && str_ieq(tok[1], "LIST")) cmd_hedge_list();
        else if (nt >= 2 && str_ieq(tok[1], "ADD")) cmd_hedge_add(tok, nt);
        else if (nt >= 2 && str_ieq(tok[1], "DEL")) cmd_hedge_del(nt >= 3 ? tok[2] : 0);
        else serial_write_str("ERR usage: HEDGE LIST | HEDGE ADD ... | HEDGE DEL <id>\n");
    }
    else if (str_ieq(tok[0], "CURVATURE")) cmd_curvature();
    else if (str_ieq(tok[0], "EVOLVE")) cmd_evolve(nt >= 2 ? tok[1] : 0);
    else if (str_ieq(tok[0], "TICK")) cmd_tick(nt >= 2 ? tok[1] : 0);
    else if (str_ieq(tok[0], "TRUST")) cmd_trust(nt >= 2 ? tok[1] : 0, nt >= 3 ? tok[2] : 0);
    else serial_write_str("ERR unknown command (try HELP)\n");
}

/* Poll serial and build a line buffer (echoes input) */
static void serial_poll(void) {
    int c;
    while ((c = serial_try_read_char()) != -1) {
        char ch = (char)c;
        if (ch == '\r') continue;

        if (ch == '\b' || ch == 127) {
            if (cmd_len > 0) cmd_len--;
            continue;
        }

        if (ch == '\n') {
            cmd_buf[cmd_len] = 0;
            serial_write_str("\n");
            dispatch_command(cmd_buf);
            cmd_len = 0;
            print_prompt();
            continue;
        }

        if (cmd_len + 1 < (uint32_t)sizeof(cmd_buf)) {
            cmd_buf[cmd_len++] = ch;
            serial_write_char(ch);
        }
    }
}

/* crude busy delay to prevent 100% host CPU; still responsive */
static void tiny_delay(void) {
    for (volatile uint32_t i = 0; i < 80000; ++i) { __asm__ __volatile__("nop"); }
}

/* ===========================
   Kernel main
   =========================== */
void kernel_main(void) {
    serial_init();
    vga_print("Topo9 Hopf/Solenoid Runtime booted. Use serial console.\n");
    serial_write_str("\nTopo9 Hopf/Solenoid Runtime booted.\nType HELP for commands.\n");

    init_seed_nodes();
    print_prompt();

    for (;;) {
        serial_poll();
        tiny_delay();
    }
}
