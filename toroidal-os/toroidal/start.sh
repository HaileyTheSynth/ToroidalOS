#!/bin/sh
# ============================================================================
# TOROIDAL OS - Boot Startup Script
# For Xiaomi Mi Mix (lithium) - 256GB Edition
# ============================================================================

export HOME=/root
export PATH=/usr/local/bin:/usr/bin:/bin:/opt/llama:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH

# Log file
LOGFILE=/var/log/toroidal.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOGFILE
}

# ============================================================================
# HARDWARE INITIALIZATION
# ============================================================================

log "TOROIDAL OS Starting..."
log "Device: Xiaomi Mi Mix (lithium)"
log "CPU: Snapdragon 821"
log "RAM: 6GB"

# Set CPU governor to performance
log "Setting CPU governor..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > $cpu 2>/dev/null
done

# Enable zram for swap (uses RAM compression)
log "Setting up ZRAM swap..."
if [ -e /sys/block/zram0 ]; then
    echo lz4 > /sys/block/zram0/comp_algorithm
    echo 2G > /sys/block/zram0/disksize
    mkswap /dev/zram0
    swapon -p 100 /dev/zram0
    log "ZRAM swap enabled (2GB)"
fi

# Set memory overcommit (important for mmap)
echo 1 > /proc/sys/vm/overcommit_memory

# ============================================================================
# AUDIO SETUP
# ============================================================================

log "Configuring audio..."

# Load ALSA state if exists
if [ -f /var/lib/alsa/asound.state ]; then
    alsactl restore
fi

# Set reasonable volume
amixer set Master 70% 2>/dev/null
amixer set Speaker 70% 2>/dev/null

# ============================================================================
# DISPLAY SETUP
# ============================================================================

log "Configuring display..."

# Set framebuffer console
if [ -e /dev/fb0 ]; then
    # Clear framebuffer
    dd if=/dev/zero of=/dev/fb0 bs=1M count=8 2>/dev/null
    
    # Show boot logo (if available)
    if [ -f /opt/toroidal/assets/logo.fb ]; then
        cat /opt/toroidal/assets/logo.fb > /dev/fb0
    fi
fi

# ============================================================================
# NETWORK SETUP (optional)
# ============================================================================

log "Starting network..."

# Start WiFi
if [ -e /sys/class/net/wlan0 ]; then
    ip link set wlan0 up 2>/dev/null
    # wpa_supplicant will be managed by NetworkManager
fi

# ============================================================================
# START TOROIDAL OS
# ============================================================================

log "Starting TOROIDAL kernel..."

cd /opt/toroidal

# Check if model exists
MODEL_PATH="/opt/models/qwen2.5-omni-3b.gguf"
MMPROJ_PATH="/opt/models/mmproj.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    log "ERROR: Model not found at $MODEL_PATH"
    log "Please download the model first:"
    log "  wget https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/resolve/main/Qwen2.5-Omni-3B-Q4_K_M.gguf -O $MODEL_PATH"
    exit 1
fi

# Start the main system
exec python3 /opt/toroidal/main.py \
    --model "$MODEL_PATH" \
    --mmproj "$MMPROJ_PATH" \
    --threads 4 \
    --context 2048 \
    2>&1 | tee -a $LOGFILE
