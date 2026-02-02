#!/bin/bash
# ============================================================================
#  TOROIDAL OS BUILD SCRIPT
#  Target: Xiaomi Mi Mix (lithium) - 256GB Edition
#  Specs: Snapdragon 821, 6GB RAM, 256GB UFS
# ============================================================================

set -e

DEVICE="lithium"
WORKDIR="$(pwd)"
OUTDIR="${WORKDIR}/out"
ROOTFS="${OUTDIR}/rootfs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[TOROIDAL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============================================================================
# PHASE 1: SETUP BUILD ENVIRONMENT
# ============================================================================
setup_env() {
    log "Setting up build environment..."
    
    mkdir -p "${OUTDIR}"/{boot,rootfs,models,kernel}
    
    # Install build dependencies (Ubuntu/Debian)
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        git \
        wget \
        curl \
        adb \
        fastboot \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        device-tree-compiler \
        libssl-dev \
        bc \
        flex \
        bison \
        libncurses-dev \
        u-boot-tools \
        qemu-user-static \
        debootstrap
}

# ============================================================================
# PHASE 2: BUILD MAINLINE KERNEL FOR MSM8996
# ============================================================================
build_kernel() {
    log "Building mainline kernel for Snapdragon 821 (MSM8996)..."
    
    cd "${OUTDIR}/kernel"
    
    # Clone mainline kernel with MSM8996 support
    if [ ! -d "linux" ]; then
        git clone --depth=1 -b linux-6.6.y \
            https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
    fi
    
    cd linux
    
    # Apply MSM8996 device tree patches
    cat > arch/arm64/boot/dts/qcom/msm8996pro-xiaomi-lithium.dts << 'DTEOF'
// SPDX-License-Identifier: GPL-2.0
/dts-v1/;

#include "msm8996pro.dtsi"
#include "pm8994.dtsi"
#include "pmi8994.dtsi"

/ {
    model = "Xiaomi Mi MIX";
    compatible = "xiaomi,lithium", "qcom,msm8996pro", "qcom,msm8996";
    
    chosen {
        stdout-path = "serial0:115200n8";
        bootargs = "earlycon=msm_geni_serial,0x75b0000 console=ttyMSM0,115200n8";
    };

    reserved-memory {
        #address-cells = <2>;
        #size-cells = <2>;
        ranges;
        
        /* Reserve memory for LLM model */
        llm_reserved: llm@a0000000 {
            reg = <0x0 0xa0000000 0x0 0x80000000>; /* 2GB for model mmap */
            no-map;
        };
    };

    /* 6.4" 1080x2040 IPS LCD */
    panel: panel {
        compatible = "xiaomi,lithium-panel";
        power-supply = <&pm8994_l14>;
        
        port {
            panel_in: endpoint {
                remote-endpoint = <&mdss_dsi0_out>;
            };
        };
    };
    
    /* Piezoelectric speaker (bone conduction) */
    sound {
        compatible = "qcom,apq8096-sndcard";
        model = "lithium";
        
        audio-routing =
            "Speaker", "WSA_SPK OUT";
    };
};

/* Enable USB Type-C */
&usb3 {
    status = "okay";
};

/* Enable WiFi */
&wifi {
    status = "okay";
    vdd-supply = <&pm8994_s4>;
};

/* Enable Bluetooth */
&bluetooth {
    status = "okay";
};
DTEOF

    # Create minimal config for Toroidal OS
    cat > arch/arm64/configs/toroidal_defconfig << 'CFGEOF'
# Toroidal OS Kernel Config for Xiaomi Mi Mix (lithium)
# Optimized for LLM inference

# Base ARM64
CONFIG_ARM64=y
CONFIG_ARCH_QCOM=y

# MSM8996 specific
CONFIG_QCOM_SMEM=y
CONFIG_QCOM_SMD=y
CONFIG_QCOM_SPMI=y
CONFIG_QCOM_TSENS=y

# Memory management - critical for LLM
CONFIG_TRANSPARENT_HUGEPAGE=y
CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS=y
CONFIG_ZSWAP=y
CONFIG_ZSWAP_DEFAULT_ON=y
CONFIG_ZRAM=y
CONFIG_MEMORY_HOTPLUG=y

# File systems
CONFIG_EXT4_FS=y
CONFIG_F2FS_FS=y
CONFIG_TMPFS=y
CONFIG_TMPFS_POSIX_ACL=y

# Input
CONFIG_INPUT_TOUCHSCREEN=y
CONFIG_TOUCHSCREEN_ATMEL_MXT=y

# Display
CONFIG_DRM=y
CONFIG_DRM_MSM=y
CONFIG_DRM_PANEL_SIMPLE=y

# Audio (piezo speaker)
CONFIG_SND=y
CONFIG_SND_SOC=y
CONFIG_SND_SOC_QCOM=y

# USB
CONFIG_USB=y
CONFIG_USB_XHCI_HCD=y
CONFIG_USB_DWC3=y
CONFIG_USB_DWC3_QCOM=y
CONFIG_USB_GADGET=y
CONFIG_USB_CONFIGFS=y
CONFIG_USB_CONFIGFS_F_FS=y

# Network
CONFIG_WLAN=y
CONFIG_ATH10K=y
CONFIG_ATH10K_PCI=y
CONFIG_BT=y
CONFIG_BT_HCIUART=y

# Camera
CONFIG_MEDIA_SUPPORT=y
CONFIG_VIDEO_V4L2=y
CONFIG_VIDEO_QCOM_CAMSS=y

# Power management
CONFIG_CPU_FREQ=y
CONFIG_CPU_FREQ_GOV_SCHEDUTIL=y
CONFIG_CPU_IDLE=y
CONFIG_ARM64_CPUIDLE=y

# Crypto (for secure operations)
CONFIG_CRYPTO_AES_ARM64_CE=y
CONFIG_CRYPTO_SHA256_ARM64=y

# Disable unnecessary features to save RAM
# CONFIG_DEBUG_INFO is not set
# CONFIG_FTRACE is not set
# CONFIG_KPROBES is not set
CFGEOF

    # Build kernel
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- toroidal_defconfig
    make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j$(nproc) Image.gz dtbs
    
    # Copy outputs
    cp arch/arm64/boot/Image.gz "${OUTDIR}/boot/"
    cp arch/arm64/boot/dts/qcom/msm8996pro-xiaomi-lithium.dtb "${OUTDIR}/boot/"
    
    log "Kernel built successfully"
}

# ============================================================================
# PHASE 3: BUILD ALPINE LINUX ROOTFS
# ============================================================================
build_rootfs() {
    log "Building Alpine Linux rootfs..."
    
    cd "${OUTDIR}"
    
    # Download Alpine minirootfs
    ALPINE_VERSION="3.19"
    ALPINE_URL="https://dl-cdn.alpinelinux.org/alpine/v${ALPINE_VERSION}/releases/aarch64/alpine-minirootfs-${ALPINE_VERSION}.0-aarch64.tar.gz"
    
    wget -c "${ALPINE_URL}" -O alpine-rootfs.tar.gz
    
    mkdir -p "${ROOTFS}"
    sudo tar xzf alpine-rootfs.tar.gz -C "${ROOTFS}"
    
    # Setup QEMU for chroot
    sudo cp /usr/bin/qemu-aarch64-static "${ROOTFS}/usr/bin/"
    
    # Configure Alpine
    sudo chroot "${ROOTFS}" /bin/sh << 'CHROOTEOF'
# Setup Alpine
cat > /etc/apk/repositories << EOF
https://dl-cdn.alpinelinux.org/alpine/v3.19/main
https://dl-cdn.alpinelinux.org/alpine/v3.19/community
EOF

apk update
apk upgrade

# Install minimal packages
apk add \
    openrc \
    busybox-initscripts \
    eudev \
    dbus \
    networkmanager \
    wpa_supplicant \
    alsa-utils \
    alsa-lib \
    python3 \
    py3-pip \
    py3-numpy \
    git \
    cmake \
    make \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    curl \
    wget \
    htop \
    nano

# Create toroidal user
adduser -D -s /bin/sh toroidal
echo "toroidal:toroidal" | chpasswd
addgroup toroidal wheel
addgroup toroidal audio
addgroup toroidal video
addgroup toroidal input

# Enable services
rc-update add devfs sysinit
rc-update add dmesg sysinit
rc-update add mdev sysinit
rc-update add hwdrivers sysinit
rc-update add networkmanager default
rc-update add dbus default
rc-update add alsa default

# Set hostname
echo "toroidal-lithium" > /etc/hostname

# Configure login
cat > /etc/inittab << EOF
::sysinit:/sbin/openrc sysinit
::sysinit:/sbin/openrc boot
::wait:/sbin/openrc default
tty1::respawn:/sbin/getty 38400 tty1
ttyMSM0::respawn:/sbin/getty -L ttyMSM0 115200 vt100
::ctrlaltdel:/sbin/reboot
::shutdown:/sbin/openrc shutdown
EOF

# Auto-start Toroidal OS
cat > /etc/local.d/toroidal.start << EOF
#!/bin/sh
cd /opt/toroidal
./start.sh &
EOF
chmod +x /etc/local.d/toroidal.start
rc-update add local default

exit
CHROOTEOF

    log "Rootfs built successfully"
}

# ============================================================================
# PHASE 4: BUILD LLAMA.CPP FOR AARCH64
# ============================================================================
build_llamacpp() {
    log "Building llama.cpp for ARM64..."
    
    cd "${OUTDIR}"
    
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggml-org/llama.cpp.git
    fi
    
    cd llama.cpp
    
    mkdir -p build-arm64
    cd build-arm64
    
    # Cross-compile for ARM64 with optimizations
    cmake .. \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=OFF \
        -DLLAMA_ARM_FMA=ON \
        -DLLAMA_ARM_DOTPROD=ON \
        -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
        -DLLAMA_BUILD_SERVER=ON
    
    make -j$(nproc)
    
    # Copy binaries to rootfs
    sudo mkdir -p "${ROOTFS}/opt/llama"
    sudo cp bin/llama-server bin/llama-cli "${ROOTFS}/opt/llama/"
    
    log "llama.cpp built successfully"
}

# ============================================================================
# PHASE 5: DOWNLOAD & QUANTIZE MODEL
# ============================================================================
download_model() {
    log "Downloading Qwen2.5-Omni-3B model..."
    
    cd "${OUTDIR}/models"
    
    # Download pre-quantized GGUF (Q3_K_S for memory efficiency)
    if [ ! -f "qwen2.5-omni-3b-q3_k_s.gguf" ]; then
        # Use official GGUF from ggml-org
        wget -c "https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/resolve/main/Qwen2.5-Omni-3B-Q4_K_M.gguf" \
            -O qwen2.5-omni-3b.gguf
        
        # Also get the multimodal projector
        wget -c "https://huggingface.co/ggml-org/Qwen2.5-Omni-3B-GGUF/resolve/main/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf" \
            -O mmproj.gguf
    fi
    
    # Copy to rootfs
    sudo mkdir -p "${ROOTFS}/opt/models"
    sudo cp *.gguf "${ROOTFS}/opt/models/"
    
    log "Model downloaded successfully"
}

# ============================================================================
# PHASE 6: INSTALL TOROIDAL KERNEL (PYTHON)
# ============================================================================
install_toroidal() {
    log "Installing Toroidal kernel..."
    
    sudo mkdir -p "${ROOTFS}/opt/toroidal"
    
    # Copy Python files (will be created separately)
    sudo cp -r "${WORKDIR}/toroidal/"* "${ROOTFS}/opt/toroidal/"
    
    # Install Python dependencies in chroot
    sudo chroot "${ROOTFS}" /bin/sh << 'CHROOTEOF'
pip3 install --break-system-packages \
    networkx \
    numpy \
    requests \
    pyaudio
CHROOTEOF

    log "Toroidal kernel installed"
}

# ============================================================================
# PHASE 7: CREATE BOOT IMAGE
# ============================================================================
create_boot_image() {
    log "Creating boot image..."
    
    cd "${OUTDIR}"
    
    # Create initramfs
    cd "${ROOTFS}"
    find . | cpio -o -H newc | gzip > "${OUTDIR}/boot/initrd.img"
    
    # Create Android boot image (required for Mi Mix)
    mkbootimg \
        --kernel "${OUTDIR}/boot/Image.gz" \
        --ramdisk "${OUTDIR}/boot/initrd.img" \
        --dtb "${OUTDIR}/boot/msm8996pro-xiaomi-lithium.dtb" \
        --cmdline "console=ttyMSM0,115200n8 androidboot.hardware=qcom root=/dev/mmcblk0p66 rootfstype=ext4 rw" \
        --base 0x80000000 \
        --kernel_offset 0x00008000 \
        --ramdisk_offset 0x01000000 \
        --dtb_offset 0x01f00000 \
        --pagesize 4096 \
        --header_version 2 \
        -o "${OUTDIR}/boot.img"
    
    # Create system image (rootfs)
    SIZE_MB=4096  # 4GB system partition
    dd if=/dev/zero of="${OUTDIR}/system.img" bs=1M count=${SIZE_MB}
    mkfs.ext4 -F "${OUTDIR}/system.img"
    
    mkdir -p /tmp/rootfs_mount
    sudo mount "${OUTDIR}/system.img" /tmp/rootfs_mount
    sudo cp -a "${ROOTFS}/"* /tmp/rootfs_mount/
    sudo umount /tmp/rootfs_mount
    
    log "Boot image created: ${OUTDIR}/boot.img"
    log "System image created: ${OUTDIR}/system.img"
}

# ============================================================================
# PHASE 8: FLASH TO DEVICE
# ============================================================================
flash_device() {
    log "Ready to flash to Xiaomi Mi Mix..."
    
    echo ""
    echo "=========================================="
    echo "  FLASHING INSTRUCTIONS"
    echo "=========================================="
    echo ""
    echo "1. Unlock bootloader (if not already):"
    echo "   - Enable Developer Options"
    echo "   - Enable OEM Unlocking"
    echo "   - Use Mi Unlock tool"
    echo ""
    echo "2. Boot into fastboot mode:"
    echo "   - Power off device"
    echo "   - Hold Volume Down + Power"
    echo ""
    echo "3. Flash images:"
    echo "   fastboot flash boot ${OUTDIR}/boot.img"
    echo "   fastboot flash system ${OUTDIR}/system.img"
    echo "   fastboot reboot"
    echo ""
    echo "4. First boot will take ~2 minutes"
    echo ""
    
    read -p "Press ENTER when device is in fastboot mode..."
    
    # Check for device
    if ! fastboot devices | grep -q .; then
        error "No device found in fastboot mode"
    fi
    
    log "Flashing boot image..."
    fastboot flash boot "${OUTDIR}/boot.img"
    
    log "Flashing system image..."
    fastboot flash system "${OUTDIR}/system.img"
    
    log "Rebooting..."
    fastboot reboot
    
    log "Flash complete! Device will boot into Toroidal OS."
}

# ============================================================================
# MAIN
# ============================================================================
main() {
    echo "============================================"
    echo "  TOROIDAL OS BUILD SYSTEM"
    echo "  Target: Xiaomi Mi Mix (lithium)"
    echo "============================================"
    echo ""
    
    case "${1:-all}" in
        setup)      setup_env ;;
        kernel)     build_kernel ;;
        rootfs)     build_rootfs ;;
        llama)      build_llamacpp ;;
        model)      download_model ;;
        toroidal)   install_toroidal ;;
        image)      create_boot_image ;;
        flash)      flash_device ;;
        all)
            setup_env
            build_kernel
            build_rootfs
            build_llamacpp
            download_model
            install_toroidal
            create_boot_image
            ;;
        *)
            echo "Usage: $0 {setup|kernel|rootfs|llama|model|toroidal|image|flash|all}"
            exit 1
            ;;
    esac
}

main "$@"
