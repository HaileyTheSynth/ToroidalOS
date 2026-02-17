#!/usr/bin/env python3
"""
TOROIDAL OS - Mi Mix Emulator Support
======================================
Provides emulation and simulation modes for testing ToroidalOS
without physical hardware.

Modes:
1. SIMULATION - Mock hardware sensors, pure Python
2. QEMU_ARM - QEMU ARM emulation of Alpine rootfs
3. ANDROID_EMULATOR - Android emulator with Mi Mix profile

Usage:
    python emulator.py --mode simulation
    python emulator.py --mode qemu --rootfs ./out/rootfs.ext4
    python emulator.py --mode android --avd lithium
"""

import os
import sys
import subprocess
import time
import signal
import argparse
import tempfile
import shutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import threading
import json
import random


class EmulatorMode(Enum):
    SIMULATION = "simulation"
    QEMU_ARM = "qemu"
    ANDROID_EMULATOR = "android"


@dataclass
class SensorReading:
    """Simulated sensor reading."""
    name: str
    value: float
    unit: str
    timestamp: float


class SimulatedHardware:
    """
    Simulated hardware for Mi Mix (lithium).

    Provides realistic sensor data and hardware state for testing
    without physical device.
    """

    # Mi Mix specs
    DEVICE = "Xiaomi Mi Mix (lithium)"
    SOC = "Snapdragon 821 (MSM8996 Pro)"
    RAM_MB = 6144
    STORAGE_GB = 256
    DISPLAY = (1080, 2040)  # Width x Height
    BATTERY_MAH = 4400

    def __init__(self, seed: int = None):
        """Initialize simulated hardware."""
        if seed:
            random.seed(seed)

        # State
        self.battery_level = random.randint(30, 95)
        self.battery_charging = False
        self.orientation = (0.0, 0.0, 9.81)  # Flat on table
        self.light_level = 350  # lux (indoor)
        self.pressure = 1013.25  # hPa (sea level)
        self.temperature = 25.0  # Celsius
        self.wifi_connected = True
        self.wifi_ssid = "SimulatedNetwork"
        self.bluetooth_enabled = True
        self.bluetooth_devices = [
            ("JBL Speaker", "AA:BB:CC:11:22:33"),
            ("Galaxy Buds", "DD:EE:FF:44:55:66"),
        ]

        # Simulated audio state
        self.audio_playing = False
        self.microphone_active = False

    def get_sensor_bits(self) -> int:
        """Get 9-bit sensor state."""
        bits = 0

        # MOTION region (bits 0-2)
        accel_mag = sum(x**2 for x in self.orientation) ** 0.5
        if abs(accel_mag - 9.81) > 1.5:  # Significant acceleration
            bits |= 0b001  # bit 0: accelerometer
        # bits 1-2: gyroscope, orientation (simulated as 0)

        # ENVIRON region (bits 3-5)
        if self.light_level > 200:
            bits |= 0b001000  # bit 3: light
        # bit 4: proximity (simulated as not near)
        if abs(self.pressure - 1013.25) > 2.0:
            bits |= 0b100000  # bit 5: barometer

        # SEMANTIC region (bits 6-8)
        if self.microphone_active:
            bits |= 0b001000000  # bit 6: audio
        # bits 7-8: camera, touch (simulated as 0)

        return bits

    def read_sensors(self) -> Dict[str, SensorReading]:
        """Get all sensor readings."""
        now = time.time()

        # Add some realistic noise
        noise = lambda v, r: v + random.uniform(-r, r)

        return {
            "accelerometer": SensorReading(
                name="accelerometer",
                value=noise(self.orientation[2], 0.1),
                unit="m/s²",
                timestamp=now
            ),
            "battery": SensorReading(
                name="battery",
                value=self.battery_level,
                unit="%",
                timestamp=now
            ),
            "light": SensorReading(
                name="light",
                value=noise(self.light_level, 10),
                unit="lux",
                timestamp=now
            ),
            "pressure": SensorReading(
                name="pressure",
                value=noise(self.pressure, 0.5),
                unit="hPa",
                timestamp=now
            ),
            "temperature": SensorReading(
                name="temperature",
                value=noise(self.temperature, 0.5),
                unit="°C",
                timestamp=now
            ),
        }

    def simulate_user_activity(self):
        """Simulate realistic user activity patterns."""
        # Randomly update some state
        if random.random() < 0.1:
            # User picked up device
            self.orientation = (
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(8, 10)
            )

        if random.random() < 0.05:
            # Battery drain
            if not self.battery_charging:
                self.battery_level = max(0, self.battery_level - random.uniform(0.1, 0.5))

        if random.random() < 0.02:
            # Light change (user moved)
            self.light_level = random.uniform(50, 800)

    def get_sysfs_paths(self) -> Dict[str, str]:
        """Get simulated sysfs paths for sensors."""
        return {
            "battery_capacity": "/sys/class/power_supply/battery/capacity",
            "battery_status": "/sys/class/power_supply/battery/status",
            "accel_x": "/sys/bus/iio/devices/iio:device0/in_accel_x_raw",
            "accel_y": "/sys/bus/iio/devices/iio:device0/in_accel_y_raw",
            "accel_z": "/sys/bus/iio/devices/iio:device0/in_accel_z_raw",
            "light": "/sys/bus/iio/devices/iio:device1/in_illuminance_raw",
            "pressure": "/sys/bus/iio/devices/iio:device2/in_pressure_input",
        }


class MiMixEmulator:
    """
    Emulator for Xiaomi Mi Mix running ToroidalOS.

    Supports multiple emulation modes:
    - SIMULATION: Pure Python, no external dependencies
    - QEMU_ARM: QEMU ARM system emulation
    - ANDROID_EMULATOR: Android emulator with Mi Mix profile
    """

    def __init__(
        self,
        mode: EmulatorMode = EmulatorMode.SIMULATION,
        rootfs_path: Optional[str] = None,
        kernel_path: Optional[str] = None,
        avd_name: Optional[str] = None,
    ):
        self.mode = mode
        self.rootfs_path = rootfs_path
        self.kernel_path = kernel_path
        self.avd_name = avd_name
        self.hardware = SimulatedHardware()
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start the emulator."""
        if self.mode == EmulatorMode.SIMULATION:
            return self._start_simulation()
        elif self.mode == EmulatorMode.QEMU_ARM:
            return self._start_qemu()
        elif self.mode == EmulatorMode.ANDROID_EMULATOR:
            return self._start_android_emulator()
        else:
            print(f"[EMULATOR] Unknown mode: {self.mode}")
            return False

    def stop(self):
        """Stop the emulator."""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _start_simulation(self) -> bool:
        """Start simulation mode (no external process)."""
        print(f"[EMULATOR] Starting simulation mode")
        print(f"[EMULATOR] Device: {self.hardware.DEVICE}")
        print(f"[EMULATOR] SoC: {self.hardware.SOC}")
        print(f"[EMULATOR] RAM: {self.hardware.RAM_MB} MB")
        self.running = True

        # Start sensor simulation thread
        self._monitor_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._monitor_thread.start()

        return True

    def _simulation_loop(self):
        """Background thread for sensor simulation."""
        while self.running:
            self.hardware.simulate_user_activity()
            time.sleep(1.0)

    def _start_qemu(self) -> bool:
        """Start QEMU ARM emulation."""
        if not self.rootfs_path or not os.path.exists(self.rootfs_path):
            print("[EMULATOR] Rootfs image required for QEMU mode")
            return False

        if not self.kernel_path or not os.path.exists(self.kernel_path):
            print("[EMULATOR] Kernel image required for QEMU mode")
            return False

        print("[EMULATOR] Starting QEMU ARM emulation...")

        # QEMU command for ARM64 (aarch64)
        qemu_cmd = [
            "qemu-system-aarch64",
            "-M", "virt",
            "-cpu", "cortex-a72",
            "-m", "2048",  # 2GB RAM for emulation
            "-kernel", self.kernel_path,
            "-drive", f"file={self.rootfs_path},format=raw,if=virtio",
            "-append", "root=/dev/vda rw console=ttyAMA0",
            "-nographic",
            "-serial", "mon:stdio",
        ]

        try:
            self.process = subprocess.Popen(
                qemu_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.running = True
            print(f"[EMULATOR] QEMU started (PID: {self.process.pid})")
            return True
        except FileNotFoundError:
            print("[EMULATOR] QEMU not found. Install: apt-get install qemu-system-arm")
            return False
        except Exception as e:
            print(f"[EMULATOR] Failed to start QEMU: {e}")
            return False

    def _start_android_emulator(self) -> bool:
        """Start Android emulator with Mi Mix profile."""
        avd = self.avd_name or "lithium"

        print(f"[EMULATOR] Starting Android emulator with AVD: {avd}")

        # Android emulator command
        emulator_cmd = [
            "emulator",
            "-avd", avd,
            "-device", "lithium",  # Mi Mix device profile
            "-memory", "2048",
            "-no-snapshot-load",
            "-no-audio",
            "-no-boot-anim",
            "-accel", "on",
        ]

        try:
            self.process = subprocess.Popen(
                emulator_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.running = True
            print(f"[EMULATOR] Android emulator started (PID: {self.process.pid})")
            print("[EMULATOR] Waiting for boot...")
            time.sleep(10)  # Wait for emulator to boot
            return True
        except FileNotFoundError:
            print("[EMULATOR] Android emulator not found.")
            print("[EMULATOR] Install Android SDK and create Mi Mix AVD:")
            print("[EMULATOR]   avdmanager create avd -n lithium -k 'system-images;android-33;default;arm64-v8a' -d 17")
            return False
        except Exception as e:
            print(f"[EMULATOR] Failed to start Android emulator: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get current emulator state."""
        return {
            "mode": self.mode.value,
            "running": self.running,
            "device": self.hardware.DEVICE,
            "sensor_bits": self.hardware.get_sensor_bits(),
            "sensors": {
                name: {"value": r.value, "unit": r.unit}
                for name, r in self.hardware.read_sensors().items()
            },
            "network": {
                "wifi_connected": self.hardware.wifi_connected,
                "wifi_ssid": self.hardware.wifi_ssid,
                "bluetooth_enabled": self.hardware.bluetooth_enabled,
            },
        }

    def execute_command(self, command: str) -> str:
        """Execute a command in the emulator."""
        if self.mode == EmulatorMode.SIMULATION:
            return self._simulate_command(command)
        elif self.mode == EmulatorMode.QEMU_ARM:
            return self._qemu_command(command)
        elif self.mode == EmulatorMode.ANDROID_EMULATOR:
            return self._adb_command(command)
        return "[EMULATOR] No command execution in current mode"

    def _simulate_command(self, command: str) -> str:
        """Simulate command execution."""
        # Simulate some common commands
        if command.startswith("cat /sys/class/power_supply"):
            return f"{self.hardware.battery_level}"
        elif command.startswith("cat /sys/bus/iio"):
            sensors = self.hardware.read_sensors()
            if "accel" in command:
                return f"{self.hardware.orientation[0]:.2f}\n{self.hardware.orientation[1]:.2f}\n{self.hardware.orientation[2]:.2f}"
            elif "light" in command:
                return f"{int(self.hardware.light_level)}"
            elif "pressure" in command:
                return f"{self.hardware.pressure:.2f}"
        elif command == "uptime":
            return "00:00:42 up 42 min, load average: 0.42, 0.42, 0.42"
        elif command == "free -m":
            return f"              total        used        free\nMem:          {self.hardware.RAM_MB}        2048        4096"
        else:
            return f"[SIMULATED] {command}"

    def _qemu_command(self, command: str) -> str:
        """Send command to QEMU via serial."""
        # Would need serial communication
        return "[QEMU] Command execution not implemented"

    def _adb_command(self, command: str) -> str:
        """Execute command via adb."""
        try:
            result = subprocess.run(
                ["adb", "shell", command],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout or result.stderr
        except Exception as e:
            return f"[ADB ERROR] {e}"


def create_avd_profile():
    """Create Mi Mix AVD profile for Android emulator."""
    profile = {
        "avdName": "lithium",
        "device": {
            "name": "Xiaomi Mi Mix",
            "manufacturer": "Xiaomi",
            "screen": {
                "width": 1080,
                "height": 2040,
                "density": 440,
                "diagonal": 6.4
            },
            "ram": 6144,
            "storage": 256,
            "sensors": [
                "accelerometer",
                "gyroscope",
                "proximity",
                "light",
                "pressure",
                "magnetic_field"
            ],
            "features": [
                "wifi",
                "bluetooth",
                "gps",
                "camera_rear",
                "camera_front",
                "microphone",
                "speaker"
            ]
        },
        "systemImage": {
            "apiLevel": 33,
            "abi": "arm64-v8a",
            "tag": "default"
        }
    }
    return profile


def main():
    parser = argparse.ArgumentParser(
        description="ToroidalOS Mi Mix Emulator"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["simulation", "qemu", "android"],
        default="simulation",
        help="Emulation mode"
    )
    parser.add_argument(
        "--rootfs",
        help="Path to rootfs image (for QEMU mode)"
    )
    parser.add_argument(
        "--kernel",
        help="Path to kernel image (for QEMU mode)"
    )
    parser.add_argument(
        "--avd",
        help="AVD name (for Android emulator mode)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--create-avd",
        action="store_true",
        help="Create Mi Mix AVD profile"
    )

    args = parser.parse_args()

    if args.create_avd:
        profile = create_avd_profile()
        print(json.dumps(profile, indent=2))
        print("\nTo create this AVD:")
        print("  1. Install Android SDK")
        print("  2. Run: avdmanager create avd -n lithium -k 'system-images;android-33;default;arm64-v8a'")
        print("  3. Edit ~/.android/avd/lithium.avd/config.ini with Mi Mix specs")
        return

    mode = {
        "simulation": EmulatorMode.SIMULATION,
        "qemu": EmulatorMode.QEMU_ARM,
        "android": EmulatorMode.ANDROID_EMULATOR,
    }[args.mode]

    emulator = MiMixEmulator(
        mode=mode,
        rootfs_path=args.rootfs,
        kernel_path=args.kernel,
        avd_name=args.avd,
    )

    # Handle Ctrl+C
    def on_signal(signum, frame):
        print("\n[EMULATOR] Shutting down...")
        emulator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Start emulator
    if not emulator.start():
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ToroidalOS Mi Mix Emulator")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Device: {emulator.hardware.DEVICE}")
    print(f"Sensors: {emulator.hardware.get_sensor_bits():09b}")
    print("=" * 60)

    if args.interactive:
        print("\nInteractive mode. Type 'help' for commands, 'quit' to exit.")
        while True:
            try:
                cmd = input("emulator> ").strip()
                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "help":
                    print("Commands:")
                    print("  state     - Show emulator state")
                    print("  sensors   - Show sensor readings")
                    print("  bits      - Show 9-bit sensor state")
                    print("  exec CMD  - Execute command")
                    print("  quit      - Exit emulator")
                elif cmd == "state":
                    print(json.dumps(emulator.get_state(), indent=2))
                elif cmd == "sensors":
                    for name, reading in emulator.hardware.read_sensors().items():
                        print(f"  {name}: {reading.value:.2f} {reading.unit}")
                elif cmd == "bits":
                    bits = emulator.hardware.get_sensor_bits()
                    print(f"  Sensor bits: {bits:09b} (0x{bits:03x})")
                    print(f"  MOTION:  {(bits >> 0) & 0b111:03b}")
                    print(f"  ENVIRON: {(bits >> 3) & 0b111:03b}")
                    print(f"  SEMANTIC: {(bits >> 6) & 0b111:03b}")
                elif cmd.startswith("exec "):
                    result = emulator.execute_command(cmd[5:])
                    print(result)
                else:
                    result = emulator.execute_command(cmd)
                    print(result)
            except EOFError:
                break
    else:
        # Just print state periodically
        try:
            while emulator.running:
                time.sleep(5)
                state = emulator.get_state()
                print(f"[EMULATOR] Sensors: {state['sensor_bits']:09b} | "
                      f"Battery: {state['sensors']['battery']['value']:.0f}% | "
                      f"Light: {state['sensors']['light']['value']:.0f} lux")
        except KeyboardInterrupt:
            pass

    emulator.stop()
    print("[EMULATOR] Goodbye!")


if __name__ == "__main__":
    main()