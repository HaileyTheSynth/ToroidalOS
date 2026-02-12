#!/usr/bin/env python3
"""
ToroidalOS — Hardware Abstraction Layer for Xiaomi Mi Mix (lithium)
=====================================================================

Maps every physical sensor on the Mi Mix into the Topo9 topological
kernel.  The central idea: each sensor class owns a *region* of the
9-bit state space, and the topological metrics (Berry phase, coherence,
curvature, solenoid history) give the reasoning engine invariants that
raw sensor values alone cannot provide.

Hardware inventory (lithium):
─────────────────────────────────────────────────────────
  SoC          Qualcomm Snapdragon 821 (MSM8996pro)
               4× Kryo cores (2×2.35 GHz + 2×2.19 GHz)
               Adreno 530 GPU @ 653 MHz
               Hexagon 680 DSP
  RAM          6 GB LPDDR4
  Storage      256 GB UFS 2.0
  Display      6.4″ IPS LCD  1080×2040  (Sharp)
  Rear cam     OmniVision OV16880  16 MP  f/2.0  PDAF
  Front cam    OmniVision OV5675    5 MP  f/2.2
  Audio        Piezoelectric ceramic earpiece + bottom speaker
               Dual microphones (noise-cancelling)
  Battery      4 400 mAh  QC 3.0  (18 W)
─────────────────────────────────────────────────────────
  SENSORS
  ├─ Accelerometer          (3-axis, ~100 Hz)
  ├─ Gyroscope              (3-axis, ~200 Hz)
  ├─ Magnetometer / Compass (3-axis)
  ├─ Barometer / Altimeter
  ├─ Light / Ambient sensor
  ├─ Ultrasonic proximity   (unique to Mi Mix — no IR)
  ├─ Gravity sensor         (virtual, from accel)
  ├─ Hall-effect sensor     (magnetic flip cover)
  ├─ GPS / GLONASS / BDS
  ├─ Fingerprint            (rear-mounted, capacitive)
  └─ NFC

Mapping to Topo9 regions
─────────────────────────────────────────────────────────
The 9-bit state is split into three 3-bit regions:

  Region 0 (bits 0-2):  MOTION    — accelerometer, gyroscope, gravity
  Region 1 (bits 3-5):  ENVIRON   — light, proximity, barometer, temp
  Region 2 (bits 6-8):  SEMANTIC  — camera, audio, touch, NFC

Cross-region access naturally builds Berry phase, which signals that
the device is being used in a multi-modal way (e.g. the user picks up
the phone [motion], in a dark room [environ], and asks a question
[semantic]).  Bridge nodes that accumulate Berry phase across all three
regions represent *holistic situational awareness*.

Linux sysfs/iio paths (Android 8.0 / kernel 3.18–4.4):
  /sys/bus/iio/devices/iio:device*  — IIO sensor hub
  /sys/class/leds/                  — LED / backlight
  /dev/input/event*                 — touchscreen, keys
  /dev/video*                       — camera via V4L2
  /dev/snd/*                        — ALSA audio
  /sys/class/power_supply/          — battery
"""

import os
import struct
import time
import math
import threading
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any
from enum import IntEnum

# ═══════════════════════════════════════════════════════════════════════════
# REGION / BIT MAPPING
# ═══════════════════════════════════════════════════════════════════════════

class Region(IntEnum):
    MOTION   = 0   # bits 0-2
    ENVIRON  = 1   # bits 3-5
    SEMANTIC = 2   # bits 6-8


# Each bit position has a semantic meaning within its region:
BIT_SEMANTICS = {
    # MOTION
    0: "accel_active",       # significant acceleration detected
    1: "rotation_active",    # significant angular velocity
    2: "orientation_change", # device orientation flipped

    # ENVIRON
    3: "ambient_bright",     # above light threshold
    4: "proximity_near",     # ultrasonic proximity triggered
    5: "pressure_change",    # barometric pressure changing

    # SEMANTIC
    6: "audio_speech",       # speech detected in mic input
    7: "camera_active",      # camera frame being processed
    8: "touch_active",       # user touching screen
}


def quantize_to_bit(value: float, lo: float, hi: float) -> int:
    """Return 1 if value is above midpoint of [lo, hi], else 0."""
    return 1 if value >= (lo + hi) / 2 else 0


def bits_to_state(bits: List[int]) -> int:
    """Convert a list of 9 bit values to a 9-bit int (MSB-first)."""
    s = 0
    for i, b in enumerate(bits):
        s |= (b & 1) << i
    return s & 0x1FF


# ═══════════════════════════════════════════════════════════════════════════
# SENSOR READERS — Linux / Android sysfs paths
# ═══════════════════════════════════════════════════════════════════════════

def _read_sysfs(path: str) -> Optional[str]:
    """Read a single sysfs file.  Returns None on failure."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, IOError):
        return None


def _read_iio_raw(device: str, channel: str) -> Optional[float]:
    """Read an IIO raw value and apply scale."""
    base = f"/sys/bus/iio/devices/{device}"
    raw = _read_sysfs(f"{base}/in_{channel}_raw")
    scale = _read_sysfs(f"{base}/in_{channel}_scale")
    if raw is None:
        return None
    r = float(raw)
    s = float(scale) if scale else 1.0
    return r * s


def _find_iio_device(name_substr: str) -> Optional[str]:
    """Find an IIO device whose 'name' file contains a substring."""
    iio_base = "/sys/bus/iio/devices"
    if not os.path.isdir(iio_base):
        return None
    for entry in os.listdir(iio_base):
        name_path = os.path.join(iio_base, entry, "name")
        n = _read_sysfs(name_path)
        if n and name_substr.lower() in n.lower():
            return entry
    return None


# ─── Accelerometer ────────────────────────────────────────────────────────

@dataclass
class AccelReading:
    x: float = 0.0   # m/s²
    y: float = 0.0
    z: float = 0.0
    magnitude: float = 0.0
    timestamp: float = 0.0


class AccelSensor:
    """3-axis accelerometer via IIO or /dev/input."""

    GRAVITY = 9.81
    THRESHOLD_ACTIVE = 1.5   # m/s² above/below 1g → "active"

    def __init__(self):
        self.device = _find_iio_device("accel") or _find_iio_device("bmi")
        self._last = AccelReading()

    def read(self) -> AccelReading:
        if self.device:
            x = _read_iio_raw(self.device, "accel_x") or 0.0
            y = _read_iio_raw(self.device, "accel_y") or 0.0
            z = _read_iio_raw(self.device, "accel_z") or 0.0
        else:
            x, y, z = 0.0, 0.0, self.GRAVITY  # stationary fallback
        mag = math.sqrt(x*x + y*y + z*z)
        self._last = AccelReading(x, y, z, mag, time.time())
        return self._last

    def to_bits(self) -> int:
        """Bit 0: 1 if acceleration deviates significantly from 1g."""
        delta = abs(self._last.magnitude - self.GRAVITY)
        return 1 if delta > self.THRESHOLD_ACTIVE else 0


# ─── Gyroscope ────────────────────────────────────────────────────────────

@dataclass
class GyroReading:
    x: float = 0.0   # rad/s
    y: float = 0.0
    z: float = 0.0
    magnitude: float = 0.0
    timestamp: float = 0.0


class GyroSensor:
    THRESHOLD_ROTATING = 0.3  # rad/s

    def __init__(self):
        self.device = _find_iio_device("gyro") or _find_iio_device("bmi")
        self._last = GyroReading()

    def read(self) -> GyroReading:
        if self.device:
            x = _read_iio_raw(self.device, "anglvel_x") or 0.0
            y = _read_iio_raw(self.device, "anglvel_y") or 0.0
            z = _read_iio_raw(self.device, "anglvel_z") or 0.0
        else:
            x, y, z = 0.0, 0.0, 0.0
        mag = math.sqrt(x*x + y*y + z*z)
        self._last = GyroReading(x, y, z, mag, time.time())
        return self._last

    def to_bits(self) -> int:
        """Bit 1: 1 if rotating."""
        return 1 if self._last.magnitude > self.THRESHOLD_ROTATING else 0


# ─── Orientation (virtual — from accel) ──────────────────────────────────

class OrientationSensor:
    """Derives device orientation from accelerometer z-axis."""

    def __init__(self, accel: AccelSensor):
        self._accel = accel
        self._last_face_up = True

    def to_bits(self) -> int:
        """Bit 2: 1 if orientation has changed (face-down, landscape, etc.)."""
        r = self._accel._last
        face_up = r.z > 0
        changed = face_up != self._last_face_up
        self._last_face_up = face_up
        return 1 if changed else 0


# ─── Light sensor ─────────────────────────────────────────────────────────

class LightSensor:
    """Ambient light via IIO or /sys/class/leds/lcd-backlight."""

    THRESHOLD_BRIGHT = 200  # lux

    def __init__(self):
        self.device = _find_iio_device("light") or _find_iio_device("apds")
        self._last_lux = 0.0

    def read(self) -> float:
        if self.device:
            v = _read_iio_raw(self.device, "illuminance")
            if v is not None:
                self._last_lux = v
                return v
        # Fallback: read backlight brightness as proxy
        bl = _read_sysfs("/sys/class/leds/lcd-backlight/brightness")
        if bl:
            self._last_lux = float(bl) / 255 * 1000  # rough lux estimate
        return self._last_lux

    def to_bits(self) -> int:
        """Bit 3: 1 if bright ambient light."""
        return 1 if self._last_lux >= self.THRESHOLD_BRIGHT else 0


# ─── Ultrasonic proximity (unique to Mi Mix) ─────────────────────────────

class ProximitySensor:
    """
    The Mi Mix uses an ultrasonic proximity sensor instead of the
    typical IR proximity.  Exposed via /dev/input/event* or IIO.
    """

    def __init__(self):
        self.device = _find_iio_device("prox") or _find_iio_device("ultrasonic")
        self._near = False
        # Fallback: try input event
        self._input_path = self._find_proximity_input()

    def _find_proximity_input(self) -> Optional[str]:
        for i in range(20):
            name_path = f"/sys/class/input/input{i}/name"
            n = _read_sysfs(name_path)
            if n and "prox" in n.lower():
                return f"/dev/input/event{i}"
        return None

    def read(self) -> bool:
        if self.device:
            v = _read_iio_raw(self.device, "proximity")
            if v is not None:
                self._near = v > 0.5
                return self._near
        return self._near

    def to_bits(self) -> int:
        """Bit 4: 1 if proximity near."""
        return 1 if self._near else 0


# ─── Barometer ────────────────────────────────────────────────────────────

class BarometerSensor:
    """Barometric pressure via IIO.  Also provides crude temperature."""

    THRESHOLD_HPA_CHANGE = 2.0  # hPa change → "weather changing"

    def __init__(self):
        self.device = _find_iio_device("pressure") or _find_iio_device("bmp")
        self._last_hpa = 1013.25
        self._baseline_hpa = 1013.25

    def read(self) -> float:
        if self.device:
            v = _read_iio_raw(self.device, "pressure")
            if v is not None:
                self._last_hpa = v / 100.0  # Pa → hPa
        return self._last_hpa

    def read_temperature(self) -> Optional[float]:
        """Some barometer chips also report temperature."""
        if self.device:
            v = _read_iio_raw(self.device, "temp")
            if v is not None:
                return v / 1000.0  # millideg → deg C
        return None

    def to_bits(self) -> int:
        """Bit 5: 1 if pressure is changing significantly."""
        delta = abs(self._last_hpa - self._baseline_hpa)
        return 1 if delta > self.THRESHOLD_HPA_CHANGE else 0


# ─── Audio / Microphone ──────────────────────────────────────────────────

class AudioSensor:
    """
    Dual-microphone input.  Detects speech activity by RMS energy.
    On the Mi Mix, the ceramic body acts as the earpiece (piezo-driven),
    but capture uses standard ALSA.

    In full integration, this would feed audio to Qwen2.5-Omni for
    transcription / understanding.  Here we just detect voice activity.
    """

    RMS_SPEECH_THRESHOLD = 500  # 16-bit PCM RMS

    def __init__(self, alsa_device: str = "hw:0,0"):
        self.alsa_device = alsa_device
        self._speech_detected = False
        self._rms = 0.0

    def read_rms(self) -> float:
        """Read a short burst and compute RMS.  Non-blocking fallback."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                            input=True, frames_per_buffer=1024)
            data = stream.read(1024, exception_on_overflow=False)
            stream.stop_stream()
            stream.close()
            p.terminate()
            samples = struct.unpack(f"{len(data)//2}h", data)
            rms = math.sqrt(sum(s*s for s in samples) / len(samples))
            self._rms = rms
            self._speech_detected = rms > self.RMS_SPEECH_THRESHOLD
        except Exception:
            pass
        return self._rms

    def to_bits(self) -> int:
        """Bit 6: 1 if speech/voice activity detected."""
        return 1 if self._speech_detected else 0


# ─── Camera ───────────────────────────────────────────────────────────────

class CameraSensor:
    """
    OmniVision OV16880 rear / OV5675 front.
    V4L2 device at /dev/video0 (rear) / /dev/video1 (front).

    Full pipeline: capture frame → Qwen2.5-Omni vision → percept node.
    Here we just track whether the camera is active / frames flowing.
    """

    def __init__(self, device: str = "/dev/video0"):
        self.device = device
        self._active = False

    def is_active(self) -> bool:
        """Check if camera is currently open by another process."""
        try:
            # On Linux, if the device is busy it's being used
            fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
            os.close(fd)
            self._active = False
        except OSError:
            self._active = True  # device busy = someone is using it
        return self._active

    def to_bits(self) -> int:
        """Bit 7: 1 if camera is capturing."""
        return 1 if self._active else 0


# ─── Touchscreen ──────────────────────────────────────────────────────────

class TouchSensor:
    """
    Capacitive multi-touch via /dev/input/event*.
    On Mi Mix: 6.4″ panel, 1080×2040.
    """

    def __init__(self):
        self._touching = False
        self._last_xy = (0, 0)
        self._input_path = self._find_touch_input()

    def _find_touch_input(self) -> Optional[str]:
        for i in range(20):
            name_path = f"/sys/class/input/input{i}/name"
            n = _read_sysfs(name_path)
            if n and ("touch" in n.lower() or "synaptics" in n.lower()
                      or "atmel" in n.lower() or "goodix" in n.lower()):
                return f"/dev/input/event{i}"
        return None

    def poll(self) -> bool:
        """Non-blocking poll for recent touch activity."""
        if not self._input_path:
            return self._touching
        try:
            # Check if there are recent events (last 200ms)
            stat = os.stat(self._input_path)
            self._touching = (time.time() - stat.st_mtime) < 0.2
        except OSError:
            pass
        return self._touching

    def to_bits(self) -> int:
        """Bit 8: 1 if user is touching screen."""
        return 1 if self._touching else 0


# ─── Battery ──────────────────────────────────────────────────────────────

@dataclass
class BatteryState:
    percent: int = 100
    charging: bool = False
    voltage_uv: int = 4200000
    temp_tenths_c: int = 250  # 25.0°C


class BatterySensor:
    """4400 mAh Li-Ion, QC 3.0."""

    BASE = "/sys/class/power_supply/battery"

    def read(self) -> BatteryState:
        pct = _read_sysfs(f"{self.BASE}/capacity")
        status = _read_sysfs(f"{self.BASE}/status")
        volt = _read_sysfs(f"{self.BASE}/voltage_now")
        temp = _read_sysfs(f"{self.BASE}/temp")
        return BatteryState(
            percent=int(pct) if pct else 100,
            charging=(status or "").lower() in ("charging", "full"),
            voltage_uv=int(volt) if volt else 4200000,
            temp_tenths_c=int(temp) if temp else 250,
        )


# ─── Hall-effect (magnetic flip cover detect) ────────────────────────────

class HallSensor:
    def __init__(self):
        self._triggered = False
        self._input_path = self._find_hall_input()

    def _find_hall_input(self):
        for i in range(20):
            n = _read_sysfs(f"/sys/class/input/input{i}/name")
            if n and "hall" in n.lower():
                return f"/dev/input/event{i}"
        return None

    def is_covered(self) -> bool:
        return self._triggered


# ═══════════════════════════════════════════════════════════════════════════
# SENSOR FUSION → TOPO9 STATE
# ═══════════════════════════════════════════════════════════════════════════

class SensorHub:
    """
    Reads all sensors at a configurable rate and fuses them into a
    9-bit Topo9 state vector.

    Lifecycle:
      hub = SensorHub()
      hub.start(poll_hz=10)    # background polling at 10 Hz
      state = hub.state()      # current 9-bit fused state
      region = hub.active_region()  # most recently changed region
      hub.stop()

    The hub also provides the raw readings for the perception engine
    to feed into Qwen2.5-Omni for higher-level understanding.
    """

    def __init__(self):
        self.accel = AccelSensor()
        self.gyro = GyroSensor()
        self.orient = OrientationSensor(self.accel)
        self.light = LightSensor()
        self.prox = ProximitySensor()
        self.baro = BarometerSensor()
        self.audio = AudioSensor()
        self.camera = CameraSensor()
        self.touch = TouchSensor()
        self.battery = BatterySensor()
        self.hall = HallSensor()

        self._state = 0
        self._prev_state = 0
        self._active_region = Region.MOTION
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_count = 0
        self._history: List[int] = []  # recent states for sparkline

    # ── polling ──

    def start(self, poll_hz: float = 10.0):
        """Start background sensor polling."""
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(1.0 / poll_hz,),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _poll_loop(self, interval: float):
        while self._running:
            self._sample()
            time.sleep(interval)

    def _sample(self):
        """Read all sensors and fuse into 9-bit state."""
        # Read raw
        self.accel.read()
        self.gyro.read()
        self.light.read()
        self.prox.read()
        self.baro.read()
        self.camera.is_active()
        self.touch.poll()
        # Audio RMS is expensive; sample less often
        if self._poll_count % 5 == 0:
            self.audio.read_rms()

        bits = [0] * 9
        # Region 0: MOTION
        bits[0] = self.accel.to_bits()
        bits[1] = self.gyro.to_bits()
        bits[2] = self.orient.to_bits()
        # Region 1: ENVIRON
        bits[3] = self.light.to_bits()
        bits[4] = self.prox.to_bits()
        bits[5] = self.baro.to_bits()
        # Region 2: SEMANTIC
        bits[6] = self.audio.to_bits()
        bits[7] = self.camera.to_bits()
        bits[8] = self.touch.to_bits()

        new_state = bits_to_state(bits)

        with self._lock:
            self._prev_state = self._state
            self._state = new_state
            # determine which region changed most
            diff = (self._state ^ self._prev_state) & 0x1FF
            r0_changed = sum((diff >> i) & 1 for i in range(0, 3))
            r1_changed = sum((diff >> i) & 1 for i in range(3, 6))
            r2_changed = sum((diff >> i) & 1 for i in range(6, 9))
            if r0_changed >= r1_changed and r0_changed >= r2_changed:
                self._active_region = Region.MOTION
            elif r1_changed >= r2_changed:
                self._active_region = Region.ENVIRON
            else:
                self._active_region = Region.SEMANTIC
            self._history.append(new_state)
            if len(self._history) > 128:
                self._history = self._history[-128:]

        self._poll_count += 1

    # ── public API ──

    def state(self) -> int:
        """Current fused 9-bit state."""
        with self._lock:
            return self._state

    def active_region(self) -> Region:
        """Region that changed most in the last sample."""
        with self._lock:
            return self._active_region

    def bits_str(self) -> str:
        """Human-readable bits grouped by region."""
        s = self.state()
        b = ''.join(str((s >> (8 - i)) & 1) for i in range(9))
        return f"{b[:3]} {b[3:6]} {b[6:]}"

    def describe(self) -> str:
        """Textual description of current sensor state."""
        s = self.state()
        active = [BIT_SEMANTICS[i] for i in range(9) if (s >> i) & 1]
        return ", ".join(active) if active else "(idle)"

    def raw_snapshot(self) -> Dict[str, Any]:
        """Full snapshot of all raw sensor values for the perception engine."""
        bat = self.battery.read()
        temp = self.baro.read_temperature()
        return {
            "accel": {"x": self.accel._last.x, "y": self.accel._last.y,
                      "z": self.accel._last.z, "mag": self.accel._last.magnitude},
            "gyro": {"x": self.gyro._last.x, "y": self.gyro._last.y,
                     "z": self.gyro._last.z, "mag": self.gyro._last.magnitude},
            "light_lux": self.light._last_lux,
            "proximity_near": self.prox._near,
            "pressure_hpa": self.baro._last_hpa,
            "temperature_c": temp,
            "audio_rms": self.audio._rms,
            "camera_active": self.camera._active,
            "touch_active": self.touch._touching,
            "battery": {"pct": bat.percent, "charging": bat.charging,
                        "temp_c": bat.temp_tenths_c / 10},
            "hall_covered": self.hall.is_covered(),
            "state_bits": self.bits_str(),
            "active_region": self._active_region.name,
            "poll_count": self._poll_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# KERNEL BRIDGE — connect SensorHub ↔ Topo9 Kernel
# ═══════════════════════════════════════════════════════════════════════════

class KernelBridge:
    """
    Bridges the physical SensorHub to the Topo9 kernel.

    Each poll cycle:
    1. Read fused sensor state from SensorHub
    2. STORE or update a "live-sensor" node in the kernel
    3. ACCESS it in the correct region → builds Berry phase
    4. When cross-region transitions happen, the kernel's coherence,
       curvature, and bridge metrics evolve naturally
    5. Periodically EVOLVE to let hyperedge sync run

    The reasoning engine queries the kernel for topological invariants
    (coherence, bridges, curvature, solenoid history) that encode
    sensor *patterns* rather than raw values.
    """

    def __init__(self, kernel, hub: SensorHub):
        """
        Args:
            kernel: The Topo9 Kernel instance (from simulate.py or kernel.c via QEMU)
            hub: SensorHub instance reading from hardware
        """
        self.kernel = kernel
        self.hub = hub
        self._sensor_node_id: Optional[int] = None
        self._tick_counter = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, update_hz: float = 5.0):
        """Start the bridge loop."""
        # Create initial sensor node
        state = self.hub.state()
        region = int(self.hub.active_region())
        self._sensor_node_id = self.kernel.store_node(state, region)

        # Create a FIBER hyperedge connecting sensor node to seed nodes
        seed_ids = list(range(min(self.kernel.node_count, 8)))
        if self._sensor_node_id is not None:
            self.kernel.hedge_add(3,  # HEDGE_FIBER
                                  seed_ids + [self._sensor_node_id])

        self._running = True
        self._thread = threading.Thread(
            target=self._bridge_loop,
            args=(1.0 / update_hz,),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _bridge_loop(self, interval: float):
        while self._running:
            self._update()
            time.sleep(interval)

    def _update(self):
        nid = self._sensor_node_id
        if nid is None:
            return

        state = self.hub.state()
        region = int(self.hub.active_region())

        # Update node state to match current sensor fusion
        self.kernel.nodes[nid].state = state & 0x1FF

        # ACCESS in the active region → drives Berry phase
        self.kernel.record_access(nid, region)
        self.kernel.update_edges_on_access(nid)
        self.kernel.ctx_push(nid)

        self._tick_counter += 1

        # Every 20 cycles, run an EVOLVE step (hyperedge sync)
        if self._tick_counter % 20 == 0:
            self.kernel.evolve_steps(1)

    # ── queries for the reasoning engine ──

    def coherence(self) -> int:
        """Current coherence score of the sensor node."""
        if self._sensor_node_id is not None:
            return self.kernel.coherence_score(self._sensor_node_id)
        return 500

    def curvature(self) -> int:
        """Current context curvature."""
        return self.kernel.curvature_scaled()

    def berry_phase(self) -> int:
        """Accumulated Berry phase of the sensor node."""
        nid = self._sensor_node_id
        if nid is not None:
            return self.kernel.nodes[nid].berry_milli
        return 0

    def is_bridge(self, min_berry: int = 1200) -> bool:
        """Is the sensor node a bridge (multi-region, high Berry)?"""
        nid = self._sensor_node_id
        if nid is not None:
            return self.kernel.is_bridge(nid, min_berry)
        return False

    def solenoid_history(self) -> List[int]:
        """State history (solenoid) of the sensor node."""
        nid = self._sensor_node_id
        if nid is not None:
            n = self.kernel.nodes[nid]
            return n.sol_hist[:n.sol_len]
        return []

    def situation_summary(self) -> Dict[str, Any]:
        """
        High-level situational summary for the reasoning engine.
        This is what makes the topological approach valuable:
        instead of raw sensor dumps, the engine gets *invariants*.
        """
        return {
            "state": self.hub.bits_str(),
            "description": self.hub.describe(),
            "active_region": self.hub.active_region().name,
            "coherence": self.coherence(),
            "curvature": self.curvature(),
            "berry_phase": self.berry_phase(),
            "is_bridge": self.is_bridge(),
            "windings": (self.kernel.nodes[self._sensor_node_id].windings
                         if self._sensor_node_id is not None else 0),
            "solenoid_depth": len(self.solenoid_history()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# DEMO / SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  ToroidalOS HAL — Xiaomi Mi Mix (lithium)")
    print("  Sensor → Topology bridge self-test")
    print("=" * 60)

    # Attempt real sensor reads
    hub = SensorHub()
    hub._sample()  # single sample

    print(f"\n  Fused state:  {hub.bits_str()}")
    print(f"  Active bits:  {hub.describe()}")
    print(f"  Region:       {hub.active_region().name}")

    snap = hub.raw_snapshot()
    print(f"\n  Raw snapshot:")
    for k, v in snap.items():
        print(f"    {k}: {v}")

    # Test with simulated kernel
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    try:
        from simulate import Kernel
        k = Kernel()
        k.init_seed_nodes()

        bridge = KernelBridge(k, hub)
        bridge._sensor_node_id = k.store_node(hub.state(), int(hub.active_region()))

        # Simulate 10 cycles of cross-region activity
        for region in [0, 1, 2, 0, 2, 1, 0, 1, 2, 2]:
            hub._state = hub.state() ^ (1 << (region * 3))  # flip a bit in that region
            hub._active_region = Region(region)
            bridge._update()

        summary = bridge.situation_summary()
        print(f"\n  After 10 simulated cycles:")
        for k2, v2 in summary.items():
            print(f"    {k2}: {v2}")

    except ImportError as e:
        print(f"\n  (Kernel import unavailable: {e})")

    print(f"\n  Self-test complete.")
