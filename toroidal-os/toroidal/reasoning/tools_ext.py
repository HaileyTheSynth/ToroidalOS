#!/usr/bin/env python3
"""
TOROIDAL OS — Extended Tools
==============================
Additional tools registered via the topo:// protocol.

- web_fetch:       HTTP GET gated by epistemic state detector
- web_search:      Keyword search gated by epistemic state detector
- sensor_request:  Active sensor query (request specific sensor reading)
- time_now:        Current time and uptime
- system_info:     Hardware and memory stats
"""

import time
import os
import json
from typing import Dict, Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus


# ============================================================================
# WEB FETCH (gated by epistemic prerequisite)
# ============================================================================

def handle_web_fetch(args: Dict[str, Any]) -> str:
    """
    Fetch a URL and return its text content.

    Gated by the epistemic state detector — only called when the system
    is in KNOWLEDGE_GAP state (doesn't know something and needs to look
    it up). This prevents indiscriminate web crawling.

    Constraints for Mi Mix (Snapdragon 821, often on mobile data):
    - 10 second timeout
    - Max 4000 chars returned (memory budget)
    - Text-only (no images, JS)
    - User-Agent identifies the system
    """
    url = args.get("url", "")
    max_chars = int(args.get("max_chars", 4000))

    if not url:
        return "No URL provided"

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL (must start with http:// or https://): {url}"

    # Block private/internal IPs
    blocked_prefixes = ["http://127.", "http://localhost", "http://10.",
                        "http://172.16.", "http://192.168.", "https://127.",
                        "https://localhost"]
    for prefix in blocked_prefixes:
        if url.startswith(prefix):
            return f"Blocked: cannot fetch internal/private URLs"

    try:
        req = Request(
            url,
            headers={
                "User-Agent": "ToroidalOS/0.1 (topological-memory; +https://github.com/toroidal-os)",
                "Accept": "text/html, text/plain, application/json",
            },
        )
        with urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(max_chars * 2)  # Read extra to account for HTML overhead

            # Decode
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            text = raw.decode(charset, errors="replace")

            # Strip HTML tags if HTML content
            if "html" in content_type.lower():
                text = _strip_html(text)

            # Truncate
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[truncated]"

            return text

    except URLError as e:
        return f"Fetch error: {e.reason}"
    except Exception as e:
        return f"Fetch error: {str(e)}"


def handle_web_search(args: Dict[str, Any]) -> str:
    """
    Perform a text search query.

    Since we can't access actual search engines programmatically
    without API keys on a resource-constrained device, this tool
    constructs a search URL that could be fetched, or (when
    available) queries a local search index.

    In practice, this would integrate with:
    - A local knowledge base built from previously fetched pages
    - DuckDuckGo Lite (text-based, no JS required)
    - A cached offline dataset
    """
    query = args.get("query", "")
    if not query:
        return "No query provided"

    max_results = int(args.get("max_results", 3))

    # Try DuckDuckGo Lite (text-only, works without JS)
    encoded_query = quote_plus(query)
    search_url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"

    try:
        req = Request(
            search_url,
            headers={
                "User-Agent": "ToroidalOS/0.1 (topological-memory)",
                "Accept": "text/html",
            },
        )
        with urlopen(req, timeout=10) as resp:
            raw = resp.read(8000)
            text = raw.decode("utf-8", errors="replace")
            text = _strip_html(text)

            # Extract meaningful lines (skip empty lines and navigation)
            lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 20]
            result_lines = lines[:max_results * 3]  # ~3 lines per result

            if result_lines:
                return f"Search results for '{query}':\n" + "\n".join(result_lines[:max_results * 3])
            else:
                return f"No results found for '{query}'"

    except Exception as e:
        return f"Search failed: {str(e)}. Try using web_fetch with a specific URL."


# ============================================================================
# ACTIVE SENSOR REQUESTS
# ============================================================================

def handle_sensor_request(args: Dict[str, Any]) -> str:
    """
    Request a specific sensor reading.

    Unlike the passive `sensors` tool (which reads the fused 9-bit state),
    this tool makes active requests for specific sensor data:
    - "battery": Current battery level and charging state
    - "location": GPS coordinates (if available)
    - "orientation": Device orientation angles
    - "light": Current ambient light level
    - "pressure": Barometric pressure (altitude proxy)

    On the Mi Mix, these read from sysfs/IIO paths.
    In simulation, returns synthetic data.
    """
    sensor = args.get("sensor", "").lower()
    if not sensor:
        return "No sensor specified. Available: battery, location, orientation, light, pressure"

    if sensor == "battery":
        return _read_battery()
    elif sensor == "location":
        return _read_location()
    elif sensor == "orientation":
        return _read_orientation()
    elif sensor == "light":
        return _read_light()
    elif sensor == "pressure":
        return _read_pressure()
    else:
        return f"Unknown sensor '{sensor}'. Available: battery, location, orientation, light, pressure"


def _read_battery() -> str:
    """Read battery from sysfs or return synthetic."""
    paths = [
        "/sys/class/power_supply/battery/capacity",
        "/sys/class/power_supply/BAT0/capacity",
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    pct = f.read().strip()
                status_path = path.replace("capacity", "status")
                status = "unknown"
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        status = f.read().strip()
                return f"Battery: {pct}% ({status})"
            except Exception:
                pass
    # Synthetic
    return "Battery: 78% (discharging) [simulated]"


def _read_location() -> str:
    """Read GPS location (requires gpsd or Android location service)."""
    # In practice, this would query gpsd or Android LocationManager
    return "Location: unavailable (GPS not active) [simulated: 51.5074°N, 0.1278°W]"


def _read_orientation() -> str:
    """Read device orientation from IIO sensors."""
    accel_paths = [
        "/sys/bus/iio/devices/iio:device0/in_accel_x_raw",
    ]
    for path in accel_paths:
        if os.path.exists(path):
            try:
                base = os.path.dirname(path)
                x = int(open(os.path.join(base, "in_accel_x_raw")).read().strip())
                y = int(open(os.path.join(base, "in_accel_y_raw")).read().strip())
                z = int(open(os.path.join(base, "in_accel_z_raw")).read().strip())
                return f"Orientation: x={x} y={y} z={z} (raw accelerometer)"
            except Exception:
                pass
    return "Orientation: flat on table (z-up) [simulated]"


def _read_light() -> str:
    """Read ambient light sensor."""
    light_paths = [
        "/sys/bus/iio/devices/iio:device0/in_illuminance_raw",
        "/sys/bus/iio/devices/iio:device1/in_illuminance_raw",
    ]
    for path in light_paths:
        if os.path.exists(path):
            try:
                val = int(open(path).read().strip())
                return f"Ambient light: {val} lux"
            except Exception:
                pass
    return "Ambient light: 350 lux (indoor, moderate) [simulated]"


def _read_pressure() -> str:
    """Read barometric pressure."""
    baro_paths = [
        "/sys/bus/iio/devices/iio:device0/in_pressure_input",
        "/sys/bus/iio/devices/iio:device1/in_pressure_input",
    ]
    for path in baro_paths:
        if os.path.exists(path):
            try:
                val = float(open(path).read().strip())
                return f"Pressure: {val:.1f} hPa (altitude ~{_hpa_to_altitude(val):.0f}m)"
            except Exception:
                pass
    return "Pressure: 1013.2 hPa (~0m altitude, sea level) [simulated]"


def _hpa_to_altitude(hpa: float) -> float:
    """Approximate altitude from barometric pressure (ISA formula)."""
    return 44330 * (1 - (hpa / 1013.25) ** 0.1903)


# ============================================================================
# SYSTEM INFO
# ============================================================================

def handle_time_now(args: Dict[str, Any]) -> str:
    """Return current time and system uptime."""
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    uptime_str = "unknown"
    if os.path.exists("/proc/uptime"):
        try:
            with open("/proc/uptime") as f:
                secs = float(f.read().split()[0])
            hours = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            uptime_str = f"{hours}h {mins}m"
        except Exception:
            pass

    return f"Time: {now}\nUptime: {uptime_str}"


def handle_system_info(args: Dict[str, Any]) -> str:
    """Return hardware and memory information."""
    lines = []

    # Memory info
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith(("MemTotal", "MemAvailable", "MemFree")):
                        lines.append(line.strip())
        except Exception:
            pass

    # CPU info
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith(("model name", "Hardware", "Processor")):
                        lines.append(line.strip())
                        break
        except Exception:
            pass

    # Load average
    if os.path.exists("/proc/loadavg"):
        try:
            with open("/proc/loadavg") as f:
                lines.append(f"Load: {f.read().strip()}")
        except Exception:
            pass

    if not lines:
        lines.append("System: Xiaomi Mi Mix (lithium)")
        lines.append("SoC: Snapdragon 821 (4x Kryo)")
        lines.append("RAM: 6GB LPDDR4")
        lines.append("[simulated — /proc not accessible]")

    return "\n".join(lines)


# ============================================================================
# WIFI & BLUETOOTH (Android Native Integration)
# ============================================================================
# These tools delegate to Android's native connectivity via:
# - adb shell commands (when running via adb)
# - Direct sysfs/property access (when running native on device)
# - Android property get/set (getprop/setprop)

def handle_wifi_status(args: Dict[str, Any]) -> str:
    """
    Get WiFi connection status.

    On Android/ToroidalOS, this queries:
    - /sys/class/net/wlan0/operstate
    - Android connectivity service via dumpsys
    - iwconfig if available
    """
    lines = []

    # Method 1: sysfs (works on native Linux/Android)
    wifi_paths = [
        "/sys/class/net/wlan0/operstate",
        "/sys/class/net/wlan0/carrier",
        "/sys/class/net/wlan0/address",
    ]
    for path in wifi_paths:
        if os.path.exists(path):
            try:
                val = open(path).read().strip()
                name = os.path.basename(path)
                lines.append(f"{name}: {val}")
            except Exception:
                pass

    # Method 2: iwconfig (if available)
    try:
        result = subprocess.run(
            ["iwconfig", "wlan0"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse key info from iwconfig output
            for line in result.stdout.split("\n"):
                if "ESSID" in line or "Access Point" in line or "Signal level" in line:
                    lines.append(line.strip())
    except Exception:
        pass

    # Method 3: Android dumpsys (when running on Android)
    try:
        result = subprocess.run(
            ["dumpsys", "wifi"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Extract Wi-Fi state from dumpsys
            for line in result.stdout.split("\n")[:20]:
                if "Wi-Fi is" in line or "mNetworkInfo" in line or "SSID" in line:
                    lines.append(line.strip())
    except Exception:
        pass

    # Method 4: getprop for Android WiFi properties
    try:
        result = subprocess.run(
            ["getprop", "wifi.interface"],
            capture_output=True, text=True, timeout=2
        )
        if result.stdout.strip():
            lines.append(f"WiFi interface: {result.stdout.strip()}")
    except Exception:
        pass

    if not lines:
        lines.append("WiFi: unavailable (no wlan0 interface)")
        lines.append("[simulated: connected to 'HomeNetwork', signal 65%]")

    return "\n".join(lines)


def handle_wifi_scan(args: Dict[str, Any]) -> str:
    """
    Scan for available WiFi networks.

    Uses iwlist or Android WifiManager scan results.
    """
    networks = []

    # Method 1: iwlist (native Linux)
    try:
        result = subprocess.run(
            ["iwlist", "wlan0", "scan"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            # Parse iwlist output
            current_ssid = None
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "ESSID:" in line:
                    # Extract SSID
                    ssid = line.split("ESSID:")[1].strip().strip('"')
                    if ssid:
                        current_ssid = ssid
                elif "Quality=" in line and current_ssid:
                    quality = line.split("Quality=")[1].split()[0]
                    networks.append(f"{current_ssid} ({quality})")
                    current_ssid = None
    except Exception:
        pass

    # Method 2: Android wpa_cli (if wpa_supplicant running)
    try:
        result = subprocess.run(
            ["wpa_cli", "scan_results"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n")[1:]:  # Skip header
                if line.strip() and "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 5:
                        networks.append(f"{parts[4]} ({parts[2]})")
    except Exception:
        pass

    if not networks:
        networks = [
            "[Simulated scan results]",
            "HomeNetwork (WPA2, signal 75%)",
            "Guest_WiFi (Open, signal 40%)",
            "Neighbor_5G (WPA3, signal 20%)",
        ]

    return "Available networks:\n" + "\n".join(networks[:10])


def handle_wifi_connect(args: Dict[str, Any]) -> str:
    """
    Connect to a WiFi network.

    Args:
        ssid: Network name
        password: Network password (optional for open networks)

    Uses wpa_supplicant or Android WifiManager.
    """
    ssid = args.get("ssid", "")
    password = args.get("password", "")

    if not ssid:
        return "No SSID provided. Usage: wifi_connect {ssid: 'NetworkName', password: 'secret'}"

    # Method 1: wpa_supplicant (native Linux/Android with wpa_cli)
    try:
        # Add network
        result = subprocess.run(
            ["wpa_cli", "add_network"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            net_id = result.stdout.strip().split()[-1]

            # Set SSID
            subprocess.run(
                ["wpa_cli", "set_network", net_id, "ssid", f'"{ssid}"'],
                capture_output=True, timeout=5
            )

            # Set password if provided
            if password:
                subprocess.run(
                    ["wpa_cli", "set_network", net_id, "psk", f'"{password}"'],
                    capture_output=True, timeout=5
                )
            else:
                # Open network
                subprocess.run(
                    ["wpa_cli", "set_network", net_id, "key_mgmt", "NONE"],
                    capture_output=True, timeout=5
                )

            # Enable network
            subprocess.run(
                ["wpa_cli", "enable_network", net_id],
                capture_output=True, timeout=5
            )

            # Save config
            subprocess.run(
                ["wpa_cli", "save_config"],
                capture_output=True, timeout=5
            )

            return f"Connecting to '{ssid}'... (check wifi_status for result)"
    except Exception as e:
        pass

    # Method 2: nmcli (NetworkManager, if available)
    try:
        if password:
            result = subprocess.run(
                ["nmcli", "device", "wifi", "connect", ssid, "password", password],
                capture_output=True, text=True, timeout=30
            )
        else:
            result = subprocess.run(
                ["nmcli", "device", "wifi", "connect", ssid],
                capture_output=True, text=True, timeout=30
            )

        if result.returncode == 0:
            return f"Connected to '{ssid}'"
        else:
            return f"Connection failed: {result.stderr.strip()}"
    except Exception:
        pass

    # Fallback: simulated
    return f"[Simulated] Would connect to '{ssid}' with password '{password[:3]}***'"


def handle_bluetooth_status(args: Dict[str, Any]) -> str:
    """
    Get Bluetooth adapter status.

    Queries BlueZ or Android Bluetooth service.
    """
    lines = []

    # Method 1: hciconfig (BlueZ on Linux)
    try:
        result = subprocess.run(
            ["hciconfig", "hci0"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n")[:5]:
                if line.strip():
                    lines.append(line.strip())
    except Exception:
        pass

    # Method 2: bluetoothctl (BlueZ)
    try:
        result = subprocess.run(
            ["bluetoothctl", "show"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n")[:10]:
                if "Powered" in line or "Discoverable" in line or "Name" in line:
                    lines.append(line.strip())
    except Exception:
        pass

    # Method 3: Android dumpsys bluetooth
    try:
        result = subprocess.run(
            ["dumpsys", "bluetooth_manager"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n")[:15]:
                if "enabled" in line.lower() or "state" in line.lower() or "name" in line.lower():
                    lines.append(line.strip())
    except Exception:
        pass

    if not lines:
        lines.append("Bluetooth: unavailable")
        lines.append("[simulated: enabled, name 'ToroidalOS', 2 paired devices]")

    return "\n".join(lines)


def handle_bluetooth_scan(args: Dict[str, Any]) -> str:
    """
    Scan for Bluetooth devices.

    Uses bluetoothctl or Android BluetoothManager.
    """
    devices = []

    # Method 1: bluetoothctl (BlueZ)
    try:
        # Start scan
        subprocess.run(
            ["bluetoothctl", "scan", "on"],
            capture_output=True, timeout=10
        )
        # Get devices
        result = subprocess.run(
            ["bluetoothctl", "devices"],
            capture_output=True, text=True, timeout=5
        )
        # Stop scan
        subprocess.run(
            ["bluetoothctl", "scan", "off"],
            capture_output=True, timeout=5
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Device "):
                    # Format: "Device AA:BB:CC:DD:EE:FF Device Name"
                    parts = line.split(" ", 2)
                    if len(parts) >= 3:
                        devices.append(f"{parts[2]} ({parts[1]})")
    except Exception:
        pass

    if not devices:
        devices = [
            "[Simulated scan results]",
            "JBL Speaker (AA:BB:CC:11:22:33)",
            "Galaxy Buds (DD:EE:FF:44:55:66)",
            "Unknown Device (11:22:33:AA:BB:CC)",
        ]

    return "Discovered devices:\n" + "\n".join(devices[:10])


def handle_bluetooth_connect(args: Dict[str, Any]) -> str:
    """
    Pair and connect to a Bluetooth device.

    Args:
        address: MAC address of the device
        name: Optional device name (for logging)
    """
    address = args.get("address", "")
    name = args.get("name", "")

    if not address:
        return "No address provided. Usage: bluetooth_connect {address: 'AA:BB:CC:DD:EE:FF'}"

    # Validate MAC address format
    import re
    if not re.match(r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$', address):
        return f"Invalid MAC address format: {address}"

    # Method 1: bluetoothctl (BlueZ)
    try:
        # Pair
        result = subprocess.run(
            ["bluetoothctl", "pair", address],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0 and "already exists" not in result.stderr:
            return f"Pairing failed: {result.stderr.strip()}"

        # Connect
        result = subprocess.run(
            ["bluetoothctl", "connect", address],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            device_name = f" '{name}'" if name else ""
            return f"Connected to{device_name} ({address})"
        else:
            return f"Connection failed: {result.stderr.strip()}"
    except Exception as e:
        pass

    # Fallback: simulated
    return f"[Simulated] Would connect to Bluetooth device {address}"


def handle_bluetooth_disconnect(args: Dict[str, Any]) -> str:
    """Disconnect from a Bluetooth device."""
    address = args.get("address", "")

    if not address:
        return "No address provided."

    try:
        result = subprocess.run(
            ["bluetoothctl", "disconnect", address],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return f"Disconnected from {address}"
        else:
            return f"Disconnect failed: {result.stderr.strip()}"
    except Exception:
        pass

    return f"[Simulated] Disconnected from {address}"


# ============================================================================
# HTML STRIPPING (minimal, no dependencies)
# ============================================================================

import re

_TAG_RE = re.compile(r'<[^>]+>', re.DOTALL)
_SCRIPT_RE = re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r'<style[^>]*>.*?</style>', re.DOTALL | re.IGNORECASE)
_ENTITY_RE = re.compile(r'&\w+;')

_ENTITY_MAP = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&apos;": "'", "&nbsp;": " ",
}


def _strip_html(html: str) -> str:
    """Strip HTML to plain text. Minimal implementation, no dependencies."""
    text = _SCRIPT_RE.sub("", html)
    text = _STYLE_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    for entity, char in _ENTITY_MAP.items():
        text = text.replace(entity, char)
    text = _ENTITY_RE.sub("", text)
    # Collapse whitespace
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)
