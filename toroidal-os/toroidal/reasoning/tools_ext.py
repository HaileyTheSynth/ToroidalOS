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
