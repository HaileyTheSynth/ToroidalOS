from __future__ import annotations
import argparse
import json
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

PROMPT_RE = re.compile(r"topo>\s*$", re.MULTILINE)

@dataclass
class OracleConfig:
    iso_path: str
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 5555
    qemu_mem: str = "64M"
    boot_timeout_s: float = 8.0
    io_timeout_s: float = 2.0

class Topo9Oracle:
    """
    Boots QEMU with a TCP serial port and speaks the kernel's line-based protocol.
    """
    def __init__(self, cfg: OracleConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None
        self._buf = b""

    def boot(self) -> None:
        if not os.path.exists(self.cfg.iso_path):
            raise FileNotFoundError(self.cfg.iso_path)

        cmd = [
            "qemu-system-i386",
            "-cdrom", self.cfg.iso_path,
            "-m", self.cfg.qemu_mem,
            "-nographic",
            "-serial", f"tcp:{self.cfg.tcp_host}:{self.cfg.tcp_port},server,nowait",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.time() + self.cfg.boot_timeout_s
        while time.time() < deadline:
            try:
                s = socket.create_connection((self.cfg.tcp_host, self.cfg.tcp_port), timeout=0.5)
                self.sock = s
                self.sock.settimeout(self.cfg.io_timeout_s)
                break
            except OSError:
                time.sleep(0.1)

        if not self.sock:
            self.shutdown()
            raise RuntimeError("Could not connect to QEMU serial TCP port")

        self.read_until_prompt(timeout_s=self.cfg.boot_timeout_s)

    def shutdown(self) -> None:
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def __enter__(self) -> "Topo9Oracle":
        self.boot()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def _recv(self) -> bytes:
        assert self.sock is not None
        try:
            chunk = self.sock.recv(4096)
            return chunk
        except socket.timeout:
            return b""

    def read_until_prompt(self, timeout_s: float = 2.0) -> str:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if PROMPT_RE.search(self._buf.decode(errors="ignore")):
                out = self._buf.decode(errors="ignore")
                self._buf = b""
                return out
            self._buf += self._recv()
        out = self._buf.decode(errors="ignore")
        self._buf = b""
        return out

    def send_cmd(self, cmd: str) -> str:
        assert self.sock is not None
        self.sock.sendall((cmd.strip() + "\n").encode("utf-8"))
        return self.read_until_prompt(timeout_s=self.cfg.io_timeout_s)

    def stats(self) -> Dict[str, Any]:
        txt = self.send_cmd("STATS")
        m = re.search(r"STATS\s+(.+)", txt)
        if not m:
            return {"raw": txt}
        fields = m.group(1).strip().split()
        out: Dict[str, Any] = {}
        for f in fields:
            if "=" in f:
                k, v = f.split("=", 1)
                try:
                    out[k] = int(v, 0)
                except ValueError:
                    out[k] = v
        out["raw"] = txt
        return out

    def curvature(self) -> int:
        txt = self.send_cmd("CURVATURE")
        m = re.search(r"scaled=(\d+)", txt)
        return int(m.group(1)) if m else -1

    def bridges(self, min_berry: int = 1200) -> List[int]:
        txt = self.send_cmd(f"BRIDGES {min_berry}")
        ids = re.findall(r"\bid=(\d+)\b", txt)
        return [int(x) for x in ids]

    def coherent(self, node_id: int, k: int = 5) -> Dict[str, Any]:
        txt = self.send_cmd(f"COHERENT {node_id} {k}")
        coh_m = re.search(r"coh=(\d+)", txt)
        neighbors = re.findall(r"\bn=(\d+)\s+s=(\d+)", txt)
        return {
            "coherence": int(coh_m.group(1)) if coh_m else None,
            "neighbors": [{"id": int(a), "score": int(b)} for a, b in neighbors],
            "raw": txt,
        }
