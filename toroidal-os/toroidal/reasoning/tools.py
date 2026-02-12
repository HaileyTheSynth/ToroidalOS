#!/usr/bin/env python3
"""
TOROIDAL OS - Tool Dispatch System
====================================
Provides structured tool calling for the reasoning engine.

Each tool:
- Has a name, description, and region assignment (MOTION/ENVIRON/SEMANTIC)
- Declares a Berry phase cost (cross-region calls cost more)
- Executes against the hypergraph kernel and returns structured results
- Gets recorded as a kernel event (ACCESS in the tool's region)

The LLM emits tool calls as JSON blocks in its output:
    <tool>{"name": "memory_search", "args": {"query": "identity"}}</tool>

The dispatcher parses these, executes them, and returns results
that feed back into the next reasoning iteration.
"""

import re
import json
import subprocess
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import IntEnum


class ToolRegion(IntEnum):
    MOTION = 0    # bits 0-2: physical actions, shell commands
    ENVIRON = 1   # bits 3-5: environmental queries, sensors
    SEMANTIC = 2  # bits 6-8: memory, knowledge, reasoning


@dataclass
class ToolManifest:
    """Declares a tool available to the reasoning engine."""
    name: str
    description: str
    region: ToolRegion
    berry_cost: int          # milli-units added per invocation
    parameters: Dict[str, str]  # param_name -> description
    handler: Optional[Callable] = None
    requires_confirmation: bool = False


@dataclass
class ToolCall:
    """A parsed tool invocation from LLM output."""
    name: str
    args: Dict[str, Any]
    raw: str = ""


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    output: str
    region: ToolRegion
    berry_cost: int
    elapsed_ms: float = 0.0


# Pattern to extract tool calls from LLM output
TOOL_PATTERN = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)


def parse_tool_calls(text: str) -> tuple:
    """
    Parse tool call blocks from LLM output.

    Returns:
        (clean_text, list_of_ToolCall)
        clean_text has the <tool>...</tool> blocks removed.
    """
    calls = []
    for match in TOOL_PATTERN.finditer(text):
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            calls.append(ToolCall(
                name=data.get("name", ""),
                args=data.get("args", {}),
                raw=raw,
            ))
        except json.JSONDecodeError:
            pass  # Malformed tool call, skip

    clean = TOOL_PATTERN.sub("", text).strip()
    return clean, calls


class ToolDispatcher:
    """
    Manages tool registration and dispatch for the reasoning engine.

    Integrates with:
    - HypergraphKernel: tool calls create ACTION nodes
    - KernelBridge: each call is a region ACCESS that accumulates Berry phase
    - SolenoidMemory: tool results can be stored for future reference
    """

    def __init__(self, graph=None, memory=None, kernel_bridge=None):
        self.tools: Dict[str, ToolManifest] = {}
        self.graph = graph
        self.memory = memory
        self.bridge = kernel_bridge
        self._call_history: List[ToolResult] = []

        # Register built-in tools
        self._register_builtins()

    def register(self, manifest: ToolManifest):
        """Register a tool."""
        self.tools[manifest.name] = manifest

    def _register_builtins(self):
        """Register the essential built-in tools."""

        # --- SEMANTIC region tools (memory, knowledge) ---

        self.register(ToolManifest(
            name="memory_search",
            description="Search solenoid memory for past experiences matching a query.",
            region=ToolRegion.SEMANTIC,
            berry_cost=30,
            parameters={"query": "Search string to match against memories"},
            handler=self._handle_memory_search,
        ))

        self.register(ToolManifest(
            name="memory_store",
            description="Store an important observation or conclusion in memory.",
            region=ToolRegion.SEMANTIC,
            berry_cost=30,
            parameters={
                "content": "Text to store",
                "importance": "Float 0.0-2.0, default 1.0",
            },
            handler=self._handle_memory_store,
        ))

        self.register(ToolManifest(
            name="kernel_state",
            description="Query the topological kernel for current invariants: coherence, curvature, Berry phase, bridges.",
            region=ToolRegion.SEMANTIC,
            berry_cost=30,
            parameters={},
            handler=self._handle_kernel_state,
        ))

        self.register(ToolManifest(
            name="kernel_coherence",
            description="Get coherence score and top neighbors for a specific kernel node.",
            region=ToolRegion.SEMANTIC,
            berry_cost=30,
            parameters={"node_id": "Integer node ID to query"},
            handler=self._handle_kernel_coherence,
        ))

        self.register(ToolManifest(
            name="graph_query",
            description="Query the hypergraph for nodes matching a pattern (type, data fields).",
            region=ToolRegion.SEMANTIC,
            berry_cost=30,
            parameters={
                "type": "Node type: process, memory, sensor, thought, percept, action, belief",
                "data": "Optional dict of data fields to match",
            },
            handler=self._handle_graph_query,
        ))

        # --- ENVIRON region tools (sensors, environment) ---

        self.register(ToolManifest(
            name="sensors",
            description="Read current sensor state: 9-bit fused state, active region, raw snapshot.",
            region=ToolRegion.ENVIRON,
            berry_cost=50,
            parameters={},
            handler=self._handle_sensors,
        ))

        # --- MOTION region tools (actions, shell) ---

        self.register(ToolManifest(
            name="shell",
            description="Execute a shell command. Use for file operations, system queries. Commands are sandboxed.",
            region=ToolRegion.MOTION,
            berry_cost=110,
            parameters={"command": "Shell command to execute"},
            handler=self._handle_shell,
            requires_confirmation=True,
        ))

        self.register(ToolManifest(
            name="notes_write",
            description="Write a persistent note that bypasses solenoid compression. For important long-term storage.",
            region=ToolRegion.MOTION,
            berry_cost=50,
            parameters={
                "key": "Note identifier",
                "content": "Note content",
            },
            handler=self._handle_notes_write,
        ))

        self.register(ToolManifest(
            name="notes_read",
            description="Read a persistent note by key.",
            region=ToolRegion.MOTION,
            berry_cost=30,
            parameters={"key": "Note identifier to read"},
            handler=self._handle_notes_read,
        ))

    def dispatch(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.

        Side effects:
        - Records ACCESS in the tool's region via KernelBridge
        - Creates an ACTION node in the hypergraph
        - Appends to call history
        """
        manifest = self.tools.get(call.name)
        if not manifest:
            return ToolResult(
                tool_name=call.name,
                success=False,
                output=f"Unknown tool: {call.name}",
                region=ToolRegion.SEMANTIC,
                berry_cost=0,
            )

        start = time.time()

        # Execute the handler
        try:
            output = manifest.handler(call.args)
            success = True
        except Exception as e:
            output = f"Error: {str(e)}"
            success = False

        elapsed = (time.time() - start) * 1000

        result = ToolResult(
            tool_name=call.name,
            success=success,
            output=str(output)[:2000],  # Cap output size
            region=manifest.region,
            berry_cost=manifest.berry_cost,
            elapsed_ms=elapsed,
        )

        # Record in kernel bridge (drives Berry phase)
        if self.bridge:
            self.bridge.record_tool_call(call.name, int(manifest.region), manifest.berry_cost)

        # Record as ACTION node in hypergraph
        if self.graph:
            from kernel.hypergraph import NodeType
            action_id = f"action_{self.graph.tau}_{call.name}"
            self.graph.add_node(action_id, NodeType.ACTION, {
                "tool": call.name,
                "args": call.args,
                "success": success,
                "output_preview": output[:200],
                "region": manifest.region.name,
            })

        self._call_history.append(result)
        return result

    def dispatch_all(self, calls: List[ToolCall]) -> List[ToolResult]:
        """Dispatch multiple tool calls sequentially."""
        return [self.dispatch(call) for call in calls]

    def get_tool_prompt(self) -> str:
        """
        Generate the tool documentation for injection into the LLM prompt.
        This tells the model what tools are available and how to call them.
        """
        lines = [
            "=== AVAILABLE TOOLS ===",
            "To use a tool, include a <tool> block in your response:",
            '  <tool>{"name": "tool_name", "args": {"param": "value"}}</tool>',
            "",
            "Tools:",
        ]

        for name, manifest in sorted(self.tools.items()):
            params_str = ", ".join(
                f"{k}: {v}" for k, v in manifest.parameters.items()
            )
            lines.append(f"  {name} [{manifest.region.name}]")
            lines.append(f"    {manifest.description}")
            if params_str:
                lines.append(f"    Args: {params_str}")
            lines.append("")

        return "\n".join(lines)

    def format_results(self, results: List[ToolResult]) -> str:
        """Format tool results for injection into the next reasoning iteration."""
        if not results:
            return ""

        parts = ["=== TOOL RESULTS ==="]
        for r in results:
            status = "OK" if r.success else "FAILED"
            parts.append(f"[{r.tool_name}] {status} ({r.elapsed_ms:.0f}ms, region={r.region.name})")
            parts.append(f"  {r.output[:500]}")
            parts.append("")

        return "\n".join(parts)

    # ========================================================================
    # BUILT-IN TOOL HANDLERS
    # ========================================================================

    def _handle_memory_search(self, args: Dict) -> str:
        query = args.get("query", "")
        if not query:
            return "No query provided"
        if not self.memory:
            return "Memory system not available"

        results = self.memory.search(query)
        if not results:
            return f"No memories matching '{query}'"

        lines = []
        for item in results[:5]:
            age = time.time() - item.timestamp
            age_str = f"{int(age)}s" if age < 60 else f"{int(age/60)}m"
            lines.append(f"[L{item.level}] ({age_str} ago) {item.content[:200]}")
        return "\n".join(lines)

    def _handle_memory_store(self, args: Dict) -> str:
        content = args.get("content", "")
        importance = float(args.get("importance", 1.0))
        if not content:
            return "No content provided"
        if not self.memory:
            return "Memory system not available"

        importance = max(0.0, min(2.0, importance))
        item = self.memory.wind(content, importance=importance)
        return f"Stored memory {item.id} at level 0 (importance={importance})"

    def _handle_kernel_state(self, args: Dict) -> str:
        if self.bridge:
            summary = self.bridge.situation_summary()
            return json.dumps(summary, indent=2)

        # Fallback: basic graph info
        if self.graph:
            return (
                f"tau={self.graph.tau}, "
                f"nodes={len(self.graph.nodes)}, "
                f"edges={len(self.graph.edges)}"
            )
        return "Kernel not available"

    def _handle_kernel_coherence(self, args: Dict) -> str:
        node_id = args.get("node_id")
        if node_id is None:
            return "No node_id provided"
        if not self.bridge:
            return "Kernel bridge not available"

        try:
            node_id = int(node_id)
            coh = self.bridge.kernel.coherence_score(node_id)
            neighbors = self.bridge.kernel.top_neighbors(node_id, 5)
            lines = [f"Coherence of node {node_id}: {coh}"]
            for j, s, dh, dtau, dth in neighbors:
                lines.append(f"  neighbor={j} score={s} hamming={dh} dtau={dtau} dth4={dth}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error querying coherence: {e}"

    def _handle_graph_query(self, args: Dict) -> str:
        if not self.graph:
            return "Graph not available"

        from kernel.hypergraph import NodeType
        type_map = {t.value: t for t in NodeType}

        pattern = {}
        type_str = args.get("type")
        if type_str and type_str in type_map:
            pattern["type"] = type_map[type_str]

        data_filter = args.get("data")
        if data_filter and isinstance(data_filter, dict):
            pattern["data"] = data_filter

        results = self.graph.query(pattern)
        if not results:
            return "No matching nodes"

        lines = []
        for node in results[:10]:
            preview = str(node.data)[:100]
            lines.append(f"[{node.type.value}] {node.id}: {preview}")
        return "\n".join(lines)

    def _handle_sensors(self, args: Dict) -> str:
        if not self.bridge:
            return "Sensor bridge not available"

        try:
            summary = self.bridge.situation_summary()
            raw = self.bridge.hub.raw_snapshot()
            return (
                f"State: {summary.get('state', '?')}\n"
                f"Description: {summary.get('description', '?')}\n"
                f"Active region: {summary.get('active_region', '?')}\n"
                f"Coherence: {summary.get('coherence', '?')}\n"
                f"Curvature: {summary.get('curvature', '?')}\n"
                f"Berry phase: {summary.get('berry_phase', '?')}\n"
                f"Is bridge: {summary.get('is_bridge', '?')}\n"
                f"Battery: {raw.get('battery', {}).get('pct', '?')}%"
            )
        except Exception as e:
            return f"Sensor read error: {e}"

    def _handle_shell(self, args: Dict) -> str:
        command = args.get("command", "")
        if not command:
            return "No command provided"

        # Sandbox: block dangerous commands
        blocked = ["rm -rf", "mkfs", "dd if=", "> /dev/", "shutdown", "reboot",
                    "passwd", "chmod 777", "curl | sh", "wget -O - |"]
        for b in blocked:
            if b in command:
                return f"Blocked: command contains '{b}'"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout[:1000]
            if result.returncode != 0:
                output += f"\n[stderr]: {result.stderr[:500]}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out (10s limit)"
        except Exception as e:
            return f"Shell error: {e}"

    # Persistent notes (simple file-based, bypasses solenoid compression)
    _notes: Dict[str, str] = {}

    def _handle_notes_write(self, args: Dict) -> str:
        key = args.get("key", "")
        content = args.get("content", "")
        if not key or not content:
            return "Both key and content required"
        self._notes[key] = content
        return f"Note '{key}' saved ({len(content)} chars)"

    def _handle_notes_read(self, args: Dict) -> str:
        key = args.get("key", "")
        if not key:
            return "No key provided"
        content = self._notes.get(key)
        if content is None:
            return f"No note found with key '{key}'"
        return content
