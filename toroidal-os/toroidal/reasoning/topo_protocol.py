#!/usr/bin/env python3
"""
TOROIDAL OS — topo:// Tool Protocol
=====================================
A manifest-driven protocol for registering tools with the reasoning engine.

Each tool is described by a JSON manifest:

    {
        "name": "web_fetch",
        "version": "1.0",
        "description": "Fetch a URL and return its content.",
        "region": "ENVIRON",
        "berry_cost": 80,
        "prerequisites": ["epistemic_check"],
        "parameters": {
            "url": {"type": "string", "description": "URL to fetch", "required": true},
            "max_chars": {"type": "integer", "description": "Max response chars", "default": 2000}
        },
        "handler": "toroidal.reasoning.tools_web:handle_web_fetch",
        "requires_confirmation": false,
        "cooldown_ms": 5000,
        "tags": ["network", "external"]
    }

The protocol adds:
- Prerequisite checks (e.g. epistemic state must be KNOWLEDGE_GAP before web fetch)
- Cooldown between invocations (rate limiting for resource-constrained hardware)
- Versioned manifests loadable from JSON files
- Lifecycle hooks: on_register, on_invoke, on_result
- Tool tagging for categorization and policy enforcement
"""

import json
import time
import importlib
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

from reasoning.tools import ToolManifest, ToolRegion, ToolDispatcher, ToolCall, ToolResult


# ============================================================================
# EXTENDED MANIFEST
# ============================================================================

@dataclass
class TopoManifest:
    """Extended tool manifest for the topo:// protocol."""
    name: str
    version: str
    description: str
    region: ToolRegion
    berry_cost: int
    parameters: Dict[str, Dict[str, Any]]
    handler_path: str = ""           # "module.path:function_name"
    handler: Optional[Callable] = None
    prerequisites: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    cooldown_ms: int = 0
    tags: List[str] = field(default_factory=list)
    _last_invoked: float = 0.0

    def to_tool_manifest(self) -> ToolManifest:
        """Convert to the base ToolManifest for the dispatcher."""
        param_descriptions = {}
        for pname, pspec in self.parameters.items():
            desc = pspec.get("description", pname)
            if pspec.get("required", False):
                desc += " (required)"
            elif "default" in pspec:
                desc += f" (default: {pspec['default']})"
            param_descriptions[pname] = desc

        return ToolManifest(
            name=self.name,
            description=self.description,
            region=self.region,
            berry_cost=self.berry_cost,
            parameters=param_descriptions,
            handler=self.handler,
            requires_confirmation=self.requires_confirmation,
        )

    def check_cooldown(self) -> Optional[str]:
        """Returns error message if still in cooldown, None if ready."""
        if self.cooldown_ms <= 0:
            return None
        elapsed_ms = (time.time() - self._last_invoked) * 1000
        if elapsed_ms < self.cooldown_ms:
            remaining = self.cooldown_ms - elapsed_ms
            return f"Tool '{self.name}' is in cooldown ({remaining:.0f}ms remaining)"
        return None

    def mark_invoked(self):
        self._last_invoked = time.time()


# ============================================================================
# PREREQUISITE REGISTRY
# ============================================================================

class PrerequisiteRegistry:
    """
    Registry of named prerequisite checks.

    A prerequisite is a callable that returns (ok: bool, reason: str).
    Tools can declare prerequisites by name; before dispatch, all
    prerequisites are checked and the tool call is blocked if any fail.
    """

    def __init__(self):
        self._checks: Dict[str, Callable] = {}

    def register(self, name: str, check: Callable):
        """Register a prerequisite check.

        check signature: () -> (bool, str)
            Returns (True, "") if passed, (False, reason) if blocked.
        """
        self._checks[name] = check

    def check_all(self, names: List[str]) -> Optional[str]:
        """Check all named prerequisites. Returns error message or None."""
        for name in names:
            check = self._checks.get(name)
            if check is None:
                return f"Unknown prerequisite: {name}"
            ok, reason = check()
            if not ok:
                return f"Prerequisite '{name}' failed: {reason}"
        return None


# ============================================================================
# TOPO PROTOCOL MANAGER
# ============================================================================

class TopoProtocol:
    """
    Manages the topo:// tool protocol.

    Wraps the ToolDispatcher with:
    - JSON manifest loading
    - Prerequisite enforcement
    - Cooldown rate limiting
    - Lifecycle hooks
    """

    def __init__(self, dispatcher: ToolDispatcher):
        self.dispatcher = dispatcher
        self.manifests: Dict[str, TopoManifest] = {}
        self.prerequisites = PrerequisiteRegistry()
        self._hooks_on_invoke: List[Callable] = []
        self._hooks_on_result: List[Callable] = []

    def load_manifest_file(self, path: str) -> List[str]:
        """
        Load tool manifests from a JSON file.

        The file can contain either a single manifest object or
        an array of manifests.

        Returns list of registered tool names.
        """
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            manifests = [data]
        elif isinstance(data, list):
            manifests = data
        else:
            raise ValueError(f"Invalid manifest format in {path}")

        registered = []
        for m in manifests:
            name = self.register_manifest(m)
            registered.append(name)

        return registered

    def load_manifests_dir(self, dirpath: str) -> List[str]:
        """Load all .json manifest files from a directory."""
        registered = []
        p = Path(dirpath)
        if not p.is_dir():
            return registered

        for f in sorted(p.glob("*.json")):
            try:
                names = self.load_manifest_file(str(f))
                registered.extend(names)
            except Exception as e:
                print(f"[topo://] Warning: failed to load {f}: {e}")

        return registered

    def register_manifest(self, data: Dict[str, Any]) -> str:
        """Register a single manifest from a dict."""
        region_map = {"MOTION": ToolRegion.MOTION, "ENVIRON": ToolRegion.ENVIRON,
                      "SEMANTIC": ToolRegion.SEMANTIC}

        manifest = TopoManifest(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            region=region_map.get(data.get("region", "SEMANTIC"), ToolRegion.SEMANTIC),
            berry_cost=data.get("berry_cost", 30),
            parameters=data.get("parameters", {}),
            handler_path=data.get("handler", ""),
            prerequisites=data.get("prerequisites", []),
            requires_confirmation=data.get("requires_confirmation", False),
            cooldown_ms=data.get("cooldown_ms", 0),
            tags=data.get("tags", []),
        )

        # Resolve handler from path
        if manifest.handler_path and not manifest.handler:
            manifest.handler = self._resolve_handler(manifest.handler_path)

        self.manifests[manifest.name] = manifest

        # Register with the base dispatcher
        tool_manifest = manifest.to_tool_manifest()
        self.dispatcher.register(tool_manifest)

        return manifest.name

    def register_handler(self, name: str, handler: Callable):
        """Attach a handler to an already-registered manifest."""
        if name in self.manifests:
            self.manifests[name].handler = handler
            # Update the dispatcher's copy too
            if name in self.dispatcher.tools:
                self.dispatcher.tools[name].handler = handler

    def dispatch(self, call: ToolCall) -> ToolResult:
        """
        Dispatch a tool call with topo:// protocol enforcement.

        Checks prerequisites and cooldowns before delegating to the
        base dispatcher.
        """
        manifest = self.manifests.get(call.name)

        if manifest:
            # Check cooldown
            cooldown_err = manifest.check_cooldown()
            if cooldown_err:
                return ToolResult(
                    tool_name=call.name,
                    success=False,
                    output=cooldown_err,
                    region=manifest.region,
                    berry_cost=0,
                )

            # Check prerequisites
            if manifest.prerequisites:
                prereq_err = self.prerequisites.check_all(manifest.prerequisites)
                if prereq_err:
                    return ToolResult(
                        tool_name=call.name,
                        success=False,
                        output=prereq_err,
                        region=manifest.region,
                        berry_cost=0,
                    )

            # Fire on_invoke hooks
            for hook in self._hooks_on_invoke:
                try:
                    hook(call, manifest)
                except Exception:
                    pass

        # Delegate to base dispatcher
        result = self.dispatcher.dispatch(call)

        if manifest:
            manifest.mark_invoked()

            # Fire on_result hooks
            for hook in self._hooks_on_result:
                try:
                    hook(call, manifest, result)
                except Exception:
                    pass

        return result

    def dispatch_all(self, calls: List[ToolCall]) -> List[ToolResult]:
        """Dispatch multiple calls through the protocol."""
        return [self.dispatch(call) for call in calls]

    def on_invoke(self, hook: Callable):
        """Register a hook called before each tool invocation.
        Signature: hook(call: ToolCall, manifest: TopoManifest) -> None
        """
        self._hooks_on_invoke.append(hook)

    def on_result(self, hook: Callable):
        """Register a hook called after each tool invocation.
        Signature: hook(call: ToolCall, manifest: TopoManifest, result: ToolResult) -> None
        """
        self._hooks_on_result.append(hook)

    def get_manifest(self, name: str) -> Optional[TopoManifest]:
        return self.manifests.get(name)

    def list_tools(self, tags: List[str] = None) -> List[str]:
        """List tool names, optionally filtered by tags."""
        if tags is None:
            return list(self.manifests.keys())
        return [
            name for name, m in self.manifests.items()
            if any(t in m.tags for t in tags)
        ]

    def _resolve_handler(self, handler_path: str) -> Optional[Callable]:
        """Resolve a 'module.path:function_name' string to a callable."""
        if ":" not in handler_path:
            return None

        module_path, func_name = handler_path.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            print(f"[topo://] Warning: could not resolve handler '{handler_path}': {e}")
            return None

    def export_manifests(self) -> List[Dict[str, Any]]:
        """Export all manifests as JSON-serializable dicts."""
        result = []
        for m in self.manifests.values():
            result.append({
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "region": m.region.name,
                "berry_cost": m.berry_cost,
                "parameters": m.parameters,
                "handler": m.handler_path,
                "prerequisites": m.prerequisites,
                "requires_confirmation": m.requires_confirmation,
                "cooldown_ms": m.cooldown_ms,
                "tags": m.tags,
            })
        return result
