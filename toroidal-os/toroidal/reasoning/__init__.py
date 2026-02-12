# TOROIDAL OS - Reasoning Package
from .self_ref import (
    SelfReferentialEngine,
    ToroidalOS,
    LLMClient,
    ReasoningResult,
    ConvergenceState
)
from .tools import ToolDispatcher, ToolManifest, ToolRegion, ToolCall, ToolResult
from .topo_protocol import TopoProtocol, TopoManifest
from .epistemic import EpistemicDetector, EpistemicState
