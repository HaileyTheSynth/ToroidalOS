# TOROIDAL OS - Reasoning Package
from .self_ref import (
    SelfReferentialEngine,
    ToroidalOS,
    LLMClient,
    ReasoningResult,
    ConvergenceState,
    PerceptionEngine,
    ActionEngine,
)
from .tools import ToolDispatcher, ToolManifest, ToolRegion, ToolCall, ToolResult
from .topo_protocol import TopoProtocol, TopoManifest
from .epistemic import EpistemicDetector, EpistemicState
from .multimodal import (
    MultimodalClient,
    AudioProcessor,
    TranscriptionResult,
    AudioGenerationResult,
    create_multimodal_client,
    transcribe_file,
)
from .tools_ext import (
    handle_web_fetch,
    handle_web_search,
    handle_sensor_request,
    handle_time_now,
    handle_system_info,
    handle_wifi_status,
    handle_wifi_scan,
    handle_wifi_connect,
    handle_bluetooth_status,
    handle_bluetooth_scan,
    handle_bluetooth_connect,
    handle_bluetooth_disconnect,
)
