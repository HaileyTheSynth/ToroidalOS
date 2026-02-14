# TOROIDAL OS - Main Package
from .memory.solenoid import SolenoidMemory, MemoryItem, LLMCompressor
from .kernel.hypergraph import HypergraphKernel, Node, Edge, NodeType
from .kernel.hypergraph import create_process, create_thought, create_percept

# TUFT Integration (optional)
try:
    from .kernel.tuft_integration import TUFTHypergraphKernel, TUFTNode
    from .kernel.tuft_integration import compute_convergence_from_tuft
    TUFT_ENABLED = True
except ImportError as e:
    print(f"[TOROIDAL] TUFT not available: {e}")
    TUFT_ENABLED = False
    TUFTHypergraphKernel = None
    TUFTNode = None
    compute_convergence_from_tuft = None