# TOROIDAL OS - Kernel Package
from .hypergraph import HypergraphKernel, Node, Edge, NodeType
from .hypergraph import create_process, create_thought, create_percept

# TUFT Integration (optional)
try:
    from .tuft_integration import TUFTHypergraphKernel, TUFTNode
    from .tuft_integration import compute_convergence_from_tuft
    TUFT_ENABLED = True
except ImportError:
    TUFT_ENABLED = False
    TUFTHypergraphKernel = None
    TUFTNode = None
    compute_convergence_from_tuft = None