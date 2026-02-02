#!/usr/bin/env python3
"""
TOROIDAL OS - Hypergraph Kernel
================================
Self-referential operating system kernel for Xiaomi Mi Mix (lithium)

The hypergraph represents all system state:
- Processes are subgraph patterns
- Memory is node data
- Time emerges from graph rewriting
"""

import json
import time
import threading
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable
from enum import Enum


class NodeType(Enum):
    PROCESS = "process"
    MEMORY = "memory"
    SENSOR = "sensor"
    THOUGHT = "thought"
    PERCEPT = "percept"
    ACTION = "action"
    BELIEF = "belief"


@dataclass
class Node:
    """A node in the hypergraph"""
    id: str
    type: NodeType
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    energy: float = 1.0  # Activation energy
    
    def touch(self):
        """Mark node as accessed"""
        self.last_accessed = time.time()
        self.access_count += 1
        self.energy = min(1.0, self.energy + 0.1)
    
    def decay(self, rate: float = 0.01):
        """Energy decay over time"""
        self.energy = max(0.0, self.energy - rate)
    
    def hash(self) -> str:
        """Content-addressable hash"""
        content = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Edge:
    """A directed edge (relation) between nodes"""
    source: str
    target: str
    relation: str
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)


@dataclass 
class Hyperedge:
    """A hyperedge connecting multiple nodes"""
    id: str
    nodes: Set[str]
    relation: str
    data: Dict[str, Any] = field(default_factory=dict)


class HypergraphKernel:
    """
    The core hypergraph that represents all system state.
    
    This implements the self-referential structure where:
    - The graph contains representations of itself
    - Processes modify the graph (including themselves)
    - Time emerges from the sequence of modifications
    """
    
    def __init__(self, max_nodes: int = 10000):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.hyperedges: Dict[str, Hyperedge] = {}
        
        # Adjacency for fast lookup
        self.outgoing: Dict[str, Set[str]] = defaultdict(set)  # node -> edge ids
        self.incoming: Dict[str, Set[str]] = defaultdict(set)  # node -> edge ids
        
        # Emergent time (τ from TUFT)
        self.tau: int = 0
        
        # Memory limits (for 6GB Mi Mix)
        self.max_nodes = max_nodes
        
        # Self-reference: the graph contains a model of itself
        self._create_self_model()
        
        # Observers for reactive updates
        self.observers: List[Callable] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def _create_self_model(self):
        """Create initial self-referential structure"""
        # Create meta-node representing the graph itself
        self_node = Node(
            id="__self__",
            type=NodeType.BELIEF,
            data={
                "description": "This node represents the hypergraph itself",
                "tau": 0,
                "node_count": 0,
                "edge_count": 0
            }
        )
        self.nodes["__self__"] = self_node
        
        # Create root process node
        root = Node(
            id="__root__",
            type=NodeType.PROCESS,
            data={
                "name": "kernel",
                "state": "running",
                "priority": 1.0
            }
        )
        self.nodes["__root__"] = root
        
        # Connect root to self
        self._add_edge("__root__", "__self__", "observes")
    
    def _update_self_model(self):
        """Update the self-referential model"""
        self_node = self.nodes.get("__self__")
        if self_node:
            self_node.data.update({
                "tau": self.tau,
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "last_update": time.time()
            })
    
    def _add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> str:
        """Internal edge creation"""
        edge_id = f"{source}->{target}:{relation}"
        edge = Edge(source=source, target=target, relation=relation, weight=weight)
        self.edges[edge_id] = edge
        self.outgoing[source].add(edge_id)
        self.incoming[target].add(edge_id)
        return edge_id
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def add_node(self, id: str, type: NodeType, data: Dict[str, Any] = None) -> Node:
        """Add a node to the hypergraph"""
        with self._lock:
            if len(self.nodes) >= self.max_nodes:
                self._garbage_collect()
            
            node = Node(id=id, type=type, data=data or {})
            self.nodes[id] = node
            
            # Connect to self-model
            self._add_edge(id, "__self__", "part_of", 0.1)
            
            self.tau += 1
            self._update_self_model()
            self._notify_observers("node_added", node)
            
            return node
    
    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> Optional[str]:
        """Add an edge between nodes"""
        with self._lock:
            if source not in self.nodes or target not in self.nodes:
                return None
            
            edge_id = self._add_edge(source, target, relation, weight)
            
            self.tau += 1
            self._update_self_model()
            
            return edge_id
    
    def add_hyperedge(self, nodes: Set[str], relation: str, data: Dict = None) -> Optional[str]:
        """Add a hyperedge connecting multiple nodes"""
        with self._lock:
            # Verify all nodes exist
            if not all(n in self.nodes for n in nodes):
                return None
            
            he_id = f"he_{self.tau}_{relation}"
            he = Hyperedge(id=he_id, nodes=nodes, relation=relation, data=data or {})
            self.hyperedges[he_id] = he
            
            self.tau += 1
            return he_id
    
    def get_node(self, id: str) -> Optional[Node]:
        """Get a node by ID"""
        node = self.nodes.get(id)
        if node:
            node.touch()
        return node
    
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Get neighboring nodes"""
        neighbors = set()
        
        if direction in ("out", "both"):
            for edge_id in self.outgoing.get(node_id, set()):
                edge = self.edges.get(edge_id)
                if edge:
                    neighbors.add(edge.target)
        
        if direction in ("in", "both"):
            for edge_id in self.incoming.get(node_id, set()):
                edge = self.edges.get(edge_id)
                if edge:
                    neighbors.add(edge.source)
        
        return list(neighbors)
    
    def query(self, pattern: Dict[str, Any]) -> List[Node]:
        """Query nodes matching a pattern"""
        results = []
        for node in self.nodes.values():
            if self._matches_pattern(node, pattern):
                node.touch()
                results.append(node)
        return results
    
    def _matches_pattern(self, node: Node, pattern: Dict) -> bool:
        """Check if node matches query pattern"""
        if "type" in pattern and node.type != pattern["type"]:
            return False
        if "data" in pattern:
            for key, value in pattern["data"].items():
                if key not in node.data or node.data[key] != value:
                    return False
        return True
    
    def get_connectivity(self, node_id: str) -> float:
        """Get connectivity score (importance) of a node"""
        if node_id not in self.nodes:
            return 0.0
        
        in_degree = len(self.incoming.get(node_id, set()))
        out_degree = len(self.outgoing.get(node_id, set()))
        
        # Also count hyperedge participation
        he_count = sum(1 for he in self.hyperedges.values() if node_id in he.nodes)
        
        node = self.nodes[node_id]
        return (in_degree + out_degree + he_count * 2) * node.energy
    
    def get_most_connected(self, n: int = 10, type_filter: NodeType = None) -> List[Node]:
        """Get the n most connected nodes"""
        candidates = self.nodes.values()
        if type_filter:
            candidates = [n for n in candidates if n.type == type_filter]
        
        sorted_nodes = sorted(
            candidates,
            key=lambda n: self.get_connectivity(n.id),
            reverse=True
        )
        return sorted_nodes[:n]
    
    # ========================================================================
    # GRAPH REWRITING (emergent computation)
    # ========================================================================
    
    def step(self):
        """
        One step of emergent time.
        
        This is the core "computation" - the graph rewrites itself.
        Processes with high connectivity get to execute.
        """
        with self._lock:
            # Decay all node energies
            for node in self.nodes.values():
                node.decay()
            
            # Get active processes (high connectivity)
            processes = self.get_most_connected(5, NodeType.PROCESS)
            
            for process in processes:
                if process.data.get("state") == "running":
                    self._execute_process(process)
            
            self.tau += 1
            self._update_self_model()
    
    def _execute_process(self, process: Node):
        """Execute a process (placeholder - will be extended)"""
        # Processes can:
        # 1. Read from graph
        # 2. Write to graph
        # 3. Create new nodes/edges
        # 4. Spawn new processes
        
        process.touch()
        
        # Record execution
        process.data["last_executed"] = time.time()
        process.data["execution_count"] = process.data.get("execution_count", 0) + 1
    
    # ========================================================================
    # GARBAGE COLLECTION
    # ========================================================================
    
    def _garbage_collect(self):
        """Remove low-energy, low-connectivity nodes"""
        # Protected nodes
        protected = {"__self__", "__root__"}
        
        # Score all nodes
        scores = []
        for node_id, node in self.nodes.items():
            if node_id in protected:
                continue
            score = self.get_connectivity(node_id) * node.energy
            scores.append((node_id, score))
        
        # Sort by score (lowest first)
        scores.sort(key=lambda x: x[1])
        
        # Remove bottom 10%
        to_remove = scores[:len(scores) // 10]
        for node_id, _ in to_remove:
            self._remove_node(node_id)
    
    def _remove_node(self, node_id: str):
        """Remove a node and its edges"""
        if node_id not in self.nodes:
            return
        
        # Remove edges
        for edge_id in list(self.outgoing.get(node_id, set())):
            self._remove_edge(edge_id)
        for edge_id in list(self.incoming.get(node_id, set())):
            self._remove_edge(edge_id)
        
        # Remove from hyperedges
        for he in self.hyperedges.values():
            he.nodes.discard(node_id)
        
        # Remove node
        del self.nodes[node_id]
    
    def _remove_edge(self, edge_id: str):
        """Remove an edge"""
        edge = self.edges.get(edge_id)
        if not edge:
            return
        
        self.outgoing[edge.source].discard(edge_id)
        self.incoming[edge.target].discard(edge_id)
        del self.edges[edge_id]
    
    # ========================================================================
    # SERIALIZATION
    # ========================================================================
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "tau": self.tau,
            "nodes": {
                k: {
                    "type": v.type.value,
                    "data": v.data,
                    "energy": v.energy
                }
                for k, v in self.nodes.items()
            },
            "edges": {
                k: {
                    "source": v.source,
                    "target": v.target,
                    "relation": v.relation,
                    "weight": v.weight
                }
                for k, v in self.edges.items()
            }
        }
    
    def to_prompt(self) -> str:
        """Generate LLM-readable summary of graph state"""
        active_processes = self.query({"type": NodeType.PROCESS, "data": {"state": "running"}})
        recent_thoughts = sorted(
            self.query({"type": NodeType.THOUGHT}),
            key=lambda n: n.created_at,
            reverse=True
        )[:5]
        
        summary = f"""SYSTEM STATE at τ={self.tau}:
- Total nodes: {len(self.nodes)}
- Total edges: {len(self.edges)}
- Active processes: {len(active_processes)}
- Recent thoughts: {len(recent_thoughts)}

ACTIVE PROCESSES:
"""
        for proc in active_processes[:5]:
            summary += f"  - {proc.id}: {proc.data.get('name', 'unnamed')}\n"
        
        summary += "\nRECENT THOUGHTS:\n"
        for thought in recent_thoughts[:3]:
            content = thought.data.get("content", "")[:100]
            summary += f"  - [{thought.id}]: {content}\n"
        
        return summary
    
    # ========================================================================
    # OBSERVER PATTERN
    # ========================================================================
    
    def add_observer(self, callback: Callable):
        """Add observer for graph changes"""
        self.observers.append(callback)
    
    def _notify_observers(self, event: str, data: Any):
        """Notify all observers"""
        for observer in self.observers:
            try:
                observer(event, data)
            except Exception as e:
                pass  # Don't let observer errors crash kernel


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_process(graph: HypergraphKernel, name: str, handler: Callable = None) -> Node:
    """Create a new process in the graph"""
    proc_id = f"proc_{graph.tau}_{name}"
    proc = graph.add_node(proc_id, NodeType.PROCESS, {
        "name": name,
        "state": "running",
        "handler": handler.__name__ if handler else None,
        "created_at": time.time()
    })
    
    # Connect to root
    graph.add_edge(proc_id, "__root__", "child_of")
    
    return proc


def create_thought(graph: HypergraphKernel, content: str, source: str = None) -> Node:
    """Record a thought in the graph"""
    thought_id = f"thought_{graph.tau}"
    thought = graph.add_node(thought_id, NodeType.THOUGHT, {
        "content": content,
        "source": source,
        "created_at": time.time()
    })
    
    # Connect to source if provided
    if source and source in graph.nodes:
        graph.add_edge(source, thought_id, "generated")
    
    return thought


def create_percept(graph: HypergraphKernel, modality: str, data: Any) -> Node:
    """Record a perception (sensor input) in the graph"""
    percept_id = f"percept_{graph.tau}_{modality}"
    percept = graph.add_node(percept_id, NodeType.PERCEPT, {
        "modality": modality,  # "audio", "vision", "text", "touch"
        "data": data,
        "timestamp": time.time()
    })
    
    return percept


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing HypergraphKernel...")
    
    graph = HypergraphKernel(max_nodes=1000)
    
    # Create some processes
    p1 = create_process(graph, "audio_listener")
    p2 = create_process(graph, "reasoner")
    
    # Create some thoughts
    t1 = create_thought(graph, "The user said hello", p1.id)
    t2 = create_thought(graph, "I should respond friendly", p2.id)
    
    # Connect thoughts
    graph.add_edge(t1.id, t2.id, "caused")
    
    # Run some steps
    for _ in range(10):
        graph.step()
    
    # Print state
    print(graph.to_prompt())
    
    print("\nTest complete!")
