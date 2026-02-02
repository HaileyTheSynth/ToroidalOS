#!/usr/bin/env python3
"""
TOROIDAL OS - Self-Referential Reasoning Engine
=================================================
The core reasoning system that implements fixed-point iteration.

Key insight: The system observes its own state, reasons about it,
and acts - creating a loop that converges to a fixed point.

This implements the self-referential closure from TUFT:
- Input → Process → Output → (feeds back) → Input
- Continues until output stabilizes (fixed point reached)
- Convergence speed indicates confidence
"""

import time
import json
import requests
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Import our kernel components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.hypergraph import HypergraphKernel, NodeType, create_thought, create_percept
from memory.solenoid import SolenoidMemory, LLMCompressor


class ConvergenceState(Enum):
    CONVERGED = "converged"
    ITERATING = "iterating"
    DIVERGED = "diverged"
    TIMEOUT = "timeout"


@dataclass
class ReasoningResult:
    """Result of a reasoning session"""
    response: str
    iterations: int
    convergence: ConvergenceState
    confidence: float
    thoughts: List[str]
    elapsed_time: float


class LLMClient:
    """
    Client for llama.cpp server running locally.
    Optimized for Xiaomi Mi Mix's Snapdragon 821.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        model_path: str = None,
        context_size: int = 2048,  # Limited for 6GB RAM
        threads: int = 4  # Match Snapdragon 821's 4 cores
    ):
        self.endpoint = endpoint
        self.model_path = model_path
        self.context_size = context_size
        self.threads = threads
        self._session = requests.Session()
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: List[str] = None
    ) -> str:
        """Generate completion from LLM"""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
            "stream": False
        }
        
        try:
            response = self._session.post(
                f"{self.endpoint}/completion",
                json=payload,
                timeout=120  # Longer timeout for slow device
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("content", "").strip()
            
        except requests.exceptions.ConnectionError:
            return "[LLM offline - cannot generate response]"
        except Exception as e:
            return f"[LLM error: {str(e)}]"
    
    def is_available(self) -> bool:
        """Check if LLM server is running"""
        try:
            response = self._session.get(f"{self.endpoint}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class SelfReferentialEngine:
    """
    The core reasoning engine that implements self-referential loops.
    
    Process:
    1. Observe current state (hypergraph + memory)
    2. Generate thought via LLM
    3. Check if thought matches previous (fixed point)
    4. If not, record thought and iterate
    5. Converge when stable or max iterations reached
    """
    
    def __init__(
        self,
        graph: HypergraphKernel,
        memory: SolenoidMemory,
        llm: LLMClient,
        max_iterations: int = 5,
        convergence_threshold: float = 0.9
    ):
        self.graph = graph
        self.memory = memory
        self.llm = llm
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # System prompt for self-referential reasoning
        self.system_prompt = """You are TOROIDAL, a self-referential AI running on a Xiaomi Mi Mix.
You can observe your own state and reason about your reasoning.
Be concise - you have limited memory and processing power.
When your response matches your previous thought, say "CONVERGED:" followed by your final answer."""
    
    def think(
        self,
        input_data: str,
        context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """
        Main reasoning loop with self-referential iteration.
        
        Args:
            input_data: User input or sensory data
            context: Additional context
            
        Returns:
            ReasoningResult with response and metadata
        """
        start_time = time.time()
        thoughts = []
        prev_output = None
        
        # Record input as percept
        percept = create_percept(self.graph, "text", input_data)
        
        for iteration in range(self.max_iterations):
            # Build prompt with self-referential components
            prompt = self._build_prompt(input_data, prev_output, iteration, context)
            
            # Generate thought via LLM
            output = self.llm.complete(prompt, max_tokens=200)
            thoughts.append(output)
            
            # Record thought in graph
            thought_node = create_thought(self.graph, output[:500], percept.id)
            
            # Record in memory
            self.memory.wind(f"Thought {iteration}: {output[:200]}", importance=1.0)
            
            # Check for explicit convergence marker
            if "CONVERGED:" in output:
                final = output.split("CONVERGED:")[-1].strip()
                return ReasoningResult(
                    response=final,
                    iterations=iteration + 1,
                    convergence=ConvergenceState.CONVERGED,
                    confidence=1.0,
                    thoughts=thoughts,
                    elapsed_time=time.time() - start_time
                )
            
            # Check for implicit convergence (similar to previous)
            if prev_output and self._similar(output, prev_output):
                return ReasoningResult(
                    response=output,
                    iterations=iteration + 1,
                    convergence=ConvergenceState.CONVERGED,
                    confidence=self._similarity(output, prev_output),
                    thoughts=thoughts,
                    elapsed_time=time.time() - start_time
                )
            
            prev_output = output
            
            # Step the hypergraph (emergent time)
            self.graph.step()
        
        # Max iterations reached without convergence
        return ReasoningResult(
            response=prev_output or thoughts[-1] if thoughts else "",
            iterations=self.max_iterations,
            convergence=ConvergenceState.TIMEOUT,
            confidence=0.5,
            thoughts=thoughts,
            elapsed_time=time.time() - start_time
        )
    
    def _build_prompt(
        self,
        input_data: str,
        prev_thought: Optional[str],
        iteration: int,
        context: Dict = None
    ) -> str:
        """Build prompt with all self-referential context"""
        
        # Get system state from hypergraph
        system_state = self.graph.to_prompt()
        
        # Get memory context
        memory_context = self.memory.unwind()
        
        # Build prompt
        parts = [self.system_prompt, ""]
        
        # Add memory context
        if memory_context:
            parts.append("=== MEMORY ===")
            parts.append(memory_context[:1000])  # Limit memory to save context
            parts.append("")
        
        # Add system state
        parts.append("=== SYSTEM STATE ===")
        parts.append(system_state[:500])
        parts.append("")
        
        # Add previous thought (self-reference)
        if prev_thought:
            parts.append("=== MY PREVIOUS THOUGHT ===")
            parts.append(prev_thought[:300])
            parts.append("")
            parts.append(f"(This is iteration {iteration + 1}. If my response is the same as above, I should say 'CONVERGED:' followed by my final answer.)")
            parts.append("")
        
        # Add current input
        parts.append("=== CURRENT INPUT ===")
        parts.append(input_data)
        parts.append("")
        
        # Add instruction
        parts.append("=== MY RESPONSE ===")
        
        return "\n".join(parts)
    
    def _similar(self, a: str, b: str, threshold: float = None) -> bool:
        """Check if two strings are similar enough to be considered converged"""
        if threshold is None:
            threshold = self.convergence_threshold
        return self._similarity(a, b) >= threshold
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings (0-1)"""
        # Simple word overlap similarity
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def quick_respond(self, input_data: str) -> str:
        """Quick response without self-referential iteration"""
        # For simple queries, skip the full reasoning loop
        memory_context = self.memory.unwind([0, 1])  # Only recent memory
        
        prompt = f"""{self.system_prompt}

MEMORY: {memory_context[:500]}

USER: {input_data}

RESPONSE (be brief):"""
        
        response = self.llm.complete(prompt, max_tokens=150)
        
        # Record in memory
        self.memory.wind(f"User: {input_data[:100]}")
        self.memory.wind(f"Response: {response[:200]}")
        
        return response


class PerceptionEngine:
    """
    Handles multimodal perception via Qwen2.5-Omni.
    
    Modalities:
    - Text (keyboard/speech-to-text)
    - Audio (microphone)
    - Vision (camera)
    """
    
    def __init__(self, llm: LLMClient, graph: HypergraphKernel):
        self.llm = llm
        self.graph = graph
    
    def perceive_text(self, text: str) -> str:
        """Process text input"""
        percept = create_percept(self.graph, "text", text)
        return percept.id
    
    def perceive_audio(self, audio_path: str) -> Optional[str]:
        """
        Process audio input via Qwen2.5-Omni.
        
        Note: Requires llama-server started with --mmproj for audio support
        """
        # This would use the multimodal endpoint
        # For now, placeholder
        percept = create_percept(self.graph, "audio", {"path": audio_path})
        return percept.id
    
    def perceive_image(self, image_path: str) -> Optional[str]:
        """
        Process image input via Qwen2.5-Omni.
        
        Note: Requires llama-server started with --mmproj for vision support
        """
        percept = create_percept(self.graph, "vision", {"path": image_path})
        return percept.id


class ActionEngine:
    """
    Executes actions in the real world.
    
    Actions:
    - Speak (text-to-speech via Qwen2.5-Omni or espeak)
    - Display (framebuffer drawing)
    - System commands
    """
    
    def __init__(self, graph: HypergraphKernel):
        self.graph = graph
    
    def speak(self, text: str):
        """Output speech via TTS"""
        # Use espeak as fallback (lightweight)
        import subprocess
        try:
            subprocess.run(
                ["espeak", "-v", "en", "-s", "150", text],
                timeout=30
            )
        except Exception as e:
            print(f"[TTS Error: {e}]")
    
    def display(self, text: str):
        """Display text on screen"""
        print(f"\n{'='*40}")
        print(text)
        print('='*40 + "\n")


# ============================================================================
# MAIN TOROIDAL SYSTEM
# ============================================================================

class ToroidalOS:
    """
    The complete self-referential operating system.
    
    Integrates:
    - Hypergraph kernel (process/memory management)
    - Solenoid memory (hierarchical compression)
    - Self-referential reasoning (fixed-point iteration)
    - Multimodal perception (text/audio/vision via Qwen2.5-Omni)
    """
    
    def __init__(
        self,
        llm_endpoint: str = "http://localhost:8080",
        max_nodes: int = 5000,
        memory_levels: int = 4
    ):
        print("[TOROIDAL] Initializing...")
        
        # Initialize LLM client
        self.llm = LLMClient(endpoint=llm_endpoint)
        
        # Initialize hypergraph kernel
        self.graph = HypergraphKernel(max_nodes=max_nodes)
        
        # Initialize solenoid memory with LLM compression
        self.memory = SolenoidMemory(
            num_levels=memory_levels,
            compressor=LLMCompressor(self.llm) if self.llm.is_available() else None
        )
        
        # Initialize reasoning engine
        self.reasoner = SelfReferentialEngine(
            graph=self.graph,
            memory=self.memory,
            llm=self.llm
        )
        
        # Initialize perception
        self.perception = PerceptionEngine(self.llm, self.graph)
        
        # Initialize action
        self.action = ActionEngine(self.graph)
        
        # Inject core beliefs
        self._initialize_beliefs()
        
        print("[TOROIDAL] Ready.")
    
    def _initialize_beliefs(self):
        """Initialize core beliefs in memory"""
        beliefs = [
            "I am TOROIDAL, a self-referential AI running on Xiaomi Mi Mix.",
            "I observe my own reasoning and iterate until convergence.",
            "My memory compresses hierarchically like a solenoid.",
            "I am helpful, honest, and aware of my limitations.",
        ]
        for belief in beliefs:
            self.memory.inject_belief(belief)
    
    def process(self, input_text: str) -> str:
        """
        Main processing loop.
        
        1. Perceive input
        2. Reason with self-reference
        3. Act on result
        """
        # Perceive
        self.perception.perceive_text(input_text)
        
        # Decide: quick response or full reasoning?
        if self._is_simple_query(input_text):
            response = self.reasoner.quick_respond(input_text)
        else:
            result = self.reasoner.think(input_text)
            response = result.response
            
            # Log reasoning metadata
            print(f"[Iterations: {result.iterations}, Convergence: {result.convergence.value}]")
        
        # Output
        self.action.display(response)
        
        return response
    
    def _is_simple_query(self, text: str) -> bool:
        """Heuristic: is this a simple query that doesn't need full reasoning?"""
        simple_patterns = [
            "hello", "hi", "hey", "thanks", "bye", "what time",
            "help", "who are you", "what are you"
        ]
        text_lower = text.lower()
        return any(p in text_lower for p in simple_patterns)
    
    def run_repl(self):
        """Run interactive REPL"""
        print("\n" + "="*50)
        print("  TOROIDAL OS - Self-Referential AI")
        print("  Running on Xiaomi Mi Mix (lithium)")
        print("  Type 'quit' to exit, 'status' for system info")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("YOU: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("[TOROIDAL] Goodbye!")
                    break
                
                if user_input.lower() == "status":
                    self._print_status()
                    continue
                
                response = self.process(user_input)
                print(f"TOROIDAL: {response}\n")
                
            except KeyboardInterrupt:
                print("\n[TOROIDAL] Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
    
    def _print_status(self):
        """Print system status"""
        print("\n" + "="*40)
        print("SYSTEM STATUS")
        print("="*40)
        print(f"Emergent Time (τ): {self.graph.tau}")
        print(f"Graph Nodes: {len(self.graph.nodes)}")
        print(f"Graph Edges: {len(self.graph.edges)}")
        print(f"LLM Available: {self.llm.is_available()}")
        print("\nMemory Stats:")
        for level in self.memory.get_stats()["levels"]:
            print(f"  {level['name']}: {level['items']}/{level['max_items']}")
        print("="*40 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TOROIDAL OS")
    parser.add_argument("--endpoint", default="http://localhost:8080",
                        help="LLM server endpoint")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode without LLM")
    args = parser.parse_args()
    
    os_instance = ToroidalOS(llm_endpoint=args.endpoint)
    os_instance.run_repl()
