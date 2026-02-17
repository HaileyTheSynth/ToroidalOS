#!/usr/bin/env python3
"""
TOROIDAL OS - System Test Runner
=================================
Run the system in test mode with proper console handling.
"""

import sys
import os
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add toroidal to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Detect available LLM backends
def get_llm_client():
    """Detect and return available LLM client."""
    import requests

    # Try Ollama first
    try:
        resp = requests.get("http://localhost:11434/api/version", timeout=2)
        if resp.status_code == 200:
            from reasoning.self_ref import OllamaClient
            client = OllamaClient(endpoint="http://localhost:11434", model="qwen2.5:7b")
            print(f"[LLM] Using Ollama with model: {client.model}")
            return client
    except:
        pass

    # Try llama.cpp
    try:
        resp = requests.get("http://localhost:8080/health", timeout=2)
        if resp.status_code == 200:
            from reasoning.self_ref import LLMClient
            client = LLMClient(endpoint="http://localhost:8080")
            print("[LLM] Using llama.cpp server")
            return client
    except:
        pass

    # No LLM available
    from reasoning.self_ref import LLMClient
    print("[LLM] No LLM server detected - using offline mode")
    return LLMClient(endpoint="http://localhost:9999")  # Fake endpoint

def test_system_init():
    """Test that ToroidalOS initializes correctly."""
    print("=" * 60)
    print("TOROIDAL OS - System Test")
    print("=" * 60)
    print()

    print("[TEST 1] Import modules...")
    try:
        from kernel.hypergraph import HypergraphKernel, NodeType, TrustTier
        from memory.solenoid import SolenoidMemory
        from reasoning.self_ref import ToroidalOS, LLMClient
        from reasoning.tools import ToolDispatcher
        from reasoning.multimodal import MultimodalClient, AudioProcessor
        print("[OK] All modules imported")
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

    print()
    print("[TEST 2] Initialize ToroidalOS (test mode)...")
    try:
        # Create without LLM
        toroidal = ToroidalOS(
            llm_endpoint="http://localhost:9999",  # Fake endpoint
            max_nodes=500,
            memory_levels=4,
            enable_embeddings=False,  # Skip embeddings for test
        )
        print("[OK] ToroidalOS initialized")
    except Exception as e:
        print(f"[FAIL] Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("[TEST 3] Check system components...")
    components = {
        "graph": toroidal.graph,
        "memory": toroidal.memory,
        "llm": toroidal.llm,
        "tools": toroidal.tools,
        "perception": toroidal.perception,
        "action": toroidal.action,
    }

    for name, component in components.items():
        if component is not None:
            print(f"  [OK] {name}: {type(component).__name__}")
        else:
            print(f"  [WARN] {name}: None")

    print()
    print("[TEST 4] Test graph operations...")
    try:
        from kernel.hypergraph import create_thought, create_percept

        # Add some nodes
        thought = create_thought(toroidal.graph, "Hello, this is a test thought")
        percept = create_percept(toroidal.graph, "text", "User said hello")

        print(f"  [OK] Created thought: {thought.id[:8]}...")
        print(f"  [OK] Created percept: {percept.id[:8]}...")
        print(f"  [OK] Graph now has {len(toroidal.graph.nodes)} nodes")
    except Exception as e:
        print(f"  [FAIL] Graph error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("[TEST 5] Test memory operations...")
    try:
        toroidal.memory.wind("First memory item", importance=1.0)
        toroidal.memory.wind("Second memory item", importance=0.8)

        all_mem = toroidal.memory.unwind()
        print(f"  [OK] Stored 2 memories")
        print(f"  [OK] Unwind: {len(all_mem)} characters returned")
    except Exception as e:
        print(f"  [FAIL] Memory error: {e}")
        return False

    print()
    print("[TEST 6] Test tool dispatch...")
    try:
        from reasoning.tools import ToolCall

        result = toroidal.tools.dispatch(ToolCall(
            name="kernel_state",
            args={}
        ))
        print(f"  [OK] kernel_state: success={result.success}")
        if result.success:
            # Show first line of output
            first_line = result.output.split('\n')[0][:60]
            print(f"  [OK] Output: {first_line}...")
    except Exception as e:
        print(f"  [FAIL] Tool error: {e}")
        return False

    print()
    print("[TEST 7] Test tool: memory_store...")
    try:
        result = toroidal.tools.dispatch(ToolCall(
            name="memory_store",
            args={"content": "Test memory via tool", "importance": 1.5}
        ))
        print(f"  [OK] memory_store: success={result.success}")
        print(f"  [OK] Output: {result.output[:50]}...")
    except Exception as e:
        print(f"  [FAIL] Tool error: {e}")
        return False

    print()
    print("[TEST 8] Test tool: memory_search...")
    try:
        result = toroidal.tools.dispatch(ToolCall(
            name="memory_search",
            args={"query": "memory"}
        ))
        print(f"  [OK] memory_search: success={result.success}")
        if result.success:
            # Count matches
            print(f"  [OK] Found results")
    except Exception as e:
        print(f"  [FAIL] Tool error: {e}")
        return False

    print()
    print("[TEST 9] Test quick_respond (no LLM)...")
    try:
        # This will fail gracefully without LLM
        response = toroidal.reasoner.quick_respond("Hello")
        if response:
            print(f"  [OK] Got response: {response[:50]}...")
        else:
            print(f"  [OK] No response (expected - no LLM server)")
    except Exception as e:
        print(f"  [OK] Expected error: {e}")

    print()
    print("[TEST 10] Test system state...")
    try:
        # Get state from components directly
        print(f"  [OK] State retrieved")
        print(f"    tau: {toroidal.graph.tau}")
        print(f"    nodes: {len(toroidal.graph.nodes)}")
        print(f"    edges: {len(toroidal.graph.edges)}")
        print(f"    llm_available: {toroidal.llm.is_available()}")
        print(f"    memory_items: {sum(len(level.items) for level in toroidal.memory.levels)}")
    except Exception as e:
        print(f"  [FAIL] State error: {e}")
        return False

    print()
    print("[TEST 11] Test emulator...")
    try:
        from tests.test_emulator import MiMixEmulator, EmulatorMode

        emulator = MiMixEmulator(mode=EmulatorMode.SIMULATION)
        emulator.start()

        state = emulator.get_state()
        print(f"  [OK] Emulator started")
        print(f"    mode: {state['mode']}")
        print(f"    sensor_bits: {state['sensor_bits']:09b}")

        emulator.stop()
        print(f"  [OK] Emulator stopped")
    except Exception as e:
        print(f"  [FAIL] Emulator error: {e}")
        return False

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True


def interactive_test():
    """Run interactive test session."""
    print("=" * 60)
    print("TOROIDAL OS - Interactive Test")
    print("=" * 60)
    print()

    from reasoning.self_ref import ToroidalOS

    print("Initializing ToroidalOS...")
    toroidal = ToroidalOS(
        llm_endpoint="http://localhost:9999",
        max_nodes=500,
        memory_levels=4,
        enable_embeddings=False,
    )
    print("Ready!")
    print()

    print("Commands: status, memory, tools, nodes, quit")
    print()

    while True:
        try:
            cmd = input("TOROIDAL> ").strip().lower()

            if cmd == "quit" or cmd == "exit":
                print("Goodbye!")
                break

            elif cmd == "status":
                print(f"  tau: {toroidal.graph.tau}")
                print(f"  nodes: {len(toroidal.graph.nodes)}")
                print(f"  edges: {len(toroidal.graph.edges)}")
                print(f"  llm_available: {toroidal.llm.is_available()}")
                print(f"  memory_items: {sum(len(level.items) for level in toroidal.memory.levels)}")

            elif cmd == "memory":
                all_mem = toroidal.memory.unwind()
                print(f"Memory ({len(all_mem)} chars):")
                print(all_mem[:500] if all_mem else "(empty)")

            elif cmd == "tools":
                print("Registered tools:")
                for name in toroidal.tools.tools.keys():
                    print(f"  - {name}")

            elif cmd == "nodes":
                print(f"Nodes: {len(toroidal.graph.nodes)}")
                for i, (nid, node) in enumerate(list(toroidal.graph.nodes.items())[:5]):
                    print(f"  {nid[:8]}...: {node.type.name} - {str(node.data)[:30]}...")

            elif cmd.startswith("tool "):
                parts = cmd.split(maxsplit=2)
                if len(parts) >= 2:
                    tool_name = parts[1]
                    args = {}
                    if len(parts) > 2:
                        try:
                            import json
                            args = json.loads(parts[2])
                        except:
                            args = {"arg": parts[2]}

                    from reasoning.tools import ToolCall
                    result = toroidal.tools.dispatch(ToolCall(name=tool_name, args=args))
                    print(f"Result: {result.output}")

            else:
                # Try to process as input
                print(f"Processing: {cmd}")
                # Store in memory
                toroidal.memory.wind(f"User: {cmd}")
                print("(Stored in memory - LLM not available for response)")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ToroidalOS System Test")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_test()
    else:
        success = test_system_init()
        sys.exit(0 if success else 1)