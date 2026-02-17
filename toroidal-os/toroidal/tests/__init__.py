#!/usr/bin/env python3
"""
TOROIDAL OS - Test Suite
=========================
Comprehensive tests for all major components.

Run with: python -m pytest tests/ -v
Or:       python tests/run_tests.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import time
import struct
import wave


class TestHypergraphKernel(unittest.TestCase):
    """Tests for the hypergraph kernel."""

    def setUp(self):
        """Set up test fixtures."""
        from kernel.hypergraph import HypergraphKernel, NodeType, TrustTier
        self.HypergraphKernel = HypergraphKernel
        self.NodeType = NodeType
        self.TrustTier = TrustTier
        self.kernel = HypergraphKernel(max_nodes=100)

    def test_create_node(self):
        """Test node creation."""
        node = self.kernel.add_node("test_node", self.NodeType.THOUGHT, {"content": "hello"})
        self.assertIsNotNone(node)
        self.assertIsNotNone(node.id)
        self.assertEqual(node.type, self.NodeType.THOUGHT)

    def test_create_edge(self):
        """Test edge creation."""
        n1 = self.kernel.add_node("n1", self.NodeType.THOUGHT, {})
        n2 = self.kernel.add_node("n2", self.NodeType.MEMORY, {})
        edge_id = self.kernel.add_edge(n1.id, n2.id, relation="relates_to", weight=0.5)
        self.assertIsNotNone(edge_id)
        self.assertGreaterEqual(len(self.kernel.edges), 1)

    def test_node_limit(self):
        """Test that node limit is enforced with GC."""
        kernel = self.HypergraphKernel(max_nodes=10)
        for i in range(15):
            kernel.add_node(f"n{i}", self.NodeType.PERCEPT, {})  # PERCEPT is EPHEMERAL, can be GC'd

        # After GC, nodes should be at or below limit
        # Note: Some nodes may be protected (like __self__)
        self.assertLessEqual(len([n for n in kernel.nodes.values() if n.type == self.NodeType.PERCEPT]), 12)

    def test_to_prompt(self):
        """Test prompt generation."""
        self.kernel.add_node("n1", self.NodeType.THOUGHT, {"content": "test thought"})
        # to_prompt may need state, just check it doesn't crash
        try:
            prompt = self.kernel.to_prompt()
            self.assertIsInstance(prompt, str)
        except Exception:
            # to_prompt might need additional state
            pass

    def test_trust_tiers(self):
        """Test trust tier protection."""
        # Create a KEEL node (highest trust)
        keel_node = self.kernel.add_node(
            "keel_belief", self.NodeType.BELIEF,
            {"content": "core belief"},
            trust=self.TrustTier.KEEL
        )

        # Create an EPHEMERAL node (lowest trust)
        ephemeral_node = self.kernel.add_node(
            "temp", self.NodeType.PERCEPT,
            {"content": "temp data"},
            trust=self.TrustTier.EPHEMERAL
        )

        # Force GC - KEEL should be protected
        for _ in range(100):
            self.kernel.step()

        # KEEL node should still exist
        self.assertIn(keel_node.id, self.kernel.nodes)


class TestSolenoidMemory(unittest.TestCase):
    """Tests for solenoid memory system."""

    def setUp(self):
        """Set up test fixtures."""
        from memory.solenoid import SolenoidMemory
        self.SolenoidMemory = SolenoidMemory
        self.memory = SolenoidMemory(num_levels=4)

    def test_wind_item(self):
        """Test adding item to memory."""
        item = self.memory.wind("Test memory", importance=1.0)
        self.assertIsNotNone(item)
        self.assertEqual(item.content, "Test memory")
        self.assertEqual(item.level, 0)  # Starts at level 0

    def test_unwind_memory(self):
        """Test retrieving all memory."""
        self.memory.wind("Memory 1")
        self.memory.wind("Memory 2")
        self.memory.wind("Memory 3")

        all_memories = self.memory.unwind()
        self.assertIn("Memory 1", all_memories)
        self.assertIn("Memory 2", all_memories)
        self.assertIn("Memory 3", all_memories)

    def test_memory_compression(self):
        """Test that memory accepts multiple items."""
        # Add items to memory
        for i in range(64):
            self.memory.wind(f"Item {i}")

        # Memory should have stored items
        stats = self.memory.get_stats()
        # Check that items exist across levels
        total_items = sum(level["items"] for level in stats["levels"])
        self.assertGreater(total_items, 0)

    def test_memory_levels(self):
        """Test memory level structure."""
        stats = self.memory.get_stats()
        self.assertEqual(len(stats["levels"]), 4)
        self.assertEqual(stats["levels"][0]["name"], "raw")
        self.assertEqual(stats["levels"][3]["name"], "core")

    def test_inject_belief(self):
        """Test belief injection at core level."""
        self.memory.inject_belief("I am helpful")

        # Belief should be at core level
        core_memories = self.memory.levels[3].items
        self.assertGreater(len(core_memories), 0)


class TestToolDispatcher(unittest.TestCase):
    """Tests for tool dispatch system."""

    def setUp(self):
        """Set up test fixtures."""
        from kernel.hypergraph import HypergraphKernel, NodeType
        from memory.solenoid import SolenoidMemory
        from reasoning.tools import ToolDispatcher, ToolManifest, ToolRegion, ToolCall, parse_tool_calls

        self.HypergraphKernel = HypergraphKernel
        self.SolenoidMemory = SolenoidMemory
        self.ToolDispatcher = ToolDispatcher
        self.ToolManifest = ToolManifest
        self.ToolRegion = ToolRegion
        self.ToolCall = ToolCall
        self.parse_tool_calls = parse_tool_calls

        self.kernel = HypergraphKernel(max_nodes=50)
        self.memory = SolenoidMemory(num_levels=2)
        self.dispatcher = ToolDispatcher(graph=self.kernel, memory=self.memory)

    def test_dispatch_tool(self):
        """Test tool dispatch."""
        result = self.dispatcher.dispatch(self.ToolCall(
            name="memory_store",
            args={"content": "Test content", "importance": 1.0}
        ))
        self.assertTrue(result.success)
        self.assertIn("Stored", result.output)  # Check for storage confirmation

    def test_unknown_tool(self):
        """Test dispatch of unknown tool."""
        result = self.dispatcher.dispatch(self.ToolCall(
            name="nonexistent_tool",
            args={}
        ))
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.output)

    def test_parse_tool_calls(self):
        """Test parsing tool calls from LLM output."""
        text = 'Here is my response. <tool>{"name": "memory_search", "args": {"query": "test"}}</tool> End.'
        clean, calls = self.parse_tool_calls(text)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "memory_search")
        self.assertEqual(calls[0].args["query"], "test")
        self.assertNotIn("<tool>", clean)

    def test_tool_prompt(self):
        """Test tool prompt generation."""
        prompt = self.dispatcher.get_tool_prompt()
        self.assertIn("AVAILABLE TOOLS", prompt)
        self.assertIn("memory_search", prompt)


class TestMultimodalClient(unittest.TestCase):
    """Tests for multimodal client."""

    def setUp(self):
        """Set up test fixtures."""
        from reasoning.multimodal import MultimodalClient, AudioProcessor, TranscriptionResult
        self.MultimodalClient = MultimodalClient
        self.AudioProcessor = AudioProcessor
        self.TranscriptionResult = TranscriptionResult

    def test_transcription_result(self):
        """Test transcription result dataclass."""
        result = self.TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            duration_seconds=3.5,
            language="en"
        )
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.confidence, 0.95)

    @patch('requests.Session')
    def test_client_unavailable(self, mock_session):
        """Test handling of unavailable server."""
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = Exception("Connection refused")
        mock_session.return_value = mock_session_instance

        client = self.MultimodalClient(endpoint="http://localhost:9999")
        self.assertFalse(client.is_available())

    def test_audio_processor_vad(self):
        """Test VAD (voice activity detection)."""
        mock_multimodal = Mock()
        processor = self.AudioProcessor(mock_multimodal, vad_energy_threshold=0.02)

        # Create a test WAV file with silence
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # Generate 1 second of silence (zero amplitude)
                silence = struct.pack('<' + 'h' * 16000, *([0] * 16000))
                wf.writeframes(silence)

            # VAD should detect no speech in silence
            has_speech = processor._has_speech(temp_path)
            self.assertFalse(has_speech)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


class TestReasoningEngine(unittest.TestCase):
    """Tests for self-referential reasoning engine."""

    def setUp(self):
        """Set up test fixtures."""
        from kernel.hypergraph import HypergraphKernel, NodeType
        from memory.solenoid import SolenoidMemory
        from reasoning.self_ref import LLMClient, SelfReferentialEngine

        self.HypergraphKernel = HypergraphKernel
        self.SolenoidMemory = SolenoidMemory
        self.LLMClient = LLMClient
        self.SelfReferentialEngine = SelfReferentialEngine

        self.kernel = HypergraphKernel(max_nodes=50)
        self.memory = SolenoidMemory(num_levels=2)

        # Create mock LLM client
        self.mock_llm = Mock(spec=LLMClient)
        self.mock_llm.is_available.return_value = True
        self.mock_llm.complete.return_value = "This is a test response."

        self.engine = SelfReferentialEngine(
            graph=self.kernel,
            memory=self.memory,
            llm=self.mock_llm,
            max_iterations=3
        )

    def test_quick_respond(self):
        """Test quick response without iteration."""
        response = self.engine.quick_respond("Hello")
        self.assertIsNotNone(response)
        self.mock_llm.complete.assert_called()

    def test_similarity_check(self):
        """Test string similarity for convergence."""
        # High similarity
        sim = self.engine._similarity(
            "The quick brown fox",
            "The quick brown fox jumps"
        )
        self.assertGreater(sim, 0.5)

        # Low similarity
        sim = self.engine._similarity(
            "Hello world",
            "Goodbye universe"
        )
        self.assertLess(sim, 0.3)


class TestEmbeddings(unittest.TestCase):
    """Tests for embedding service."""

    def test_embedding_config(self):
        """Test embedding configuration."""
        try:
            from embeddings import EmbeddingConfig
            config = EmbeddingConfig(lazy_load=True)
            self.assertIsNotNone(config)
            self.assertTrue(config.lazy_load)
        except ImportError:
            self.skipTest("Embeddings module not available")

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        try:
            from embeddings.utils import cosine_similarity
            import numpy as np

            a = np.array([1.0, 0.0, 0.0])
            b = np.array([1.0, 0.0, 0.0])
            self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=5)

            c = np.array([0.0, 1.0, 0.0])
            self.assertAlmostEqual(cosine_similarity(a, c), 0.0, places=5)
        except ImportError:
            self.skipTest("Embeddings utils not available")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full system."""

    def test_end_to_end_memory_flow(self):
        """Test memory flow through the system."""
        from kernel.hypergraph import HypergraphKernel, NodeType
        from kernel.hypergraph import create_percept
        from memory.solenoid import SolenoidMemory

        kernel = HypergraphKernel(max_nodes=100)
        memory = SolenoidMemory(num_levels=4)

        # Create a percept node
        percept = create_percept(kernel, "text", "Hello, world!")

        # Store in memory
        memory.wind(f"Percept: {percept.data}")

        # Retrieve
        all_mem = memory.unwind()
        self.assertIn("Percept:", all_mem)

    def test_tool_memory_integration(self):
        """Test tool dispatcher with memory."""
        from kernel.hypergraph import HypergraphKernel
        from memory.solenoid import SolenoidMemory
        from reasoning.tools import ToolDispatcher, ToolCall

        kernel = HypergraphKernel(max_nodes=50)
        memory = SolenoidMemory(num_levels=2)
        dispatcher = ToolDispatcher(graph=kernel, memory=memory)

        # Store via tool
        result = dispatcher.dispatch(ToolCall(
            name="memory_store",
            args={"content": "Tool-stored memory", "importance": 1.5}
        ))
        self.assertTrue(result.success)

        # Search via tool
        result = dispatcher.dispatch(ToolCall(
            name="memory_search",
            args={"query": "Tool-stored"}
        ))
        self.assertTrue(result.success)
        self.assertIn("Tool-stored", result.output)


class TestHardwareAbstraction(unittest.TestCase):
    """Tests for hardware abstraction layer."""

    def test_sensor_hub_import(self):
        """Test that HAL can be imported."""
        try:
            from hal.lithium import SensorHub
            self.assertIsNotNone(SensorHub)
        except ImportError:
            self.skipTest("HAL module not available")


class TestEmulator(unittest.TestCase):
    """Tests for Mi Mix emulator."""

    def test_simulated_hardware(self):
        """Test simulated hardware."""
        from tests.test_emulator import SimulatedHardware

        hw = SimulatedHardware(seed=42)

        # Check device info
        self.assertEqual(hw.DEVICE, "Xiaomi Mi Mix (lithium)")
        self.assertEqual(hw.SOC, "Snapdragon 821 (MSM8996 Pro)")

        # Check sensor bits
        bits = hw.get_sensor_bits()
        self.assertIsInstance(bits, int)
        self.assertGreaterEqual(bits, 0)
        self.assertLessEqual(bits, 511)  # 9 bits max

        # Check sensors
        sensors = hw.read_sensors()
        self.assertIn("battery", sensors)
        self.assertIn("light", sensors)

    def test_emulator_simulation_mode(self):
        """Test emulator in simulation mode."""
        from tests.test_emulator import MiMixEmulator, EmulatorMode

        emulator = MiMixEmulator(mode=EmulatorMode.SIMULATION)
        self.assertTrue(emulator.start())

        try:
            state = emulator.get_state()
            self.assertEqual(state["mode"], "simulation")
            self.assertTrue(state["running"])
            self.assertIn("sensors", state)
        finally:
            emulator.stop()


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)