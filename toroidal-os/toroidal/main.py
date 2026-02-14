#!/usr/bin/env python3
"""
TOROIDAL OS - Main Entry Point
===============================
Self-Referential Operating System for Xiaomi Mi Mix (lithium)

Hardware: Snapdragon 821 • 6GB RAM • 256GB UFS
Model: Qwen2.5-Omni-3B (GGUF Q3_K_S)

This file starts:
1. llama.cpp server with Qwen2.5-Omni
2. Audio input handler (microphone)
3. Display output (framebuffer)
4. Main reasoning loop
"""

import os
import sys
import time
import signal
import subprocess
import threading
import argparse


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning.self_ref import ToroidalOS


class LlamaServer:
    """Manage llama.cpp server process"""
    
    def __init__(
        self,
        model_path: str,
        mmproj_path: str = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        context_size: int = 2048,
        threads: int = 4,
        gpu_layers: int = 0  # CPU only for SD821
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.host = host
        self.port = port
        self.context_size = context_size
        self.threads = threads
        self.gpu_layers = gpu_layers
        self.process = None
    
    def start(self):
        """Start llama.cpp server"""
        cmd = [
            "/opt/llama/llama-server",
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.context_size),
            "-t", str(self.threads),
            "-ngl", str(self.gpu_layers),
            "--mlock",  # Lock model in RAM
        ]
        
        # Add multimodal projector if available
        if self.mmproj_path and os.path.exists(self.mmproj_path):
            cmd.extend(["--mmproj", self.mmproj_path])
        
        print(f"[LLM] Starting server: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # Wait for server to be ready
        time.sleep(10)  # Give it time to load model
        
        return self.is_running()
    
    def stop(self):
        """Stop llama.cpp server"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None
    
    def is_running(self) -> bool:
        """Check if server is running"""
        if not self.process:
            return False
        return self.process.poll() is None


class AudioHandler:
    """Handle microphone input for voice interaction"""
    
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        self.thread = None
    
    def start(self):
        """Start audio capture thread"""
        try:
            import pyaudio
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            print("[AUDIO] Microphone handler started")
        except ImportError:
            print("[AUDIO] PyAudio not available - voice disabled")
    
    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _capture_loop(self):
        """Main audio capture loop"""
        import pyaudio
        import wave
        import tempfile
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 3
        
        p = pyaudio.PyAudio()
        
        while self.running:
            try:
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
                
                frames = []
                for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    if not self.running:
                        break
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wf = wave.open(f.name, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    
                    # Callback with audio file
                    self.callback(f.name)
                    os.unlink(f.name)
                    
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")
                time.sleep(1)
        
        p.terminate()


class DisplayHandler:
    """Handle framebuffer display output"""
    
    def __init__(self, fb_device: str = "/dev/fb0"):
        self.fb_device = fb_device
        self.width = 1080
        self.height = 2040
    
    def clear(self):
        """Clear display"""
        try:
            with open(self.fb_device, 'wb') as fb:
                fb.write(b'\x00' * (self.width * self.height * 4))
        except OSError as exc:
            print(f"[DISPLAY] Failed to clear framebuffer {self.fb_device}: {exc}")
    
    def draw_text(self, text: str, x: int = 50, y: int = 100):
        """Draw text on display (simplified - would need proper font rendering)"""
        # For now, just print to console
        print(f"[DISPLAY] {text}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TOROIDAL OS - Self-Referential AI for Xiaomi Mi Mix"
    )
    parser.add_argument(
        "--model", "-m",
        default="/opt/models/qwen2.5-omni-3b.gguf",
        help="Path to GGUF model"
    )
    parser.add_argument(
        "--mmproj",
        default="/opt/models/mmproj.gguf",
        help="Path to multimodal projector"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="LLM server port"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Number of CPU threads"
    )
    parser.add_argument(
        "--context", "-c",
        type=int,
        default=2048,
        help="Context size"
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice input"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (no LLM server)"
    )
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║                    TOROIDAL OS                          ║
    ║         Self-Referential Operating System               ║
    ║              Xiaomi Mi Mix (lithium)                    ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # Initialize components
    llm_server = None
    audio_handler = None
    
    exit_code = 0

    def shutdown(signum=None, frame=None):
        """Clean shutdown"""
        print("\n[TOROIDAL] Shutting down...")
        if audio_handler:
            audio_handler.stop()
        if llm_server:
            llm_server.stop()
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    try:
        # Start LLM server (unless in test mode)
        if not args.test:
            print("[TOROIDAL] Starting LLM server...")
            llm_server = LlamaServer(
                model_path=args.model,
                mmproj_path=args.mmproj,
                port=args.port,
                threads=args.threads,
                context_size=args.context
            )
            
            if not llm_server.start():
                print("[ERROR] Failed to start LLM server")
                print("[INFO] Running in limited mode without LLM")
        
        # Initialize main OS
        toroidal = ToroidalOS(
            llm_endpoint=f"http://127.0.0.1:{args.port}",
            max_nodes=5000,
            memory_levels=4
        )
        
        # Start voice handler (optional)
        if not args.no_voice:
            def on_audio(audio_path):
                # Process audio through Qwen2.5-Omni
                # For now, placeholder
                pass
            
            audio_handler = AudioHandler(on_audio)
            audio_handler.start()
        
        # Run main REPL
        toroidal.run_repl()
        
    except Exception as e:
        exit_code = 1
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
