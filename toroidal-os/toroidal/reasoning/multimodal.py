#!/usr/bin/env python3
"""
TOROIDAL OS - Multimodal Integration
=====================================
Qwen2.5-Omni audio/text integration via llama.cpp server.

Supports:
- Audio transcription (STT): Send WAV audio, get text
- Audio generation (TTS): Generate speech from text
- Multimodal chat: Audio + text combined

API Format (llama.cpp server):
- /completion with multimodal_data array (base64 encoded)
- /v1/chat/completions with image_url content parts
- Media markers: <__media__> as placeholders
"""

import base64
import json
import os
import tempfile
import time
import wave
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import requests


class AudioFormat(Enum):
    """Supported audio formats for Qwen2.5-Omni"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


@dataclass
class TranscriptionResult:
    """Result of audio transcription"""
    text: str
    confidence: float
    duration_seconds: float
    language: str = "en"


@dataclass
class AudioGenerationResult:
    """Result of TTS audio generation"""
    audio_data: bytes  # Raw audio bytes
    format: AudioFormat
    duration_seconds: float
    sample_rate: int


class MultimodalClient:
    """
    Client for Qwen2.5-Omni multimodal capabilities via llama.cpp server.

    Optimized for Xiaomi Mi Mix constraints:
    - Max audio chunk: 30 seconds (memory budget)
    - Sample rate: 16kHz (speech optimized)
    - Timeout: 60s (slow CPU inference)
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        timeout: int = 60,
        max_audio_seconds: int = 30,
        sample_rate: int = 16000,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_audio_seconds = max_audio_seconds
        self.sample_rate = sample_rate
        self._session = requests.Session()

    def is_available(self) -> bool:
        """Check if llama.cpp server is running with multimodal support"""
        try:
            resp = self._session.get(f"{self.endpoint}/health", timeout=5)
            if resp.status_code != 200:
                return False
            # Check if mmproj is loaded by trying a minimal multimodal call
            # (some servers don't advertise this, so just check health)
            return True
        except Exception:
            return False

    def transcribe_audio(
        self,
        audio_path: str,
        language: str = "en",
        context: str = ""
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio file to text using Qwen2.5-Omni.

        Args:
            audio_path: Path to audio file (WAV preferred)
            language: Language hint (en, zh, etc.)
            context: Optional context to improve transcription

        Returns:
            TranscriptionResult or None on failure
        """
        if not os.path.exists(audio_path):
            print(f"[MULTIMODAL] Audio file not found: {audio_path}")
            return None

        # Read and encode audio
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            print(f"[MULTIMODAL] Failed to read audio: {e}")
            return None

        # Get audio duration
        duration = self._get_audio_duration(audio_path)
        if duration > self.max_audio_seconds:
            print(f"[MULTIMODAL] Audio too long ({duration}s > {self.max_audio_seconds}s max)")
            # Could truncate, but for now just fail
            return None

        # Build prompt for transcription
        prompt_parts = [
            "Transcribe the following audio. Output only the spoken text, no commentary.",
        ]
        if context:
            prompt_parts.append(f"Context: {context}")
        if language != "en":
            prompt_parts.append(f"Language: {language}")

        prompt = "\n".join(prompt_parts) + "\n\nAudio: <__media__>\n\nTranscription:"

        # Call llama.cpp with multimodal data
        payload = {
            "prompt": prompt,
            "multimodal_data": [audio_b64],
            "n_predict": 512,  # Generous for transcription
            "temperature": 0.1,  # Low temp for accuracy
            "stop": ["\n\n", "Audio:", "Transcription:"],
            "stream": False,
        }

        try:
            start = time.time()
            resp = self._session.post(
                f"{self.endpoint}/completion",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            text = result.get("content", "").strip()
            if not text:
                return None

            return TranscriptionResult(
                text=text,
                confidence=0.9,  # Placeholder; could be derived from tokens
                duration_seconds=duration,
                language=language,
            )

        except requests.exceptions.ConnectionError:
            print("[MULTIMODAL] Server not reachable")
            return None
        except requests.exceptions.Timeout:
            print(f"[MULTIMODAL] Transcription timed out ({self.timeout}s)")
            return None
        except Exception as e:
            print(f"[MULTIMODAL] Transcription error: {e}")
            return None

    def generate_speech(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
    ) -> Optional[AudioGenerationResult]:
        """
        Generate speech from text using Qwen2.5-Omni TTS capability.

        Note: Qwen2.5-Omni can generate audio tokens. This method
        extracts the audio output if the model produces it.

        Args:
            text: Text to synthesize
            voice: Voice style (if supported)
            speed: Speech speed multiplier

        Returns:
            AudioGenerationResult or None on failure
        """
        # Build prompt for TTS
        # Qwen2.5-Omni uses special tokens for audio generation
        prompt = f"<|audio|>{text}<|/audio|>"

        payload = {
            "prompt": prompt,
            "n_predict": 1024,  # Audio tokens take more space
            "temperature": 0.7,
            "stream": False,
        }

        try:
            resp = self._session.post(
                f"{self.endpoint}/completion",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            # Check if audio was generated
            # The response may include special markers or base64 audio
            content = result.get("content", "")

            # Look for audio output markers
            # This depends on how Qwen2.5-Omni encodes audio output
            # For now, return None if no audio detected
            if "<|audio_out|" in content or "audio_data:" in content:
                # Extract audio data
                audio_b64 = self._extract_audio_from_response(content)
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    return AudioGenerationResult(
                        audio_data=audio_bytes,
                        format=AudioFormat.WAV,
                        duration_seconds=len(audio_bytes) / (self.sample_rate * 2),
                        sample_rate=self.sample_rate,
                    )

            # Fallback: no audio generated
            print("[MULTIMODAL] No audio output in response")
            return None

        except Exception as e:
            print(f"[MULTIMODAL] Speech generation error: {e}")
            return None

    def chat_with_audio(
        self,
        audio_path: str,
        text: str = "",
        history: List[Dict] = None,
    ) -> Tuple[str, Optional[bytes]]:
        """
        Multimodal chat: combine audio and text input, get text + optional audio output.

        Args:
            audio_path: Path to audio file
            text: Optional text to accompany audio
            history: Conversation history

        Returns:
            (text_response, audio_bytes_or_none)
        """
        # Encode audio
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception:
            return "[Audio input failed]", None

        # Build chat prompt
        if history is None:
            history = []

        messages = history.copy()

        # Add user message with audio
        user_content = []
        if text:
            user_content.append({"type": "text", "text": text})
        user_content.append({
            "type": "image_url",  # llama.cpp uses image_url for all media
            "image_url": {"url": f"data:audio/wav;base64,{audio_b64}"}
        })
        messages.append({"role": "user", "content": user_content})

        # Call chat endpoint
        payload = {
            "model": "qwen2.5-omni",
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.7,
        }

        try:
            resp = self._session.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            text_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Check for audio in response
            audio_out = None
            # Would extract if model outputs audio

            return text_response, audio_out

        except Exception as e:
            return f"[Chat error: {e}]", None

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception:
            # Fallback: estimate from file size (rough)
            try:
                size = os.path.getsize(audio_path)
                # Assume 16kHz, 16-bit mono = 32000 bytes/sec
                return size / 32000
            except Exception:
                return 0.0

    def _extract_audio_from_response(self, content: str) -> Optional[str]:
        """Extract base64 audio data from model response"""
        # Look for markers like <|audio_out|>base64...<|/audio_out|>
        import re
        match = re.search(r'<\|audio_out\|>([^<]+)<\|/audio_out\|>', content)
        if match:
            return match.group(1)

        # Alternative: data:audio/wav;base64,...
        match = re.search(r'data:audio/[^;]+;base64,([A-Za-z0-9+/=]+)', content)
        if match:
            return match.group(1)

        return None


class AudioProcessor:
    """
    High-level audio processing for ToroidalOS.

    Handles:
    - Voice activity detection (simple energy-based)
    - Chunking long audio into processable segments
    - Integration with PerceptionEngine and ActionEngine
    """

    def __init__(
        self,
        multimodal: MultimodalClient,
        sample_rate: int = 16000,
        vad_energy_threshold: float = 0.02,
    ):
        self.multimodal = multimodal
        self.sample_rate = sample_rate
        self.vad_threshold = vad_energy_threshold

    def process_audio_chunk(
        self,
        audio_path: str,
        context: str = ""
    ) -> Optional[str]:
        """
        Process an audio chunk: detect speech, transcribe, return text.

        Args:
            audio_path: Path to audio file
            context: Context for better transcription

        Returns:
            Transcribed text or None if no speech detected
        """
        # Simple VAD: check if audio has enough energy
        if not self._has_speech(audio_path):
            print("[AUDIO] No speech detected in chunk")
            return None

        # Transcribe
        result = self.multimodal.transcribe_audio(audio_path, context=context)
        if result:
            return result.text
        return None

    def _has_speech(self, audio_path: str) -> bool:
        """Simple energy-based voice activity detection"""
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                # Calculate RMS energy
                import struct
                samples = struct.unpack(f"<{len(frames)//2}h", frames)
                rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                # Normalize
                rms_norm = rms / 32768.0
                return rms_norm > self.vad_threshold
        except Exception:
            # If we can't analyze, assume it has speech
            return True

    def chunk_audio(
        self,
        audio_path: str,
        chunk_seconds: float = 10.0,
        output_dir: str = None
    ) -> List[str]:
        """
        Split long audio into chunks for processing.

        Returns:
            List of paths to chunk files
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        chunks = []

        try:
            with wave.open(audio_path, "rb") as wf:
                rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                total_frames = wf.getnframes()
                chunk_frames = int(chunk_seconds * rate)

                chunk_idx = 0
                offset = 0

                while offset < total_frames:
                    wf.setpos(offset)
                    chunk_data = wf.readframes(chunk_frames)

                    chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx}.wav")
                    with wave.open(chunk_path, "wb") as chunk_wf:
                        chunk_wf.setnchannels(channels)
                        chunk_wf.setsampwidth(sampwidth)
                        chunk_wf.setframerate(rate)
                        chunk_wf.writeframes(chunk_data)

                    chunks.append(chunk_path)
                    offset += chunk_frames
                    chunk_idx += 1

        except Exception as e:
            print(f"[AUDIO] Chunking failed: {e}")

        return chunks


# Convenience functions for integration

def create_multimodal_client(endpoint: str = "http://localhost:8080") -> MultimodalClient:
    """Create a configured multimodal client."""
    return MultimodalClient(
        endpoint=endpoint,
        timeout=60,
        max_audio_seconds=30,
        sample_rate=16000,
    )


def transcribe_file(audio_path: str, endpoint: str = "http://localhost:8080") -> Optional[str]:
    """Quick transcription of an audio file."""
    client = create_multimodal_client(endpoint)
    result = client.transcribe_audio(audio_path)
    return result.text if result else None