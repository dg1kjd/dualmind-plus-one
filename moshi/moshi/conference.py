# Title: DualMind+1 Moshi Edition Conference Server
# Description: This module implements a conference server for DualMind+1 Moshi Edition, allowing two instances to communicate with each other while enabling user monitoring and intervention. It handles audio transfer and processing between processes.
# Author: Jens David
# Copyright: 2026 Jens David Consulting
# License: MIT

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
DualMind+1 Moshi Edition Conference Server

Two PersonaPlex instances talking to each other, with user able to monitor and intervene.
- Left (GPU 0): PersonaPlex A
- Right (GPU 1): PersonaPlex B  
- User can intervene via microphone, audio mixed all-to-all
"""

import argparse
import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Optional, List, Dict, Any
import json
import multiprocessing as mp
from multiprocessing import Queue, Value, Array
import ctypes

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import torch.multiprocessing as torch_mp
import random


# ============================================================
# SHARED MEMORY RING BUFFER
# Zero-copy audio transfer between processes
# ============================================================

class SharedAudioBuffer:
    """Lock-free single-producer single-consumer ring buffer using shared memory"""
    def __init__(self, frame_size: int, num_frames: int = 8):
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.buffer_size = frame_size * num_frames
        
        # Shared memory: audio data + write/read indices
        # Use lock=True to get SynchronizedArray with get_obj()
        self._data = Array(ctypes.c_float, self.buffer_size, lock=True)
        self._write_idx = Value(ctypes.c_long, 0, lock=True)
        self._read_idx = Value(ctypes.c_long, 0, lock=True)
    
    def _get_data_array(self) -> np.ndarray:
        """Get numpy view of data buffer - must be called in each process separately"""
        return np.frombuffer(self._data.get_obj(), dtype=np.float32)
    
    def write(self, frame: np.ndarray) -> bool:
        """Write a frame. Returns False if buffer is full."""
        write_pos = self._write_idx.value
        read_pos = self._read_idx.value
        
        # Check if full (leave one slot empty to distinguish full from empty)
        next_write = (write_pos + 1) % self.num_frames
        if next_write == read_pos:
            return False  # Buffer full
        
        # Write frame to buffer - get numpy view in THIS process context
        np_data = self._get_data_array()
        start = write_pos * self.frame_size
        frame_len = min(len(frame), self.frame_size)
        np_data[start:start + frame_len] = frame[:frame_len]
        
        # Update write index
        self._write_idx.value = next_write
        return True
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame. Returns None if buffer is empty."""
        write_pos = self._write_idx.value
        read_pos = self._read_idx.value
        
        if read_pos == write_pos:
            return None  # Buffer empty
        
        # Read frame from buffer - get numpy view in THIS process context
        np_data = self._get_data_array()
        start = read_pos * self.frame_size
        frame = np_data[start:start + self.frame_size].copy()
        
        # Update read index
        self._read_idx.value = (read_pos + 1) % self.num_frames
        return frame


class SharedOutputBuffer:
    """Shared buffer for worker output: audio + text + level"""
    def __init__(self, frame_size: int, max_text_len: int = 256):
        self.frame_size = frame_size
        self.max_text_len = max_text_len
        
        # Audio data
        self._audio = Array(ctypes.c_float, frame_size, lock=True)
        self._audio_len = Value(ctypes.c_int, 0, lock=True)
        self._level = Value(ctypes.c_float, 0.0, lock=True)
        
        # Text data (ASCII bytes)
        self._text = Array(ctypes.c_char, max_text_len, lock=True)
        self._text_len = Value(ctypes.c_int, 0, lock=True)
        
        # Frame counter for detecting new data
        self._frame_id = Value(ctypes.c_long, 0, lock=True)
        self._last_read_id = 0
    
    def _get_audio_array(self) -> np.ndarray:
        """Get numpy view of audio buffer - must be called in each process separately"""
        return np.frombuffer(self._audio.get_obj(), dtype=np.float32)
    
    def write(self, audio: np.ndarray, text: str, level: float):
        """Write output from worker process"""
        audio_len = min(len(audio), self.frame_size)
        if audio_len > 0:
            # Create numpy view in THIS process context
            np_audio = self._get_audio_array()
            np_audio[:audio_len] = audio[:audio_len]
        self._audio_len.value = audio_len
        self._level.value = level
        
        text_bytes = text.encode('utf-8')[:self.max_text_len - 1]
        self._text.get_obj()[:len(text_bytes)] = text_bytes
        self._text_len.value = len(text_bytes)
        
        self._frame_id.value += 1
    
    def read(self) -> Optional[tuple]:
        """Read output. Returns None if no new data since last read."""
        frame_id = self._frame_id.value
        if frame_id == self._last_read_id:
            return None
        
        self._last_read_id = frame_id
        
        audio_len = self._audio_len.value
        # Create numpy view in THIS process context
        np_audio = self._get_audio_array()
        audio = np_audio[:audio_len].copy() if audio_len > 0 else np.array([], dtype=np.float32)
        level = self._level.value
        
        text_len = self._text_len.value
        text = bytes(self._text.get_obj()[:text_len]).decode('utf-8') if text_len > 0 else ""
        
        return audio, text, level

from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog

logger = setup_logger(__name__)


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


# ============================================================
# MULTIPROCESSING PERSONA WORKER
# Runs in separate process with dedicated CUDA context
# Uses shared memory for zero-copy audio transfer
# ============================================================

def persona_worker_process(
    name: str,
    device_str: str,
    mimi_weight: str,
    moshi_weight: str,
    tokenizer_path: str,
    voice_prompt_dir: str,
    ctrl_queue: Queue,       # Control messages only: ("config", ...) ("reset",) ("system_prompts",) ("shutdown",)
    status_queue: Queue,     # Status messages: ("ready",) ("system_prompts_done",) ("text", str)
    input_buffer: SharedAudioBuffer,   # Shared memory for input audio (from mixer)
    output_buffer: SharedOutputBuffer, # Shared memory for output audio+text+level
    sample_rate: int = 24000,
    frame_rate: float = 12.5,
):
    """
    Standalone persona worker running in its own process.
    Has exclusive ownership of its GPU and CUDA context.
    Uses shared memory buffers for zero-copy audio I/O.
    """
    import torch
    from .models import loaders, LMGen
    import sentencepiece
    import numpy as np
    
    device = torch.device(device_str)
    frame_size = int(sample_rate / frame_rate)
    
    # Set CUDA device for this process
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    print(f"[{name}] Worker process started on {device} (PID={os.getpid()})")
    
    # Load models in this process
    print(f"[{name}] Loading models...")
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * frame_rate),
        sample_rate=sample_rate,
        device=device,
        frame_rate=frame_rate,
    )
    
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    
    # Warmup
    print(f"[{name}] Warming up...")
    with torch.no_grad():
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
            codes = mimi.encode(chunk)
            _ = other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    
    print(f"[{name}] Ready!")
    status_queue.put(("ready",))
    
    # State
    current_voice = None
    current_prompt = None
    audio_level = 0.0
    running = True
    
    # Pre-allocate buffers to avoid repeated allocation
    output_accum = np.zeros(frame_size * 2, dtype=np.float32)
    output_len = 0
    
    def configure(voice_prompt: str, text_prompt: str):
        nonlocal current_voice, current_prompt
        voice_path = os.path.join(voice_prompt_dir, voice_prompt)
        if not os.path.exists(voice_path):
            print(f"[{name}] Voice not found: {voice_path}")
            return
        
        if lm_gen.voice_prompt != voice_path:
            if voice_path.endswith('.pt'):
                lm_gen.load_voice_prompt_embeddings(voice_path)
            else:
                lm_gen.load_voice_prompt(voice_path)
        
        lm_gen.text_prompt_tokens = text_tokenizer.encode(
            wrap_with_system_tags(text_prompt)
        ) if text_prompt else None
        current_voice = voice_prompt
        current_prompt = text_prompt
    
    def reset_streaming():
        nonlocal audio_level, output_len
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        audio_level = 0.0
        output_len = 0
    
    # Diagnostic counters
    diag_frame_count = [0]
    diag_code_count = [0]
    diag_token_count = [0]
    diag_none_count = [0]
    diag_input_rms_sum = [0.0]
    
    def process_frame(audio_chunk: np.ndarray) -> tuple:
        """Process exactly one frame of audio"""
        nonlocal audio_level, output_len
        new_text = ""
        output_len = 0
        
        diag_frame_count[0] += 1
        # Track input RMS to verify user audio reaches worker
        input_rms = float(np.sqrt(np.mean(audio_chunk ** 2))) if len(audio_chunk) > 0 else 0.0
        diag_input_rms_sum[0] += input_rms
        
        with torch.no_grad():
            chunk_tensor = torch.from_numpy(audio_chunk).to(device=device)[None, None]
            codes = mimi.encode(chunk_tensor)
            _ = other_mimi.encode(chunk_tensor)
            
            num_codes = codes.shape[-1]
            diag_code_count[0] += num_codes
            
            for c in range(num_codes):
                tokens = lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    diag_none_count[0] += 1
                    continue
                
                diag_token_count[0] += 1
                main_pcm = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
                main_pcm_np = main_pcm.cpu().numpy()[0, 0]
                
                # Append to pre-allocated buffer
                pcm_len = len(main_pcm_np)
                if output_len + pcm_len <= len(output_accum):
                    output_accum[output_len:output_len + pcm_len] = main_pcm_np
                    output_len += pcm_len
                
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    new_text += _text
        
        if output_len > 0:
            audio_level = float(np.sqrt(np.mean(output_accum[:output_len] ** 2)))
        
        # Log diagnostics every 20 frames (~1 second)
        if diag_frame_count[0] % 20 == 0:
            avg_input_rms = diag_input_rms_sum[0] / 20
            diag_input_rms_sum[0] = 0.0
            print(f"[{name}] frames={diag_frame_count[0]} codes={diag_code_count[0]} tokens={diag_token_count[0]} none={diag_none_count[0]} output_len={output_len} level={audio_level:.4f} in_rms={avg_input_rms:.4f}")
        
        return output_accum[:output_len], new_text, audio_level
    
    def process_system_prompts():
        """Process voice and text prompts synchronously"""
        with torch.no_grad():
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()
    
    # Main loop - blocking wait for frame signals (synchronized to mic timing)
    try:
        while running:
            # Wait for next message (blocking with timeout)
            try:
                msg = ctrl_queue.get(timeout=0.5)
            except:
                continue
            
            if msg is None or msg[0] == "shutdown":
                print(f"[{name}] Shutdown received")
                break
            elif msg[0] == "config":
                configure(msg[1], msg[2])
            elif msg[0] == "reset":
                reset_streaming()
            elif msg[0] == "system_prompts":
                process_system_prompts()
                status_queue.put(("system_prompts_done",))
            elif msg[0] == "frame_ready":
                # Read audio from shared input buffer
                frame = input_buffer.read()
                if frame is not None:
                    # Process the frame
                    output_pcm, new_text, level = process_frame(frame)
                    
                    # Write output to shared buffer
                    if len(output_pcm) == 0:
                        output_pcm = np.zeros(frame_size, dtype=np.float32)
                    output_buffer.write(output_pcm, new_text, level)
                    
                    # Signal output ready (with text if any)
                    status_queue.put(("output_ready", new_text, level))
                else:
                    # No frame available, send empty output
                    output_buffer.write(np.zeros(frame_size, dtype=np.float32), "", 0.0)
                    status_queue.put(("output_ready", "", 0.0))
    
    except KeyboardInterrupt:
        print(f"[{name}] Interrupted")
    except Exception as e:
        print(f"[{name}] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[{name}] Worker process ending")


@dataclass
class PersonaHandle:
    """Handle for a persona worker process with shared memory buffers"""
    name: str
    device: torch.device
    process: mp.Process
    ctrl_queue: Queue           # Control messages (config, reset, etc)
    status_queue: Queue         # Status messages (ready, text, etc)
    input_buffer: SharedAudioBuffer    # Shared memory for input audio
    output_buffer: SharedOutputBuffer  # Shared memory for output audio


class ConferenceServer:
    """
    Conference server using multiprocessing for GPU isolation.
    Each persona runs in its own process with dedicated CUDA context.
    Uses shared memory for zero-copy audio transfer.
    """
    def __init__(
        self,
        device_a: torch.device,
        device_b: torch.device,
        mimi_weight: str,
        moshi_weight: str,
        tokenizer_path: str,
        voice_prompt_dir: str,
    ):
        self.device_a = device_a
        self.device_b = device_b
        self.mimi_weight = mimi_weight
        self.moshi_weight = moshi_weight
        self.tokenizer_path = tokenizer_path
        self.voice_prompt_dir = voice_prompt_dir
        self.lock = asyncio.Lock()
        self.sample_rate = 24000
        self.frame_rate = 12.5
        self.frame_size = int(self.sample_rate / self.frame_rate)
        
        # Spawn worker processes
        logger.info("Spawning worker processes...")
        self.persona_a = self._spawn_worker("PersonaA", device_a)
        self.persona_b = self._spawn_worker("PersonaB", device_b)
        
        # Wait for both workers to be ready
        logger.info("Waiting for workers to initialize...")
        self._wait_for_ready(self.persona_a)
        self._wait_for_ready(self.persona_b)
        logger.info("Conference server ready!")

    def _spawn_worker(self, name: str, device: torch.device) -> PersonaHandle:
        """Spawn a persona worker process with shared memory buffers"""
        ctrl_queue = mp.Queue(maxsize=16)
        status_queue = mp.Queue(maxsize=64)
        
        # Shared memory buffers for audio (zero-copy)
        input_buffer = SharedAudioBuffer(self.frame_size, num_frames=8)
        output_buffer = SharedOutputBuffer(self.frame_size * 2)  # Output can be larger
        
        process = mp.Process(
            target=persona_worker_process,
            args=(
                name,
                str(device),
                self.mimi_weight,
                self.moshi_weight,
                self.tokenizer_path,
                self.voice_prompt_dir,
                ctrl_queue,
                status_queue,
                input_buffer,
                output_buffer,
                self.sample_rate,
                self.frame_rate,
            ),
            daemon=True,
        )
        process.start()
        logger.info(f"Spawned {name} worker process (PID={process.pid})")
        
        return PersonaHandle(
            name=name,
            device=device,
            process=process,
            ctrl_queue=ctrl_queue,
            status_queue=status_queue,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
        )
    
    def _wait_for_ready(self, handle: PersonaHandle, timeout: float = 120.0):
        """Wait for a worker to signal ready"""
        try:
            msg = handle.status_queue.get(timeout=timeout)
            if msg[0] == "ready":
                logger.info(f"{handle.name} worker ready")
            else:
                logger.warning(f"{handle.name} unexpected message: {msg}")
        except Exception as e:
            logger.error(f"{handle.name} failed to initialize: {e}")
            raise
    
    def shutdown(self):
        """Shutdown all worker processes"""
        for handle in [self.persona_a, self.persona_b]:
            try:
                handle.ctrl_queue.put(("shutdown",))
                handle.process.join(timeout=5.0)
                if handle.process.is_alive():
                    handle.process.terminate()
            except Exception as e:
                logger.error(f"Error shutting down {handle.name}: {e}")

    async def handle_conference(self, request):
        """WebSocket handler for conference sessions using multiprocessing workers"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        clog.log("info", f"Conference connection from {request.remote}")

        # Parse configuration from query params
        voice_a = request.query.get("voice_a", "NATF1.pt")
        voice_b = request.query.get("voice_b", "NATM1.pt")
        prompt_a = request.query.get("prompt_a", "You enjoy having a good conversation.")
        prompt_b = request.query.get("prompt_b", "You enjoy having a good conversation.")
        
        close = False
        user_opus_reader = sphn.OpusStreamReader(self.sample_rate)
        output_opus_writer = sphn.OpusStreamWriter(self.sample_rate)
        
        # Asyncio queues for coordination within main process
        user_to_mixer: asyncio.Queue = asyncio.Queue(maxsize=8)
        mixer_to_user: asyncio.Queue = asyncio.Queue(maxsize=8)
        text_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        
        # Stats
        stats = {'opus_bytes_sent': 0, 'mixer_cycles': 0, 'a_frames': 0, 'b_frames': 0}

        async def recv_loop():
            """Receive WebSocket messages and decode user audio"""
            nonlocal close
            try:
                logger.info("[recv_loop] Started")
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"[recv_loop] WS error: {ws.exception()}")
                        break
                    elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    
                    data = message.data
                    if len(data) == 0:
                        continue
                    
                    kind = data[0]
                    if kind == 1:  # User audio
                        user_opus_reader.append_bytes(data[1:])
                        stats['user_opus_bytes'] = stats.get('user_opus_bytes', 0) + len(data) - 1
                    elif kind == 10:  # Config update
                        try:
                            config = json.loads(data[1:].decode('utf-8'))
                            if 'voice_a' in config:
                                self.persona_a.ctrl_queue.put(("config", config['voice_a'], config.get('prompt_a', '')))
                            if 'voice_b' in config:
                                self.persona_b.ctrl_queue.put(("config", config['voice_b'], config.get('prompt_b', '')))
                        except Exception as e:
                            logger.error(f"[recv_loop] Config error: {e}")
            except Exception as e:
                logger.error(f"[recv_loop] Exception: {e}")
            finally:
                logger.info("[recv_loop] Ending")
                close = True

        async def user_input_loop():
            """Read decoded user PCM and send to mixer"""
            logger.info("[UserInput] Started")
            # Accumulate partial frames
            user_accum = np.zeros(self.frame_size * 2, dtype=np.float32)
            user_accum_len = 0
            
            try:
                while not close:
                    await asyncio.sleep(0.005)
                    user_pcm = user_opus_reader.read_pcm()
                    
                    if user_pcm.shape[-1] > 0:
                        # Accumulate incoming PCM
                        pcm_len = user_pcm.shape[-1]
                        if user_accum_len + pcm_len <= len(user_accum):
                            user_accum[user_accum_len:user_accum_len + pcm_len] = user_pcm
                            user_accum_len += pcm_len
                        
                        # Send complete frames
                        while user_accum_len >= self.frame_size:
                            frame = user_accum[:self.frame_size].copy()
                            # Shift remaining data
                            user_accum[:user_accum_len - self.frame_size] = user_accum[self.frame_size:user_accum_len]
                            user_accum_len -= self.frame_size
                            
                            stats['user_frames'] = stats.get('user_frames', 0) + 1
                            try:
                                user_to_mixer.put_nowait(frame)
                            except asyncio.QueueFull:
                                try:
                                    user_to_mixer.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                user_to_mixer.put_nowait(frame)
            except asyncio.CancelledError:
                pass
            finally:
                logger.info("[UserInput] Ended")

        async def process_bridge():
            """
            Bridge between asyncio and multiprocessing:
            - Frame-locked: waits for both workers to complete before next frame
            - Uses shared memory for audio data (zero-copy)
            - Uses queues for synchronization signals
            """
            logger.info("[ProcessBridge] Started")
            silence = np.zeros(self.frame_size, dtype=np.float32)
            
            # Latest outputs from each persona
            a_output = silence.copy()
            a_level = 0.0
            b_output = silence.copy()
            b_level = 0.0
            
            last_stats_time = time.time()
            
            try:
                while not close:
                    # === Get user frame (timing master) ===
                    try:
                        user_frame = user_to_mixer.get_nowait()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.001)
                        continue
                    
                    stats['mixer_cycles'] += 1
                    
                    # === Mix for each destination using PREVIOUS frame outputs ===
                    # Track user audio level for diagnostics
                    user_rms = float(np.sqrt(np.mean(user_frame ** 2))) if len(user_frame) > 0 else 0.0
                    stats['user_rms_sum'] = stats.get('user_rms_sum', 0.0) + user_rms
                    stats['user_rms_count'] = stats.get('user_rms_count', 0) + 1
                    
                    # To A: user + B (A hears user and B)
                    mix_for_a = np.zeros(self.frame_size, dtype=np.float32)
                    b_len = min(len(b_output), self.frame_size)
                    user_len = min(len(user_frame), self.frame_size)
                    if b_len > 0:
                        mix_for_a[:b_len] = 0.5 * b_output[:b_len]
                    if user_len > 0:
                        mix_for_a[:user_len] += 0.5 * user_frame[:user_len]
                    
                    # To B: user + A (B hears user and A)
                    mix_for_b = np.zeros(self.frame_size, dtype=np.float32)
                    a_len = min(len(a_output), self.frame_size)
                    if a_len > 0:
                        mix_for_b[:a_len] = 0.5 * a_output[:a_len]
                    if user_len > 0:
                        mix_for_b[:user_len] += 0.5 * user_frame[:user_len]
                    
                    # === Write to shared memory and signal workers ===
                    self.persona_a.input_buffer.write(mix_for_a)
                    self.persona_b.input_buffer.write(mix_for_b)
                    self.persona_a.ctrl_queue.put(("frame_ready",))
                    self.persona_b.ctrl_queue.put(("frame_ready",))
                    
                    # === Wait for both workers to complete (with timeout) ===
                    a_done = False
                    b_done = False
                    wait_start = time.time()
                    timeout = 0.2  # 200ms timeout per frame
                    
                    while (not a_done or not b_done) and (time.time() - wait_start < timeout):
                        # Check A
                        if not a_done:
                            try:
                                msg = self.persona_a.status_queue.get_nowait()
                                if msg[0] == "output_ready":
                                    a_done = True
                                    stats['a_frames'] += 1
                                    # Read from shared buffer
                                    result = self.persona_a.output_buffer.read()
                                    if result is not None:
                                        a_output, a_text, a_level = result
                                        if len(a_output) == 0:
                                            a_output = silence.copy()
                                        if a_text:
                                            try:
                                                text_queue.put_nowait(("A", a_text))
                                            except asyncio.QueueFull:
                                                pass
                            except:
                                pass
                        
                        # Check B
                        if not b_done:
                            try:
                                msg = self.persona_b.status_queue.get_nowait()
                                if msg[0] == "output_ready":
                                    b_done = True
                                    stats['b_frames'] += 1
                                    # Read from shared buffer
                                    result = self.persona_b.output_buffer.read()
                                    if result is not None:
                                        b_output, b_text, b_level = result
                                        if len(b_output) == 0:
                                            b_output = silence.copy()
                                        if b_text:
                                            try:
                                                text_queue.put_nowait(("B", b_text))
                                            except asyncio.QueueFull:
                                                pass
                            except:
                                pass
                        
                        if not a_done or not b_done:
                            await asyncio.sleep(0.0005)  # Brief yield
                    
                    # === Mix for user output: A + B ===
                    mix_for_user = np.zeros(self.frame_size, dtype=np.float32)
                    a_len = min(len(a_output), self.frame_size)
                    b_len = min(len(b_output), self.frame_size)
                    if a_len > 0:
                        mix_for_user[:a_len] += 0.5 * a_output[:a_len]
                    if b_len > 0:
                        mix_for_user[:b_len] += 0.5 * b_output[:b_len]
                    
                    # === Send to user output queue ===
                    try:
                        mixer_to_user.put_nowait((mix_for_user, a_level, b_level))
                    except asyncio.QueueFull:
                        try:
                            mixer_to_user.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        mixer_to_user.put_nowait((mix_for_user, a_level, b_level))
                    
                    # === Stats logging ===
                    now = time.time()
                    if now - last_stats_time >= 1.0:
                        # Calculate RMS levels to verify audio isn't silent
                        a_rms = float(np.sqrt(np.mean(a_output ** 2))) if len(a_output) > 0 else 0.0
                        b_rms = float(np.sqrt(np.mean(b_output ** 2))) if len(b_output) > 0 else 0.0
                        mix_rms = float(np.sqrt(np.mean(mix_for_user ** 2))) if len(mix_for_user) > 0 else 0.0
                        user_opus = stats.get('user_opus_bytes', 0)
                        user_frames = stats.get('user_frames', 0)
                        user_rms_avg = stats.get('user_rms_sum', 0.0) / max(1, stats.get('user_rms_count', 1))
                        logger.info(f"[STATS] mixer={stats['mixer_cycles']} a={stats['a_frames']} b={stats['b_frames']} opus={stats['opus_bytes_sent']} user_opus={user_opus} user_frames={user_frames} | a_rms={a_rms:.4f} b_rms={b_rms:.4f} mix_rms={mix_rms:.4f} user_rms={user_rms_avg:.4f}")
                        stats['mixer_cycles'] = 0
                        stats['a_frames'] = 0
                        stats['b_frames'] = 0
                        stats['opus_bytes_sent'] = 0
                        stats['user_opus_bytes'] = 0
                        stats['user_frames'] = 0
                        stats['user_rms_sum'] = 0.0
                        stats['user_rms_count'] = 0
                        last_stats_time = now
                        
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[ProcessBridge] Exception: {e}")
                import traceback
                traceback.print_exc()
            finally:
                logger.info("[ProcessBridge] Ended")

        async def send_loop():
            """Send mixed audio and text to WebSocket client"""
            logger.info("[send_loop] Started")
            try:
                while not close:
                    await asyncio.sleep(0.001)
                    
                    # Send audio from mixer
                    try:
                        while not mixer_to_user.empty():
                            mix_for_user, a_level, b_level = mixer_to_user.get_nowait()
                            if len(mix_for_user) > 0:
                                output_opus_writer.append_pcm(mix_for_user)
                            
                            level_msg = json.dumps({'level_a': a_level, 'level_b': b_level}).encode('utf-8')
                            await ws.send_bytes(b"\x04" + level_msg)
                    except asyncio.QueueEmpty:
                        pass
                    
                    # Send text
                    try:
                        while not text_queue.empty():
                            name, text = text_queue.get_nowait()
                            await ws.send_bytes(b"\x02" + name.encode('utf-8') + text.encode('utf-8'))
                    except asyncio.QueueEmpty:
                        pass
                    
                    # Send opus
                    msg = output_opus_writer.read_bytes()
                    if len(msg) > 0:
                        stats['opus_bytes_sent'] += len(msg)
                        await ws.send_bytes(b"\x01" + msg)
                        
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[send_loop] Exception: {e}")
            finally:
                logger.info("[send_loop] Ended")

        async with self.lock:
            seed_all(42424242)
            
            # Configure and reset workers
            clog.log("info", "Configuring personas...")
            self.persona_a.ctrl_queue.put(("config", voice_a, prompt_a))
            self.persona_b.ctrl_queue.put(("config", voice_b, prompt_b))
            self.persona_a.ctrl_queue.put(("reset",))
            self.persona_b.ctrl_queue.put(("reset",))
            
            # Process system prompts in workers
            clog.log("info", "Processing system prompts...")
            self.persona_a.ctrl_queue.put(("system_prompts",))
            self.persona_b.ctrl_queue.put(("system_prompts",))
            
            # Wait for both to finish
            prompts_done = 0
            timeout = time.time() + 60.0
            while prompts_done < 2 and time.time() < timeout:
                try:
                    msg = self.persona_a.status_queue.get_nowait()
                    if msg[0] == "system_prompts_done":
                        prompts_done += 1
                        clog.log("info", "PersonaA system prompts done")
                except:
                    pass
                try:
                    msg = self.persona_b.status_queue.get_nowait()
                    if msg[0] == "system_prompts_done":
                        prompts_done += 1
                        clog.log("info", "PersonaB system prompts done")
                except:
                    pass
                await asyncio.sleep(0.01)
            
            if prompts_done < 2:
                logger.error("Timeout waiting for system prompts")
            
            # Send handshake
            await ws.send_bytes(b"\x00")
            clog.log("info", "Conference started!")
            
            # Launch async tasks
            tasks = [
                asyncio.create_task(recv_loop()),
                asyncio.create_task(user_input_loop()),
                asyncio.create_task(process_bridge()),
                asyncio.create_task(send_loop()),
            ]
            
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            await ws.close()
            clog.log("info", "Conference ended")
        
        return ws

    async def handle_voices(self, request):
        """Return available voices"""
        voices = []
        if os.path.exists(self.voice_prompt_dir):
            for f in os.listdir(self.voice_prompt_dir):
                if f.endswith('.pt'):
                    voices.append(f)
        return web.json_response(sorted(voices))


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> str:
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    
    logger.info("Retrieving voice prompts...")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"
    
    if not voices_dir.exists():
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    
    return str(voices_dir)


def main():
    # Use 'spawn' for proper CUDA context isolation in worker processes
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="DualMind+1 Moshi Edition Conference Server")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8999, type=int)
    parser.add_argument("--device-a", type=str, default="cuda:0", help="Device for PersonaA (default: cuda:0)")
    parser.add_argument("--device-b", type=str, default="cuda:1", help="Device for PersonaB (default: cuda:1)")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--ssl", type=str, help="Directory with key.pem and cert.pem")
    parser.add_argument("--static", type=str, help="Path to static files for conference UI")
    args = parser.parse_args()

    # Get model paths
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    
    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    
    device_a = torch.device(args.device_a)
    device_b = torch.device(args.device_b)
    
    server = ConferenceServer(
        device_a=device_a,
        device_b=device_b,
        mimi_weight=args.mimi_weight,
        moshi_weight=args.moshi_weight,
        tokenizer_path=args.tokenizer,
        voice_prompt_dir=args.voice_prompt_dir,
    )
    
    app = web.Application()
    app.router.add_get("/api/conference", server.handle_conference)
    app.router.add_get("/api/voices", server.handle_voices)
    
    # Serve static files
    if args.static and os.path.exists(args.static):
        async def handle_root(_):
            return web.FileResponse(os.path.join(args.static, "index.html"))
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=args.static, follow_symlinks=True)
    
    protocol = "http"
    ssl_context = None
    if args.ssl:
        ssl_context, protocol = create_ssl_context(args.ssl)
    
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Conference server at {protocol}://{host_ip}:{args.port}")
    
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()
