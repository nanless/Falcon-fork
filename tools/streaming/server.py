"""Loopback-only WebSocket service for the 120 ms stateful prototype.

The client, not this server, acquires audio, tracks the face, and selects three
mouth images at or before the interval-start audio-clock timestamps. The model
uses the whole current block but requests no later block or output guard. Timing fields are server-local and do not measure
capture, network transit or playback.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import io
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from tools.streaming.common import checkpoint_format_for_visual_mode, SAMPLES, FRAMES, SR, FPS
from tools.streaming.session import NonFiniteOutputError, StreamingSession

MAX_MESSAGE_BYTES = 100_000
MOUTH_BYTES = FRAMES * 96 * 96
MAX_JPEG_BYTES = 16_000
logger = logging.getLogger(__name__)


def _decode_base64(value, size, label):
    if not isinstance(value, str) or len(value) != 4 * ((size + 2) // 3):
        raise ValueError(f"{label} has incorrect encoded length")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"invalid {label} base64") from exc
    if len(raw) != size:
        raise ValueError(f"{label} has incorrect decoded length")
    return raw


def decode_chunk(data):
    if not isinstance(data, dict) or data.get("type") != "chunk":
        raise ValueError("expected a chunk object")
    audio = np.frombuffer(_decode_base64(data.get("audio_f32le"), SAMPLES * 4, "audio"), dtype="<f4").copy()
    if not np.isfinite(audio).all():
        raise ValueError("audio must be finite")
    # Exactly one encoding must be present, so timing and input semantics are unambiguous.
    if ("mouth_gray_u8" in data) == ("mouth_jpeg" in data):
        raise ValueError("provide exactly one of mouth_gray_u8 and mouth_jpeg")
    if "mouth_gray_u8" in data:
        mouth = np.frombuffer(_decode_base64(data["mouth_gray_u8"], MOUTH_BYTES, "mouth"), dtype=np.uint8).copy().reshape(FRAMES, 96, 96)
    else:
        encoded_frames = data["mouth_jpeg"]
        if not isinstance(encoded_frames, list) or len(encoded_frames) != FRAMES:
            raise ValueError("mouth_jpeg must contain three base64 JPEG frames")
        frames = []
        for encoded in encoded_frames:
            if not isinstance(encoded, str) or not encoded or len(encoded) > 4 * ((MAX_JPEG_BYTES + 2) // 3):
                raise ValueError("JPEG frame encoded size is invalid")
            try:
                frame_bytes = base64.b64decode(encoded, validate=True)
                if len(frame_bytes) > MAX_JPEG_BYTES:
                    raise ValueError("JPEG frame exceeds size limit")
                with Image.open(io.BytesIO(frame_bytes)) as image:
                    if image.format != "JPEG" or image.size != (96, 96):
                        raise ValueError("mouth frame must be a 96x96 JPEG")
                    image.load()
                    frames.append(np.asarray(image.convert("L"), dtype=np.uint8).copy())
            except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ValueError("invalid JPEG frame") from exc
        mouth = np.stack(frames)
    # Keep sample clocks as integers: they identify capture time relative to the
    # first audio sample, after client-side camera/audio-clock synchronization.
    return audio, mouth, data.get("sequence"), data.get("start_sample"), data.get("video_samples"), data.get("valid_samples", SAMPLES)


def _same_origin(ws: WebSocket):
    host = ws.headers.get("host", "").lower()
    if not (host.startswith("127.0.0.1:") or host.startswith("localhost:")):
        return False
    origin = ws.headers.get("origin")
    if not origin:  # native clients have no Origin
        return True
    parsed = urlparse(origin)
    return parsed.scheme in ("http", "https") and parsed.netloc.lower() == host


def create_app(model, training_status="unknown", device="cpu", warmup_ms=None):
    if model.training:
        raise ValueError("shared inference model must be in eval mode")
    model_format = checkpoint_format_for_visual_mode(getattr(model, "visual_mode", "causal"),
                                                    getattr(model, "synthesis_mode", "nonnegative"),
                                                    getattr(model, "attention_history_chunks", 0))
    app = FastAPI(title="Falcon 120ms streaming")
    lock = asyncio.Lock()  # Shared model call is serialized; session state stays private.
    active_sessions = 0
    max_sessions = 8
    repo_root = Path(__file__).resolve().parents[2]

    @app.middleware("http")
    async def browser_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            "media-src 'self' blob:; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
            "worker-src 'self' blob:; frame-ancestors 'none'"
        )
        return response

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse("/stream-ui/")

    @app.get("/status")
    async def status():
        return {"format": model_format, "sample_rate": SR,
                "chunk_samples": SAMPLES, "video_fps": FPS, "frames": FRAMES,
                "guard_samples": 0, "device": str(device), "training_status": training_status, "warmup_ms": warmup_ms,
                "scope": "model service; excludes capture, tracking, network and playback"}

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        nonlocal active_sessions
        if not _same_origin(ws) or active_sessions >= max_sessions:
            await ws.close(code=1008)
            return
        active_sessions += 1
        try:
            await ws.accept()
            session = StreamingSession.from_model(model, training_status=training_status)
            await ws.send_json({"type": "ready", "format": model_format, "chunk_samples": SAMPLES, "video_frames": FRAMES,
                                "guard_samples": 0, "sample_rate": SR, "training_status": training_status, "warmup_ms": warmup_ms})
            while True:
                    raw = await ws.receive_text()
                    received = time.perf_counter()
                    try:
                        if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                            raise ValueError("message too large")
                        data = json.loads(raw)
                        if isinstance(data, dict) and data.get("type") == "reset":
                            session.reset()
                            await ws.send_json({"type": "reset", "next_sequence": 0, "next_sample": 0})
                            continue
                        audio, mouth, sequence, start, video_times, valid_samples = decode_chunk(data)
                        decoded = time.perf_counter()
                        async with lock:
                            acquired = time.perf_counter()
                            # Concurrent connections do not block the event loop while
                            # the model runs. The lock protects the shared model.
                            output = await asyncio.to_thread(session.push, audio, mouth,
                                                             sequence=sequence, start_sample=start,
                                                             video_samples=video_times, valid_samples=valid_samples)
                            inferred = time.perf_counter()
                        out64 = base64.b64encode(output.astype("<f4", copy=False).tobytes()).decode("ascii")
                        encoded = time.perf_counter()
                        await ws.send_json({"type": "result", "sequence": sequence,
                                            "start_sample": start, "end_sample": start + valid_samples,
                                            "valid_samples": valid_samples,
                                            "audio_f32le": out64, "guard_samples": 0,
                                            "timing_ms": {"decode": (decoded-received)*1000,
                                                          "queue": (acquired-decoded)*1000,
                                                          "session": (inferred-acquired)*1000,
                                                          "encode": (encoded-inferred)*1000,
                                                          "server_before_send": (encoded-received)*1000}})
                    except NonFiniteOutputError:
                        # push checks output before committing any session state.
                        logger.warning("non-finite streaming output at sequence %s, sample %s",
                                       session.next_sequence, session.next_sample)
                        await ws.send_json({"type": "error", "message": "non-finite model output",
                                            "expected_sequence": session.next_sequence,
                                            "expected_start_sample": session.next_sample})
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                        await ws.send_json({"type": "error", "message": str(exc),
                                            "expected_sequence": session.next_sequence,
                                            "expected_start_sample": session.next_sample})
        except WebSocketDisconnect:
            pass
        finally:
            active_sessions -= 1

    # The worker references its MediaPipe model under /static/vendor.
    app.mount("/stream-ui", StaticFiles(directory=repo_root / "tools/streaming/web", html=True, check_dir=False), name="stream-ui")
    app.mount("/studio-assets", StaticFiles(directory=repo_root / "studio/web"), name="studio-assets")
    app.mount("/static/vendor/mediapipe", StaticFiles(directory=repo_root / "studio/web/vendor/mediapipe"), name="mediapipe")
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cpu-threads", type=int, default=4)
    a = p.parse_args()
    if a.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    torch.set_num_threads(a.cpu_threads)
    seed = StreamingSession(a.checkpoint, device=a.device)
    warmup_started = time.perf_counter()
    zero_audio = np.zeros(SAMPLES, dtype=np.float32)
    zero_mouth = np.zeros((FRAMES, 96, 96), dtype=np.uint8)
    for i in range(3):
        start = i * SAMPLES
        seed.push(zero_audio, zero_mouth, sequence=i, start_sample=start,
                  video_samples=[start + n * 640 for n in range(FRAMES)])
    warmup_ms = (time.perf_counter() - warmup_started) * 1000
    seed.reset()
    print(f"warmup_ms={warmup_ms:.1f}; warmup state discarded", flush=True)
    app = create_app(seed.model, seed.training_status, a.device, warmup_ms)
    uvicorn.run(app, host="127.0.0.1", port=a.port, ws_max_size=MAX_MESSAGE_BYTES,
                log_level="info", access_log=False)


if __name__ == "__main__":
    main()
