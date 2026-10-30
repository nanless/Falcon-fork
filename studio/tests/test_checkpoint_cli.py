"""Checkpoint overrides: CLI parsing, /api/load, and /api/status filename + SHA-256.

These tests never download weights: Engine loads are served by a fake FalconRuntime.
"""
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import falcon_runtime
import server
from server import Engine, checkpoint_name, normalize_checkpoint, parse_args


def write_checkpoint(tmp_path: Path, name: str = "custom-falcon.pt") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake checkpoint bytes for sha256\n")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_runtime(monkeypatch):
    """Replace FalconRuntime so Engine loads are instant and never touch the network."""
    calls = []

    class FakeRuntime:
        def __init__(self, variant="small", checkpoint_path=None):
            from falcon_runtime import VARIANTS
            self.variant = variant
            self.checkpoint_path = checkpoint_path
            self.device = "cpu"
            calls.append((variant, str(checkpoint_path) if checkpoint_path else None))

        def load(self, device="auto"):
            self.device = device
            return self

        def status(self):
            from falcon_runtime import VARIANTS, file_sha256
            if self.checkpoint_path:
                spec = str(self.checkpoint_path)
                name = spec.rpartition("/")[2] if spec.startswith("hf://") else Path(spec).name
                digest = None if spec.startswith("hf://") else file_sha256(spec)
            else:
                name = VARIANTS[self.variant]["weights"]
                digest = None
            return {"ready": True, "variant": self.variant,
                    "architecture": VARIANTS[self.variant]["architecture"], "device": self.device,
                    "requested_device": self.device, "checkpoint": name, "checkpoint_sha256": digest,
                    "sample_rate": 16000, "video_fps": 25, "window_seconds": 2.0,
                    "audio_weight_keys": 0, "video_weight_keys": 0, "parameters": 0,
                    "last_inference_ms": None, "warmup_ms": 0.0, "error": None, "warning": None,
                    "mode": "windowed_audio_visual"}

        def infer(self, audio, frames):
            return np.zeros(32000, np.float32)

    monkeypatch.setattr(falcon_runtime, "FalconRuntime", FakeRuntime)
    return calls


def test_parser_accepts_local_and_hf_checkpoint(tmp_path):
    local = write_checkpoint(tmp_path)
    args = parse_args(["--checkpoint", str(local), "--no-browser", "--port", "7999"])
    assert args.variant == "small"
    assert args.checkpoint == str(local.resolve())

    args = parse_args(["--checkpoint", "hf://someone/Falcon-Custom/model.pt"])
    assert args.checkpoint == "hf://someone/Falcon-Custom/model.pt"
    assert parse_args([]).checkpoint is None


@pytest.mark.parametrize("spec", ["missing.pt", "hf://model.pt"])
def test_parser_rejects_missing_local_or_malformed_hf(tmp_path, monkeypatch, spec):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", spec])


def test_normalize_and_display_name(tmp_path):
    local = write_checkpoint(tmp_path)
    assert normalize_checkpoint(str(local)) == str(local.resolve())
    assert normalize_checkpoint(None) is None
    assert checkpoint_name(str(local), "small") == local.name
    assert checkpoint_name("hf://org/repo/model.pt", "small") == "model.pt"
    assert checkpoint_name(None, "big") == "falcon-big.pt"
    for bad in (5, "", "hf://only-file.pt", str(tmp_path / "absent.pt")):
        with pytest.raises(ValueError):
            normalize_checkpoint(bad)


def test_status_reports_local_checkpoint_filename_and_sha(tmp_path, fake_runtime):
    local = write_checkpoint(tmp_path)
    engine = Engine(device="cpu", variant="small", checkpoint=str(local))

    status = engine.status()  # before loading
    assert status["checkpoint"] == local.name
    assert status["checkpoint_sha256"] == sha256(local)

    engine.start_load()
    engine.load_thread.join(5)
    assert not engine.load_thread.is_alive()
    assert fake_runtime[-1] == ("small", str(local))
    status = engine.status()
    assert status["checkpoint"] == local.name
    assert status["checkpoint_sha256"] == sha256(local)


def test_api_load_accepts_checkpoint_and_status_echoes_it(tmp_path, fake_runtime):
    local = write_checkpoint(tmp_path)
    engine = Engine(device="cpu", variant="small")
    with TestClient(server.create_app(engine, autoload=False)) as client:
        response = client.post("/api/load", json={"variant": "small", "checkpoint": str(local)})
        assert response.status_code == 200, response.text
        engine.load_thread.join(5)
        assert not engine.load_thread.is_alive()
        status = client.get("/api/status").json()
    assert status["checkpoint"] == local.name
    assert status["checkpoint_sha256"] == sha256(local)
    assert fake_runtime[-1] == ("small", str(local))


def test_api_load_supports_hf_checkpoint_without_download(tmp_path, fake_runtime):
    engine = Engine(device="cpu", variant="small")
    with TestClient(server.create_app(engine, autoload=False)) as client:
        response = client.post("/api/load", json={"variant": "small", "checkpoint": "hf://org/repo/model.pt"})
        assert response.status_code == 200, response.text
        engine.load_thread.join(5)
        status = client.get("/api/status").json()
    assert status["checkpoint"] == "model.pt"
    assert status["checkpoint_sha256"] is None
    assert fake_runtime[-1] == ("small", "hf://org/repo/model.pt")


@pytest.mark.parametrize("payload", [{"checkpoint": 5}, {"checkpoint": ""}, {"checkpoint": "hf://model.pt"}])
def test_api_load_rejects_bad_checkpoint(payload, fake_runtime):
    engine = Engine(device="cpu", variant="small")
    with TestClient(server.create_app(engine, autoload=False)) as client:
        response = client.post("/api/load", json=payload)
    assert response.status_code == 400, response.text
    assert engine.runtime is None and not engine.loading


def test_switching_checkpoint_reloads_and_switching_variant_resets(tmp_path, fake_runtime):
    first = write_checkpoint(tmp_path, "first.pt")
    second = write_checkpoint(tmp_path, "second.pt")
    engine = Engine(device="cpu", variant="small", checkpoint=str(first))
    engine.start_load()
    engine.load_thread.join(5)
    engine.start_load()  # same variant and same checkpoint: no-op
    assert len(fake_runtime) == 1
    engine.start_load(checkpoint=str(second))
    engine.load_thread.join(5)
    assert len(fake_runtime) == 2 and engine.checkpoint == str(second.resolve())
    engine.start_load("big")  # variant switch drops the small-only checkpoint
    engine.load_thread.join(5)
    assert engine.checkpoint is None
    assert fake_runtime[-1] == ("big", None)


def test_main_wires_cli_checkpoint_into_engine(tmp_path, monkeypatch):
    local = write_checkpoint(tmp_path)
    captured = {}
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kwargs: captured.update(app=app)))
    monkeypatch.setattr(sys, "argv", ["server.py", "--checkpoint", str(local), "--no-browser", "--port", "7999"])
    server.main()
    engine = captured["app"].state.engine
    assert engine.checkpoint == str(local.resolve())
    assert engine.status()["checkpoint"] == local.name


def test_file_sha256_matches_hashlib(tmp_path):
    local = write_checkpoint(tmp_path)
    assert falcon_runtime.file_sha256(local) == sha256(local)
    assert falcon_runtime.file_sha256(tmp_path / "absent.pt") is None
