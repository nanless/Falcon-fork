"""Studio backend verification (studio-backend / task-6).

Covers, without downloading any Falcon weights:
  A. falcon_runtime: constants, VARIANTS, weight resolution, pick_device,
     preprocess_mouth numerics, model construction / parameter counts,
     state_dict key+shape equality vs the training code (look2hear),
     a synthetic checkpoint round-trip through FalconRuntime for BOTH variants,
     bit-identical forward vs the training-code model on CPU,
     strict-loading failure modes and infer() input validation.
  B. server: decode_request wire contract, OutputSlicer offsets/crossfade,
     same_origin truth table, Engine state/status merge, /api/status,
     /api/load, /ws result/error schemas and limits.

Run (repo root):
  PYTHONPATH=$PWD/docs/verification/scripts/pil_stub:$PWD/docs/verification/stubs:$PWD \
    /root/.venvs/falcon/bin/python docs/verification/scripts/verify_studio_backend.py

docs/verification/scripts/pil_stub is a minimal Pillow stand-in created only
because Pillow is absent from the read-only verification venv (true image
decoding is NOT tested; see the doc's §7/§8). Without it part B cannot import
server.py. Drop it from PYTHONPATH once Pillow is installed.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
STUDIO = REPO / "studio"
sys.path.insert(0, str(STUDIO))

import falcon_runtime as R  # noqa: E402
from falcon_model.models.Falcon_Contrastive import FalconSpeakerContrastive  # noqa: E402
from falcon_model.models.Speaker_Contrastive_v2 import SpeakerContrastiveV2  # noqa: E402
from falcon_model.videomodels.frcnn_videomodel import FRCNNVideoModel  # noqa: E402

SECTION = lambda s: print(f"\n=== {s} ===")  # noqa: E731


def build_studio(variant: str):
    """Mirror FalconRuntime.load() model construction (no checkpoint)."""
    cfg = yaml.safe_load((STUDIO / "configs" / R.VARIANTS[variant]["config"]).read_text())
    if "train_conf" in cfg:
        cfg = cfg["train_conf"]
    audio_cfg = copy.deepcopy(cfg["audionet"]["audionet_config"])
    video_cfg = copy.deepcopy(cfg["videonet"]["videonet_config"])
    video_cfg["pretrain"] = None
    arch = cfg["audionet"]["audionet_name"]
    assert arch == R.VARIANTS[variant]["architecture"], (arch, variant)
    audio = R.MODEL_CLASSES[arch](sample_rate=R.SAMPLE_RATE, **audio_cfg)
    video = FRCNNVideoModel(**video_cfg)
    return audio, video, cfg


def make_ckpt(audio, video, path: Path, epoch: int = 714):
    state = {"audio_model." + k: v for k, v in audio.state_dict().items()}
    state.update({"video_model." + k: v for k, v in video.state_dict().items()})
    torch.save({"state_dict": state, "model_name": type(audio).__name__, "epoch": epoch}, path)
    return path


def count(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- A. runtime
SECTION("A1 constants / variants / weight resolution")
print("SAMPLE_RATE", R.SAMPLE_RATE, "VIDEO_FPS", R.VIDEO_FPS,
      "WINDOW_SAMPLES", R.WINDOW_SAMPLES, "WINDOW_FRAMES", R.WINDOW_FRAMES,
      "-> 2s*16k =", 2 * R.SAMPLE_RATE, "2s*25fps =", 2 * R.VIDEO_FPS)
print("VARIANTS", json.dumps(R.VARIANTS, sort_keys=True))
print("variant_for('JusperLee/Falcon')", R.variant_for("JusperLee/Falcon"),
      "| ('JusperLee/Falcon-Big')", R.variant_for("JusperLee/Falcon-Big"),
      "| ('big')", R.variant_for("big"))
try:
    R.variant_for("someone/Other")
except ValueError as exc:
    print("variant_for bad -> ValueError:", exc)
print("weights_source small/big (cache-only, no network):", R.weights_source("small"), R.weights_source("big"))
print("resolve_checkpoint('small', Path('/tmp/x.pt')) ->", R.resolve_checkpoint("small", Path("/tmp/x.pt")))
try:
    R.resolve_checkpoint("small", "hf://model.pt")
except ValueError as exc:
    print("bad hf:// spec -> ValueError:", exc)
print("pick_device('auto') on this host:", R.pick_device("auto"),
      "| explicit cpu:", R.pick_device("cpu"))
for bad in ("tpu", "gpu", None):
    try:
        R.pick_device(bad)
    except ValueError as exc:
        print(f"pick_device({bad!r}) -> ValueError: {exc}")

SECTION("A2 model construction, parameter counts, config mutation")
studio_models = {}
for variant in ("small", "big"):
    audio, video, cfg = build_studio(variant)
    studio_models[variant] = (audio, video)
    print(f"{variant}: audio={count(audio):,} video={count(video):,} total={count(audio)+count(video):,} "
          f"({(count(audio)+count(video))/1e6:.3f} M)  arch={type(audio).__name__}")
    print(f"   original config still has repeats/shared after build="
          f"{'repeats' in cfg['audionet']['audionet_config']['audio_params']}/"
          f"{'shared' in cfg['audionet']['audionet_config']['audio_params']} "
          f"(deepcopy kept Separator.dict.pop from mutating the source)")
    print(f"   video_bottleneck params={count(audio.video_bottleneck)} out_chan={audio.video_bottleneck.out_chan} "
          f"conv={type(audio.video_bottleneck.conv).__name__}")

small_audio, small_video = studio_models["small"]
big_audio, big_video = studio_models["big"]
s_keys, b_keys = set(small_audio.state_dict()), set(big_audio.state_dict())
print("small vs big audio state_dict keys equal:", s_keys == b_keys, "| n=", len(s_keys))
same_shape = [k for k in s_keys & b_keys if small_audio.state_dict()[k].shape != big_audio.state_dict()[k].shape]
print("keys with different shapes small->big:", len(same_shape), "e.g.", same_shape[:4])

SECTION("A3 state_dict equality vs training code (look2hear)")
from look2hear.models.Falcon_Contrastive import FalconSpeakerContrastive as TrainFSC  # noqa: E402
from look2hear.videomodels.frcnn_videomodel import FRCNNVideoModel as TrainFRCNN  # noqa: E402

train_cfg = yaml.safe_load((REPO / "configs" / "Falcon.yml").read_text())
train_audio_cfg = copy.deepcopy(train_cfg["audionet"]["audionet_config"])
train_video_cfg = copy.deepcopy(train_cfg["videonet"]["videonet_config"])
train_video_cfg.pop("pretrain", None)
train_audio = TrainFSC(sample_rate=16000, **train_audio_cfg)
train_video = TrainFRCNN(**train_video_cfg)


def describe_keys(a, b, name_a="studio", name_b="training"):
    ka, kb = set(a.state_dict()), set(b.state_dict())
    same = ka == kb
    shapes = all(a.state_dict()[k].shape == b.state_dict()[k].shape for k in ka & kb) if same else None
    print(f"{name_a} vs {name_b}: keys_equal={same} shapes_equal={shapes} "
          f"n_{name_a}={len(ka)} n_{name_b}={len(kb)} only_{name_a}={sorted(ka-kb)[:3]} only_{name_b}={sorted(kb-ka)[:3]}")
    return same and shapes


describe_keys(small_audio, train_audio)
describe_keys(small_video, train_video)
print("=> training ckpt 'audio_model.*' keys load into the Studio copy with strict=True:", describe_keys(small_audio, train_audio))

SECTION("A4 preprocess_mouth numerics")
frames = np.full((50, 96, 96), 107, np.uint8)
x = R.preprocess_mouth(frames)
print("shape/dtype:", tuple(x.shape), x.dtype,
      "| value(gray=107):", float(x[0, 0, 0, 0, 0]),
      "| formula (107/255-.421)/.165 =", (107 / 255 - 0.421) / 0.165)
print("range for gray 0 / 255:", float(R.preprocess_mouth(np.zeros((50, 96, 96), np.uint8))[0, 0, 0, 0, 0]),
      "/", float(R.preprocess_mouth(np.full((50, 96, 96), 255, np.uint8))[0, 0, 0, 0, 0]),
      "-> NOT [-1,1]; it is z-score over [0,1] pixels")
border = np.zeros((50, 96, 96), np.uint8)
border[:, :4, :] = border[:, 92:, :] = 255
print("ignoring outer 4 px (96->88 CenterCrop) max|z|:", float(R.preprocess_mouth(border).abs().max()))
for bad in (np.zeros((49, 96, 96), np.uint8), np.zeros((50, 88, 88), np.uint8), np.zeros((50, 96, 96), np.float32)):
    try:
        R.preprocess_mouth(bad)
    except ValueError as exc:
        print("  reject:", exc)

SECTION("A5 synthetic-checkpoint round trip through FalconRuntime (CPU)")
tmp = Path(tempfile.mkdtemp(prefix="falcon-studio-verify-"))
results = {}
for variant in ("small", "big"):
    audio, video = studio_models[variant]
    ckpt = make_ckpt(audio, video, tmp / f"fake-{variant}.pt")
    t0 = time.perf_counter()
    rt = R.FalconRuntime(variant, ckpt).load("cpu")
    load_s = time.perf_counter() - t0
    st = rt.status()
    print(f"{variant}: ready={st['ready']} device={st['device']} arch={st['architecture']} "
          f"audio_keys={st['audio_weight_keys']} video_keys={st['video_weight_keys']} "
          f"parameters={st['parameters']:,} warmup_ms={st['warmup_ms']} load_s={load_s:.2f}")
    audio_in = np.random.default_rng(7).standard_normal(R.WINDOW_SAMPLES).astype(np.float32) * 0.05
    frames_in = np.random.default_rng(8).integers(0, 256, (R.WINDOW_FRAMES, 96, 96), dtype=np.uint8)
    out = rt.infer(audio_in, frames_in)
    print(f"   infer -> shape={out.shape} dtype={out.dtype} finite={bool(np.isfinite(out).all())} "
          f"last_inference_ms={rt.last_inference_ms} range=[{out.min():.3f},{out.max():.3f}]")
    results[variant] = (rt, audio_in, frames_in, out)

SECTION("A6 Studio runtime vs training-code forward (same synthetic weights, CPU)")
for variant in ("small", "big"):
    rt, audio_in, frames_in, out_studio = results[variant]
    train_model = TrainFSC(sample_rate=16000, **copy.deepcopy(train_cfg["audionet"]["audionet_config"])) \
        if variant == "small" else None
    if train_model is None:
        # The training repo at this commit has no SpeakerContrastiveV2; compare against
        # a separately built Studio model loaded from the same state_dict instead.
        ref = studio_models["big"][0]
        ref.load_state_dict({k: v for k, v in rt.audio_model.state_dict().items()})
    else:
        ref = train_model
        ref.load_state_dict({k: v for k, v in rt.audio_model.state_dict().items()})
        train_video_ref = TrainFRCNN(**copy.deepcopy(train_video_cfg))
        train_video_ref.load_state_dict({k: v for k, v in rt.video_model.state_dict().items()})
        # FRCNNVideoModel overrides train() without returning self, so eval() returns None
        print("  FRCNNVideoModel.eval() returns:", repr(train_video_ref.eval()),
              "(falcon_runtime.py:178 comment verified)")
        train_video_ref.requires_grad_(False)
    ref.eval().requires_grad_(False)
    with torch.inference_mode():
        mouth = R.preprocess_mouth(frames_in)
        wav = torch.from_numpy(audio_in).unsqueeze(0)
        if variant == "small":
            embed = train_video_ref(mouth)
        else:
            embed = rt.video_model(mouth)
        sep = ref(wav, embed)
        out_ref = sep[0, 0].float().cpu().numpy()
    print(f"{variant}: max|studio-training| = {np.abs(out_studio - out_ref).max():.3e} "
          f"(bit-identical={np.array_equal(out_studio, out_ref)})"
          f"{'' if variant == 'small' else '  [training class absent -> self-consistency only]'}")

SECTION("A7 strict loading and error paths")
audio, video = studio_models["small"]
good = tmp / "good.pt"
make_ckpt(audio, video, good)
strict_ckpt = tmp / "strict.pt"
state = torch.load(good, weights_only=True)["state_dict"]
bad_state = dict(state)
first_audio = next(k for k in bad_state if k.startswith("audio_model."))
bad_state[first_audio + "_renamed"] = bad_state.pop(first_audio)
torch.save({"state_dict": bad_state}, strict_ckpt)
for name, ckpt, variant in [
    ("audio keys only", tmp / "audio_only.pt", "small"),
    ("renamed audio key", strict_ckpt, "small"),
    ("small ckpt with big config", good, "big"),
    ("missing file", tmp / "nope.pt", "small"),
]:
    if name == "audio keys only":
        only = {k: v for k, v in state.items() if k.startswith("audio_model.")}
        torch.save({"state_dict": only}, ckpt)
    try:
        R.FalconRuntime(variant, ckpt).load("cpu")
        print(f"{name}: NOT REJECTED (unexpected)")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
plain = tmp / "plain_state_dict.pt"
torch.save(state, plain)  # no {"state_dict": ...} wrapper
rt_plain = R.FalconRuntime("small", plain).load("cpu")
print("plain prefixed state_dict (no 'state_dict' key) accepted:", rt_plain.ready, rt_plain.device)

SECTION("A8 infer() input validation")
rt = R.FalconRuntime("small")  # never loaded
for name, call in [
    ("not loaded", lambda: rt.infer(np.zeros(32000, np.float32), frames)),
    ("float64 audio", lambda: rt_plain.infer(np.zeros(32000, np.float64), frames)),
    ("wrong audio length", lambda: rt_plain.infer(np.zeros(16000, np.float32), frames)),
    ("NaN audio", lambda: rt_plain.infer(np.full(32000, np.nan, np.float32), frames)),
    ("list frames", lambda: rt_plain.infer(np.zeros(32000, np.float32), [[0]])),
]:
    try:
        call()
        print(f"{name}: NOT REJECTED")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: {type(exc).__name__}: {str(exc)[:100]}")

# ---------------------------------------------------------------- B. protocol
SECTION("B0 import server (needs Pillow; stub on PYTHONPATH)")
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

print("server constants: SR,FPS,WINDOW,HOP,GUARD,FADE,MAX_MESSAGE_BYTES,MAX_SAFE_INTEGER =",
      server.SR, server.FPS, server.WINDOW, server.HOP, server.GUARD, server.FADE,
      server.MAX_MESSAGE_BYTES, server.MAX_SAFE_INTEGER)
print("absolute chunk/right-context: start=WINDOW-GUARD-HOP=", server.WINDOW - server.GUARD - server.HOP,
      "end=WINDOW-GUARD=", server.WINDOW - server.GUARD,
      "| output samples=", server.HOP, f"({server.HOP/server.SR*1000:.0f} ms)",
      "| guard=", server.GUARD, f"({server.GUARD/server.SR*1000:.0f} ms)",
      "| fade=", server.FADE, f"({server.FADE/server.SR*1000:.0f} ms)")


def payload(seed=1, **over):
    image = io.BytesIO()
    Image.new("L", (96, 96), 107).save(image, format="JPEG")
    frame = base64.b64encode(image.getvalue()).decode()
    data = {"type": "infer", "id": 1, "audio": base64.b64encode(np.zeros(server.WINDOW, "<f4").tobytes()).decode(),
            "frames": [frame] * 50, "end_sample": server.WINDOW, "reset": True,
            "window_seconds": 2, "hop_seconds": 0.5}
    data.update(over)
    return data


audio, frames, endpoint = server.decode_request(payload())
print("decode_request valid ->", audio.shape, audio.dtype, frames.shape, frames.dtype, "endpoint", endpoint)
print("default end_sample ->", server.decode_request({k: v for k, v in payload().items() if k != "end_sample"})[2])
print("wire sizes: audio b64 chars", len(payload()["audio"]), "(expect 4*((32000*4+2)//3) =",
      4 * ((server.WINDOW * 4 + 2) // 3), ") | per-frame cap b64 chars", 40000,
      "| -> max frames in 2MB msg ~", (server.MAX_MESSAGE_BYTES - len(payload()["audio"])) // 40000,
      "but frame cap allows", (server.MAX_MESSAGE_BYTES - len(payload()["audio"])) // 50, "chars/frame")
invalid = {
    "type": {"type": "nope"}, "window_seconds": {"window_seconds": 3},
    "hop_seconds": {"hop_seconds": 1}, "reset not bool": {"reset": 1},
    "audio not str": {"audio": 5}, "audio bad length": {"audio": "AAAA"},
    "frames not list": {"frames": "x"}, "49 frames": {"frames": [payload()["frames"][0]] * 49},
    "frame too long": {"frames": ["A" * 40001] * 50},
    "end_sample bool": {"end_sample": True}, "end_sample -1": {"end_sample": -1},
    "end_sample float": {"end_sample": 32000.5}, "end_sample > 2^53-1": {"end_sample": 2 ** 53},
    "bad base64 frame": {"frames": ["!!!!"] * 50},
}
for name, over in invalid.items():
    try:
        server.decode_request(payload(**over))
        print(f"  {name}: NOT REJECTED")
    except ValueError as exc:
        print(f"  {name}: ValueError: {exc}")

SECTION("B1 OutputSlicer offsets / crossfade / clipping")
sl = server.OutputSlicer()
pred = np.arange(server.WINDOW, dtype=np.float32) / 100000.0
chunk, contiguous, clipped = sl.take(pred, server.WINDOW, reset=True)
print("first take: n=", chunk.size, "contiguous=", contiguous, "clipped=", clipped,
      "first/last=", float(chunk[0]), float(chunk[-1]),
      "expected absolute chunk starts at", server.WINDOW - server.GUARD - server.HOP)
predec = np.arange(server.WINDOW, dtype=np.float32) / 100000.0
next_pred = np.arange(server.HOP, server.HOP + server.WINDOW, dtype=np.float32) / 100000.0
prev_future = sl.future.copy()
chunk2, contiguous2, _ = sl.take(next_pred, server.WINDOW + server.HOP)
print("second take (endpoint+HOP): contiguous=", contiguous2,
      "| crossfade samples=", server.FADE, "first sample=", float(chunk2[0]),
      "future(new chunk start after fade)=", float(chunk2[server.FADE]))
joined = np.r_[chunk, chunk2]
expected_timeline = np.arange(server.WINDOW - server.GUARD - server.HOP,
                              server.WINDOW - server.GUARD + server.HOP) / 100000.0
print("joined (2 hops) covers one continuous absolute timeline:",
      np.allclose(joined, expected_timeline, rtol=0, atol=1e-7),
      "max|diff|=", float(np.abs(joined - expected_timeline).max()))
gap_chunk, gap_contiguous, _ = sl.take(np.zeros(server.WINDOW, np.float32), server.WINDOW + 3 * server.HOP)
print("gap (endpoint jump 3*HOP, zero prediction): contiguous=", gap_contiguous,
      "non-zero samples=", int(np.count_nonzero(gap_chunk)))
loud = np.zeros(server.WINDOW, np.float32)
loud[server.WINDOW - server.GUARD - server.HOP:server.WINDOW - server.GUARD] = 2.0
clip_chunk, _, clipped = server.OutputSlicer().take(loud, server.WINDOW)
print("clipping: max=", float(clip_chunk.max()), "clipped_samples=", clipped)
for name, pred_bad in [("short pred", np.zeros(100, np.float32)), ("NaN pred", np.full(server.WINDOW, np.nan, np.float32))]:
    try:
        server.OutputSlicer().take(pred_bad, server.WINDOW)
    except ValueError as exc:
        print(f"  {name}: ValueError: {exc}")

SECTION("B2 same_origin truth table")
cases = [
    (None, "127.0.0.1:7860", True, "no Origin (CLI/tests)"),
    ("http://localhost:7860", "localhost:7860", True, "http same host:port"),
    ("http://127.0.0.1:7860", "127.0.0.1:7860", True, "http ipv4"),
    ("http://[::1]:7860", "[::1]:7860", True, "http ipv6"),
    ("https://localhost:7860", "localhost:7860", True, "https same host:port"),
    ("http://localhost:9999", "localhost:7860", False, "port mismatch"),
    ("http://127.0.0.1:7860", "localhost:7860", False, "host string mismatch (netloc)"),
    ("https://evil.example", "127.0.0.1:7860", False, "external origin"),
    ("file:///etc/passwd", "127.0.0.1:7860", False, "non-http scheme"),
    ("http://localhost.evil.example", "localhost.evil.example", False, "non-local hostname"),
]
for origin, host, expect, what in cases:
    got = server.same_origin(origin, host)
    print(f"  {str(origin):32s} host={host:22s} -> {got} (expect {expect}) [{what}]"
          f"{'' if got == expect else '  <-- MISMATCH'}")

SECTION("B3 Engine status / state machine object")


class FakeRuntime:
    def status(self):
        return {"ready": True, "device": "cpu", "architecture": "Fake", "sample_rate": 16000,
                "video_fps": 25, "window_seconds": 2.0, "parameters": 1, "warmup_ms": 1.0,
                "last_inference_ms": None, "error": None, "warning": None,
                "audio_weight_keys": 1, "video_weight_keys": 1,
                "requested_device": "cpu", "checkpoint": "fake.pt", "variant": "small",
                "mode": "windowed_audio_visual"}


eng = server.Engine()
print("fresh Engine.status():", json.dumps(eng.status(), ensure_ascii=False, sort_keys=True))
eng.runtime = FakeRuntime()
print("with ready runtime -> ready:", eng.status()["ready"], "| loading:", eng.status()["loading"])
eng.loading = True
print("runtime + loading=True -> ready:", eng.status()["ready"], "| loading:", eng.status()["loading"])
eng.loading = False
eng.error = "boom"
print("runtime + error -> status['error']:", eng.status()["error"], "ready:", eng.status()["ready"])
eng.error = None
eng.infer_lock.acquire()
try:
    eng.infer(np.zeros(32000, np.float32), np.zeros((50, 96, 96), np.uint8))
except RuntimeError as exc:
    print("infer while lock held -> RuntimeError:", exc)
finally:
    eng.infer_lock.release()
print("status keys:", sorted(server.Engine().status()))


class StubEngine(server.Engine):
    """Protocol-only engine: never downloads weights, returns a deterministic window."""
    busy = False

    def status(self):
        return {"ready": True, "device": "test", "app": "falcon-studio", "variant": "small",
                "variants": [{"id": "small", "name": "Falcon Small", "available": True,
                              "source": "cached", "checkpoint": "falcon.pt", "repo": "JusperLee/Falcon"}]}

    def start_load(self, variant=None):
        pass

    def infer(self, audio, frames):
        return np.linspace(-0.2, 0.2, server.WINDOW, dtype=np.float32), 25


SECTION("B4 HTTP endpoints / headers / limits")
client = TestClient(server.create_app(StubEngine(), autoload=False))
r = client.get("/api/status")
print("GET /api/status ->", r.status_code, json.dumps(r.json(), sort_keys=True))
print("security headers:", {k: r.headers.get(k) for k in
                            ("x-content-type-options", "referrer-policy", "permissions-policy",
                             "cache-control", "content-security-policy")})
print("POST /api/load {} ->", client.post("/api/load", json={}).status_code)
print("POST /api/load bad json ->", client.post("/api/load", content="bad").status_code)
print("POST /api/load list ->", client.post("/api/load", json=[]).status_code)
print("POST /api/load >1024B ->", client.post("/api/load", content="x" * 1025,
                                             headers={"Content-Type": "application/json"}).status_code)
print("GET /api/status with evil Origin ->", client.get("/api/status",
      headers={"Origin": "https://evil.example"}).status_code)
print("GET / ->", client.get("/").status_code, "| static index:", client.get("/static/index.html").status_code)
print("GET /static/vendor/mediapipe/vision_bundle.mjs ->",
      client.get("/static/vendor/mediapipe/vision_bundle.mjs").status_code)
wasm = client.get("/static/vendor/mediapipe/wasm/vision_wasm_module_internal.wasm")
print("wasm ->", wasm.status_code, wasm.headers.get("content-type"))

SECTION("B5 WebSocket protocol schema")
with client.websocket_connect("/ws") as ws:
    ws.send_json(payload())
    result = ws.receive_json()
    out = np.frombuffer(base64.b64decode(result["audio"]), "<f4")
    field_types = {k: type(v).__name__ for k, v in result.items()}
    print("result keys:", sorted(result))
    print("result types:", json.dumps(field_types, sort_keys=True))
    print("audio b64 chars:", len(result["audio"]), "-> decoded samples:", out.size, out.dtype,
          "| expected", server.HOP)
    print("timing: inference_ms=", result["inference_ms"], "rtf=", result["rtf"],
          "(=ms/500)", "window_rtf=", result["window_rtf"], "(=ms/2000)")
    print("timestamps: start_sample=", result["start_sample"], "end_sample=", result["end_sample"],
          "expected", server.WINDOW - server.GUARD - server.HOP, server.WINDOW - server.GUARD)
    print("flags: contiguous=", result["contiguous"], "clipped_samples=", result["clipped_samples"],
          "output_delay_ms=", result["output_delay_ms"], "device=", result["device"])
    ws.send_json(payload(id=2, reset=False, end_sample=server.WINDOW + server.HOP))
    r2 = ws.receive_json()
    print("second window contiguous=", r2["contiguous"], "start_sample=", r2["start_sample"],
          "end_sample=", r2["end_sample"])
    ws.send_text("not json")
    err = ws.receive_json()
    print("error schema:", sorted(err), "->", err["type"], "id=", err["id"], "message=", err["message"])
    bad = payload()
    bad["frames"] = bad["frames"][:49]
    ws.send_json(bad)
    print("validation error message:", ws.receive_json()["message"])
    ws.send_json(payload(id=True))
    print("bool id error:", ws.receive_json()["message"])
    ws.send_json(payload())  # session survives errors
    print("session survives errors:", ws.receive_json()["type"])

with client.websocket_connect("/ws") as ws:
    try:
        with client.websocket_connect("/ws", headers={"Origin": "https://evil.example"}):
            pass
    except Exception as exc:  # noqa: BLE001
        print("evil-origin ws rejected:", type(exc).__name__, str(exc)[:80])

SECTION("B6 max message size / PNG frames / amplitude bound / load errors")
image_png = io.BytesIO()
Image.new("L", (96, 96), 0).save(image_png, format="PNG")
png = payload(frames=[base64.b64encode(image_png.getvalue()).decode()] * 50)
print("PNG frames accepted:", server.decode_request(png)[1].shape)
amp = np.zeros(server.WINDOW, np.float32); amp[:] = 4.0
ok = payload(audio=base64.b64encode(amp.astype("<f4").tobytes()).decode())
print("|audio| = 4.0 accepted:", server.decode_request(ok)[0].max())
amp[:] = 4.0001
over = payload(audio=base64.b64encode(amp.astype("<f4").tobytes()).decode())
try:
    server.decode_request(over)
except ValueError as exc:
    print("|audio| > 4 rejected:", exc)

with client.websocket_connect("/ws") as ws:
    ws.send_text("x" * server.MAX_MESSAGE_BYTES)
    print("message == MAX_MESSAGE_BYTES (2,000,000 chars) ->", ws.receive_json()["type"], "(json error, session alive)")
    ws.send_text("x" * (server.MAX_MESSAGE_BYTES + 1))
    msg = ws.receive()
    print("message > MAX_MESSAGE_BYTES -> close frame:", msg)


class FailLoadEngine(StubEngine):
    def start_load(self, variant=None):
        raise RuntimeError("请先停止所有分离会话，再切换模型。")


with TestClient(server.create_app(FailLoadEngine(), autoload=False)) as failing:
    print("POST /api/load when start_load raises RuntimeError ->", failing.post("/api/load", json={}).status_code)
with TestClient(server.create_app(server.Engine(), autoload=False)) as real_engine:
    print("POST /api/load unknown variant (real Engine, no download) ->",
          real_engine.post("/api/load", json={"variant": "bogus"}).status_code,
          "| start_load('bogus') raises ValueError before any download")

SECTION("B7 engine error propagation + status types after load")
saved = R.FalconRuntime
class BoomRuntime:
    def __init__(self, *a, **k):
        raise RuntimeError("simulated checkpoint failure")
R.FalconRuntime = BoomRuntime
eng3 = server.Engine()
eng3._load()
R.FalconRuntime = saved
print("Engine._load catches exception -> error=", eng3.error, "| loading=", eng3.loading,
      "| status():", json.dumps(eng3.status(), ensure_ascii=False, sort_keys=True))
rt_status = results["small"][0].status()
print("FalconRuntime.status() after load:", json.dumps(rt_status, sort_keys=True))
print("FalconRuntime.status() types:", {k: type(v).__name__ for k, v in rt_status.items()})

print("\nOK: studio backend verification finished")
