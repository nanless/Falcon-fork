"""Actual WebSocket service messages, model state and clock validation."""
import base64
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from look2hear.models.FalconStreaming import FalconStreaming
from tools.streaming.server import create_app


def payload(sequence=0, video_samples=None):
    start = 2560 * sequence
    a = np.sin(np.arange(2560, dtype=np.float32) * 0.03) * 0.2
    v = np.random.default_rng(77).integers(0, 256, (4,96,96), dtype=np.uint8)
    enc = lambda raw: base64.b64encode(raw).decode("ascii")
    return {"type": "chunk", "sequence": sequence, "start_sample": start,
            "video_samples": video_samples if video_samples is not None else [start + k * 640 for k in range(4)],
            "audio_f32le": enc(a.astype("<f4").tobytes()), "mouth_gray_u8": enc(v.tobytes())}


@pytest.fixture(scope="module")
def client():
    torch.set_num_threads(2)
    torch.manual_seed(32)
    model = FalconStreaming().eval()
    with TestClient(create_app(model, "random_untrained"), base_url="http://127.0.0.1:8765") as c:
        yield c


def test_websocket_reset_reconnect_isolation(client):
    with client.websocket_connect("ws://127.0.0.1:8765/ws") as a, client.websocket_connect("ws://127.0.0.1:8765/ws") as b:
        assert a.receive_json()["format"] == "falcon-streaming-v3"
        assert b.receive_json()["format"] == "falcon-streaming-v3"
        a.send_json(payload()); r0 = a.receive_json()
        b.send_json(payload()); r1 = b.receive_json()
        assert r0["type"] == "result" and r0["audio_f32le"] == r1["audio_f32le"]
        a.send_json(payload(1)); assert a.receive_json()["sequence"] == 1
        b.send_json(payload(1)); assert b.receive_json()["sequence"] == 1
        a.send_json({"type":"reset"}); assert a.receive_json()["next_sequence"] == 0
        a.send_json(payload()); assert a.receive_json()["audio_f32le"] == r0["audio_f32le"]
    with client.websocket_connect("ws://127.0.0.1:8765/ws") as c:
        assert c.receive_json()["type"] == "ready"
        c.send_json(payload()); assert c.receive_json()["audio_f32le"] == r0["audio_f32le"]


def test_bad_input_cannot_advance_state(client):
    with client.websocket_connect("ws://127.0.0.1:8765/ws") as ws:
        ws.receive_json()
        bad = payload(); bad["audio_f32le"] = "a"; ws.send_json(bad)
        assert ws.receive_json()["expected_sequence"] == 0
        ws.send_text("{" + " " * 100001); assert ws.receive_json()["type"] == "error"
        ws.send_json(payload(1)); assert ws.receive_json()["expected_sequence"] == 0
        ws.send_json(payload(video_samples=[1,640,1280,1920])); assert ws.receive_json()["type"] == "error"
        ws.send_json(payload(video_samples=[-99999,640,1280,1920])); assert ws.receive_json()["type"] == "error"
        ws.send_json(payload()); assert ws.receive_json()["type"] == "result"


def test_websocket_rejects_cross_origin(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("ws://127.0.0.1:8765/ws", headers={"origin":"http://attacker.example"}):
            pass


def test_connection_limit_is_released(client):
    from contextlib import ExitStack
    from starlette.websockets import WebSocketDisconnect
    with ExitStack() as stack:
        for _ in range(8):
            ws = stack.enter_context(client.websocket_connect("ws://127.0.0.1:8765/ws"))
            assert ws.receive_json()["type"] == "ready"
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("ws://127.0.0.1:8765/ws"):
                pass
    with client.websocket_connect("ws://127.0.0.1:8765/ws") as ws:
        assert ws.receive_json()["type"] == "ready"


def test_browser_static_resources_and_security_headers(client):
    response = client.get("/studio-assets/video-worker.js")
    assert response.status_code == 200
    assert "wasm-unsafe-eval" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/vendor/mediapipe/face_landmarker.task").status_code == 200
    assert client.get("/", follow_redirects=False).headers["location"] == "/stream-ui/"


def _image64(size=(96,96), kind="JPEG"):
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(2026).integers(0,256,size,dtype=np.uint8)).save(buffer,format=kind)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_jpeg_input_matches_equivalent_raw_gray(client):
    from tools.streaming.server import decode_chunk
    request = payload()
    del request["mouth_gray_u8"]
    request["mouth_jpeg"] = [_image64()]*4
    _, decoded, _, _, _, _ = decode_chunk(request)
    raw = payload()
    raw["mouth_gray_u8"] = base64.b64encode(decoded.tobytes()).decode("ascii")
    with client.websocket_connect("ws://127.0.0.1:8765/ws") as a, client.websocket_connect("ws://127.0.0.1:8765/ws") as b:
        a.receive_json(); b.receive_json()
        a.send_json(request); x = a.receive_json()
        b.send_json(raw); y = b.receive_json()
        assert x["type"] == y["type"] == "result"
        assert x["audio_f32le"] == y["audio_f32le"]


def test_jpeg_rejects_bad_format_size_ambiguity_and_recovers(client):
    good = payload(); del good["mouth_gray_u8"]
    good["mouth_jpeg"] = [_image64()]*4
    wrong_length = dict(good, mouth_jpeg=good["mouth_jpeg"][:3])
    wrong_dimension = dict(good, mouth_jpeg=[_image64((97,96))]*4)
    png = dict(good, mouth_jpeg=[_image64(kind="PNG")]*4)
    broken = dict(good, mouth_jpeg=[base64.b64encode(b"bad image").decode()]*4)
    invalid_b64 = dict(good, mouth_jpeg=["@@@@"]*4)
    oversized = dict(good, mouth_jpeg=["A"*22000]*4)
    ambiguous = dict(good, mouth_gray_u8=payload()["mouth_gray_u8"])
    with client.websocket_connect("ws://127.0.0.1:8765/ws") as ws:
        ws.receive_json()
        for bad in (wrong_length,wrong_dimension,png,broken,invalid_b64,oversized,ambiguous):
            ws.send_json(bad); result=ws.receive_json()
            assert result["type"] == "error" and result["expected_sequence"] == 0
        ws.send_json(good); assert ws.receive_json()["type"] == "result"


def test_status_uses_current_checkpoint_format(client):
    from tools.streaming.common import CHECKPOINT_FORMAT
    response = client.get('/status')
    assert response.status_code == 200
    assert response.json()['format'] == CHECKPOINT_FORMAT == 'falcon-streaming-v3'


def test_partial_final_protocol_enforces_zero_padding_and_requires_reset(client):
    req=payload()
    audio=np.zeros(2560,dtype='<f4');audio[:20]=0.1
    req['audio_f32le']=base64.b64encode(audio.tobytes()).decode('ascii')
    req['video_samples']=[-2000]*4
    req['valid_samples']=20
    with client.websocket_connect('ws://127.0.0.1:8765/ws') as ws:
        ws.receive_json()
        invalid=dict(req); bad=audio.copy();bad[20]=.2
        invalid['audio_f32le']=base64.b64encode(bad.tobytes()).decode('ascii')
        ws.send_json(invalid); error=ws.receive_json()
        assert error['type']=='error' and error['expected_sequence']==0
        ws.send_json(req); result=ws.receive_json()
        assert result['type']=='result' and result['valid_samples']==20 and result['end_sample']==20
        assert len(base64.b64decode(result['audio_f32le']))==2560*4
        again=payload(1);again['start_sample']=20
        ws.send_json(again); assert ws.receive_json()['type']=='error'
        ws.send_json({'type':'reset'});assert ws.receive_json()['next_sample']==0
        ws.send_json(payload());assert ws.receive_json()['type']=='result'
