"""Check offline and live inference agree at a 120 ms partial-tail boundary."""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

from look2hear.models.FalconStreaming import FalconStreaming
from tools.streaming.common import FRAMES, SAMPLES, SAMPLES_PER_FRAME, SR
from tools.streaming.session import StreamingSession


def test_offline_inference_matches_live_session_with_partial_final_chunk(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(712)
    rng = np.random.default_rng(713)
    model = FalconStreaming(channels=8, dilations=(1, 2)).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({"format": "falcon-streaming-v4", "model_config": {"channels": 8, "dilations": [1, 2]},
                "state_dict": model.state_dict()}, checkpoint)
    audio = rng.normal(0, 0.1, SAMPLES + 281).astype(np.float32)
    mouth = rng.integers(0, 256, (FRAMES + 1, 96, 96), dtype=np.uint8)
    input_path, mouth_path, output_path = tmp_path / "input.wav", tmp_path / "mouth.pt", tmp_path / "output.wav"
    sf.write(input_path, audio, SR, subtype="FLOAT")
    torch.save(torch.from_numpy(mouth), mouth_path)
    process = subprocess.run(
        [sys.executable, "-m", "tools.streaming.infer", "--checkpoint", str(checkpoint),
         "--audio", str(input_path), "--mouth", str(mouth_path), "--output", str(output_path), "--device", "cpu"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, process.stderr
    offline, rate = sf.read(output_path, dtype="float32")
    assert rate == SR and len(offline) == len(audio)
    session = StreamingSession(checkpoint)
    first = session.push(audio[:SAMPLES], mouth[:FRAMES], sequence=0, start_sample=0,
                         video_samples=[i * SAMPLES_PER_FRAME for i in range(FRAMES)])
    valid = len(audio) - SAMPLES
    tail_audio = np.pad(audio[SAMPLES:], (0, SAMPLES - valid))
    tail_mouth = np.repeat(mouth[FRAMES:FRAMES + 1], FRAMES, axis=0)
    second = session.push(tail_audio, tail_mouth, sequence=1, start_sample=SAMPLES,
                          video_samples=[SAMPLES + i * SAMPLES_PER_FRAME for i in range(FRAMES)], valid_samples=valid)
    np.testing.assert_allclose(offline, np.concatenate([first, second[:valid]]), atol=1e-7, rtol=1e-6)
