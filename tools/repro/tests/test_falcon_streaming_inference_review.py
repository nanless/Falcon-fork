"""Regression checks for checkpoint identity at inference entry points."""
import os
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
import torch
from tools.streaming.session import StreamingSession


def test_v1_checkpoint_cannot_be_used_for_session_or_file_inference(tmp_path):
    checkpoint = tmp_path / 'old-v1.pt'
    torch.save({'format':'falcon-streaming-v1', 'state_dict':{}}, checkpoint)
    with pytest.raises(ValueError, match='falcon-streaming-v2.*v1'):
        StreamingSession(checkpoint)
    audio, mouth, output = tmp_path / 'mix.wav', tmp_path / 'mouth.pt', tmp_path / 'out.wav'
    sf.write(audio, np.zeros(160, np.float32), 16000)
    torch.save(torch.zeros(1,96,96,dtype=torch.uint8), mouth)
    command = [sys.executable,'-m','tools.streaming.infer','--checkpoint',str(checkpoint),
               '--audio',str(audio),'--mouth',str(mouth),'--output',str(output),'--device','cpu']
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ,'CUDA_VISIBLE_DEVICES':''}, timeout=120)
    assert result.returncode != 0
    assert 'falcon-streaming-v2' in result.stderr and 'v1' in result.stderr
    assert not output.exists()


def test_partial_final_chunk_allows_old_padded_video_only_and_closes_session():
    from look2hear.models.FalconStreaming import FalconStreaming
    torch.set_num_threads(1)
    session = StreamingSession.from_model(FalconStreaming().eval(), max_frame_age_samples=2560)
    audio=np.zeros(2560,np.float32); audio[:20]=.1
    mouth=np.zeros((4,96,96),np.uint8)
    # Frame is 125 ms old at start, valid for the first 20 ms; it is older
    # than 160 ms in padded bins, which do not describe real audio.
    times=[-2000]*4
    bad=audio.copy();bad[20]=.25
    with pytest.raises(ValueError,match='must be zero'):
        session.push(bad,mouth,sequence=0,start_sample=0,video_samples=times,valid_samples=20)
    assert session.next_sequence==0 and not session.finished
    with pytest.raises(ValueError,match='too old'):
        session.push(audio,mouth,sequence=0,start_sample=0,video_samples=[-3000]*4,valid_samples=20)
    result=session.push(audio,mouth,sequence=0,start_sample=0,video_samples=times,valid_samples=20)
    assert result.shape==(2560,) and np.isfinite(result).all()
    assert session.finished and session.next_sample==20
    with pytest.raises(ValueError,match='reset'):
        session.push(audio,mouth,sequence=1,start_sample=20,video_samples=[0]*4,valid_samples=20)
    session.reset()
    assert not session.finished and session.next_sample==0
