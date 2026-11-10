"""Read-only CPU probe; run from Falcon-fork root with PYTHONDONTWRITEBYTECODE=1."""
import copy
import yaml
import torch
from look2hear.models import FalconSpeakerContrastive

torch.set_num_threads(4)
torch.manual_seed(42)
with open('configs/Falcon.yml', encoding='utf-8') as f:
    params = yaml.safe_load(f)['audionet']['audionet_config']
ckpt = torch.load('/root/.cache/falcon_weights/falcon.pt', map_location='cpu', weights_only=False)
state = {k[len('audio_model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('audio_model.')}
print('audio keys', len(state))
x = torch.randn(2, 32000) * .1
v = torch.randn(2, 512, 50)
changed = v.clone()
changed[0] = torch.randn_like(changed[0])

def model(**opts):
    m = FalconSpeakerContrastive(**dict(copy.deepcopy(params), **opts)).eval()
    m.load_state_dict(state, strict=True)
    return m

def rms(x):
    return float(x.square().mean().sqrt())

with torch.inference_mode():
    for batch_first in (False, True):
        m = model(video_lstm_batch_first=batch_first)
        y, z = m(x, v), m(x, changed)
        solo = m(x[1:2], v[1:2])
        diff = z[1] - y[1]
        print('batch_first', batch_first, 'max', float(diff.abs().max()), 'delta_rms', rms(diff), 'reference_rms', rms(y[1]), 'solo_delta_rms', rms(solo[0] - y[1]))
    for causal in (False, True):
        m = model(paper_causal=causal)
        shapes = []
        handles = [layer.register_forward_hook(lambda module, args, output: shapes.append(tuple(output.shape))) for layer in m.separator.audio_net.downsample_layers]
        m(x[:1], v[:1])
        for h in handles: h.remove()
        print('paper_causal', causal, 'shapes', shapes)
