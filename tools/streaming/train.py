"""Train the causal model over sequential chunks using Falcon's paired JSON indexes."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
if __package__:
    from .common import checkpoint_format_for_visual_mode, SAMPLES, FRAMES, IndexedAudioVisualDataset, new_checkpoint, read_checkpoint
else:
    from common import checkpoint_format_for_visual_mode, SAMPLES, FRAMES, IndexedAudioVisualDataset, new_checkpoint, read_checkpoint
from look2hear.models.FalconStreaming import FalconStreaming

class Synthetic(Dataset):
    def __init__(self, chunks): self.chunks = chunks
    def __len__(self): return 32
    def __getitem__(self, i):
        gen = torch.Generator().manual_seed(i)
        target = torch.randn(self.chunks * SAMPLES, generator=gen) * .1
        mix = target + torch.randn(target.shape, generator=gen) * .05
        video = torch.randn(self.chunks * FRAMES, 88, 88, generator=gen)
        return mix, target, video

def neg_si_snr(output, target):
    # Whole-sequence scale-invariant SNR; labels align with emitted sample indices.
    output = output - output.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    projection = (output * target).sum(-1, keepdim=True) * target / (target.square().sum(-1, keepdim=True) + 1e-8)
    noise = output - projection
    return (-10 * torch.log10((projection.square().sum(-1) + 1e-8) / (noise.square().sum(-1) + 1e-8))).mean()

p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--index-dir", help="directory containing Falcon mix.json and target.json")
g.add_argument("--synthetic", action="store_true", help="deterministic generated inputs for smoke tests only")
p.add_argument("--output", required=True, help="new checkpoint path; existing files are never overwritten")
p.add_argument("--resume", help="resume a streaming checkpoint and its optimizer, if saved")
p.add_argument("--chunks", type=int, default=10, help="number of 120ms chunks per training example")
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--batch-size", type=int, default=2)
p.add_argument("--lr", type=float, default=None, help="learning rate; default 0.0003 for new training, preserves checkpoint rate on resume unless explicitly set")
p.add_argument("--workers", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cpu")
a = p.parse_args()
if min(a.chunks, a.steps, a.batch_size) < 1 or a.workers < 0 or (a.lr is not None and (not np.isfinite(a.lr) or a.lr <= 0)): p.error("invalid training parameter")
if Path(a.output).exists(): raise FileExistsError(f"output already exists: {a.output}")
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dataset = Synthetic(a.chunks) if a.synthetic else IndexedAudioVisualDataset(a.index_dir, a.chunks)
loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=a.workers)
ckpt = read_checkpoint(a.resume, a.device) if a.resume else None
model_config = ckpt["model_config"] if ckpt else {"channels": 320, "dilations": [1, 2, 4, 8, 16, 32]}
model = FalconStreaming(**model_config).to(a.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003 if a.lr is None else a.lr)
start = 0
if a.resume:
    model.load_state_dict(ckpt["state_dict"], strict=True)
    start = int(ckpt["steps"])
    if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    # A requested override must win over the optimizer state; without --lr,
    # continue with the checkpoint rate (or Adam default for weights-only).
    if a.lr is not None:
        for group in optimizer.param_groups: group["lr"] = a.lr
print(f"learning_rates={[group['lr'] for group in optimizer.param_groups]}", flush=True)
model.train()
iterator = iter(loader)
for step in range(a.steps):
    try: mixture, target, video = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        mixture, target, video = next(iterator)
    mixture, target, video = (x.to(a.device) for x in (mixture, target, video))
    optimizer.zero_grad(set_to_none=True)
    # forward walks the same stateful chunk API used by inference, preserving gradients.
    output = model(mixture, video)
    if output.shape != target.shape: raise RuntimeError(f"output {output.shape} != target {target.shape}")
    loss = neg_si_snr(output, target)
    if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    if not torch.isfinite(grad): raise RuntimeError("non-finite gradient")
    optimizer.step()
    print(f"step={start + step + 1} loss={loss.item():.5f} grad_norm={grad.item():.5f}", flush=True)
new_checkpoint(a.output, {"format": checkpoint_format_for_visual_mode(model.visual_mode, model.synthesis_mode, model.attention_history_chunks, model.subband_refiner, model.subband_width, model.subband_depth), "state_dict": model.state_dict(), "model_config": model_config,
                          "optimizer": optimizer.state_dict(), "steps": start + a.steps,
                          "training_status": "synthetic_smoke" if a.synthetic else "real_data_optimization", "data": "synthetic-smoke" if a.synthetic else str(a.index_dir),
                          "chunks": a.chunks, "seed": a.seed})
print(f"saved {a.output}")
