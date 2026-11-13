"""Index-bound sample exclusions for shape-preserving RealSSA training."""
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset


def load_exclusions(path, index_sha256, dataset):
    """Bind each record to the exact index, row, TAR and member byte range."""
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("index_sha256") != index_sha256 or
            manifest.get("predicate") != "raw 16000 Hz PCM16 ch0 equals -32768 for every sample"):
        raise ValueError("exclusion manifest does not match index SHA or exclusion predicate")
    members = manifest.get("members")
    if not isinstance(members, list) or manifest.get("count") != len(members):
        raise ValueError("exclusion manifest count differs")
    excluded = {}
    for row in members:
        index = row["index"]
        if type(index) is not int or not 0 <= index < len(dataset) or index in excluded:
            raise ValueError(f"invalid or duplicate exclusion index {index}")
        sample = dataset.samples[index]
        if (row["id"] != sample["id"] or row["shard"] != dataset.shards[sample["shard"]] or
                row["mix_location"] != sample["fields"]["mix"]):
            raise ValueError(f"exclusion row differs from index at {index}")
        offset, size = row["mix_location"]
        with Path(row["shard"]["path"]).open("rb") as stream:
            stream.seek(offset)
            wav = stream.read(size)
        if len(wav) != size or hashlib.sha256(wav).hexdigest() != row["mix_wav_sha256"]:
            raise ValueError(f"exclusion WAV SHA differs at {index}")
        excluded[index] = row["id"]
    return excluded, hashlib.sha256(raw).hexdigest()


class IndexedSamples(Dataset):
    """Append original index without changing sampler length, order or crop."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (*self.dataset[index], index)


def batch_exclusions(indices, excluded, device):
    indices = indices.tolist()
    hits = {i: excluded[i] for i in indices if i in excluded}
    mask = torch.tensor([i in excluded for i in indices], device=device, dtype=torch.bool)
    return mask, hits


def replace_excluded_inputs(mask, *inputs):
    """Use finite zero placeholders before model execution, including on empty local ranks."""
    return tuple(x.masked_fill(mask.reshape((-1,) + (1,) * (x.ndim - 1)), 0) for x in inputs)


def training_mean(values, excluded=None):
    """Use the original per-rank denominator, including excluded rows."""
    return values.mean() if excluded is None else torch.where(excluded, 0, values).mean()


def require_retained_global(mask):
    """Fail on every rank before any optimizer update when all rows are excluded."""
    retained = (~mask).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(retained)
    if not retained.item():
        raise RuntimeError("training batch has no unexcluded samples across either rank")
