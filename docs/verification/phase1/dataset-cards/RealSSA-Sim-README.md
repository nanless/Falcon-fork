---
pretty_name: RealSSA-Sim
license: cc-by-nc-nd-4.0
language:
- en
task_categories:
- audio-to-audio
tags:
- audio
- video
- audio-visual
- target-speaker-extraction
- simulated
configs:
- config_name: default
  data_files:
  - split: train
    path: tr/*.tar
  - split: validation
    path: cv/*.tar
  - split: test
    path: tt/*.tar
extra_gated_heading: Request access to RealSSA-Sim
extra_gated_prompt: >-
  RealSSA-Sim is derived from LRS3 (TED talks, CC BY-NC-ND 4.0), LibriSpeech (CC BY 4.0)
  and several noise collections with non-commercial or per-clip terms. Access is limited to
  non-commercial academic research, and requests are reviewed and approved manually. Read
  "Access and license" in the dataset card before submitting.
extra_gated_fields:
  Full name: text
  Affiliation: text
  Country: country
  Intended research use: text
  I will use this dataset only for non-commercial academic research: checkbox
  I agree to the terms of LRS3, the TED talks license and the other source datasets listed in the dataset card: checkbox
  I will not redistribute the data or share my access with anyone: checkbox
extra_gated_button_content: Submit access request
---

# RealSSA-Sim

RealSSA-Sim is the simulated counterpart of [RealSSA-Real](https://hf.co/datasets/JusperLee/RealSSA-Real) for audio-visual target-speaker extraction. Each example has a 4-channel 16 kHz mixture, the 4-channel target speech, the target speaker's mouth crops, and a JSON file describing how the mixture was simulated. All sources are static, in simulated rooms of four scene types: `small_meeting`, `medium_indoor`, `large_public` and `outdoor`.

| Split | Directory | Examples | Shards |
|---|---|---:|---:|
| train | `tr/` | 100,826 | 202 |
| validation | `cv/` | 9,133 | 19 |
| test | `tt/` | 9,169 | 19 |

Four training examples (`00001_01_87828`, `00002_05_59973`, `00020_00_46434`, `50007_35934`) have no JSON file or mouth crops and are not included.

## Falcon project and resources

RealSSA accompanies **Falcon: Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World Acoustic Scenes** (NeurIPS 2026).

| Resource | Link | What it provides |
|---|---|---|
| Source code | [JusperLee/Falcon](https://github.com/JusperLee/Falcon) | Training, evaluation, offline inference and Falcon Studio |
| Project page | [cslikai.cn/Falcon](https://cslikai.cn/Falcon/) | Method overview and demonstration samples |
| Paper model | [JusperLee/Falcon](https://huggingface.co/JusperLee/Falcon) | falcon.pt; 0.5 M trainable parameters, plus the frozen visual encoder |
| Larger inference model | [JusperLee/Falcon-Big](https://huggingface.co/JusperLee/Falcon-Big) | falcon-big.pt; Studio's Big option, separate from the paper model |
| Browser demo | [Falcon Studio Space](https://huggingface.co/spaces/JusperLee/Falcon-Studio) | Record up to 60 s, then separate on ZeroGPU |
| Local / live inference | [Falcon Studio guide](https://github.com/JusperLee/Falcon/blob/main/studio/README.md) | Local inference, live microphone separation and meeting routing |
| Space implementation | [space/](https://github.com/JusperLee/Falcon/tree/main/space) | Source and deployment instructions for the ZeroGPU demo |
| Simulated benchmark | [RealSSA-Sim](https://huggingface.co/datasets/JusperLee/RealSSA-Sim) | Paired multichannel mixtures, target speech and RGB mouth crops |
| Real-recorded benchmark | [RealSSA-Real](https://huggingface.co/datasets/JusperLee/RealSSA-Real) | Paired AV test inputs, enrollment audio and recording-level transcripts |

The Space is a record-then-separate demo. Use the local Studio for live separation and the Python/CLI examples below for dataset inference. Dataset access is reviewed separately; a public model or demo does not grant access to either dataset. Code and weights use Apache 2.0; the datasets retain their own access and usage conditions.

## Access and license

RealSSA-Sim contains the faces and voices of real people from TED talks. Access is gated on Hugging Face, and each request is reviewed and approved manually.

**Permitted use:** non-commercial academic research. Publishing aggregate results is allowed.

**Prohibited:**

- Any commercial use.
- Sharing access, or redistributing the data or derived files that contain the speakers' faces or voices, such as mouth crops or separated speech estimates.
- Identifying, tracking or profiling speakers, or creating synthetic media of them.

The dataset is released under **CC BY-NC-ND 4.0**, the license of the TED talks it is built from. Every source below keeps its own terms, and users must follow them. Mixing, reverberation and cropping create adapted versions of the source material. Rights holders can ask for content to be removed at [thucs.kaili@gmail.com](mailto:thucs.kaili@gmail.com), and we will remove it.

| Component | Source | Mixtures containing it | Terms |
|---|---|---:|---|
| Target speech, mouth crops, transcripts | LRS3 (TED and TEDx talks) | 100% | TED talks: CC BY-NC-ND 4.0; LRS3 terms apply |
| Interfering speech | LibriSpeech train-clean-360 | 100% | CC BY 4.0 |
| Noise | WHAM! noise | 77.8% | See original terms |
| Noise | BBC Sound Effects | 64.5% | RemArc licence: non-commercial only, credit the BBC |
| Noise | AudioSet | 28.4% | YouTube clips; copyright stays with the uploaders |
| Noise | VGGSound | 20.5% | YouTube clips; copyright stays with the uploaders |
| Noise | Clotho v2 | 9.2% | Freesound clips; per-clip licenses |
| Noise | FreeSound | 6.3% | Per-clip licenses |
| Noise | DCASE (synthetic) | 1.8% | See original terms |
| Noise | SoundBible | 1.3% | Per-clip licenses |
| Noise | VoiceBank-DEMAND | 1.2% | CC BY 4.0 |
| Noise | ESC-50 | 1.0% | CC BY-NC |
| Noise | FSD50K | 0.6% | Per-clip Creative Commons licenses |
| Noise | AVE | 0.4% | YouTube clips; copyright stays with the uploaders |
| Noise | MUSIC21 | 0.2% | YouTube clips; copyright stays with the uploaders |
| Noise | DCASE (real) | 0.1% | See original terms |

## Layout

The data is stored as WebDataset tar shards of 500 examples, `{tr,cv,tt}/{tr,cv,tt}-NNNNNN.tar`. Train examples are shuffled across shards. Each example `<id>` is four consecutive tar members:

```text
<id>.mix.wav      4-channel PCM16 mixture, 16 kHz
<id>.target.wav   4-channel PCM16 target speech, 16 kHz
<id>.json         scene, room (dimensions, mic center, T60), components (type, source, SIR, position), target position, transcript
<id>.mouths.pt    uint8 RGB mouth crops, torch tensor [T,96,96,3]
```

Component types are `near` and `far` (interfering speech) and `env` and `sudden` (noise). The `path` fields in the JSON files are the generation-time file paths and only identify the source clips.


## Download a split or one shard

Request access on this page and wait for manual approval, then authenticate with the approved account:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

Start with one test shard instead of downloading the complete dataset (about 522 GB):

```python
from huggingface_hub import hf_hub_download, snapshot_download

shard = hf_hub_download(
    repo_id="JusperLee/RealSSA-Sim", repo_type="dataset",
    filename="tt/tt-000000.tar", local_dir="RealSSA-Sim",
)
# Optional: download all test shards. Use tr/* or cv/* for other splits.
# snapshot_download(
#     repo_id="JusperLee/RealSSA-Sim", repo_type="dataset",
#     allow_patterns=["README.md", "tt/*.tar"], local_dir="RealSSA-Sim",
# )
```

## Read one example

Install NumPy, SoundFile, PyTorch and OpenCV in your inference environment. This example reads the four fields from one local shard without extracting the archive or using metadata source paths:

```python
import io
import json
import tarfile
import numpy as np
import soundfile as sf
import torch

with tarfile.open(shard, "r:*") as archive:
    members = {m.name: m for m in archive.getmembers() if m.isfile()}
    key = next(name[:-len(".mix.wav")] for name in members if name.endswith(".mix.wav"))

    def read_field(suffix):
        with archive.extractfile(members[key + suffix]) as stream:
            return stream.read()

    mix, sr = sf.read(io.BytesIO(read_field(".mix.wav")), dtype="float32", always_2d=True)
    target, target_sr = sf.read(io.BytesIO(read_field(".target.wav")), dtype="float32", always_2d=True)
    metadata = json.loads(read_field(".json"))
    mouths = torch.load(io.BytesIO(read_field(".mouths.pt")), map_location="cpu", weights_only=True).numpy()

assert sr == target_sr == 16000
assert mix.shape[1] == target.shape[1] == 4
assert mouths.dtype == np.uint8 and mouths.shape[1:] == (96, 96, 3)
print(key, mix.shape, target.shape, mouths.shape)
```

The audio arrays are `[samples, channels]`; the mouth array is `[frames, height, width, RGB]`. The JSON transcript describes the target utterance. Source paths in metadata identify generation inputs and are not portable local paths. A shard contains at most 500 examples; the final shard of each split is smaller.

## Run Falcon on the example

Clone the [code repository](https://github.com/JusperLee/Falcon) and install the Studio environment:

```bash
git clone https://github.com/JusperLee/Falcon.git
cd Falcon
python -m pip install -r studio/requirements.txt
```

Continue the Python example above from the repository root. This explicitly selects microphone channel 0 and converts RGB mouth crops to grayscale. It is an inference example, not an assertion of the paper's channel protocol:

```python
import sys
import cv2

sys.path.insert(0, "studio")
from falcon_runtime import FalconRuntime
from infer_file import extract

audio = np.ascontiguousarray(mix[:, 0], dtype=np.float32)
mouth = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in mouths])
frames = min(len(mouth), len(audio) // 640, len(target) // 640)
audio = audio[:frames * 640]
mouth = mouth[:frames]
clean_target = target[:frames * 640, 0]  # same channel and duration for paired metrics

model = FalconRuntime.from_pretrained("JusperLee/Falcon", device="auto")
estimate = extract(model, audio, mouth)
sf.write("falcon_estimate.wav", estimate, 16000, subtype="PCM_16")
```

Use `JusperLee/Falcon-Big` for the larger inference variant. Weights download automatically on first use. `extract` performs whole-clip inference and applies the model's validation transform (center crop to 88×88, divide by 255, normalize with mean 0.421/std 0.165). Do not normalize mouth pixels a second time. Whole-clip inference requires at least one second; `FalconRuntime.infer` instead accepts exactly 2 s of audio and 50 mouth frames.

### Training and evaluation notes

- Preserve the published train/validation/test split. Use the paired target on the same microphone channel and aligned duration for reference-based metrics. Report channel selection, visual conversion and segmentation with results.
- The released TAR layout is not read directly by the current folder-based training loader. Adapt samples into `tr/`, `cv/`, `tt/` with `mix/`, `target/`, `mouths/`, or implement a WebDataset loader. Convert multichannel audio and RGB mouth tensors explicitly. See [data preprocessing](https://github.com/JusperLee/Falcon/tree/main/DataPreProcess) and [Falcon configuration](https://github.com/JusperLee/Falcon/blob/main/configs/Falcon.yml).
- The shards ship mixtures and target speech, not the original individual near-speaker signals. Generation-time `components[].path` entries are provenance only. The original speaker-contrastive loss needs valid near-speaker negatives; the indexing script's fallback to the mixture is not an equivalent reproduction of that setup.
- Published data and model inference were not re-evaluated as part of this documentation update.

## Citation

When using RealSSA or Falcon, cite the accompanying paper:

```bibtex
@inproceedings{sang2026falcon,
  title     = {Efficient Streaming Audio-Visual Target Speaker Extraction for Real-World Acoustic Scenes},
  author    = {Sang, Wendi and Li, Kai and Li, Yifan and Huang, Jianqiang and Hu, Xiaolin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

