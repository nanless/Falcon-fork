---
pretty_name: RealSSA-Real
language:
- en
task_categories:
- automatic-speech-recognition
tags:
- audio
- video
- target-speaker-extraction
- speaker-verification
- real-world
size_categories:
- n<1K
configs:
- config_name: inference
  default: true
  drop_labels: true
  data_files:
  - split: test
    path:
    - metadata.jsonl
    - audio/inference/**
    - references/**
- config_name: original_recordings
  drop_labels: true
  data_files:
  - split: test
    path:
    - views/original/metadata.jsonl
    - audio/original_recordings/**
    - references/**
extra_gated_heading: Request access to RealSSA-Real
extra_gated_prompt: >-
  RealSSA-Real contains the faces, voices and speech content of real, identifiable
  people. Participants authorized collection and use of their recordings for this
  research dataset only. Access requests are reviewed and approved manually. Read
  "Access, consent and permitted use" in the dataset card before submitting.
extra_gated_fields:
  Full name: text
  Affiliation: text
  Country: country
  Intended research use: text
  I will use this dataset only for the permitted research purpose described in the dataset card: checkbox
  I will not attempt to identify, re-identify or contact any participant: checkbox
  I will not redistribute the data or share my access with anyone: checkbox
extra_gated_button_content: Submit access request
---

# RealSSA-Real

RealSSA-Real provides real-scene audiovisual inputs, dataset-local target-speaker identities, independent speaker enrollment audio, and recording-level reference transcripts. **Start with the `inference` view for audiovisual model inference:** it has 54 paired physical segments from 40 logical recordings, 10 speakers and 8 scenes, totaling 83.9905 minutes of native audio. Every segment has its matching audio, face video, mouth ROI and landmarks.

The separate `original_recordings` view has 34 logical recordings extracted from 44 available original video files, totaling 71.1154 minutes. These audio files have a different extraction and segmentation history. Their relationship to inference segments is recorded, but they must not be substituted for the audio paired with the face/mouth arrays.

Model evaluation results are not included.

Each view has a single `test` split. There is no invented train/dev split. No model checkpoints, separated estimates, generic ZIP/TAR archive or DOCX file are included.

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

## Access, consent and permitted use

RealSSA-Real contains **personal data of real, identifiable people**: face videos, mouth crops and facial landmarks; voices in scene recordings and speaker enrollment recordings; transcripts of what they say; and the original videos with their capture metadata, such as device model and recording time (GPS location removed). Participants authorized the collection and use of their recordings for this research dataset. **Any other use is strictly prohibited.**

Access is gated on Hugging Face. Each request is reviewed and approved manually by the dataset owners, who may decline or revoke access. Approval is personal to the requesting account.

**Permitted use:** non-commercial research on audio-visual target-speaker extraction and speech separation, including the WER and speaker-similarity evaluations described in this card. Publishing aggregate results, such as WER or similarity scores, is allowed.

**Prohibited:**

- Any use outside the permitted purpose, including commercial use.
- Sharing access, or redistributing the data or derived files that contain participants' faces, facial landmarks or voices, such as mouth crops or separated speech estimates.
- Identifying, re-identifying, tracking, profiling or contacting any participant, including matching their faces or voices against real identities or external databases.
- Creating voice clones, talking-face or face-swap videos, or other synthetic media of participants.
- Showing participants' faces or voices in public papers, slides, demos or websites without the dataset owners' written permission.

Store downloaded copies only in access-controlled locations that unapproved users cannot read. Delete them when the approved use ends, or when the dataset owners ask, for example because a participant withdraws.

**Contact:** [thucs.kaili@gmail.com](mailto:thucs.kaili@gmail.com) for access questions, permission to show participants' faces or voices publicly, withdrawal or removal requests from participants, and reports of misuse.

## Directory layout

```text
audio/inference/             54 native stereo PCM16 WAV inputs, 16 kHz
audio/original_recordings/   34 full original-extraction mono float32 WAVs, 16 kHz
faces/                      54 target-face MP4s, 224x224, 25 fps
mouth/                      54 NPZ files: data = uint8 [T,96,96]
landmarks/                  54 NPZ files: data = float32 [T,68,2]
references/                 ONE set of 10 selected-channel mono PCM16 enrollment WAVs
raw/videos/                 44 original videos, byte-identical except blanked GPS tags
raw/transcripts/            40 archived-source transcripts, TXT
raw/references/             10 original enrollment recordings as float32 WAVs, 48 kHz
raw/reference_texts/        10 enrollment reading texts, TXT
transcripts/reference/      Later evaluation-reference TXT files that differ from raw TXT
metadata/                   Recording/segment/speaker joins, provenance and audit reports
metadata.jsonl              AudioFolder metadata for 54 inference segments
views/original/metadata.jsonl  AudioFolder metadata for the 34-recording audio view
scripts/                    Loaders and independent validation
```

All media paths in the main manifests are relative to the downloaded repository root. **Resolve GT using the manifest's path:** if an evaluation reference is byte-identical to its raw TXT, both roles share that one raw file. There are 40 GT entries, not 40 additional copied files. No symlinks are needed.

## Download and install

The repository ID is `JusperLee/RealSSA-Real`. Access and use are governed by the research-use conditions above.

This is a **gated dataset**. Request access on its [Hugging Face page](https://hf.co/datasets/JusperLee/RealSSA-Real) and wait for manual approval. Then log in with the approved account before calling `snapshot_download`, or `load_dataset` with the Hub ID: run `hf auth login` (`huggingface-cli login` with older `huggingface_hub` versions) or set `HF_TOKEN`. Unapproved or unauthenticated downloads fail.

```python
from huggingface_hub import snapshot_download

root = snapshot_download(repo_id='JusperLee/RealSSA-Real', repo_type='dataset', local_dir='RealSSA-Real')
```

Download the **full snapshot** when using face/mouth inputs. `load_dataset` alone exposes their paths and does not necessarily download the auxiliary NPZ and MP4 files.

For a reproducible lightweight loader environment, use Python 3.10+:

```bash
cd /path/to/RealSSA-Real
python -m pip install -r requirements.txt
# Install ffmpeg separately if you want to decode the face MP4s.
```

The path-loading examples were tested with Datasets 3.6.0 and SoundFile. Newer Datasets versions may require the corresponding PyTorch/TorchCodec audio backend. Our NumPy/SoundFile loader does not require PyTorch or TorchCodec. Your own model environment and trained checkpoint are separate requirements for a neural forward pass.

## Quick start: load paired AV input

```bash
python scripts/load_inference.py /path/to/RealSSA-Real --index 0 --align-tail
```

```python
import sys
from pathlib import Path

root = Path('/path/to/RealSSA-Real')
sys.path.insert(0, str(root / 'scripts'))
from load_inference import load_sample, iter_face_frames

x = load_sample(root, index=0, align_tail=True, channel='mean')
audio = x['audio']                  # float32 [samples], 16000 Hz
mouth = x['mouth']                  # uint8 [frames,96,96], no normalization applied
landmarks = x['landmarks']          # float32 [frames,68,2], original coordinates
enrollment = x['speaker_reference'] # float32 [samples], independent 16000 Hz reference
speaker_id = x['metadata']['speaker_id']

assert len(audio) == len(mouth) * 640
# For a face-input model, decode RGB frames lazily instead of loading a full video:
for rgb in iter_face_frames(x):     # uint8 [224,224,3], matching the selected frame limit
    pass                           # feed frames into your own face preprocessing
```

`inference_id` is a physical segment, such as `small_indoor/study_room/2-3`. `recording_key` is its logical recording, such as `small_indoor/study_room/2`. Use IDs in the manifests; do not infer speaker identity by matching bare numeric filenames across scenes.

## Run the released Falcon checkpoint

The [Falcon repository](https://github.com/JusperLee/Falcon) provides a complete inference runtime. Install its Studio environment separately from the dataset loader:

```bash
git clone https://github.com/JusperLee/Falcon.git
cd Falcon
python -m pip install -r studio/requirements.txt
```

Run this example from the Falcon repository root after downloading the full approved dataset snapshot:

```python
import sys
from pathlib import Path
import soundfile as sf

root = Path("/path/to/RealSSA-Real")
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, "studio")
from load_inference import load_sample
from falcon_runtime import FalconRuntime
from infer_file import extract

sample = load_sample(root, index=0, align_tail=True, channel="mean")
model = FalconRuntime.from_pretrained("JusperLee/Falcon", device="auto")
estimate = extract(model, sample["audio"], sample["mouth"])
sf.write("falcon_estimate.wav", estimate, 16000, subtype="PCM_16")
print(sample["metadata"]["inference_id"], estimate.shape)
```

Use `JusperLee/Falcon-Big` to select the larger Studio variant; weights are downloaded on first use. The example uses the declared channel-mean and tail-alignment policies. The dataset loader returns uint8 grayscale mouth pixels; Falcon applies its own center crop and normalization internally, so do not pre-normalize them. `extract` processes a whole clip; `FalconRuntime.infer` is for exactly 2 s / 50 frames. Report the channel and alignment policy when comparing results. This integration example has been checked against the published APIs; no checkpoint inference or metric recomputation was performed for this documentation update.

For quantitative evaluation, use the recording-level WER and speaker-reference protocols below. Independent enrollment audio is not a clean synchronized target, so it cannot be used as the SI-SDR reference. The browser Space demonstrates record-then-separate use; the local Python path supports these dataset assets. Do not upload or display participant media in a public demo without the written permission required by the access terms.

## Run your separation model

Use the model's **trained visual transform, input layout and checkpoint**. The dataset deliberately supplies unnormalized mouth pixels and native face crops because different architectures use different crops, means/stds and tensor layouts. Do not apply the speaker enrollment reading text as the scene transcript.

For a model whose API is `model(audio_BN, mouth_B1THW)`, the integration looks like this:

```python
import torch

def separate_one(model, model_visual_transform, sample, device):
    model = model.to(device).eval()
    audio_tensor = torch.from_numpy(sample['audio']).unsqueeze(0).to(device)  # [B,N]
    visual = model_visual_transform(sample['mouth'])                        # [T,H,W]
    mouth_tensor = torch.as_tensor(visual).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(audio_tensor, mouth_tensor)  # model-specific output

# Load your own trained checkpoint and evaluation-time visual transform, then:
# estimate = separate_one(model, model_visual_transform, x, device='cuda')
```

This is an API example, not a claim that all models share that signature. Face-input models should use the RGB face frames instead. Speaker-conditioned models may use `x['speaker_reference']`; the reference is not a time-aligned clean target waveform. Apply the model's own output handling and save its estimated waveform at 16 kHz. No particular trained model was run as part of the data-integrity checks.

## Time alignment and channel policy

- The canonical AV files come from the **same historical preprocessing directory** for each segment. They are copied byte-for-byte from one canonical export, `SpeakerContrastiveV2_small`'s inputs; no `speaker1_est.wav` is shipped.
- The stored inference WAV is **stereo PCM16 at 16 kHz**. The loader derives a float32 channel mean on demand, without peak normalization; use `channel='0'` or `'1'` for controlled channel experiments. There is no second saved mono copy. The historical models' internal downmix was not established by their logs, so this channel choice is declared explicitly.
- All 54 face videos were fully decoded: **125,943 frames**, regular 25 fps presentation timestamps. Mouth arrays have the same count as face in 52 segments; two have one fewer mouth frame. Native audio extends beyond mouth by at most **70 ms**.
- `align_tail=True` explicitly chooses `T=min(mouth frames, landmark frames, face frames, floor(audio samples/640))`, keeps the first `T` visual frames and first `T*640` audio samples, and reports the discarded audio sample count. Only the returned arrays are trimmed; files are unchanged. Consume at most `x['face_frame_limit']` face frames. With `align_tail=False`, native lengths are preserved.
- This policy makes tensor durations equal; it is not a new forced alignment or proof of sub-frame lip synchronization. Do not pair `audio/original_recordings` with these visual arrays by filename: the original extraction differs in codec timing and physical segmentation.

## GT is for a whole logical recording

**There are 40 logical-recording GT entries. There are not 54 independently annotated segment transcripts.** In the inference metadata, `recording_reference_text` and `recording_reference_file` refer to the full logical recording. `reference_text_scope` is explicitly `whole_logical_recording_not_this_segment`.

For ASR/WER, process parts and concatenate predictions in `part_order` before comparing with GT:

```python
from load_inference import read_recordings

record = read_recordings(root)['small_indoor/study_room/2']
predictions = []
for segment_id in record['segment_ids']:
    part = load_sample(root, inference_id=segment_id, align_tail=True)
    # estimated_audio = your_separator(part, ...)
    # predictions.append(your_asr(estimated_audio))
    pass
hypothesis = ' '.join(predictions)  # compare only after all parts have been processed
reference = record['reference_text']
```

The 40 evaluation references were verified byte-for-byte against the frozen evaluation snapshot and the current source `check_version` TXT files. Raw TXT is the separately extracted archived document version; **34 of the 40 raw texts differ from evaluation GT**, so raw TXT must not silently replace GT. The six identical cases share one file. Enrollment reading texts are separately stored in `raw/reference_texts/`.

GT remains the existing evaluation-reference annotation. This release audit checks its version and recording association; it does not claim a fresh 84-minute verbatim listening audit, word timestamps, diarization, or aligned clean target speech. Evaluate WER and speaker similarity with their stated protocols; clean-target SI-SDR is not supported by an unavailable clean reference waveform.

## Speaker reference mapping

The final human-reviewed cross-scene speaker map is preserved in `metadata/speaker_mapping.json`. IDs are dataset-local labels. Target IDs **2 and 8 use the corrected final assignments**. All 10 prepared reference WAVs were independently re-created from their original recordings and selected channels; PCM samples matched exactly.

| Speaker | Original target ID | Selected channel (zero-based) |
|---|---:|---:|
| speaker_01 | 4 | 0 |
| speaker_02 | 10 | 1 |
| speaker_03 | 2 | 1 |
| speaker_04 | 6 | 1 |
| speaker_05 | 5 | 1 |
| speaker_06 | 7 | 0 |
| speaker_07 | 8 | 1 |
| speaker_08 | 3 | 1 |
| speaker_09 | 9 | 0 |
| speaker_10 | 1 | 1 |

## Hugging Face loading

```python
from datasets import load_dataset, Audio

# Local downloaded snapshot; after approval you can also use 'JusperLee/RealSSA-Real'.
ds = load_dataset(str(root), name='inference', split='test')
ds = ds.cast_column('audio', Audio(decode=False)).cast_column('reference', Audio(decode=False))
assert len(ds) == 54
row = ds[0]
print(row['audio']['path'], row['reference']['path'], row['mouthroi'])

original = load_dataset(str(root), name='original_recordings', split='test')
original = original.cast_column('audio', Audio(decode=False)).cast_column('reference', Audio(decode=False))
assert len(original) == 34
```

The AudioFolder `audio` feature points to the native file; Datasets decoder defaults may downmix audio automatically. Use `Audio(decode=False)` and our SoundFile loader when controlling channels. Auxiliary face/mouth/landmark paths are repo-root-relative strings. Load them from a complete snapshot with `load_sample`.

## Original data and deduplication

`raw/videos/` retains the **44 original video files** with their capture metadata, except GPS location. In 28 files the location strings were overwritten in place with spaces; `raw/videos_manifest.jsonl` marks them with `gps_location_removed`. All other bytes, including every audio and video sample, are identical to the originals. Six logical recordings have unextractable original videos: billiard hall 4–5 and internet cafe 1–4. Their historical processed AV inputs still exist in the 54-segment inference view. The 34 original-extraction WAVs remain available for the corresponding original-video WER protocol.

There is no `data/` directory, repeated enrollment set, second metadata-only copy of each video, saved duplicate inference mono waveform, intermediate original stereo WAV cache, DOCX, or source archive. The two remaining audio directories represent different acquisition/preprocessing histories, not aliases of the same waveform. `metadata/deduplication.json` records removed redundant representations and retained asset paths. Every media asset has one canonical path.

Independently repeated landmark/mouth preprocessing can vary, so canonical published crops are not claimed to reproduce every historical model run bit-for-bit. Use the same published inputs when comparing models.

## Verify and release status

```bash
python scripts/validate_release.py /path/to/RealSSA-Real
```

Validation checks all 54 AV pairs, 40 recording/GT joins, 10 speaker references, optional tail alignment, the 34 original-audio entries, raw TXT format and path portability. The independent source audit is in `metadata/source_audit.json`; detailed limits and results are in `metadata/validation_report.json`.

Normal Hugging Face cache symlinks are supported. Generated `.cache/`, `.git/` and `__pycache__/` files are ignored by the inventory check. The runtime validation report is replaced atomically, without modifying a cached Hub blob.

A separate general-purpose dataset license is not specified. Follow the access and permitted-use conditions above; the code/weight Apache 2.0 license does not apply to participant recordings.


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

