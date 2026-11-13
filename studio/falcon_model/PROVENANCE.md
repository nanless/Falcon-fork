# Falcon model code in Falcon Studio

`falcon_model/` is an inference-only copy of the model code in the training code
(`look2hear/` at the repository root):

| Studio file | Source |
|---|---|
| `models/Falcon_Contrastive.py`, `models/Falcon.py` | `FalconSpeakerContrastive` (Falcon, the paper model, `configs/Falcon.yml`) |
| `models/Speaker_Contrastive_v2.py`, `models/SISNRRepel.py` | `SpeakerContrastiveV2` (Falcon-Big, `configs/FalconBig.yml`) |
| `videomodels/frcnn_videomodel.py`, `resnet.py`, `shufflenetv2.py` | `FRCNNVideoModel`, the frozen lip-reading encoder |
| `layers/normalizations.py`, `layers/activations.py` | layer helpers |

Inference-only changes; no learned layers are removed:

- `timm`'s `DropPath` is replaced by `drop_path.py`. DropPath is the identity in evaluation mode.
- The base class keeps only the parameter-free constructor and the sample-rate accessor, so
  PyTorch Lightning is not needed.
- Unused imports (`ResNet1D`, a `sys.path` change) and package-level imports of training-only
  datasets, losses and metrics are removed.
- Small and Big use an independent zero-state VideoBlock LSTM step per sample and video frame by default, preserving the released single-item semantics. The explicit temporal-recurrence configuration remains available.
- On Apple Metal (MPS), STFT and inverse STFT run on the CPU because complex tensors are not fully
  supported there. FFT parameters, windows and output length are unchanged; everything else runs
  on the selected device.

`falcon_runtime.py` builds each model from its configuration (deep-copied, because the original
`Separator` consumes `repeats` and `shared` with `dict.pop()`) and loads both `audio_model.*` and
`video_model.*` with `strict=True`, so a mismatched checkpoint fails instead of running a partial
network.

Mouth crops follow the validation transform of the training code: uint8 grayscale at 25 fps,
divided by 255, center crop 96×96 → 88×88, normalized with mean 0.421 and std 0.165. Audio is mono
16 kHz without peak or RMS normalization.

In one recorded CPU test with `falcon.pt`, this copy and the training code gave bit-identical output
(11,724,385 parameters, maximum absolute difference 0).

The released files `falcon.pt` (Hugging Face `JusperLee/Falcon`) and `falcon-big.pt`
(`JusperLee/Falcon-Big`) contain `state_dict`, the architecture name, and epoch metadata 714 and 715, respectively.
Optimizer and Lightning state and training hyper-parameters are not included. The full training dataset, checkpoint lineage, and identity with the weights used for a particular
paper result remain to be confirmed; see the [reproduction status](../../docs/STATUS.md).

Inference uses rolling two-second windows. The model exposes no streaming state, so this is
windowed inference, not sample-by-sample streaming.
