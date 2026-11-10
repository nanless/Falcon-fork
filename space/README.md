---
title: Falcon Studio
emoji: 🦅
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12.12"
app_file: app.py
license: apache-2.0
short_description: Audio-visual target speaker extraction in your browser
models:
  - JusperLee/Falcon
  - JusperLee/Falcon-Big
preload_from_hub:
  - JusperLee/Falcon falcon.pt
  - JusperLee/Falcon-Big falcon-big.pt
---

# Falcon Studio · Hugging Face Space

**English** | [中文](README_zh.md)

Paper: [arXiv:2610.32749](https://arxiv.org/abs/2610.32749) (NeurIPS 2026)

The [Falcon Studio](https://github.com/JusperLee/Falcon/tree/main/studio) interface on a ZeroGPU Space. Record a clip with your camera and microphone; Falcon extracts the voice of the speaker whose mouth is in the frame.

ZeroGPU attaches a GPU for each call and counts it against a daily quota, so the live Studio's loop of one call every 0.5 s does not fit. This Space records first and separates afterwards. For live separation, headphone monitoring and meeting output, run Falcon Studio locally.

## How it works

1. The browser captures audio at 16 kHz and tracks the mouth with MediaPipe Face Landmarker, as in Falcon Studio. Tracking runs in the browser; full camera frames are not uploaded, but mouth crops are sent after recording.
2. A recording is up to 60 s. When it stops, the audio and the 96×96 grayscale mouth crops, each with its capture time, are sent to the Space. The start and end of the recording without any mouth crop are trimmed.
3. The server creates 2 s windows at 0.5 s intervals, chooses the closest mouth crop for each slot, emits the 0.5 s block ending 80 ms before each window end and crossfades adjacent blocks over 20 ms. Audio is zero-padded at the boundaries and boundary video holds the first or last crop. One request runs windows in batches of up to 32. The geometry resembles Studio, but the current model interprets the batch axis as an LSTM sequence, so later windows can depend on earlier windows’ video; Studio processes each window separately. A real-audio, real-mouth example differed between batch 1 and batch 4; the overall quality effect needs measurement across more samples.
4. The page shows the GPU time, and you can play, compare with the mixture and export both as WAV.

Tracking follows position rather than identity. After face loss the page may hold the last crop, so verify the target if another person enters that region. Space can also select a distant nearest frame across a capture gap. In the source repository, `docs/understanding/20261110-evaluation-and-apps.md` explains the window protocol and evidence, and `docs/STATUS.md` records reproduction status.

Both models load when the Space starts: [`JusperLee/Falcon`](https://huggingface.co/JusperLee/Falcon) (the paper model) and [`JusperLee/Falcon-Big`](https://huggingface.co/JusperLee/Falcon-Big).

## Privacy and quota

Only the 16 kHz audio and the 96×96 grayscale mouth crops leave the browser; the full camera frames are never uploaded. The server code (`space/app.py`, `space/clip.py`) has no write path — no temporary files, no saved request, nothing written to disk — so a recording is held in memory for the separation only, and the result stays in your browser. The one thing the page persists is the interface language, in `localStorage['falcon.language']`. Opened from the Space page, a separation counts against your own ZeroGPU quota. Anonymous visitors get the smallest quota and can be refused ("ZeroGPU runs limit"); signing in to Hugging Face gives more quota and a higher queue priority. The source contains historical GPU timing examples without per-run measurement files; actual waiting time also includes upload and queueing and depends on hardware and load.

## Development

The Space is assembled from the Studio runtime. `build.py` copies `studio/falcon_runtime.py`,
`studio/falcon_model/`, `studio/configs/` and `studio/web/` byte for byte, then replaces
`web/app.js`, `web/index.html` and adds `web/i18n-space.js` from `space/web/`. `studio/server.py`,
the FastAPI + WebSocket service that re-runs a 2 s window every 0.5 s, is not part of the build,
so the Space cannot serve a live session: there is no WebSocket endpoint, and one HTTP call
takes one finished recording and returns the separated audio.

A build is 54 files and 39,865,947 B (39.87 MB). `web/vendor/mediapipe` accounts for
39,380,889 B of that, 98.8%. The three `.wasm` binaries add up to 34.47 MB, but
`FilesetResolver.forVisionTasks(..., true)` fetches only the module variant (12.08 MB);
the other 23.36 MB (`vision_wasm_internal.*`, `vision_wasm_nosimd_internal.*`) is never
requested and is dead weight in the upload.

The Space runs on the ZeroGPU runtime (`sdk_version: 6.29.1`): **Python 3.12 with
`torch==2.13.0`**, as pinned in `space/requirements.txt`. That is a **different
environment from the local Studio setup**: the repository's `falcon` conda env is
Python 3.11 with `torch 2.8` (see `AGENTS.md` §2). Do not install
`space/requirements.txt` into `falcon`; `space/tests/test_clip.py` runs in the `falcon`
env, while the HTTP/SSE tests need the Space deps from that file.

From the GitHub repository root, with Python 3.12:

```bash
pip install -r space/requirements.txt gradio==6.29.1
python space/build.py --out /tmp/falcon-space    # copies studio/ runtime and space/web
(cd /tmp/falcon-space && python app.py)          # returns to the repository root after stopping
# Publishing deletes existing target Space files not present in the new build:
python space/build.py --push JusperLee/Falcon-Studio
```

Two checks need only Python and Node and also pass in the `falcon` env:

```bash
python -m pytest space/tests/test_static.py -q    # 3 passed
node space/tests/check_i18n.mjs                  # i18n: 226 Chinese strings translated
```

`space/tests/test_space.py` starts `app.py` in-process and needs `pip install gradio spaces`;
the `falcon` env has neither, so `python -m pytest space/tests` ends with 4 errors there
(errors, not skips). Those four cases drive the HTTP and Server-Sent-Events endpoints and only
mean something in the ZeroGPU runtime.

`space/` holds only what differs from Falcon Studio: `app.py` (Gradio server mode, `@spaces.GPU`), `clip.py` (window schedule and request checks), `web/index.html` and `web/app.js` (record, then separate), and `web/i18n-space.js`. Everything else is copied from `studio/` by `build.py`.
