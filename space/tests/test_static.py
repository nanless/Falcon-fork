"""Static checks for the assembled Space — no gradio, spaces, torch or GPU required.

These are the build/file assertions and the i18n audit that used to live in
``test_space.py``. They only run ``space/build.py`` (standard library) and Node, so they
work in the repository's ``falcon`` environment and in CI without the Space runtime.
The HTTP/SSE round trips that need ``gradio``/``spaces`` stay in ``test_space.py``
(marked ``slow``).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    out = tmp_path_factory.mktemp("space-static") / "build"
    subprocess.run([sys.executable, str(SPACE / "build.py"), "--out", str(out)], check=True)
    return out


def test_build_merges_space_files(build):
    assert (build / "web" / "vendor" / "mediapipe" / "face_landmarker.task").is_file()
    assert "import spaceMessages from './i18n-space.js'" in (build / "web" / "i18n.js").read_text()
    assert (build / "web" / "app.js").read_text().startswith("// Falcon Studio on a ZeroGPU Space")
    assert not list(build.rglob("__pycache__")) and not list(build.rglob("*.test.cjs"))
    assert (build / "README.md").read_text().startswith("---\ntitle: Falcon Studio")


def test_build_reuses_the_studio_runtime_unchanged(build):
    studio = SPACE.parent / "studio"
    for name in ("falcon_runtime.py", "configs/Falcon.yml", "configs/FalconBig.yml",
                 "web/capture-worklet.js", "web/video-worker.js", "web/playback-resampler.js",
                 "web/vendor/mediapipe/manifest.json"):
        assert (build / name).is_file(), f"assembled Space is missing {name}"
    assert (build / "falcon_runtime.py").read_bytes() == (studio / "falcon_runtime.py").read_bytes()
    assert (build / "configs" / "Falcon.yml").read_bytes() == (studio / "configs" / "Falcon.yml").read_bytes()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_chinese_string_has_an_english_translation(build):
    subprocess.run(["node", str(SPACE / "tests" / "check_i18n.mjs"), str(build)], check=True)
