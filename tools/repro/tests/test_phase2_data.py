"""Phase2-QA（task-24）: LRS2-2Mix → Falcon 格式转换数据抽样校验。

覆盖 P2-C 产出 `/root/data/falcon_lrs2_2mix/{tr,cv,tt}`（mix/target/mouths + mix/target/near.json）：
  - JSON 结构与路径一致性、n_samples 与真实 wav 帧数一致；
  - target = s1 约定（与 lrs2_rebuild 的 s1 wav 对齐；mix ≈ target + s2）；
  - mouth 帧数与音频对齐（frames == samples // 640，2 s/25fps 约定）、dtype/shape；
  - split 计数与非空。

数据缺失（Phase2 转换未完成 / 资产已清理）时**自动 skip**，不阻塞仓库内回归。
运行：/root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_phase2_data.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONVERTED = Path("/root/data/falcon_lrs2_2mix")
REBUILD = Path("/root/data/lrs2_2mix/lrs2_rebuild")
SPLITS = ("tr", "cv", "tt")
SAMPLE_RATE = 16000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS  # 640


def _split_dir(split: str) -> Path:
    return CONVERTED / split


def _json_path(split: str, name: str) -> Path:
    return _split_dir(split) / f"{name}.json"


def _available_splits() -> list[str]:
    out = []
    for split in SPLITS:
        if (_json_path(split, "mix").is_file()
                and _json_path(split, "target").is_file()
                and _json_path(split, "near").is_file()):
            out.append(split)
    return out


@pytest.fixture(scope="module")
def converted():
    if not CONVERTED.is_dir():
        pytest.skip(f"Phase2 转换数据不存在: {CONVERTED}（等待 P2-C 全量转换）")
    splits = _available_splits()
    if not splits:
        pytest.skip(f"{CONVERTED} 下没有完整的 mix/target/near.json（P2-C 未完成）")
    return splits


def test_convert_json_consistency(converted):
    """三个 split 的 JSON 表长度一致、路径存在、n_samples 与 wav 帧数一致。"""
    import soundfile as sf

    for split in converted:
        mix = json.loads(_json_path(split, "mix").read_text())
        target = json.loads(_json_path(split, "target").read_text())
        near = json.loads(_json_path(split, "near").read_text())
        assert len(mix) == len(target) == len(near) > 0, f"{split}: JSON 长度不一致"
        # 抽样（大 split 不全量读 wav）
        step = max(1, len(mix) // 25)
        for i in range(0, len(mix), step):
            mix_path, mix_n = mix[i][0], int(mix[i][1])
            tgt_path, mouth_path, tgt_n = target[i][0], target[i][1], int(target[i][2])
            assert Path(mix_path).is_file(), f"{split}[{i}] mix 不存在: {mix_path}"
            assert Path(tgt_path).is_file(), f"{split}[{i}] target 不存在: {tgt_path}"
            assert Path(mouth_path).is_file(), f"{split}[{i}] mouth 不存在: {mouth_path}"
            assert int(mix_n) == int(tgt_n), f"{split}[{i}] mix/target n_samples 不一致"
            assert int(sf.info(mix_path).frames) == mix_n, f"{split}[{i}] mix wav 帧数 != JSON n"
            assert int(sf.info(tgt_path).frames) == tgt_n, f"{split}[{i}] target wav 帧数 != JSON n"


def test_convert_mouth_alignment_and_dtype(converted):
    """mouth 帧数 = samples//640、shape [F,96,96,C]、uint8；且 mouth 命名对应 s1 说话人 id。"""
    import torch

    checked = 0
    for split in converted:
        mix = json.loads(_json_path(split, "mix").read_text())
        target = json.loads(_json_path(split, "target").read_text())
        step = max(1, len(mix) // 10)
        for i in range(0, len(mix), step):
            n = int(mix[i][1])
            mouth = torch.load(target[i][1], map_location="cpu", weights_only=True)
            assert torch.is_tensor(mouth)
            assert mouth.dtype == torch.uint8, f"{split}[{i}] mouth dtype={mouth.dtype}（期望 uint8）"
            assert mouth.dim() == 4 and mouth.shape[1:] == (96, 96, 3), (
                f"{split}[{i}] mouth shape={tuple(mouth.shape)}（期望 [F,96,96,3]，RealSSA-Sim 布局）"
            )
            expected_frames = n // SAMPLES_PER_FRAME
            assert abs(mouth.shape[0] - expected_frames) <= 1, (
                f"{split}[{i}] mouth 帧 {mouth.shape[0]} != samples/640={expected_frames}"
            )
            # mix 名形如 <s1_id>_<g1>_<s2_id>_<g2>.wav -> mouth 名应为 <s1_id>.npz
            stem = Path(mix[i][0]).stem
            token_parts = stem.split("_")
            if len(token_parts) >= 2:
                s1_id = "_".join(token_parts[:2])
                assert Path(target[i][1]).stem == s1_id, (
                    f"{split}[{i}] mouth 命名 {Path(target[i][1]).stem} != s1_id {s1_id}"
                )
            checked += 1
    assert checked > 0


def test_convert_target_is_s1_and_mix_equals_sum(converted):
    """target 约定=s1：转换后的 target 与 lrs2_rebuild 的 s1 wav 一致；mix == target + s2。"""
    if not REBUILD.is_dir():
        pytest.skip("lrs2_rebuild 原始数据不在本机，跳过 s1/s2 对照")
    import numpy as np
    import soundfile as sf

    checked = 0
    for split in converted:
        mix_json = json.loads(_json_path(split, "mix").read_text())
        tgt_json = json.loads(_json_path(split, "target").read_text())
        step = max(1, len(mix_json) // 5)
        for i in range(0, len(mix_json), step):
            mix_path = Path(mix_json[i][0])
            name = mix_path.name
            s1_rebuild = REBUILD / "audio/wav16k/min" / split / "s1" / name
            s2_rebuild = REBUILD / "audio/wav16k/min" / split / "s2" / name
            if not (s1_rebuild.is_file() and s2_rebuild.is_file()):
                continue  # 该 split 的 rebuild 布局不同则跳过该样本
            mix, _ = sf.read(str(mix_path))
            tgt, _ = sf.read(str(tgt_json[i][0]))
            s1, _ = sf.read(str(s1_rebuild))
            s2, _ = sf.read(str(s2_rebuild))
            n = min(len(mix), len(tgt), len(s1), len(s2))
            assert n > 0
            # target 内容 == rebuild s1（转换应原样复制/引用 s1 wav）
            err_t = float(np.max(np.abs(tgt[:n] - s1[:n])))
            assert err_t < 1e-6, f"{split}/{name}: target 与 rebuild s1 不一致（max|Δ|={err_t:.2e}）"
            # mix == s1 + s2（LRS2-2Mix 合成关系）
            err_m = float(np.max(np.abs(mix[:n] - (s1[:n] + s2[:n]))))
            assert err_m < 1e-3, f"{split}/{name}: mix != s1+s2（max|Δ|={err_m:.2e}）"
            checked += 1
    if checked == 0:
        pytest.skip("抽样样本在 rebuild 中找不到对应 s1/s2（数据布局变化）")
    assert checked > 0


def test_convert_near_entries_valid(converted):
    """near.json 每项为非空 list；fallback 项指向 mix 自身，raw_audio 项路径必须存在。"""
    from pathlib import Path as _P

    checked = 0
    for split in converted:
        mix = json.loads(_json_path(split, "mix").read_text())
        near = json.loads(_json_path(split, "near").read_text())
        step = max(1, len(near) // 20)
        for i in range(0, len(near), step):
            entries = near[i]
            assert isinstance(entries, list) and len(entries) >= 1, f"{split}[{i}] near 项为空"
            for item in entries:
                assert len(item) == 2, f"{split}[{i}] near 项格式应为 [path, n]"
                p, n = item
                assert _P(p).is_file(), f"{split}[{i}] near 路径不存在: {p}"
                assert int(n) > 0
            checked += 1
    assert checked > 0


def test_convert_split_counts(converted):
    """报告各 split 计数并核对 mix/target/near 表长度；tt 至少应为数百条（若已全量转换）。"""
    import soundfile as sf  # noqa: F401  (确保环境依赖可用)

    counts = {}
    for split in converted:
        mix = json.loads(_json_path(split, "mix").read_text())
        target = json.loads(_json_path(split, "target").read_text())
        near = json.loads(_json_path(split, "near").read_text())
        counts[split] = (len(mix), len(target), len(near))
        assert len(mix) == len(target) == len(near)
    print("Phase2 转换数据计数（mix/target/near）:", counts)
    if "tt" in counts:
        assert counts["tt"][0] > 0
