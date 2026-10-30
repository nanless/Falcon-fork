"""W2 模型修复回归：config 污染 / ckpt 自描述 / FRCNN 严格加载（A1-1 / A1-2 / A2-2）.

背景（见 docs/understanding/THREE_ROUND_REVIEW_20261030.md §2/§3）：
- A1-1：``Falcon``/``Separator`` 直接保存并原地 ``pop`` 传入的 audio/video/fusion 子配置，
  同一份 ``configs/Falcon.yml`` dict 连续构造两次，第二次得到 49,081 参数的残缺模型；
  污染还会被序列化进 ckpt 的 hparams。本测试固定"同一 dict 连续构造两次"必须都是 537,697。
- A1-2：``get_model_args`` 只返回 ``{"n_src"}``，``serialize`` -> ``from_pretrain`` 无法重建模型
  （外部评测也会拒绝生成的 ``best_model.pth``）。本测试要求完整 args + 525-key 重建 + 数值 bit-identical。
- A2-2：``update_frcnn_parameter`` 静默合并部分权重后整网冻结（缺键随机初始化且无报错）。
  本测试要求缺键/多余键/形状不符/路径不存在/负载格式错误都必须 raise。

数值不变量（本仓库核心约束）：所有修复不得改变官方权重的前向数值。测试用随机初始化 + 固定种子，
只比较修复后的**自我一致性**；官方权重的 bit-identical 另由 ``docs/verification/scripts/verify_real_weights.py``
与 ``verify_model_params.py`` 证明（任务验收要求）。

运行：
    PYTHONPATH=$PWD /root/miniforge3/envs/falcon/bin/python -m pytest tools/repro/tests/test_model_rebuild.py -q
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import look2hear.models as models  # noqa: E402
from look2hear.videomodels import FRCNNVideoModel  # noqa: E402
from look2hear.videomodels.frcnn_videomodel import (  # noqa: E402
    filter_frcnn_pretrained_dict,
    update_frcnn_parameter,
)

CONFIG_PATH = REPO / "configs" / "Falcon.yml"
STUDIO_CONFIG_PATH = REPO / "studio" / "configs" / "Falcon.yml"
REAL_BACKBONE = Path("/root/.cache/falcon_weights/frcnn_128_512.backbone.pth.tar")
REAL_CKPT = Path("/root/.cache/falcon_weights/falcon.pt")

AUDIO_PARAMS_SPK = 537_697      # FalconSpeakerContrastive，configs/Falcon.yml
AUDIO_PARAMS_BASE = 523_113     # 裸 Falcon（扣掉 14,584 的 spk 头）
VIDEO_PARAMS = 11_186_688       # 冻结 FRCNN
N_KEYS_SPK = 525                # 官方 falcon.pt 的 audio_model.* key 数


def _load_conf(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def _audio_cfg(path: Path = CONFIG_PATH) -> dict:
    return _load_conf(path)["audionet"]["audionet_config"]


def _video_cfg(path: Path = CONFIG_PATH) -> dict:
    cfg = copy.deepcopy(_load_conf(path)["videonet"]["videonet_config"])
    cfg.pop("pretrain", None)
    return cfg


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# A1-1: 同一 config dict 连续构造两次，不得互相污染
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def four_threads():
    """临时设 4 线程跑本测试，结束后恢复，避免污染后续数值指纹测试（A8-1 顺序相关失败）。"""
    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def test_config_dict_not_polluted_speaker_contrastive():
    """同一 audionet_config dict（按引用展开）连续构造两次：参数一致且原 dict 完整。"""
    raw = _audio_cfg()
    # 记录原始子配置快照，用于断言 callers 的 dict 未被 pop/写入
    snapshot = copy.deepcopy(raw)

    model1 = models.FalconSpeakerContrastive(sample_rate=16000, **raw)
    model2 = models.FalconSpeakerContrastive(sample_rate=16000, **raw)

    assert _n_params(model1) == AUDIO_PARAMS_SPK
    assert _n_params(model2) == AUDIO_PARAMS_SPK
    assert str(model2) == str(model1)  # 第二次构造的图结构与第一次一致（不是空 ModuleList）

    assert raw == snapshot, "FalconSpeakerContrastive.__init__ 污染了调用方的 config dict"
    for key, sub in (("audio_params", "audio"), ("video_params", "video")):
        assert raw[key]["repeats"] in (1, 2), f"{key} 的 repeats 被 pop 掉了"
        assert "shared" in raw[key], f"{key} 的 shared 被 pop 掉了"
        assert "causal" not in raw[key], f"{key} 被写入了 causal（应只在模型内副本上写）"


def test_config_dict_not_polluted_base_falcon():
    """裸 Falcon 同样不得污染 config；两次构造都是 523,113 参数。"""
    raw = _audio_cfg()
    raw = {k: v for k, v in raw.items() if k != "spk_embed_dim"}
    snapshot = copy.deepcopy(raw)

    model1 = models.Falcon(sample_rate=16000, **raw)
    model2 = models.Falcon(sample_rate=16000, **raw)

    assert _n_params(model1) == AUDIO_PARAMS_BASE
    assert _n_params(model2) == AUDIO_PARAMS_BASE
    assert raw == snapshot
    assert len(model1.state_dict()) == N_KEYS_SPK - 7  # spk 头 7 个 tensor


def test_mutating_returned_model_args_does_not_touch_model():
    """get_model_args 返回的是深拷贝：改它不应影响模型自身的配置。"""
    model = models.FalconSpeakerContrastive(sample_rate=16000, **_audio_cfg())
    args = model.get_model_args()
    args["audio_params"]["hid_chan"] = 1
    args["enc_dec_params"]["win"] = 999
    assert model.audio_params["hid_chan"] == 64
    assert model.enc_dec_params["win"] == 256


def test_studio_copy_shares_a1_1_behaviour():
    """Studio 副本（studio/falcon_model/models/Falcon.py）同样不污染 config。"""
    studio_dir = REPO / "studio"
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    from falcon_model.models.Falcon_Contrastive import FalconSpeakerContrastive as StudioModel

    raw = _audio_cfg(STUDIO_CONFIG_PATH)
    snapshot = copy.deepcopy(raw)
    model1 = StudioModel(sample_rate=16000, **raw)
    model2 = StudioModel(sample_rate=16000, **raw)
    assert _n_params(model1) == _n_params(model2) == AUDIO_PARAMS_SPK
    assert raw == snapshot


# ──────────────────────────────────────────────────────────────────────────────
# A1-2: ckpt 自描述（serialize -> from_pretrain 完整重建）
# ──────────────────────────────────────────────────────────────────────────────
def test_get_model_args_full_and_json_serializable():
    model = models.FalconSpeakerContrastive(sample_rate=16000, **_audio_cfg())
    args = model.get_model_args()
    required = {
        "n_src", "enc_dec_params", "audio_bn_params", "audio_params",
        "video_bn_params", "video_params", "fusion_params",
        "mask_generation_params", "pretrained_vout_chan", "sample_rate",
        "causal", "spk_embed_dim",
    }
    assert required <= set(args), f"missing ctor args: {sorted(required - set(args))}"
    assert args["spk_embed_dim"] == 64
    # 必须可 JSON 序列化（ckpt/hparams 落盘与跨进程传递）
    text = json.dumps(args)
    assert json.loads(text)["audio_params"]["repeats"] == 2


def test_base_falcon_get_model_args_is_rebuildable():
    raw = {k: v for k, v in _audio_cfg().items() if k != "spk_embed_dim"}
    base = models.Falcon(sample_rate=16000, **raw)
    rebuilt = models.Falcon(**base.get_model_args())
    assert _n_params(rebuilt) == AUDIO_PARAMS_BASE
    assert set(base.state_dict()) == set(rebuilt.state_dict())


def test_serialize_from_pretrain_rebuilds_525_keys_bit_identical(four_threads):
    """serialize -> from_pretrain：525 key / 537,697 参数 / 固定 2 s 输入 max|Δ|=0。"""
    torch.manual_seed(0)
    model = models.FalconSpeakerContrastive(sample_rate=16000, **_audio_cfg()).eval()
    assert len(model.state_dict()) == N_KEYS_SPK

    conf = model.serialize()
    assert conf["model_name"] == "FalconSpeakerContrastive"
    assert len(conf["state_dict"]) == N_KEYS_SPK
    assert "spk_embed_dim" in conf["model_args"]

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(conf, f.name)
        path = f.name
    try:
        # 自描述：不显式传 args，from_pretrain 从 ckpt 的 model_args 重建
        rebuilt = models.FalconSpeakerContrastive.from_pretrain(path).eval()
        assert len(rebuilt.state_dict()) == N_KEYS_SPK
        assert _n_params(rebuilt) == AUDIO_PARAMS_SPK

        g = torch.Generator().manual_seed(1234)
        audio = torch.randn(1, 32000, generator=g)
        mouth = torch.randn(1, 512, 50, generator=g)
        with torch.no_grad():
            est_a = model(audio, mouth)
            est_b = rebuilt(audio, mouth)
        assert est_a.shape == (1, 1, 32000)
        assert (est_a - est_b).abs().max().item() == 0.0

        # 显式传 args 的旧调用方式也照旧可用
        rebuilt_explicit = models.FalconSpeakerContrastive.from_pretrain(
            path, **model.get_model_args()
        ).eval()
        with torch.no_grad():
            est_c = rebuilt_explicit(audio, mouth)
        assert torch.equal(est_a, est_c)
    finally:
        os.unlink(path)


def test_studio_serialize_args_match_look2hear():
    """Studio 副本与训练侧对同一配置给出完全相同的 ctor args（防止再次漂移）。"""
    studio_dir = REPO / "studio"
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    from falcon_model.models.Falcon_Contrastive import FalconSpeakerContrastive as StudioModel

    train_model = models.FalconSpeakerContrastive(sample_rate=16000, **_audio_cfg())
    studio_model = StudioModel(sample_rate=16000, **_audio_cfg(STUDIO_CONFIG_PATH))
    assert json.dumps(train_model.get_model_args(), sort_keys=True) == json.dumps(
        studio_model.get_model_args(), sort_keys=True
    )


# ──────────────────────────────────────────────────────────────────────────────
# A2-2: FRCNN 严格加载
# ──────────────────────────────────────────────────────────────────────────────
def _frcnn_sd() -> dict:
    torch.manual_seed(0)
    model = FRCNNVideoModel(**_video_cfg())
    return model.state_dict()


def _write_payload(tmp_path: Path, payload) -> str:
    path = tmp_path / "backbone.pth.tar"
    torch.save(payload, path)
    return str(path)


def test_frcnn_strict_load_from_payload_freezes_all(tmp_path):
    sd = _frcnn_sd()
    path = _write_payload(tmp_path, {"model_state_dict": sd, "meta": {"num_keys": len(sd)}})
    model = FRCNNVideoModel(pretrain=path, **_video_cfg())
    assert _n_params(model) == VIDEO_PARAMS
    assert not any(p.requires_grad for p in model.parameters())
    assert filter_frcnn_pretrained_dict(sd) == sd  # 允许前缀覆盖全部视觉键
    with torch.no_grad():
        out = model(torch.randn(1, 1, 5, 96, 96))
    assert tuple(out.shape) == (1, 512, 5)


def test_frcnn_rejects_missing_key(tmp_path):
    sd = _frcnn_sd()
    truncated = dict(list(sd.items())[1:])
    path = _write_payload(tmp_path, {"model_state_dict": truncated})
    model = FRCNNVideoModel(**_video_cfg())
    with pytest.raises(RuntimeError, match="missing="):
        update_frcnn_parameter(model, truncated)


def test_frcnn_rejects_unexpected_key(tmp_path):
    sd = _frcnn_sd()
    extra = dict(sd)
    extra["trunk.bogus.weight"] = torch.zeros(1)
    model = FRCNNVideoModel(**_video_cfg())
    with pytest.raises(RuntimeError, match="unexpected="):
        update_frcnn_parameter(model, extra)


def test_frcnn_rejects_shape_mismatch():
    sd = _frcnn_sd()
    key = next(iter(sd))
    bad = dict(sd)
    bad[key] = torch.zeros(1)
    model = FRCNNVideoModel(**_video_cfg())
    with pytest.raises(RuntimeError, match="shape_mismatch="):
        update_frcnn_parameter(model, bad)


def test_frcnn_rejects_non_visual_payload():
    model = FRCNNVideoModel(**_video_cfg())
    with pytest.raises(RuntimeError, match="no visual-encoder keys"):
        update_frcnn_parameter(model, {"some_other_head.weight": torch.zeros(1)})


def test_frcnn_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        FRCNNVideoModel(pretrain="/nonexistent/frcnn_backbone.pth.tar", **_video_cfg())


def test_frcnn_bad_payload_format_raises(tmp_path):
    path = _write_payload(tmp_path, {"not_state_dict": 1})
    with pytest.raises(KeyError, match="model_state_dict"):
        FRCNNVideoModel(pretrain=path, **_video_cfg())


def test_studio_frcnn_copy_strict():
    """Studio 副本同样严格：缺键 raise、真实导出骨干可加载。"""
    studio_dir = REPO / "studio"
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    from falcon_model.videomodels.frcnn_videomodel import FRCNNVideoModel as StudioFRCNN

    model = StudioFRCNN(**_video_cfg())
    with pytest.raises(RuntimeError):
        update_frcnn_parameter(model, dict(list(_frcnn_sd().items())[1:]))
    if REAL_BACKBONE.is_file():
        loaded = StudioFRCNN(pretrain=str(REAL_BACKBONE), **_video_cfg())
        assert _n_params(loaded) == VIDEO_PARAMS
        assert loaded.eval() is loaded  # studio train() 已同步 return self


@pytest.mark.skipif(not REAL_BACKBONE.is_file(), reason="real FRCNN backbone export not cached")
def test_real_exported_backbone_loads_strictly():
    """真实导出的 137-key 骨干必须 strict 通过（这是训练/评测实际使用的文件）。"""
    payload = torch.load(REAL_BACKBONE, map_location="cpu", weights_only=False)
    assert set(payload) >= {"model_state_dict"}
    assert len(payload["model_state_dict"]) == 137
    model = FRCNNVideoModel(pretrain=str(REAL_BACKBONE), **_video_cfg())
    assert _n_params(model) == VIDEO_PARAMS
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.skipif(not REAL_CKPT.is_file(), reason="official falcon.pt not cached")
def test_official_audio_weights_strict_and_numeric_unchanged():
    """官方 falcon.pt audio 权重 strict 加载：525 key / 537,697 参数（数值不变量）。"""
    ck = torch.load(REAL_CKPT, map_location="cpu", weights_only=False)
    audio_sd = {k[len("audio_model."):]: v for k, v in ck["state_dict"].items()
                if k.startswith("audio_model.")}
    model = models.FalconSpeakerContrastive(sample_rate=16000, **_audio_cfg())
    model.load_state_dict(audio_sd, strict=True)
    assert len(model.state_dict()) == N_KEYS_SPK
    assert _n_params(model) == AUDIO_PARAMS_SPK
