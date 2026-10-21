#!/usr/bin/env python3
"""Falcon 复现阶段 1 · 统一数据获取脚本（极小依赖：huggingface_hub + curl）。

数据集（HF dataset repos）：
  lrs2-2mix      JusperLee/LRS2-2Mix      公开，单文件 lrs2.tar.gz ≈16.47 GB（纯音频 + 重建脚本）
  realssa-sim    JusperLee/RealSSA-Sim    gated(manual)，242 shards ≈521.5 GB（4ch 16k + mouth crops）
  realssa-real   JusperLee/RealSSA-Real   gated(manual)，422 文件 ≈12.84 GB（真实场景 AV + enrollment）
  （LRS3-2Mix / VoxCeleb2-2Mix 的 HF 仓库只有 README，无数据文件 → 不在本脚本下载范围。）

设计原则：
  * 公开数据集直接下载；gated 数据集**不自动申请**，只打印申请入口与所需步骤。
  * LRS2 的 16.47 GB 大文件使用 curl 断点续传（中断后重跑同一命令即可续传）。
  * RealSSA-Sim 默认只取 README + 1 个 test shard（≈2.4 GB）作为管线验证；全量请显式 `--full`（522 GB）。
  * `--dry-run` 不写盘：列出匹配文件、大小、gated/授权状态与申请链接。
  * 支持 HF_ENDPOINT（例如 https://hf-mirror.com）与标准 http(s)_proxy 环境变量；
    本机实测 hf-mirror.com TLS 不可达，走 huggingface.co + 本机 mihomo 代理（见 docs/DATA_ACCESS_GUIDE.md）。

用法：
  python tools/repro/fetch_data.py --dataset lrs2-2mix --dry-run
  python tools/repro/fetch_data.py --dataset lrs2-2mix                 # 断点续传下载到 /root/data/lrs2_2mix
  python tools/repro/fetch_data.py --dataset realssa-sim --dry-run     # 查看 gated 状态与申请入口
  python tools/repro/fetch_data.py --dataset realssa-sim --shard tt-000000   # 获批后只取一个 test shard
  python tools/repro/fetch_data.py --dataset realssa-real --full       # 获批后全量（≈12.84 GB）
  python tools/repro/fetch_data.py --dataset all --dry-run
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import time

try:
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少 huggingface_hub：请用 falcon 环境运行，或 pip install huggingface_hub>=2\n")
    raise SystemExit(1)

DATASETS = {
    "lrs2-2mix": {
        "repo": "JusperLee/LRS2-2Mix",
        "repo_type": "dataset",
        "gated": False,
        "default_out": "/root/data/lrs2_2mix",
        "default_patterns": ["lrs2.tar.gz"],
        "expected_bytes": 16470508590,
        "probe": "lrs2.tar.gz",
        "access_url": "https://huggingface.co/datasets/JusperLee/LRS2-2Mix",
        "note": "公开；单文件 lrs2.tar.gz（16.47 GB）内含 lrs2_rebuild/（纯音频，无视频/嘴部）",
    },
    "realssa-sim": {
        "repo": "JusperLee/RealSSA-Sim",
        "repo_type": "dataset",
        "gated": "manual",
        "default_out": "/root/data/realssa_sim",
        "default_patterns": ["README.md", "tt/tt-000000.tar"],
        "expected_bytes": 521532645652,
        "probe": "tt/tt-000000.tar",
        "access_url": "https://huggingface.co/datasets/JusperLee/RealSSA-Sim",
        "note": "gated(manual)；全量 242 shards≈521.5 GB；默认仅 README+1 个 test shard 作为管线验证",
    },
    "realssa-real": {
        "repo": "JusperLee/RealSSA-Real",
        "repo_type": "dataset",
        "gated": "manual",
        "default_out": "/root/data/realssa_real",
        "default_patterns": ["README.md", "INFERENCE.md"],
        "expected_bytes": 12843097039,
        "probe": "INFERENCE.md",
        "access_url": "https://huggingface.co/datasets/JusperLee/RealSSA-Real",
        "note": "gated(manual)；全量 422 文件≈12.84 GB；含 faces/mouth/landmarks/enrollment（12 GB 为 raw/ 原始视频）",
    },
}


def human_bytes(n: int | None) -> str:
    """十进制单位（与 HF 数据集卡片的 GB 口径一致：1 GB = 10^9 B）。"""
    if not n:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return f"{n} B"


def resolve_endpoint(args) -> str:
    return args.endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co"


def setup_proxy(args) -> None:
    """把 --proxy 写进 HTTP(S)_PROXY，令 huggingface_hub(requests) 与 curl 都使用它。"""
    if not args.proxy:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(key, args.proxy)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")
    print(f"[proxy] 使用 {args.proxy}（仅本进程）")


def repo_files(api: HfApi, spec: dict) -> tuple[list[tuple[str, int]], str | None]:
    """返回 [(rfilename, size)] 与 gated 状态；metadata 对 gated 数据集仍可读。"""
    try:
        info = api.dataset_info(spec["repo"], files_metadata=True)
    except RepositoryNotFoundError:
        raise SystemExit(
            f"[ERROR] 数据集 {spec['repo']} 不存在或无权访问（RepositoryNotFound）。"
            f" 检查 HF_ENDPOINT/网络：{spec['access_url']}"
        )
    except HfHubHTTPError as exc:
        raise SystemExit(f"[ERROR] 读取 {spec['repo']} 元数据失败：{exc}（可能网络/代理问题）")
    files = [(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
    return files, info.gated


def match_patterns(files: list[tuple[str, int]], patterns: list[str]) -> list[tuple[str, int]]:
    matched = []
    for name, size in files:
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            matched.append((name, size))
    # 去重并保持稳定顺序
    seen, out = set(), []
    for name, size in matched:
        if name not in seen:
            seen.add(name)
            out.append((name, size))
    return out


def shard_patterns(shards: list[str]) -> list[str]:
    patterns = []
    for raw in shards:
        token = raw.strip().rstrip("/")
        if not token:
            continue
        if token.endswith(".tar") and "/" in token:
            patterns.append(token)
        elif token in ("tr", "cv", "tt"):
            patterns.append(f"{token}/*.tar")
        elif "/" in token:  # e.g. tt/tt-000000.tar
            patterns.append(token)
        else:  # e.g. tt-000000
            patterns.append(f"*/{token}.tar")
    return patterns


def check_gated_access(spec: dict, endpoint: str) -> tuple[bool, str | None]:
    """read-only 探测：dry_run HEAD 一个非 README 文件，判断当前 token 是否已获批。"""
    probe = spec["probe"]
    try:
        info = hf_hub_download(
            repo_id=spec["repo"], filename=probe, repo_type=spec["repo_type"],
            endpoint=endpoint, dry_run=True,
        )
        return True, f"dry_run OK, file_size={getattr(info, 'file_size', '?')}"
    except GatedRepoError:
        return False, "GatedRepoError 403：当前 HF 账号未获批准"
    except Exception as exc:  # 网络错误等
        return False, f"{type(exc).__name__}: {str(exc)[:180]}"


def print_gated_instructions(spec: dict) -> None:
    print(f"\n  [GATED] {spec['repo']}（{spec['gated']}）当前账号无下载权限。")
    print(f"  1) 打开申请页：{spec['access_url']}")
    print("  2) 填写申请人/单位/国家/用途，勾选非商用学术研究/不转发/遵守源许可，等待人工批准。")
    print("  3) 批准后在本机执行 `hf auth login`（或设置 HF_TOKEN 环境变量）再重跑本脚本。")
    print("  4) 申请与审批由数据集所有者人工处理；本脚本不会提交申请，也不会绕过 gated 限制。")


def resumable_download(url: str, out_path: str, expected_size: int | None, args) -> bool:
    """公开大文件：curl -C - 断点续传；返回是否成功且大小吻合。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    have = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    if expected_size and have >= expected_size:
        print(f"[skip] 已存在完整文件：{out_path}（{human_bytes(have)}）")
        return True
    if have:
        print(f"[resume] {out_path} 已有 {human_bytes(have)}，从断点继续")
    cmd = [
        "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
        "-C", "-", "--connect-timeout", "30", "--speed-time", "90", "--speed-limit", "65536",
        "-sS", "-o", out_path,
    ]
    if args.max_time:
        cmd += ["--max-time", str(args.max_time)]
    cmd.append(url)
    print(f"[download] curl -C - ... {os.path.basename(out_path)}  ← {url.split('?')[0]}")
    started = time.time()
    rc = subprocess.call(cmd)
    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    elapsed = max(time.time() - started, 1e-6)
    print(f"[curl] rc={rc} bytes={size} ({human_bytes(size)}) 用时 {elapsed:.1f}s 平均 {human_bytes(int(size / elapsed))}/s")
    if rc != 0:
        return False
    if expected_size and size < expected_size:
        print(f"[WARN] 大小不匹配：{size} < {expected_size}，可重跑本命令续传")
        return False
    return True


def plan_dataset(args, name: str, endpoint: str, out_root: str | None = None) -> int:
    spec = DATASETS[name]
    if out_root:
        out_dir = out_root
    elif args.out and len(DATASETS) == 1:
        out_dir = args.out
    else:
        out_dir = spec["default_out"]
    api = HfApi(endpoint=endpoint)
    print("=" * 78)
    print(f"[{name}] {spec['repo']}  gated={spec['gated']}  {spec['note']}")
    print(f"  endpoint = {endpoint}")
    print(f"  out      = {out_dir}")

    files, gated = repo_files(api, spec)
    if args.pattern:
        patterns = list(args.pattern)
    elif args.shard:
        patterns = shard_patterns(args.shard)
    elif args.full:
        patterns = ["*"]
    else:
        patterns = spec["default_patterns"]
    matched = match_patterns(files, patterns)
    if not matched:
        print(f"  [ERROR] 没有文件匹配 patterns={patterns}；可先 `--dry-run` 查看文件列表片段")
        return 1
    total = sum(size for _, size in matched)
    print(f"  patterns = {patterns}")
    print(f"  matched  = {len(matched)} 个文件，共 {human_bytes(total)}")
    for fname, fsize in matched[:12]:
        print(f"     - {fname}  {human_bytes(fsize)}")
    if len(matched) > 12:
        print(f"     ... 其余 {len(matched) - 12} 个")
    if args.full and name == "realssa-sim" and not args.dry_run:
        print(f"  [WARN] --full 将下载 {human_bytes(spec['expected_bytes'])}；请确认磁盘（建议 ≥600 GB 空余）")
    if args.dry_run:
        if gated:
            ok, why = check_gated_access(spec, endpoint)
            print(f"  gated 授权检查：{'✅ 已获批' if ok else '❌ 未获批'}（{why}）")
            if not ok:
                print_gated_instructions(spec)
        else:
            print("  gated = False（公开，可直接下载）")
        print("  [dry-run] 未写入磁盘。")
        return 0

    if gated:
        ok, why = check_gated_access(spec, endpoint)
        if not ok:
            print(f"  授权检查失败：{why}")
            print_gated_instructions(spec)
            return 3

    # 实际下载
    if name == "lrs2-2mix" and len(matched) == 1 and matched[0][0] == "lrs2.tar.gz":
        url = hf_hub_url(spec["repo"], "lrs2.tar.gz", repo_type="dataset", endpoint=endpoint)
        ok = resumable_download(url, os.path.join(out_dir, "lrs2.tar.gz"), spec["expected_bytes"], args)
        return 0 if ok else 1

    try:
        snapshot_download(
            repo_id=spec["repo"], repo_type=spec["repo_type"], local_dir=out_dir,
            allow_patterns=[fname for fname, _ in matched],
            endpoint=endpoint, max_workers=args.max_workers,
        )
    except GatedRepoError:
        print_gated_instructions(spec)
        return 3
    except HfHubHTTPError as exc:
        print(f"  [ERROR] 下载失败：{exc}")
        return 1
    print(f"  [done] 已保存到 {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Falcon 数据获取（LRS2-2Mix 公开 / RealSSA-Sim、RealSSA-Real gated）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["realssa-sim", "realssa-real", "lrs2-2mix", "all"], required=True)
    parser.add_argument("--out", default=None,
                        help="输出根目录；单个数据集时直接用它（默认见内置 default_out），all 时为父目录")
    parser.add_argument("--pattern", action="append", default=None,
                        help="repo 内文件 glob，可重复（如 --pattern 'tt/*.tar'）；与 --shard/--full 互斥")
    parser.add_argument("--shard", action="append", default=None,
                        help="RealSSA shard 简写：tt / tt-000000 / tt/tt-000000.tar，可重复")
    parser.add_argument("--full", action="store_true", help="下载整个 snapshot（RealSSA-Sim 约 522 GB！）")
    parser.add_argument("--dry-run", action="store_true", help="只列出文件/大小/gated 状态，不写盘")
    parser.add_argument("--endpoint", default=None, help="覆盖 HF_ENDPOINT（默认取环境变量或 huggingface.co）")
    parser.add_argument("--proxy", default=None, help="HTTP(S) 代理，如 http://127.0.0.1:17897")
    parser.add_argument("--max-workers", type=int, default=4, help="并行下载数（非 LRS2 大文件）")
    parser.add_argument("--max-time", type=int, default=0, help="curl 单次最长秒数（0=不限）")
    args = parser.parse_args(argv)

    if args.pattern and (args.shard or args.full):
        parser.error("--pattern 不能与 --shard/--full 同时使用")
    setup_proxy(args)
    endpoint = resolve_endpoint(args)
    if "hf-mirror.com" in endpoint:
        print("[warn] 检测到 HF_ENDPOINT=hf-mirror.com；本机沙箱实测该域名 TLS 不可达，")
        print("       若失败可改 --endpoint https://huggingface.co 并配置代理（--proxy 或 http(s)_proxy）。")

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = {}
    for name in names:
        if args.out and len(names) > 1:
            out_root = os.path.join(args.out, name)
        elif args.out and len(names) == 1:
            out_root = args.out
        else:
            out_root = None
        results[name] = plan_dataset(args, name, endpoint, out_root)
    print("=" * 78)
    for name, rc in results.items():
        print(f"  {name:14s} -> {'OK' if rc == 0 else ('GATED/NO-ACCESS' if rc == 3 else 'FAILED')} (rc={rc})")
    return 0 if all(rc == 0 for rc in results.values()) else (3 if 3 in results.values() else 1)


if __name__ == "__main__":
    raise SystemExit(main())
