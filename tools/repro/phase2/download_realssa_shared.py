#!/usr/bin/env python3
"""把 RealSSA-Sim / RealSSA-Real 下载到组内共享存储，下载完成后迁移到 TOS 冷端。

规则（按用户 2026-10-23 指令）：
- 下载目录：/root/group-shared/voiceprint/data/multimodal/{realssa_sim,realssa_real}
- 优先镜像；镜像不可用时走本机 mihomo 代理（http://127.0.0.1:17897）。
  本机实测：hf-mirror 对 gated 数据 redirect 后不可用（超时），huggingface.co 直连超时，
  代理单流 ~8 MB/s 可用 ⇒ 默认用代理。
- 实时进度：`_download/realssa_status.json`（5s 更新）+ `_download/realssa_monitor.log`（60s 一行）。
- 全部完成后：`rsync --remove-source-files` 迁移到
  /aispeech-cfs/voiceprint/data/multimodal/{realssa_sim,realssa_real}，写 `_archive_done.json`。

用法：
  python download_realssa_shared.py --stage all            # 下载 + 归档（后台跑）
  python download_realssa_shared.py --stage download --only real
  python download_realssa_shared.py --stage archive        # 只做迁移
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- 固定配置 -------------------------------------------------------------
HF_ENDPOINT = "https://huggingface.co"
PROXY = "http://127.0.0.1:17897"
SHARED = Path("/root/group-shared/voiceprint/data/multimodal")
TOS = Path("/aispeech-cfs/voiceprint/data/multimodal")
STATUS_DIR = SHARED / "_download"
TOKEN_FILE = Path.home() / ".cache/huggingface/token"

REPOS = {
    "real": {
        "repo_id": "JusperLee/RealSSA-Real",
        "name": "realssa_real",
        "patterns": None,  # 全部文件
        "local_seed": Path("/root/data/RealSSA-Real"),
        "seed_subsets": None,
    },
    "sim": {
        "repo_id": "JusperLee/RealSSA-Sim",
        "name": "realssa_sim",
        "patterns": ("README.md", ".gitattributes", "tr/", "cv/", "tt/"),
        "local_seed": Path("/root/data/RealSSA-Sim"),
        "seed_subsets": ("tt", "README.md", ".gitattributes"),
    },
}

CHUNK = 1 << 20  # 1 MiB
LOCK = threading.Lock()
PROGRESS: dict[str, dict[str, int]] = {}   # rel -> {"size","done"}
STOP = threading.Event()
CURRENT: dict[str, object] = {"repo": None, "phase": "idle", "total_files": 0, "done_files": 0,
                              "expected_bytes": 0, "started": time.time(), "last_bytes": 0,
                              "last_t": time.time(), "speed": 0.0, "note": ""}


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    try:
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATUS_DIR / "realssa_download.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def token() -> str:
    return TOKEN_FILE.read_text().strip()


def api_files(repo_id: str) -> list[dict]:
    """返回 [{path,size,lfs_sha256}]，用 HF API 的 recursive tree。"""
    from huggingface_hub import HfApi
    api = HfApi(endpoint=HF_ENDPOINT, token=token())
    out = []
    for f in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        if type(f).__name__ != "RepoFile":
            continue
        sha = None
        lfs = getattr(f, "lfs", None)
        if isinstance(lfs, dict):
            sha = lfs.get("oid")
        elif lfs is not None:
            sha = getattr(lfs, "oid", None)
        out.append({"path": f.path, "size": int(f.size or 0), "lfs_sha256": sha})
    return out


def pick(meta: dict, files: list[dict]) -> list[dict]:
    pats = meta["patterns"]
    if not pats:
        return files
    sel = [f for f in files if f["path"] in pats or any(f["path"].startswith(p) for p in pats if p.endswith("/"))]
    return sel


def seed(meta: dict) -> None:
    """把本地已下载的文件先拷进目标目录，减少重复下载。"""
    src = meta["local_seed"]
    dst = SHARED / meta["name"]
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    subsets = meta["seed_subsets"]
    if subsets is None:
        log(f"seed {meta['name']}: rsync 全部（排除 .cache）")
        subprocess.run(["rsync", "-a", "--exclude", ".cache", f"{src}/", f"{dst}/"], check=False)
    else:
        for sub in subsets:
            p = src / sub
            if not p.exists():
                continue
            if p.is_dir():
                subprocess.run(["rsync", "-a", f"{p}/", f"{dst}/{sub}/"], check=False)
            else:
                shutil.copy2(p, dst / sub)
        log(f"seed {meta['name']}: {','.join(subsets)}")


def download_one(session, repo: str, rel: str, dest: Path, size: int) -> None:
    """带 Range 续传的下载；完成后原子替换。"""
    from requests.exceptions import RequestException
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".incomplete")
    url = f"{HF_ENDPOINT}/datasets/{repo}/resolve/main/{rel}?download=true"
    headers = {"Authorization": f"Bearer {token()}", "Accept-Encoding": "identity"}
    for attempt in range(1, 8):
        start = part.stat().st_size if part.exists() else 0
        if start > size:
            part.unlink(); start = 0
        if start == size and size > 0:
            os.replace(part, dest)
            with LOCK:
                PROGRESS[rel] = {"size": size, "done": size}
            return
        h = dict(headers)
        if start:
            h["Range"] = f"bytes={start}-"
        try:
            with session.get(url, headers=h, stream=True, timeout=(20, 180),
                             proxies={"http": PROXY, "https": PROXY}, allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    raise RequestException(f"HTTP {r.status_code}")
                mode = "ab" if start and r.status_code == 206 else "wb"
                if mode == "wb":
                    start = 0
                done = start
                with open(part, mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        with LOCK:
                            PROGRESS[rel] = {"size": size, "done": done}
                if size and part.stat().st_size != size:
                    raise RequestException(f"size mismatch {part.stat().st_size} != {size}")
            os.replace(part, dest)
            with LOCK:
                PROGRESS[rel] = {"size": size, "done": size}
            return
        except Exception as e:  # noqa: BLE001
            log(f"retry {rel} attempt={attempt}: {type(e).__name__}: {str(e)[:160]}")
            time.sleep(min(60, 3 * attempt))


def status_writer() -> None:
    while not STOP.wait(5):
        with LOCK:
            total_expected = CURRENT["expected_bytes"] or 0
            done = sum(v["done"] for v in PROGRESS.values())
            n_done = sum(1 for v in PROGRESS.values() if v["size"] and v["done"] >= v["size"])
            now = time.time()
            dt = max(1e-6, now - CURRENT["last_t"])
            inst = (done - CURRENT["last_bytes"]) / dt
            CURRENT["speed"] = inst
            CURRENT["last_bytes"] = done
            CURRENT["last_t"] = now
            pct = (100.0 * done / total_expected) if total_expected else 0.0
            eta = (total_expected - done) / inst if inst > 1e3 and total_expected else 0
            snap = {
                "updated": time.strftime("%FT%TZ", time.gmtime(now)),
                "repo": CURRENT["repo"], "phase": CURRENT["phase"],
                "total_files": CURRENT["total_files"], "done_files": n_done,
                "expected_bytes": total_expected, "downloaded_bytes": done,
                "percent": round(pct, 2),
                "speed_mbps": round(inst / 1e6, 2),
                "speed_mibps": round(inst / 1048576, 2),
                "eta_seconds": int(eta),
                "note": CURRENT["note"],
                "files": {k: {"size": v["size"], "done": v["done"]} for k, v in PROGRESS.items()},
            }
            STATUS_DIR.mkdir(parents=True, exist_ok=True)
            tmp = STATUS_DIR / "realssa_status.json.tmp"
            tmp.write_text(json.dumps(snap, ensure_ascii=False, indent=1))
            tmp.replace(STATUS_DIR / "realssa_status.json")
        if int(time.time()) % 60 < 6:
            line = (f"progress repo={snap['repo']} phase={snap['phase']} "
                    f"{snap['downloaded_bytes']/1e9:.1f}/{snap['expected_bytes']/1e9:.1f} GB "
                    f"({snap['percent']:.1f}%) {snap['speed_mibps']:.1f} MiB/s "
                    f"files={snap['done_files']}/{snap['total_files']} "
                    f"eta={snap['eta_seconds']/3600:.1f}h {snap['note']}")
            try:
                with open(STATUS_DIR / "realssa_monitor.log", "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%F %T')}] {line}\n")
            except Exception:
                pass


def run_repo(key: str, workers: int) -> bool:
    import requests
    meta = REPOS[key]
    meta_name = meta["name"]
    files = pick(meta, api_files(meta["repo_id"]))
    total = sum(f["size"] for f in files)
    log(f"=== {key} ({meta['repo_id']}) files={len(files)} expected={total/1e9:.2f} GB -> {SHARED/meta_name}")
    seed(meta)
    with LOCK:
        PROGRESS.clear()
        CURRENT.update({"repo": key, "phase": "download", "total_files": len(files),
                        "expected_bytes": total, "started": time.time(),
                        "last_bytes": 0, "last_t": time.time(), "note": ""})
    dest_root = SHARED / meta_name
    todo = []
    for f in files:
        d = dest_root / f["path"]
        if d.exists() and d.stat().st_size == f["size"]:
            with LOCK:
                PROGRESS[f["path"]] = {"size": f["size"], "done": f["size"]}
            continue
        todo.append(f)
    log(f"{key}: already-present={len(files)-len(todo)}, to-download={len(todo)}")
    if not todo:
        return True
    sessions = [requests.Session() for _ in range(workers)]

    def worker(i: int, item: dict) -> None:
        download_one(sessions[i % workers], meta["repo_id"], item["path"],
                     dest_root / item["path"], item["size"])

    ok = True
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, i, it) for i, it in enumerate(todo)]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                ok = False
                log(f"{key}: file failed: {type(e).__name__}: {str(e)[:200]}")
    # 校验尺寸
    bad = []
    for f in files:
        d = dest_root / f["path"]
        if not d.exists() or d.stat().st_size != f["size"]:
            bad.append(f["path"])
    manifest = {"repo_id": meta["repo_id"], "dir": str(dest_root), "files": len(files),
                "bytes": total, "bad": bad, "updated": time.strftime("%FT%TZ", time.gmtime())}
    (dest_root / "_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    log(f"{key}: done. bad={len(bad)} {bad[:3]}")
    return ok and not bad


def archive() -> None:
    STATS = []
    for key, meta in REPOS.items():
        src = SHARED / meta["name"]
        dst = TOS / meta["name"]
        if not src.exists():
            log(f"archive: {src} 不存在，跳过")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        with LOCK:
            PROGRESS.clear()
            CURRENT.update({"repo": key, "phase": "archive", "total_files": 0,
                            "expected_bytes": sum(p.stat().st_size for p in src.rglob('*') if p.is_file()),
                            "started": time.time(), "last_bytes": 0, "last_t": time.time(),
                            "note": "rsync -> TOS"})
        log(f"archive {key}: rsync {src} -> {dst}")
        r = subprocess.run(["rsync", "-a", "--remove-source-files", "--info=progress2",
                            f"{src}/", f"{dst}/"], check=False)
        # 清理空目录
        subprocess.run(["find", str(src), "-type", "d", "-empty", "-delete"], check=False)
        ok = r.returncode == 0 and not src.exists()
        log(f"archive {key}: rsync rc={r.returncode} src_removed={not src.exists()}")
        STATS.append({"dataset": key, "src": str(src), "dst": str(dst), "ok": ok})
    (STATUS_DIR / "_archive_done.json").write_text(json.dumps({
        "finished": time.strftime("%FT%TZ", time.gmtime()),
        "results": STATS, "tos_root": str(TOS)}, ensure_ascii=False, indent=1))
    log(f"archive done: {STATS}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["download", "archive", "all"], default="all")
    ap.add_argument("--only", choices=["real", "sim", "both"], default="both")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.environ["HTTPS_PROXY"] = PROXY
    os.environ["HTTP_PROXY"] = PROXY
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (TOS).mkdir(parents=True, exist_ok=True)
    log(f"START stage={args.stage} only={args.only} workers={args.workers} proxy={PROXY}")
    t = threading.Thread(target=status_writer, daemon=True)
    t.start()
    ok = True
    if args.stage in ("download", "all"):
        keys = ["real", "sim"] if args.only == "both" else [args.only]
        for k in keys:
            if not run_repo(k, args.workers):
                ok = False
                log(f"{k}: download 未完全成功，继续其它；归档阶段会跳过失败项")
    if args.stage in ("archive", "all") and ok:
        archive()
    elif args.stage in ("archive", "all") and not ok:
        log("跳过归档：下载未全部成功")
    STOP.set()
    log(f"EXIT ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
