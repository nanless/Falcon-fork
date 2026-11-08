# 贡献指南

本仓库是 [JusperLee/Falcon](https://github.com/JusperLee/Falcon) 的复现 fork：上游负责模型与论文，本仓库负责在 A800 机器上的复现、证据与工程化。下面的命令与数字都来自本机实测，依据是 [AGENTS.md](AGENTS.md)、[docs/STATUS.md](docs/STATUS.md) 和 `.github/workflows/tests.yml`。

## 1. 运行环境

- 依赖只装在 conda 环境 `falcon`（`/root/miniforge3/envs/falcon`）：Python 3.11、torch 2.8.0+cu128、`typeguard<3`。换其他解释器会在 `import look2hear.models` 处失败。
- 重建步骤见 [AGENTS.md](AGENTS.md) §5。`pypesq` 兼容层是 site-packages 里的手写文件，**不在仓库内**，新环境要按该节手动创建。
- 非交互式命令直接用 `/root/miniforge3/envs/falcon/bin/python`，不必先激活环境。

## 2. 测试

| 套件 | 有 GPU | 无 GPU |
|---|---|---|
| `pytest tools/repro/tests -q` | 168 passed，约 147 s | 168 passed，约 170 s |
| `pytest studio/tests -q` | 51 passed / 2 skipped | 53 passed / 0 skipped |
| Studio Node 测试 | 8 通过 | 8 通过 |

`tools/repro/tests` 在本机用 1 张 GPU（`CUDA_VISIBLE_DEVICES=0`）跑；它读 `/root/data`、`/root/.cache/falcon_weights`、`/root/tmp`，并校验本机环境锁，只有在这台机器上才有意义。

## 3. CI 跑什么、不跑什么

`.github/workflows/tests.yml` 有两个 job：

- `frontend`：8 个 Studio Node 测试、Space 组装与 i18n（226 条中文串）、Space 静态测试、环境锁文件检查。
- `python`：Falcon 训练/评测模块导入、`pytest studio/tests -q`、`pytest space/tests/test_clip.py -q -m "not slow"`。

**不跑** `tools/repro/tests`：它依赖本机数据与 conda 环境锁，在托管 runner 上必然失败（`5371df7` 起不再作为 CI 门禁）。这 168 条是本地合并门禁，PR 里不用等它变绿。

## 4. 提交流程

1. 动手前先记基线：`git status --short` 与 `git diff --stat`，改完再比一次。
2. 证据按追加方式维护：`docs/verification/**` 下的 csv/log/json 不覆盖已有行；修正结论时新增带日期的文件，写明替代哪一份、为什么。
3. 权重不进 Git：`.pt` / `.ckpt` 留在 `/root/.cache/falcon_weights`，提交会让每次 clone 多出近 180 MB，且历史删不干净。
4. 提交由用户决定：默认只改工作区，交付时报清改了哪些文件、各增删多少行。

## 5. 文档

- 中文按 `plain-chinese-writing` 的口径：结论先行、数字先行；事实要标清是实测、推断还是未验证。
- 改动落在 `docs/`，并在 [docs/CHANGELOG.md](docs/CHANGELOG.md) 记一行（提交号 + 一句话 + 影响面）。
- 证据代际与索引见 [docs/verification/README.md](docs/verification/README.md)；复现计划见 [docs/REPRODUCTION_PLAN.md](docs/REPRODUCTION_PLAN.md)。

## 6. 联系与上游

- 上游：`JusperLee/Falcon`（GitHub）。模型、论文与原始实现的问题转到上游。
- 本仓库的问题：在本仓库提 issue 或 PR，附上 `git rev-parse HEAD`、环境版本与复现命令。
