# 理解文档规范（所有 subagent 必读）

本文件约定本轮“多 subagent 多轮理解 Falcon 项目”的产出格式、证据标准与协作规则。
目标：对项目达到**可验证、可复现、可传授**的完整理解，而不是泛泛的 README 复述。

## 1. 分析对象

- 仓库根：`/root/code/github_repos/Falcon-fork`
- 项目：Falcon —— NeurIPS 2026 论文《Efficient Streaming Audio-Visual Target Speaker Extraction for
  Real-World Acoustic Scenes》的官方代码，包含两条产品线：
  1. **训练/评测代码**（仓库根 + `look2hear/`，PyTorch Lightning）
  2. **Falcon Studio**（`studio/`，实时浏览器 Demo：FastAPI/WebSocket 服务 + 纯前端 JS 应用）
- 核心承诺（待验证）：因果（causal）单次 encoder–decoder 的 AVTSE；0.5M 可训练参数；RealSSA 等 4 个
  benchmark SOTA；Studio 端到端“2s 窗口 / 0.5s hop”实时推理。

## 2. 语言与风格

- 文档用**中文**撰写，专有名词、类名、函数名、配置键保持英文原文。
- 不要大段粘贴源码；引用代码用 `` `path:line` `` 或 `path:line-line`，必要时引用 ≤5 行关键片段。
- 结论先行：每节开头一句话结论，然后给证据。

## 3. 证据标准（最重要）

- 每条非显然的技术断言都必须给出 **`文件路径:行号`** 证据；跨文件结论给多条。
- 断言分三级并显式标注：
  - `[代码]`：直接读源码确认；
  - `[推断]`：由源码/文档合理推断，未被代码直接证明；
  - `[验证]`：通过实际运行命令/脚本确认（记录命令与输出摘要）。
- 不确定就写进“未解疑问”，**禁止**把猜测写成事实。
- 行号以当前 `main`（commit `7576854`）为准；若发现行号漂移，给出准确行号。

## 4. 每个子系统的文档模板（固定 8 节，便于交叉评审）

```
# <0X> <子系统名>
## 1. 子系统职责与总体位置
## 2. 文件清单与逐文件解析
   （每个文件：职责、关键类/函数签名、输入输出与张量形状/单位、关键常数）
## 3. 关键调用链与数据流（带 file:line）
## 4. 关键实现细节与设计动机（为什么这样设计；与论文/README 的对应）
## 5. 对外接口契约（谁调用它 / 它依赖谁 / 形状、采样率、帧率、dtype、坐标系约定）
## 6. 可验证断言表
   | 编号 | 断言 | 证据 path:line | 验证方式 | 置信度 |
## 7. 未解疑问 / 需其它子系统确认的问题（Q1..Qn，注明需谁确认）
## 8. 本轮验证活动记录（跑过的命令、结果、失败原因）
```

文档末尾维护 `## 9. 变更日志（Changelog）`：每轮 review 后追加“日期/轮次 + 改了什么 + 采纳了哪条 review 意见”。

## 5. 交叉评审文档模板（放在 `docs/reviews/roundN/`）

```
# 评审：<被评审文档> —— 评审人 <name>
## 1. 结论（通过 / 需修订：列 P0/P1 问题数）
## 2. 断言逐条核对表（✅正确 / ❌错误 / ❓无法确认；指出证据错误）
## 3. 错误清单（P0 事实错误 / P1 重要遗漏或误导 / P2 细节瑕疵），每条给证据与修正建议
## 4. 遗漏的重要行为（代码里有、文档没写）
## 5. 跨模块一致性问题（接口/形状/命名/文档 vs 代码）
## 6. 给作者的问题清单（需作者回答或补充验证）
```

## 6. 协作规则

- **禁止修改任何源码**（`*.py / *.js / *.cfg / *.yml / *.md` 源码与配置）；只允许写自己负责的
  `docs/understanding/0X-*.md` 与 `docs/reviews/**` 下分配给你的评审文件。不要动 `assets/`、
  `studio/web/vendor/`（38MB 第三方 MediaPipe 资产，只做清单级说明）。
- 写前先读；同一文件不要被两个 agent 同时编辑。若发现别人文档有疑问，写进自己的评审或
  `Q` 清单，不要直接改别人的文档。
- 所有 subagent 共享同一工作目录与文件系统，你写的文件对其他 agent 和 Lead 立即可见。
- 每个 teammate 在共享任务板上有一个任务，开始做前 `team_task_get` 拿到最新 revision 并 `claim`；
  阶段完成时汇报给 Lead（`send_message`），全部轮次结束后再 `complete`。

## 7. 环境事实（影响验证手段，务必遵守）

- **唯一允许的运行环境**：专属 conda 环境 `falcon`（`/root/miniforge3/envs/falcon`，Python 3.11，
  torch/pytorch_lightning/timm/fastapi/pytest 等已装齐）。**所有 Python 命令必须在该环境执行**，
  见仓库根 `AGENTS.md` 与 `docs/understanding/VERIFY-ENV.md`。
- 禁止使用：系统 `python3`（3.8，无依赖）、`/root/code/github_repos/Wav2TextGrid/.venv`、
  早期临时环境 `/root/.venvs/falcon`（**已删除**，仅历史日志引用）、其它项目的 conda env。
- `timm`/`typeguard` 的 stub（`docs/verification/stubs/`）仍保留给**无依赖场景**复跑历史脚本；
  在 `falcon` 环境中已装真包（`typeguard<3` 以兼容 `check_argument_types`），可不再注入 stub。
- `node v22` 可用：`node --check`、`node studio/tests/check_*.mjs`、`node studio/web/*.test.cjs`
  等纯 JS 测试可直接运行（**已知可通过**，例如 `node studio/tests/check_i18n.mjs`）。
- 网络可用（可访问 pypi / huggingface.co），但**不要下载大模型权重**、不要 pip 安装 torch；
  如确需安装小依赖，先向 Lead 报告。
- 未下载 Falcon 权重（`~/.cache/huggingface` 中无 Falcon），`pretrain_zoo/` 不存在，
  因此“加载真实权重 / 端到端跑通”默认不可做，改为静态核对 + 可运行的最小验证。
- 不要修改 git 状态（不要 commit / checkout / 清理），只读代码。

## 8. 轮次说明

- **R1 理解**：每人产出自己子系统的深度文档（写入 `docs/understanding/`）。
- **R2 交叉评审**：每人按 Lead 指派评审另一位同学的文档，产出 `docs/reviews/round1/`。
- **R3 修正深化**：作者按评审意见修订文档、回答 Q 清单、补齐遗漏。
- **R4 复核**：换不同评审人再评一轮（`docs/reviews/round2/`），并做一次“执行级验证”。
- **R5 汇总**：Lead 写主理解文档 `docs/understanding/FINAL_PROJECT_UNDERSTANDING.md` 并做最终一致性检查。
