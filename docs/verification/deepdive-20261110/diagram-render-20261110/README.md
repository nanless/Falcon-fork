# Mermaid 流程图渲染复核（2026-11-10）

逐一提取项目 Markdown 中的 Mermaid 代码块，用 Mermaid 11.17.2 与本机 Chrome 156.0.8078.80 渲染为 SVG，再生成下列 PNG 并查看。每张表中的编号对应当次提取的源码位置；后续改动可能使行号移动，源码摘要可用于判断图是否变化。原历史状态图的重叠已修正并拆分，其原稿以纯文本留存。

截图展示当次源码的渲染布局，业务含义见相邻说明与代码。截图采用白底和 Mermaid 默认主题，按图自身尺寸保存。

| 图 | 文档及位置 | 图源码 SHA-256 前 12 位 | 图像尺寸（px） | 渲染图 |
| --- | --- | --- | --- | --- |
| 01 | [README.md:21](../../../../README.md) | `104f0819efab` | 542 × 718 | [PNG](01.png) |
| 02 | [README_zh.md:26](../../../../README_zh.md) | `c6542fc8b837` | 490 × 742 | [PNG](02.png) |
| 03 | [README_zh.md:142](../../../../README_zh.md) | `3a1c47b3556c` | 1187 × 174 | [PNG](03.png) |
| 04 | [docs/REPRODUCTION_REPORT.md:12](../../../REPRODUCTION_REPORT.md) | `e160b2a3788b` | 538 × 614 | [PNG](04.png) |
| 05 | [docs/understanding/07-studio-frontend.md:341](../../../understanding/07-studio-frontend.md) | `06723bb98177` | 313 × 828 | [PNG](05.png) |
| 06 | [docs/understanding/07-studio-frontend.md:360](../../../understanding/07-studio-frontend.md) | `ccd9695fafb6` | 293 × 244 | [PNG](06.png) |
| 07 | [docs/understanding/20261110-data-and-training.md:10](../../../understanding/20261110-data-and-training.md) | `65e9f29dace3` | 1336 × 438 | [PNG](07.png) |
| 08 | [docs/understanding/20261110-data-and-training.md:66](../../../understanding/20261110-data-and-training.md) | `defeab92e780` | 808 × 774 | [PNG](08.png) |
| 09 | [docs/understanding/20261110-data-and-training.md:121](../../../understanding/20261110-data-and-training.md) | `c12be7ac5f0a` | 862 × 918 | [PNG](09.png) |
| 10 | [docs/understanding/20261110-evaluation-and-apps.md:10](../../../understanding/20261110-evaluation-and-apps.md) | `63f13cccc219` | 948 × 718 | [PNG](10.png) |
| 11 | [docs/understanding/20261110-evaluation-and-apps.md:125](../../../understanding/20261110-evaluation-and-apps.md) | `9a3ed07c0ef5` | 974 × 663 | [PNG](11.png) |
| 12 | [docs/understanding/20261110-model-and-streaming.md:10](../../../understanding/20261110-model-and-streaming.md) | `348345200843` | 500 × 1102 | [PNG](12.png) |
| 13 | [docs/understanding/20261110-model-and-streaming.md:71](../../../understanding/20261110-model-and-streaming.md) | `4769629cb28d` | 629 × 1230 | [PNG](13.png) |
| 14 | [docs/understanding/20261110-model-and-streaming.md:120](../../../understanding/20261110-model-and-streaming.md) | `2091574e414e` | 1138 × 210 | [PNG](14.png) |
| 15 | [docs/verification/deepdive-20261110/README.md:21](../README.md) | `fa4093a3ef24` | 1111 × 384 | [PNG](15.png) |

本轮对两张入口图和模型总图改用纵向布局；训练采样图手动指定换行；历史 Studio 状态图拆分为两图，转换编号及行号在原文下方的表格中。

[返回证据导航](../README.md)
