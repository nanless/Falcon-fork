# 项目图片逐项复核（2026-11-10）

## 范围和方法

在 Mermaid 渲染产物生成前，对服务器仓库原有静态图片，按 PNG、JPEG、GIF、SVG、WebP、BMP、PDF 扩展名扫描文件，排除 `.git` 和 `node_modules`。发现 13 张图片，均在 Git 文件清单中；对全部 Git 跟踪文件再按实际 MIME 类型扫描，结果同为这 13 张，该次扫描没有发现其他同类原有文件；此后新增的 Mermaid 渲染图由独立渲染记录覆盖，不计入本表。将图片只读复制到本机临时目录，逐张以原尺寸查看；同时用 Pillow 校验可解码性、记录尺寸并计算 SHA-256。表中哈希仅为便于识别的前 12 位，不作为完整性验证的替代。

本表核对图片内容与引用位置。[Mermaid 图渲染记录](diagram-render-20261110/README.md)列出了流程图的源码与渲染结果。

## 逐张记录

| 文件 | 像素 | SHA-256 前 12 位 | 实际查看与用途 |
| --- | --- | --- | --- |
| [`assets/Overall_Pipeline.png`](../../../assets/Overall_Pipeline.png) | 1897 × 413 | `05159a90caeb` | 混合音频、目标嘴部视觉输入和模型输出的概览；中英文根 README 与模型说明引用，主题吻合。 |
| [`assets/TFRB.png`](../../../assets/TFRB.png) | 1970 × 380 | `591b525f88a0` | 时间频率细化结构；RNN 与时间操作使用抽象记号。当前根 README 不直接嵌入。 |
| [`assets/logo-source.png`](../../../assets/logo-source.png) | 1254 × 1254 | `8c6b6db6a990` | 透明背景猎鹰标志源图；没有发现直接引用。 |
| [`assets/logo-tile.png`](../../../assets/logo-tile.png) | 512 × 512 | `9b8a371e29ac` | 方形底色的猎鹰标志；Studio 中英文 README 的深色模式 source 使用。 |
| [`assets/logo.png`](../../../assets/logo.png) | 320 × 320 | `570bc71a8bf2` | 透明背景猎鹰标志；根 README 和 Studio README 使用。 |
| [`assets/realssa_scenes.jpg`](../../../assets/realssa_scenes.jpg) | 1372 × 420 | `87fc64a4d594` | 四格场景图片，依次呈现室内会谈、公共室内空间和街景；未发现当前 Markdown/HTML 直接引用。数据规模与许可可查阅对应数据卡。 |
| [`assets/separator.png`](../../../assets/separator.png) | 1793 × 866 | `11d901b76efa` | 编码器、多层 GCI 和 TFRB、解码器示意；中英文根 README 引用，主题吻合。 |
| [`project_page/figures/Overall_Pipeline.pdf.png`](../../../project_page/figures/Overall_Pipeline.pdf.png) | 2411 × 524 | `63b24cbe2d13` | 项目页面使用的总体结构示意，与其标题相符；与 assets 中同名图视觉内容对应，但不是同一文件。 |
| [`project_page/figures/Separator.pdf.png`](../../../project_page/figures/Separator.pdf.png) | 1662 × 803 | `d4b92b1f24a8` | 项目页面使用的分离器示意，主题正确；与 assets 版本在模块编号与运算符标注上存在差异，见下文。 |
| [`project_page/figures/TFRB.pdf.png`](../../../project_page/figures/TFRB.pdf.png) | 3242 × 625 | `3d862ec87eab` | 项目页面使用的细化模块示意，明确写了 Bi-LSTM、Uni-LSTM 和掩码注意力；应结合所用配置及实现判断实际行为。 |
| [`project_page/logo.png`](../../../project_page/logo.png) | 128 × 128 | `c4b9f622420f` | 页面图标和品牌标志，内容与用途相符；与 Studio 网页标志文件的完整 SHA-256 相同。 |
| [`studio/tests/browser/portrait.jpg`](../../../studio/tests/browser/portrait.jpg) | 820 × 1024 | `a6f11efaa834` | 单人正面肖像，用于浏览器人脸检测测试；图片来源和许可信息尚待核实。 |
| [`studio/web/logo.png`](../../../studio/web/logo.png) | 128 × 128 | `c4b9f622420f` | Studio 网页图标和品牌标志，内容与用途相符；与项目页图标完整哈希相同。 |

## 两版结构图的区别

[README 所用分离器图](../../../assets/separator.png)带有 GCI 层编号和 φ 运算符，[项目页分离器图](../../../project_page/figures/Separator.pdf.png)省略这些标注；[assets 中的 TFRB 图](../../../assets/TFRB.png)用抽象算子记号，[项目页 TFRB 图](../../../project_page/figures/TFRB.pdf.png)则直接标注 Bi-LSTM 和 Uni-LSTM。两版图各有侧重。具体执行顺序、加法与插值见 [`AudioBlock.forward`](../../../look2hear/models/Falcon.py) 和[模型与流式处理说明](../../understanding/20261110-model-and-streaming.md)。

原有图片保留在各自目录，使用许可需结合来源材料核实。
