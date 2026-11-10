import json,hashlib,subprocess
from pathlib import Path
repo=Path("/root/code/github_repos/Falcon-fork");root=repo/"docs/understanding"; out=Path("/tmp/falcon_full_review_20261110/historical-understanding-review.json")
# These are targeted cross-checks of selected statements, not claims of full reading.
notes={
"deepdive-20261024/00-README-门牌.md":("历史索引与已推翻观点","21–25列部分纠偏，但未涵盖2026-11-10发现的LSTM跨样本与因果裁剪","保留历史快照与现状分层的导航意义"),
"deepdive-20261024/00-SYNTHESIS.md":("2026-10-24九域总纲","23–24把57帧简写为整体、称RealSSA-Sim闭合；非同协议且全窗依赖；49行A800延迟缺原始记录","目录分工和历史工作脉络可作导航"),
"deepdive-20261024/01-model-core.md":("模型形状、参数和因果","17,249中的窗口级因果概念不是逐帧因果；此前未覆盖LSTM批次串扰与causal stride额外裁剪","权重strict加载、参数537697及concat[2]死参数有后续佐证"),
"deepdive-20261024/02-layers-dsp-video.md":("DSP归一化和视觉骨干","12行称每级与视频融合；实际单处视频加法，AudioBlock内部四尺度；57帧只属限定单遍","win256/hop128、cLNhw全窗统计、视觉前端±2帧"),
"deepdive-20261024/03-training-system.md":("旧训练入口、配置和DDP","164行200 epoch退火属旧配置；当前paper v2为1000且K=1退火不影响损失；DDP风险仍需多卡实测","训练入口修复历史、分桶all_gather条件风险"),
"deepdive-20261024/04-losses-metrics.md":("PIT/NASS和指标","471行论文未公开及待确认损失的旧说法受后续论文/配置校正；RealSSA分母与协议需复核","SI-SDRi逐句基线相减、SDR512与PESQ参数的定义"),
"deepdive-20261024/05-data-pipeline.md":("数据索引、转换和存储","243,291,412行采用full3被起点错误作废；资产体积与near内容是历史快照","索引格式、mix/target/mouth对应关系和near变更历史"),
"deepdive-20261024/06-studio-backend.md":("Studio后端、协议和性能","16,402行A800延迟缺原始测量文件；154行不能作为完整算法/端到端延迟；遗漏infer_file格式问题","2秒窗、0.5秒hop、80ms guard及strict加载"),
"deepdive-20261024/07-studio-frontend-space.md":("前端、Space和项目页","263行的合成mock等价测试不能证明真实模型批次等价；445论文未公开已过期","两套UI采集、Space组装与时间窗口结构"),
"deepdive-20261024/08-verification-ops-env.md":("验证资产、环境与追溯","140,181行full3已作废；早期git未跟踪证据状态已变，不能当当前盘点","typeguard限制、pypesq兼容层和证据分级方法"),
"deepdive-20261024/09-benchmark-alignment.md":("论文基准与复现规划","13行RealSSA-Sim闭合应降为数值吻合非同协议；89,154 full3与115的57帧为过期/缺限定","分离L1结构、L2评测、L3训练、L4消融框架"),
"00-CONVENTIONS.md":("历史审阅约定","初始项目与模型数字是旧任务语境，并非当前代码验证或可用环境指令","事实/验证/推断分开标记和逐项复核规则"),
"00-project-map.md":("早期仓库地图","开头明确假设，causal概称及目录/行数不应作为现状；遗漏后续Space批处理风险","按训练评测、Studio划分及初始问题清单"),
"00-round-plan.md":("历史多轮审阅流程","验收表和当时负责人不代表2026-11-10无问题或全仓库完成","理解、review、追证的协作流程与证据要求"),
"01-model-core.md":("R3模型源码结构","23行eval strict=False是旧入口；因果措辞与VideoBlock解释受后续全窗及批次实验证伪/限定","参数、形状、dead concat及复数mask机制"),
"02-layers-dsp.md":("层、DSP和归一化","window-causal不可推为帧因果；有关视频LSTM时间解释需以现源码为准","GroupNorm全窗和CumulateLN逐频累计的数学定义"),
"03-training-system.md":("旧版训练系统及故障","12–13的comet_logger P0已修；strict=False和config污染是修复前行为","训练入口装配及历史缺陷复现与评审脉络"),
"04-losses-metrics-video.md":("旧版损失、指标及视觉","200 epoch和评测入口行为是旧代码；本轮K=1时beta不影响loss，P2A批次造成跨样本依赖","指标公式、目标基线和视觉特征接口"),
"05-data-pipeline.md":("旧版数据加载与JSON","450行mono collapse P0是修复前；train_samples是否YAML显式声明取具体配置；epoch不等全量","JSON契约、2秒/25fps、dtype及near时间对齐警示"),
"06-studio-backend.md":("历史后端与运行时","本篇使用PIL stub、合成权重；不能证明真实权重行为；现infer_file存在PCM32/float和训练mouth .pt边界","runtime strict加载、2秒窗与消息校验流程"),
"07-studio-frontend.md":("历史实时前端","跟踪丢失可能沿用旧ROI，自动重锁时可能切人；旧版行号和UI语义须对当前代码核实","PCM采集、WebSocket单飞、时间戳及录音导出结构"),
"08-ops-tests-docs.md":("历史CI、启动与配置","28行comet P0已修；现CI不跑本地repro证据套件，当前测试166通过2跳过","环境启动路径、测试分类与Host/Origin防护边界"),
}
files=sorted((root/"deepdive-20261024").glob("*.md"))+sorted(root.glob("0*.md")); assert set(str(p.relative_to(root)) for p in files)==set(notes)
rows=[]
for p in files:
 rel=str(p.relative_to(root));data=p.read_bytes();lines=data.decode("utf8",errors="replace").splitlines();topic,conflict,retained=notes[rel]
 history=subprocess.check_output(["git","log","-1","--format=%h %as %s","--",str(p.relative_to(repo))],cwd=repo,text=True).strip()
 rows.append({"path":str(p.relative_to(repo)),"lines":len(lines),"sha256":hashlib.sha256(data).hexdigest(),"git_last_change":history,"baseline":"2beec38 plus later annotations" if rel.startswith("deepdive") and "README" not in rel else ("historical directory guide 2026-11-09" if "README" in rel else "7576854 for R3 files; original plan/map preliminary"),"topic":topic,"targeted_conflict":conflict,"potentially_reusable":retained,"read_status":"targeted headings, first paragraphs and matched claims; NOT complete line-by-line reading","review_limit":"Statements are cross-checked selectively; retained information must remain conditional on source version and evidence scope."})
out.write_text(json.dumps({"scope":"20261024 deepdive and flat 00–08","file_count":len(rows),"total_lines":sum(x["lines"] for x in rows),"files":rows},ensure_ascii=False,indent=2)+"\n")
print("files",len(rows),"lines",sum(x["lines"] for x in rows),"output",out)
