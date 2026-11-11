"""Render the project dataset and storage relationship diagram."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=160)
fig.patch.set_facecolor('#f7f8fb'); ax.set_facecolor('#f7f8fb')
ax.set_xlim(0, 15); ax.set_ylim(0, 8.5); ax.axis('off')
def text(x,y,s,size=12,color='#1e293b',weight='normal',ha='center',va='center'):
    ax.text(x,y,s,fontproperties=font,fontsize=size,color=color,weight=weight,ha=ha,va=va,linespacing=1.5)
def box(x,y,w,h,title,detail,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=0.16',facecolor='white',edgecolor=color,linewidth=1.5))
    text(x+w/2,y+h-.28,title,12,color,'bold'); text(x+w/2,y+.38,detail,9,'#475569')
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,linewidth=1.5,color='#8191a5'))
text(.4,8.08,'Falcon 数据从哪里来，怎样进入训练与评测',20,'#14213d','bold','left')
text(.4,7.55,'先分清原始媒体、可训练样本和独立测试；具体数量与状态以文档的核验日期为准',10,'#526175','normal','left')
box(.4,5.75,3.0,1.15,'已转换的 LRS2-2Mix','混音 · 目标语音 · 嘴部帧\n训练 / 验证 / 测试索引','#2563eb')
box(4.4,5.75,3.0,1.15,'项目训练入口','旧版 Falcon 或新版流式模型\n按各自的数据入口读取','#2563eb')
box(8.4,5.75,3.0,1.15,'模型与检查点','训练输出独立保存\n记录数据划分与配置','#2563eb')
box(12.2,5.75,2.35,1.15,'独立测试','LRS2 / RealSSA\n按各自协议','#2563eb')
for a,b in [(3.4,4.4),(7.4,8.4),(11.4,12.2)]:arrow(a,6.325,b,6.325)
box(.4,3.65,3.0,1.15,'其他视听语料','单人音视频 · 真实场景\n混音包 · 标注或元数据','#0f766e')
box(4.4,3.65,3.0,1.15,'核查与转换','目标身份 / 音视频对齐\n许可 / 划分 / 文件格式','#0f766e')
box(8.4,3.65,3.0,1.15,'构造或转换样本','目标语音 + 嘴图 + 混音\n需要时构造干扰与混合','#0f766e')
box(12.2,3.65,2.35,1.15,'训练与评测','划分不交叉\n记录来源','#0f766e')
for a,b in [(3.4,4.4),(7.4,8.4),(11.4,12.2)]:arrow(a,4.225,b,4.225)
box(.4,1.48,3.0,1.15,'数据登记与原路径','共享目录中的索引与状态\n先核对位置和版本','#a16207')
box(4.4,1.48,3.0,1.15,'TOS 冷端挂载','共享大包 / 数据集归档\nAVSpeech 分卷 / RealSSA','#a16207')
box(8.4,1.48,3.0,1.15,'按需取回','选择库与分片 → 读取回执\n目标路径与容量单独核对','#a16207')
box(12.2,1.48,2.35,1.15,'恢复后核验','文件清单 / 格式\n时间轴 / 划分','#a16207')
for a,b in [(3.4,4.4),(7.4,8.4),(11.4,12.2)]:arrow(a,2.055,b,2.055)
text(.4,.52,'来源：项目配置、数据索引、多模态登记表与冷端归档记录；具体范围见各页证据。',9,'#526175','normal','left')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
out=Path(__file__).resolve().parent/'images'/'data-paths.png';out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,facecolor=fig.get_facecolor(),bbox_inches='tight');plt.close(fig)
print(out)
