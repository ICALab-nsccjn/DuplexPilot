# DuplexPilot V3：按Lychee-FD论文的论证组织重写

## 阅读与编译

- `main.pdf`：本次完整论文，ICLR单栏、匿名审稿版式。
- `main.tex` + `appendix.tex`：正式学术行文正文和附录，不再将执行记录作为摘要。
- `references.bib`：40条实际引用的资料；31篇正式会议/期刊论文，6篇相关预印本/技术报告，3项必要官方文档/标准。
- `figures/*.tex`：可编辑架构图和事务时序图；PDF/PNG随包提供。
- `scripts/plot_results.py`：从所附报告数据表再生实验图。
- `scripts/verify_manuscript.py`：检查引用、文件、计量边界和编译结果。
- `audit/main_v2_to_v3.diff`、`audit/references_v2_to_v3.diff`：相对实际V2的逐行变化；V2原稿在audit目录保留。

依赖LaTeX发行版中的TikZ、algorithm/algpseudocode、natbib、xurl等。已有图文件时，直接编译main即可；完整构建：

```bash
bash build.sh
python scripts/verify_manuscript.py
# 仅需重绘实验图时：
python scripts/plot_results.py
```

当前构建结果：**正文11页，声明、参考文献和附录合计后共20页。**本轮按用户要求先与参考论文的论述规模看齐，没有通过缩小字号或修改边距压成9页。按相同的PDF英文字词统计法，参考ACL正文约5,072词，V3正文约5,457词（含图表文字，属于近似比较）。ACL双栏页数不能直接等同ICLR单栏页数。初次投稿若仍限9页，需要后续内容压缩与官方样式重新预检；本文件不宣称超页版本可直接提交。

## 实际修改

1. 摘要采用“问题—难点—方法—核心发现”，删除96/92/4、环境编号、审计/封版语句。引言突出科学问题、设计洞察、贡献，不逐项列运行账本。
2. 第2章改为Related Work；第3章先解释状态耦合，再介绍表示、桥接/声学、联合事务和高效实现。
3. 第4章按照基线、数据/实现/指标、原生系统参考、冷恢复结果、资源权衡、传输对照、正确性组织；用Metrics/Results/Discussion式小段落，而非逐轮执行总结。
4. Lychee-FD原论文分数进入Table 2上半部分，原生N1记录与vLLM-Omni现有运行进入下半部分。L-best/F-cold/A-hot/L-hybrid继续构成同模型冷恢复主表。
5. 16%仍作为核心技术性能结果；明确相对同步联合卸载、来自独立匹配实验，不转移为击败vLLM-Omni或Lychee原论文的比例。
6. 重画可编辑的架构和时序图，修复V2图中标题、CPU框和底部说明重叠。实验图从所附数据再生成；不添加不存在的误差条。
7. 执行计数、4个失败、版本映射和历史负结果放在实验/附录的相应位置。没有删除实质限制或将未知结果改成成功。

## 基线不能混成同一计时

- Lychee-FD论文的637ms、826ms等为其原生FDBench/FullDuplexBench指标，不是CPU冷恢复时延。
- 原生Lychee N1历史记录为共同身份修复之后的真实运行，不能宣称是毫无改动的上游代码，也不能把该历史轨迹与当前A-hot混成一行。
- vLLM-Omni实际33块、32.52秒是**生成的音频长度**；没有共同冷恢复/浏览器时延。因此没有人为填入某个速度数，也未计算外部加速比。
- 同模型主矩阵保留E1/E2独立汇总、Hybrid更快但不释放GPU1、GPU0净额外占用等实际事实。

## 数据与文献核验事项

`audit/evidence_map.json`把各表图与实际来源绑定；`audit/reference_verification.csv`列出40条资料的出版社/作者原始入口、发表身份、引用章节及用途。

源数据情况未被文字润色改变：此前提供的逐次CSV是26通过、1未验收、69未开始的进度导出，与封版报告最终92/96不匹配。V3主图主表继续使用封版报告明确给出的聚合值，不能据此生成缺失逐次记录或置信区间。投稿前仍需要作者核对最终导出、环境映射与摘要统计。机器图表数据保留为报告转录，不冒充远端原始回执。

用户提供的Lychee-FD文件首页标注ACL 2026；主文9页，全部17页。本轮学习其论证结构，没有把“首次/SOTA”等措辞移植为我们未证实的结论。所给PDF标题为“Hierarchical Intelligent Acoustic-Semantic Modeling: Modality Separation and Alignment for Full-Duplex SLMs”；ACL出版社页面的规范标题为“Hierarchical Acoustic-Semantic Modeling: Modality Separation and Semantic Coherence for Full-Duplex SLMs”，作者名字也有个别拼写差异。Bib采用出版社元数据，表中原生数字明确来自用户上传版本，未静默混同两个版本。

该上传论文引用FDBench时，其参考文献中存在指向流体仿真同名benchmark的条目。本轮没有照抄该错误条目；原生指标按Lychee-FD原论文归属引用，不将其与HumDial或流体数据集混用。

## 样式与可提交性

随包`iclr2027_conference.sty`原样继承V2，其SHA256在BUILD_VERIFICATION.json中；没有调整字体、行距、版心来压页数。该文件是此前从官方网页转录的样式，尚未与官方ZIP作字节级同一性核验。最终提交前应使用ICLR官方完整样式包复核，重新编译检查9页限制、匿名性、AI使用声明与引用。可成功编译不等于已完成官方格式核验。

## 本轮没有做的事情

没有新增模型实验、人评或ASR；没有补写外部性能；没有按共享模型权重继承原论文的质量分数；没有改变既有性能数字。正文已经按论文形式写作，但最终逐次数据、质量与代码—伪代码对齐仍需要研究团队确认。
