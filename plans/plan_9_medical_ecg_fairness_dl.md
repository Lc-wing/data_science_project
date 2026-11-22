# Plan 9（v2）：12 导联临床心电图（PTB-XL / PTB-XL+）异常筛查的亚组公平性与概率校准审计 —— 临床统计/ML 主分析 + 12 导联波形 DL 与 4B 临床参数 LLM 微调附加模块

> **主题来源：** 医药健康与数字医疗质量评估（PhysioNet 开放临床心电基准 PTB-XL / PTB-XL+）
> **建议文件名：** `CS482_GroupX_ECGFairnessCalibrationDL.pdf`
> **算力需求：** 阶段 A 用 CPU 即可；阶段 B 用 Colab L4 / A100（`bf16` / QLoRA，约 20–50 分钟；所有模型 ≤ 5B）
> **⚠️ 强制安全边界（遵循 `clinical-decision-support` 技能）：** **本项目仅用于课程数据科学与模型评估研究，严禁用于真实患者临床诊断、分诊、预警或治疗决策（Not for patient care or live clinical use）。**

### 📊 难度与成功率评估（基于 22 天工期 11/22–12/14 与 3–5 人团队的主观规划估计——**跨领域推荐第 2**）

| 评估维度 | 评级 | 说明与关键前提 |
| --- | --- | --- |
| **数据获取难度** | ★☆☆☆☆ 很低（1/5） | PhysioNet **PTB-XL v1.0.3**（21,799 份 12 导联 ECG，18,869 名患者）与 **PTB-XL+ v1.0.0**（529–782 个临床形态/间期特征表），全部 **CC-BY 4.0 完全开放下载，无需 PhysioNet 资质审批、无需伦理审批**，且官方自带防患者泄漏的 `strat_fold` 10 折切分 |
| **全阶段实现难度** | ★★★☆☆ 中等（3/5） | 阶段 A **2.5/5**（直接基于 PTB-XL+ 表格特征做多标签逻辑回归、LightGBM、年龄/性别亚组公平性与 ECE 校准审计、决策曲线分析 DCA）；阶段 B **3.5/5**（Colab 上训练 12 导联 `1D-ResNet-SE / ECG-PatchTransformer` 波形模型及 `Qwen3-4B` 非诊断客观参数 QLoRA） |
| **达成最低要求成功率（阶段 A）** | **90%–95%** | 数据质量属全球医学 AI 标杆级，无清洗死结、无患者切分歧义 |
| **实现最终目标成功率（阶段 A + 阶段 B ≤ 5B）** | **82%–90%** | 100 Hz 波形总大小可控（约 1.7 GB，Colab 拉取顺畅），1D 波形 DL 与 4.0B 客观参数 QLoRA 均可高效完成 |

---

## 0. 选题合规性与科研技能把关要点（对应 [评审报告](00_review_report.md)）

| 科研技能把关项 | 避开的常见医药健康 AI 项目陷阱 | 本计划的规范设计 |
| --- | --- | --- |
| `clinical-decision-support` | 声称模型“可用于临床自动诊断”；仅报一个总 AUROC 而忽略**概率校准**、**高龄/性别亚组偏差**与**临床净收益** | 全文置顶“仅限科研方法学评估”硬边界；以 **鉴别力（Macro-AUROC/PR-AUC）+ 校准度（ECE/Brier）+ 年龄/性别亚组公平性 + 决策曲线分析（DCA）** 四维框架开展研究 |
| `scientific-critical-thinking` / 防泄漏 | **隐蔽文本泄漏：** PTB-XL 的 `report` 文本本身就是医生诊断结论，绝不能拿来预测诊断标签！**患者重复记录泄漏：** 同一患者有多份 ECG | **严禁将医生诊断报告文本 `report` 输入任何预测模型！** 严格遵循官方患者级 `strat_fold`（1–8 训练，9 验证与校准，10 独立测试，确保同一 `patient_id` 永不跨折） |
| `statistical-analysis` / `scientific-writing` | 二分类/多标签只报相对比值或仅报准确率；混淆探索与验证 | 每个超类明确报告 **阳性例数/分母（患病率）**、灵敏度/特异度、绝对风险差与 95% CI（DeLong / 患者级 bootstrap）；多重亚组比较做 **Holm 校正** |
| `shap` / `transformers` | 忽略医学相关性特征对 SHAP 的影响；把深度网络 softmax 误当校准后的患病概率 | TreeSHAP 仅从折 1–8 抽取背景集并核对可加性；所有深度/大模型均在折 9 做 **Temperature Scaling** 校准后，才在折 10 评估 ECE 与 DCA |

---

## 1. 研究问题与意义

**主研究问题：** 在 18,869 名患者的 21,799 份 12 导联心电图基准（PTB-XL / PTB-XL+）中，基于临床电生理形态特征（PTB-XL+ `ECGDeli` 特征）构建的可解释模型（正则化 Logistic 与 LightGBM）在五大诊断超类（正常 `NORM`、心肌梗死 `MI`、ST/T 改变 `STTC`、传导阻滞 `CD`、心室肥大 `HYP`）上的鉴别力与概率校准是否存在显著的 **年龄（<50 / 50–70 / ≥70 岁）与性别亚组差异**？相比临床手工特征，端到端 12 导联深度波形网络（~10M–20M）与基于客观电生理参数的 **4.0B 预训练大模型 QLoRA** 能否缩小困难类别（如 `HYP`）与老年亚组的性能缺口？

| 假设 | 内容 | 检验方式 |
| --- | --- | --- |
| **H1（验证性）** | 在五大诊断超类中，困难类别（尤其 `HYP` 心室肥大）与高龄亚组（≥70 岁）在 LightGBM 上的 AUROC/校准误差（ECE）显著劣于中年与 `NORM` 类别 | 官方测试折（fold 10）亚组 AUROC DeLong 检验 + 患者级 bootstrap ECE 差值 95% CI（Holm 校正） |
| **H2（验证性）** | 在折 9 上执行 Temperature Scaling / Isotonic 校准后，各亚组概率校准误差（ECE、Brier Score）与特定风险阈值下的决策曲线净收益（Net Benefit）显著改善 | 校准前后折 10 配对患者级 bootstrap 对比 |
| **H3（验证性，阶段 B 延伸）** | 在完全相同的官方患者切分下，端到端 12 导联深度波形模型（`1D-ResNet-SE / ECG-PatchTransformer`）在 Macro-AUROC（尤其是 `HYP` 与 `MI`）上优于仅使用 PTB-XL+ 客观形态特征的 LightGBM 与 `Qwen3-4B` QLoRA | 折 10 跨模型 DeLong 检验（Holm 校正）+ 亚组公平性对比 |

**受益者：** 医学 AI 评估与质控研究者（暴露高龄/性别亚组校准隐患）、心电算法研发团队（量化“手工电生理特征 vs 端到端波形 DL vs 4B 大模型”的精度-可解释性权衡）、数字健康课程与方法学教学。

**结论边界：** 数据采集自 1989–1996 年德国单中心设备（Schiller），存在显著历史时期、人群地域与设备域偏移；**无外部医院验证，绝不可外推至现代临床场景或用于真实患者诊疗**。

---

## 2. 数据收集与准备

### 2.1 Day-1（11/22）数据可得性核查清单

| # | 数据 | 主要来源 | 核查动作 | 备选方案 |
| --- | --- | --- | --- | --- |
| 1 | 临床标签、患者元数据与 100 Hz 12 导联波形 | **PTB-XL v1.0.3**（`physionet.org/content/ptb-xl/`，CC-BY 4.0） | 下载 `ptbxl_database.csv`、`scp_statements.csv` 与 `records100/`（约 1.7 GB，用 `wfdb` 读取），核实 21,799 条记录与 `strat_fold` 分布 | 若网络拉取 1.7 GB 慢，可从 PhysioNet AWS Open Data 镜像 `s3://physionet-open/ptb-xl/1.0.3/` 免账号高速下载至 Colab |
| 2 | 预提取客观临床 ECG 形态/间期特征 | **PTB-XL+ v1.0.0**（`physionet.org/content/ptb-xl-plus/`，CC-BY 4.0） | 下载开源 `ECGDeli` 特征表（529 列，覆盖 12 导联 P/QRS/ST/T 波幅、时限与间期），核对 `ecg_id` 对齐率 | 若仅用 PTB-XL，可用开源 `neurokit2` 在 100 Hz 波形上自提核心间期与心率变异性特征 |

### 2.2 数据规模与变量
- **分析单元：** 单份 10 秒 12 导联 ECG 记录（`ecg_id`，聚类/切分单元为独立患者 `patient_id`）
- **样本量：** **21,799 份 ECG / 18,869 名患者**（标注有效超类样本约 21,388 份，透明报告无超类标签记录的剔除计数）
- **目标变量（5 维多标签，可共存）：** `NORM`、`MI`、`STTC`、`CD`、`HYP`（严格按照官方 `scp_statements.csv` 映射，仅保留诊断置信度 `likelihood >= 50` 或采用官方基准全部映射并明确披露）。
- **输入特征：**
  - *阶段 A（表格）：* 患者年龄、性别 + PTB-XL+ `ECGDeli` 客观电生理形态特征（529 列经训练折内缺失率过滤与中位数插补）；**严禁使用 `report` 文本与商业软件自动诊断结论 `12SL` 语句（防目标泄漏！）**。
  - *阶段 B（波形与客观参数）：* 原始 100 Hz 12 导联波形张量 `(12 × 1000)`（`wfdb` 加载，仅用训练折统计量做逐导联 Z-score 标准化）；客观电生理参数摘要（用于 4B LLM 路线 B）。

### 2.3 清洗与预处理
1. **强制防泄漏检查：** 从特征表彻底删除 `report`、`scp_codes`、`infarction_stadium*` 及 PTB-XL+ 中商业自动诊断语句列；仅保留客观波形测量值与人口统计学特征。
2. **缺失与信号质控：** 统计 `ECGDeli` 特征缺失率（>30% 缺失的特征在训练集内剔除，其余仅在训练集拟合中位数插补 + 缺失指示项）；记录官方标记的 `baseline_drift` / `static_noise` / `burst_noise` 信号质量标签用于鲁棒性亚组审计！
3. **冻结切分（官方患者级 10 折，阶段 B 共用）：**
   - 训练集：`strat_fold` 1–8（约 17,400 份）
   - 验证与校准集：`strat_fold` 9（约 2,190 份）
   - 独立测试集：`strat_fold` 10（约 2,200 份，**严格一次性最终评估**）

---

## 阶段 A —— 最低要求（11/22–12/03）

### 3. 探索性数据分析（EDA）
- 五大超类的阳性率、共病热力图（如 `MI` 与 `STTC` 共现）及年龄/性别分布表（图 1 左，明确分子/分母）
- 关键临床测量值（如 QRS 时限、ST 段偏移、R 波振幅）在各超类与年龄亚组间的分布差异（图 1 右）
- 信号噪声标记（基线漂移/工频噪声）分布统计

### 4. 方法与理由

| 用途 | 方法 | 选择理由 |
| --- | --- | --- |
| 临床统计基线 | **L2 正则化多标签 Logistic 回归**（One-vs-Rest，年龄/性别 + 核心电生理间期与波幅特征） | 医学领域最标准的可解释风险模型，直接输出比值比（OR）与校准概率 |
| 经典 ML 主模型 | **多标签 LightGBM**（529 维 `ECGDeli` 客观特征 + 年龄/性别）+ **折 9 概率校准（Temperature Scaling / Isotonic）** | 捕捉跨导联形态特征的复杂非线性组合（如 Sokolow–Lyon 肥大指标的推广） |
| 亚组公平性与效用审计（H1/H2） | 在折 10 按 **性别（男/女）× 年龄（<50 / 50–70 / ≥70）× 噪声质量** 分层评估 AUROC、PR-AUC、ECE、灵敏度（固定 90% 特异度）与 **决策曲线净收益（DCA）** | 紧扣 `clinical-decision-support` 规范，超越单一 AUROC 评分 |
| 解释 | TreeSHAP（背景集仅取自折 1–8 训练集，按超类分别解释预测概率并核对 `model.classes_` 与可加性；仅说明模型行为） | 检验模型依赖的是否为真实电生理特征（如 V5/V6 R 波、V1 S 波对 `HYP` 的贡献） |

### 5. 评估、假设与不确定性
- **多标签总指标（折 10）：** Macro-AUROC（官方主指标）、Macro-PR-AUC、各超类独立 AUROC（带 DeLong 95% CI）、Macro-Brier Score、Macro-ECE。
- **公平性与校准统计检验：** 亚组间 AUROC 差异用 DeLong 检验，ECE 与敏感度差异用患者级 bootstrap 95% CI，所有亚组多重比较做 **Holm 校正**。
- **噪声鲁棒性检验：** 对比“干净 ECG vs 含基线漂移/噪声 ECG”的性能衰减。
- **核心局限：** 1989–1996 年历史德国队列的地域/设备偏移、标签本身带有双心内科医生人工判读变异性、无随访预后结局（仅横截面 ECG 诊断语句）。

### 6. 结果与讨论（报告中怎么呈现）
- **图 1：** 五大超类共病/年龄分布 + 折 10 各年龄/性别亚组校准前后可靠性曲线与 ECE 对比（回答 H1、H2）
- **表 1：** Logistic vs LightGBM（校准前/后）在折 10 的五大超类 AUROC、Macro-PR-AUC、Macro-ECE 与老年/噪声亚组差距
- **图 2：** TreeSHAP（`MI` 与 `HYP` 关键电生理特征）+ 决策曲线分析（DCA）净收益对比
- **讨论：** 回答 H1–H2、探讨医学 AI 校准与亚组偏差的治理意义、重申非临床应用边界与历史单中心局限。

### ✅ 最低要求对照表

| Requirement | 本计划中的对应内容 |
| --- | --- |
| 1. Research Question | 第 1 节：主问题 + H1–H3 + 医疗 AI 质控受益者 + 强制临床安全边界 |
| 2. Data Collection & Preparation | 第 2 节：PhysioNet PTB-XL/PTB-XL+（CC-BY 4.0）、严禁诊断文本泄漏规则、官方患者级 `strat_fold` |
| 3. Data Analysis | 第 3、4 节：EDA + 正则化 Logistic（临床统计）+ LightGBM + 概率校准 + 亚组公平性/DCA 审计 |
| 4. Method Evaluation | 第 5 节：官方折 10 独立测试、DeLong/bootstrap + Holm 校正、噪声鲁棒性与局限 |
| 5. Results & Discussion | 第 6 节：逐条回应假设、医学 AI 治理洞见与域偏移局限 |

---

## 阶段 B —— DL 附加模块（12/04–12/10，前提是通过 12/03 检查点，所有模型 ≤ 5B）

**输入：** 阶段 A **冻结** 的官方患者级 `strat_fold`（1–8 训练，9 验证/校准，10 测试）、100 Hz 原始 12 导联波形及非诊断客观 `ECGDeli` 测量值。

### 路线 A（推荐，从零训练 12 导联端到端波形深度模型，约 8M–20M 参数）
- **模型：** **`1D-ResNet-SE`（如 `xresnet1d101` 风格通道注意力残差网）** 与 **`ECG-PatchTransformer`**（`d_model=256–384, n_heads=8, n_layers=6`，约 **8M–20M** 参数），直接输入 100 Hz 12 导联波形 `(12 × 1000)`，输出 5 超类多标签概率（BCEWithLogitsLoss + Cosine LR + Mixup/随机导联遮蔽数据增强 + 折 9 Early Stopping 与 **Temperature Scaling** 校准，跑 5 个随机种子；并做 1M vs 15M 容量消融）。
- **核心验证：** 重点检验端到端波形深度学习在 `HYP`（心室肥大）与高龄亚组上是否显著缩小了与简单类别的差距（回答 H3）。

### 路线 B（预训练大模型微调，≤ 5B —— 严禁使用诊断文本 `report`！）
- **4B 客观电生理参数 LLM QLoRA（严格 ≤ 5B，零文本泄漏）：** 将每份 ECG 的 **纯客观测量数值**（年龄、性别、心率、PR/QRS/QT/QTc 间期、电轴、关键导联 R/S/ST/T 波幅汇总——**绝不包含任何医生诊断 `report` 或自动诊断结论**）序列化为规范医学测量简报，在 Colab A100/L4 上加载 **`Qwen/Qwen3-4B-Instruct-2507`（4.0B，Apache-2.0）**（可选与 **`Qwen/Qwen3-1.7B` 1.7B** 作尺度对照），接 5 类多标签分类头并施加 **QLoRA（`r=16, alpha=32`）** 在折 1–8 上微调，折 9 做早停与 Temperature Scaling 校准，折 10 一次性评估！
- **科学对照价值：** 直接回答一个非常前沿的跨学科问题——**“在客观心电图参数上微调 4.0B 通用大模型，能否借其医学教材预训练先验超越 LightGBM，又与直接看原始 12 导联波形的 15M 1D-CNN/Transformer 差距几何？”**
- **规范：** 记录 `transformers` / `peft` 版本、checkpoint `revision`、序列化模板、随机种子（至少 3 种子）及折 9 温度缩放系数。

### 对比表（报告里放 1 张小表）

| 模型（参数量，均 ≤ 5B） | 折 10 Macro-AUROC | `HYP` AUROC | ≥70 岁亚组 AUROC | Macro-ECE（校准后） | DeLong p vs LightGBM（Holm） |
| --- | --- | --- | --- | --- | --- |
| 正则化 Logistic / LightGBM（阶段 A，客观特征） | | | | | — |
| `1D-ResNet-SE` (~10M，12 导联波形，5 种子) | | | | | |
| `ECG-PatchTransformer` (~15M，12 导联波形，5 种子) | | | | | |
| `Qwen3-4B-Instruct-2507` (4.0B QLoRA，仅客观参数，零诊断文本) | | | | | |

**解读规则：** 预期端到端 12 导联波形网络（~10M–15M）很可能超越仅看汇总参数的 4.0B LLM，因为原始波形保留了手工间期丢失的细微形态信息；无论高低，均如实报告并讨论“数据模态信息量 vs 模型参数规模”的本质规律。

**放弃条件：** 若 12/08 前 `Qwen3-4B` 全量 1.7 万条文本序列化微调耗时偏长，可优先确保 `1D-ResNet-SE` + `ECG-PatchTransformer` 双波形架构完整落地，并将 `Qwen3-4B` 控制在 1–2 个 epoch 的轻量 QLoRA 评估（明示训练预算）。

---

## 7. 报告页数预算（A4、单栏、11pt、1.5 倍行距）

| 章节 | 页数 |
| --- | --- |
| 1 Introduction, RQ & Research-Only Safety Boundary | 0.6 |
| 2 PTB-XL / PTB-XL+ Data & Anti-Leakage Protocol | 0.8 |
| 3 Methods（Logistic / LightGBM / Calibration / Fairness / DCA） | 0.8 |
| 4 Results（图 1、表 1、图 2） | 1.1 |
| 5 Discussion, Subgroup Equity & External-Validity Limits | 0.5 |
| 6 Extension: 12-Lead Waveform DL & 4B Objective-Feature LLM（1 小表 + 1 段） | 0.6 |
| **合计** | **4.4** |

## 8. 分工与时间表

| 角色 | 职责 | 可核查交付物 |
| --- | --- | --- |
| A 医学数据与防泄漏负责人 | 下载 PTB-XL / PTB-XL+、核查 `ECGDeli` 特征与 100 Hz 波形、自动断言剔除 `report`/诊断泄漏列 | 冻结特征表、缺失与噪声审计日志、图 1 |
| B 临床统计与公平性负责人 | 多标签 Logistic、折 9 概率校准、年龄/性别/噪声亚组 DeLong 与 bootstrap 审计、DCA | 亚组审计表、校准与 DCA 图 |
| C 机器学习负责人 | 多标签 LightGBM、TreeSHAP 电生理特征归因 | 表 1、图 2 |
| D DL / 4B 模型负责人（3 人组由 C 兼任） | Colab 上执行 `1D-ResNet-SE`、`ECG-PatchTransformer` 与 `Qwen3-4B` 客观参数 QLoRA | 阶段 B 对比表与日志 |
| 全员 / E | 报告撰写、安全边界声明、数字复核、格式与 AI 声明 | 最终 PDF |

| 日期 | 里程碑 |
| --- | --- |
| 11/22–11/24 | 下载 PTB-XL / PTB-XL+（CC-BY 4.0），验证无诊断文本泄漏与官方 `strat_fold`；冻结 RQ/H1–H3 |
| 11/25–11/29 | 完成 EDA、Logistic、LightGBM、折 9 校准、TreeSHAP 与亚组公平性/DCA 分析 |
| 11/30–12/02 | 完成折 10 阶段 A 汇总，写出阶段 A 完整报告草稿 |
| **12/03** | **Go/No-Go 检查点**：A 草稿 ≥ 3 页且 5 部分齐全、数据与切分冻结 |
| 12/04–12/10 | 阶段 B（Colab A100/L4 完成 12 导联波形 DL 与 `Qwen3-4B` 客观参数 QLoRA） |
| 12/11–12/13 | 整合报告、核对全部数字、格式检查、贡献声明与 AI 声明 |
| 12/14 | 提交 Moodle（避开 12/15 Quiz 2） |

## 9. 学术诚信与 AI 声明要点
- 规范引用 Wagner et al. (2020) PTB-XL 论文、Strodthoff et al. (2023) PTB-XL+ 论文及 PhysioNet CC-BY 4.0 协议。
- 必须在报告显眼位置声明：**本研究仅作学术方法学评估，不构成医疗器械认证，不得用于任何真实患者诊疗、分诊或用药决策**。
- AI 声明如实写明计划构思、代码调试、语言润色及全部指标的独立脚本复现核验过程。
