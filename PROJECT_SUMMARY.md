# Sundial 岩爆预测项目整体情况说明

> 更新日期：2026-09-27
> 项目根目录：`G:\1A岩爆预测\sundial-rockburst`
> 对应仓库：`github.com/zhouyllll/sundial-rockburst`（main 分支已同步至提交 `8cb57f8`）

---

## 一、项目概述

本项目解决的是**岩爆概率预警**问题：从连续采集的 DAS（分布式光纤声学传感）波形 bin 文件出发，预测**未来 5、10、30 分钟内发生岩爆的概率**，且满足 `P5 ≤ P10 ≤ P30`。

核心思路是两级模型：

1. **冻结的 Sundial 预训练模型**（`thuml/sundial-base-128m`，只推理、不微调）：从最近 30 分钟连续波形特征预测未来特征走势；
2. **一个 9 参数的三时窗离散时间风险模型**（可训练）：把连续特征、Sundial 预测结果（及可选的微震统计）融合，输出三个时窗的累计风险概率。

项目从原始 bin 到训练结果全自动，**不需要手填任何中间 CSV**，一条 `run` 命令即可完成索引 → 特征提取 → Sundial 推理 → 标签构建 → 训练 → 测试 → 最新预测的完整链路。

---

## 二、方法部分

### 2.1 整体流水线

```text
连续bin → 自动读取 → 10秒波形特征 → 冻结Sundial预测未来变化 ─┐
                                                           ├→ 9参数概率模型 → P5/P10/P30
（可选）微震bin → 自动读取 → 前24小时片段数量、强度、时间统计 ─┘
真实岩爆时间清单 ────────────────────────────→ 生成训练标签
```

- **每完成一个 bin 推进一次预测**，不固定按一分钟推进；
- **微震分支可选**：无外部微震识别结果时省略 `--microseismic-dir`，程序自动关闭该分支，训练/流式预测两侧配置必须一致；
- `run` 一次完成全流程；`stream` 用训练好的模型按文件时间流式回放，不需要标签。

### 2.2 数据与格式

| 项 | 说明 |
|---|---|
| bin 格式 | 无文件头、小端 int32、采样点×通道交错；采样率与通道数从文件名解析（如 `5000Hz-0041pt`） |
| 文件名 | 自动解析起始时间，时长按文件字节数推算 |
| 岩爆标签 | 支持"每天时刻范围"（如 `19:39:23-19:39:24`），程序按子文件夹 bin 日期自动补日期、规范化分隔符、校验唯一配对 |
| 数据规模 | 6232 个连续 bin、27 条岩爆标签 → 4287 个有效样本 |
| 数据切分 | 按 `recording_group`（记录子文件夹）时间顺序约 70%/15%/15% 划分训练/验证/测试；时间重叠或窗口可能跨界的文件夹合并为同一组，边界剔除跨界标签 |

### 2.3 特征

- **连续波形特征**：每 10 秒提取（如 `mean_square` 能量代理），保留最近 30 分钟窗口；
- **Sundial 预测特征**：把 `log1p(mean_square)` 序列交给冻结的 Sundial，预测未来 5/10/30 分钟走势，融合为 6 维特征；
- **瞬态特征（消融实验）**：最近 5 分钟内——峰值/峰均比、STA/LTA、短窗（100ms/50ms 步长）能量 q99、超背景阈值短窗计数（`overthreshold_count`）；
- **微震统计（可选）**：前 24 小时片段数量、能量代理量、距最近事件时间。

### 2.4 模型与训练

- Sundial：固定权重（SHA `fcea0e6f41de14521a75787d77ea228bb7e66757a7c6b85278e8113913998dd4`），只推理；
- 概率模型：**三时窗离散时间风险模型**——3 个时窗截距 + 6 个共享特征系数，默认正则强度 1，不做大范围参数搜索；输出各时窗"至少发生一次岩爆"的累计概率，保证单调性；
- 样本权重：`--event-balanced` 让每个事件的正样本总权重相近（不丢样本），缓解相邻正常窗口主导损失的问题；验证/测试保持原始分布；
- 评估协议：**测试集只做最终一次评估**，不参与特征选择或阈值调参；验证集 30 分钟 AP 用于选择特征方案。

### 2.5 报警阈值与评估口径

- **自动阈值选择**（`--auto-threshold`）：只在验证集 30 分钟阈值扫描中选工作点，默认要求验证集事件召回率 ≥0.4（`--min-validation-recall`）、孤立误报段数 ≤2（`--max-isolated-false-alarms`），并支持 `--min-active-bins` 连续 bin 激活抑制孤立尖峰；若无阈值同时满足，记录 `no_operating_point`，**不再用 0.90 全静默阈值冒充方案**；
- **告警分类**：`boundary_near_event_episodes`（onset 落在窗口边界容差内）、`post_event_carryover_episodes`（事件后一个刷新间隔内）、`isolated_false_alarm_episodes`（真正与事件无关的孤立误报），严格误报保留在 `false_alarm_episodes`；
- **指标**：AP（排序指标，不是概率百分比）、Brier、事件检出率、报警段数、误报率/24h、报警时间占比、阈值扫描表 `threshold_sweep`。

### 2.6 CLI 一览

```text
python -m rockburst run                # 一键训练（索引→特征→Sundial→标签→训练→测试→预测）
python -m rockburst stream             # 按文件时间流式回放预测
python -m rockburst predict            # 指定时刻预测
python -m rockburst evaluate-forecast  # 单独评估 Sundial 未来特征预测
python -m rockburst demo / demo-raw    # 合成数据流程验证
```

---

## 三、实验历程：每次实验做了什么、结果是什么

### 3.0 流程验证期（2026-09-20 ~ 09-21）

- **做了什么**：搭建完整流水线与测试体系；用合成 bin（`demo-raw`：6 小时连续波形 + 30 个微震片段 + 5 次合成岩爆）端到端验证"bin → 特征 → 标签 → 训练 → 测试 → 预测"；实现 `stream` 流式回放、按记录文件夹分组的 70/15/15 切分、可选微震分支。
- **结果**：流程验证通过，测试从 24 项逐步增至 46 项全通过；流式回放与批量结果在三个时窗上逐点一致（容差 1e-12）。此阶段使用明确的 `persistence` 持值基线，**不是 Sundial、不构成性能结论**。

### 3.1 experiment_01 —— 首次真实数据 + Sundial（2026-09-22）

- **做了什么**：首次在真实数据上跑通 Sundial 全链路。输入 6232 个连续 bin、27 条岩爆标签（无微震目录，分支关闭），得到 4287 个样本（train 2963 / validation 615 / test 644）。
- **结果**：测试集 30 分钟 AP = **0.1785**（5min 0.0605 / 10min 0.0879）。这是第一版基线，切分方式尚未采用按记录分组。

### 3.2 experiment_02_causal_fix —— 因果性修正（2026-09-23 01:32）

- **做了什么**：修复两处方法学问题——① 每个被使用的 10 秒特征块必须严格在预测时刻前已完成（不允许未来数据混入历史窗口）；② 切分改为 `recording_group_chronological`（按记录子文件夹整体成组，25 组，按时间顺序 70/15/15）。样本变为 train 3026 / validation 589 / test 672。
- **结果**：修正后的基线测试集 30 分钟 AP = **0.2009**（5min 0.0589 / 10min 0.0877）。此后所有实验沿用此样本切分。

### 3.3 experiment_03 / 04 —— 瞬态特征消融（2026-09-23）

- **做了什么**：在**同一批样本、同一分组、同一时间切分**下，对比 baseline 与 5 个瞬态特征方案（`plus_peak_crest` / `plus_stalta` / `plus_energy_q99` / `plus_overthreshold_count` / `plus_all_transient`）。exp04 起按验证集 30min AP 自动给出 `recommended_variant`。
- **结果（测试集 30min AP，阈值 0.5 下事件检出 0/3）**：

| 方案 | 验证30 AP | 测试30 AP | 相对基线 |
|---|---|---|---|
| baseline | 0.314 | **0.201** | — |
| plus_peak_crest | 0.303 | 0.187 | -7% |
| plus_stalta | 0.293 | 0.221 | +10% |
| plus_energy_q99 | 0.303 | 0.186 | -7% |
| **plus_overthreshold_count** | **0.342** | **0.405** | **+102%** |
| plus_all_transient | 0.324 | 0.359 | +78% |

- **关键发现**：`overthreshold_count`（超背景阈值短窗计数）使 30min 测试 AP 提升约 **102%**，被推荐为首选候选（exp04 `recommended_variant=plus_overthreshold_count`）；但**阈值 0.5 下测试事件检出仍为 0/3**——排序能力改善、概率却未跨过报警阈值，暴露出概率校准问题。
- 同期 `evaluate-forecast` 诊断（见 3.6）证明 Sundial 对特征走势的预测优于持值基线 17%～22%。

### 3.4 experiment_05_balanced_abl —— 事件均衡训练（2026-09-23 23:50）

- **做了什么**：针对"所有方案正负概率集中在约 0.32～0.40"的问题，加入 `--event-balanced`（训练集每个事件正样本总权重相近、不丢样本）重跑消融。
- **结果（关键改善）**：

| 方案 | 验证30 AP | 测试30 AP | 验证检出(阈0.5) | 测试检出(阈0.5) |
|---|---|---|---|---|
| **baseline** | **0.499** | **0.449** | 4/5 | 2/3 |
| plus_stalta | 0.475 | 0.420 | 4/5 | 1/3 |
| plus_peak_crest | 0.454 | 0.387 | 4/5 | 1/3 |
| plus_overthreshold_count | 0.388 | 0.500 | 3/5 | 2/3 |
| plus_energy_q99 | 0.364 | 0.305 | 4/5 | 2/3 |
| plus_all_transient | 0.359 | 0.495 | 2/5 | 2/3 |

- **结论**：事件均衡显著拉开正负样本概率（30min 最大概率从约 0.33 提升到约 0.52），baseline 验证 30min AP 从 0.314 提升到 **0.499**；阈值 0.5 下验证集检出 4/5、测试集检出 2/3。`recommended_variant=baseline`（均衡后瞬态特征增益不再明显）。

### 3.5 experiment_06_auto_thr —— 验证集自动阈值（2026-09-26 15:27）

- **做了什么**：在均衡模型上启用 `--auto-threshold --min-active-bins 2 --max-isolated-false-alarms 2`（要求验证集 30min 召回 ≥0.4 且孤立误报 ≤2 段，连续 2 个 bin 才形成报警），让程序只凭验证集选择现场工作点。
- **结果（关键问题）**：
  - **自动阈值选中 0.9，事件召回率 0**（验证集 eligible 5 检出 0，测试集 3 检出 0，报警段 0）；
  - **根因**：均衡模型 30min 概率输出被压缩在 **0.5 附近**（全数据最大概率 ≈0.52、正样本 max ≈0.52），任何 ≥0.6 的阈值都无法触及；唯一有召回（0.8）的 0.5 档误报远超 2 段限制，规则在约束下只能退化为全静默方案；
  - 测试集指标（同 exp05 baseline）：30min AP 0.449 / Brier 0.252；最新预测（zone_A）p_5min=0.107、p_10min=0.209、p_30min=0.507，**低于选中阈值，当前不触发告警**；
  - 校准状态：`not_independently_calibrated`——小样本（27 条标签）、相邻样本相关，概率未做独立校准。

### 3.6 evaluate-forecast_30 —— Sundial 预测能力单独诊断（2026-09-23）

- **做了什么**：不读标签、不训练，直接对比 Sundial 对连续特征未来时窗均值的预测与实测，以持值基线为对照（2681 个起点）。
- **结果**：Sundial 的 MAE 比持值基线低 **17.4%（5min）/ 22.2%（10min）/ 21.6%（30min）**，且 RMSE 改善更明显。这是**特征走势预测**诊断，不等价于岩爆检出能力。

---

## 四、结果汇总

| 实验 | 核心改动 | 测试30min AP | 关键结论 |
|---|---|---|---|
| exp_01 | 首跑真实Sundial链路 | 0.179 | 流程可用，旧切分基线 |
| exp_02 | 因果修正+分组切分 | 0.201 | 基线确立（后续基准） |
| exp_03/04 | 瞬态特征消融 | 0.405(+overthr) | 超阈值计数特征 +102%，但阈值0.5检出0/3 |
| exp_05 | 事件均衡 | 0.449(baseline) | 概率分离显著改善，检出2/3 |
| exp_06 | 验证集自动阈值 | 0.449 | 选中0.9→召回0，概率压缩在0.5附近 |
| forecast_eval | Sundial预测诊断 | — | 较持值基线低17~22% |

**当前最新模型状态**（experiment_06_auto_thr / exp05 均衡基线）：
- 测试集：5min AP 0.038 / 10min AP 0.139 / 30min AP 0.449，Brier 0.047/0.095/0.252；
- 最新预测：p_5min=0.107、p_10min=0.209、p_30min=0.507（zone_A，低于阈值不告警）；
- 产物：`artifacts\experiment_06_auto_thr\`（summary.json / run/model.json / run/metrics.json / prediction.json / ablation/ 等，其中 9 个结果文件已推送 GitHub）。

---

## 五、当前已知问题与风险

1. **概率压缩在 0.5 附近**：30min 输出最大概率 ≈0.52，自动阈值只能选到全静默方案 → 需要改进校准/训练目标或增加数据，而非单纯调阈值；
2. **样本量极小**：仅 27 条岩爆标签、测试集 3 个可检事件，所有结果应视为**探索性**，多方案共享测试集不可当作独立重复实验；
3. **概率未独立校准**：相邻样本相关，未做置信区间/校准；
4. **微震分支未启用**：当前全部实验无外部微震识别结果，微震特征分支关闭；
5. **evaluate-forecast 起点重叠**：2681 个起点互不独立，仅作诊断。

## 六、下一步建议

1. **概率校准**：为均衡模型的 30min 输出做独立概率校准（如 Platt/温度缩放或 isotonic），目标是把正样本概率推向 0.5 以上，让自动阈值能选到有召回的工作点；
2. **放宽/重审误报约束**：在验证集上检查 0.5 档报警段的真实误报构成（isolated vs boundary vs post-event），确认 2 段限制是否过严；
3. **增加独立测试记录**：用新的记录子文件夹复核 overthreshold_count 与均衡 basline 的对比，避免测试集复用；
4. **启用微震分支**：整理外部微震识别结果后重训对照；
5. **做"Sundial 特征贡献"对照**：去掉 Sundial 预测特征，区分平均趋势预测与瞬态特征各自贡献。

---

## 附录 A：关键产物路径

| 产物 | 路径 |
|---|---|
| 方法文档 | `README.md` |
| 实验日志 | `EXPERIMENT_LOG.md` |
| 验证记录 | `VALIDATION.md` |
| 实验01~06 | `artifacts\experiment_01` … `artifacts\experiment_06_auto_thr` |
| Sundial 预测诊断 | `artifacts\forecast_eval_30\forecast_metrics.json` |
| 最新模型 | `artifacts\experiment_06_auto_thr\run\model.json` |
| 最新预测 | `artifacts\experiment_06_auto_thr\prediction.json` |

## 附录 B：环境与运行

- 正确解释器：`C:\Users\Dell\miniconda3\envs\yb\python.exe`（含 numpy / torch 2.6.0+cu118 / transformers 4.49）；
- Sundial 权重：`models\sundial-base-128m`（固定 SHA，见 2.4）；
- 训练命令示例：

```bash
python -u -m rockburst run \
  --continuous-dir "G:\1A岩爆预测\岩爆数据集" \
  --labels "G:\1A岩爆预测\rockbursts_full.csv" \
  --model-path models\sundial-base-128m \
  --monitor 16-35 --device cuda \
  --transient-ablation --event-balanced --auto-threshold \
  --min-active-bins 2 --max-isolated-false-alarms 2 \
  --output artifacts\experiment_06_auto_thr
```

- 仓库同步：`git pull --ff-only` + `git submodule update --init --recursive`。
