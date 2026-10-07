# Sundial 岩爆预测：从 bin 到训练

准备岩爆前后约1小时的连续bin；有外部微震识别结果时，可选提供过去24小时微震片段。模型预测未来5、10、30分钟内发生岩爆的概率。**每完成一个bin推进一次，不固定按一分钟推进。**

**需要连续bin目录和真实岩爆时间清单；微震bin目录可选。不用手填4张CSV，程序自动生成内部索引、特征和标签。**

## 1. 整体流程

```text
连续bin → 自动读取 → 10秒波形特征 → 冻结Sundial预测未来变化 ─┐
                                                           ├→ 小型概率模型 → P5/P10/P30
（可选）微震bin → 自动读取 → 前24小时片段数量、强度、时间统计 ─┘
真实岩爆时间清单 ────────────────────────────→ 生成训练标签
```

- Sundial使用已有预训练权重，只推理，不微调。
- 训练的是后面的9参数岩爆概率模型。
- 微震片段目录可选：有外部识别结果时提供；尚未整理时省略，本项目会关闭微震特征分支。本项目不重新训练识别器。
- `run`一次执行索引、提取特征、Sundial推理、标签构建、训练、测试和最新一次预测。
- `stream`使用训练好的模型按文件时间流式回放，一次输出一条预测，不需要岩爆标签。
- 当前项目已用真实岩爆数据完成 P0 概率校准与 P1 平替对照（见第 14 节）；`persistence` 仍是无权重的流程验证基线，不是Sundial。

## 2. 下载和安装

建议Linux/WSL、Python 3.12。仓库公开，可直接克隆：

```bash
git clone --recurse-submodules https://github.com/zhouyllll/sundial-rockburst.git
cd sundial-rockburst
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

本项目可以独立运行，不需要原有GUI。

## 3. 没有真实数据，先跑完整演示

```bash
python -m rockburst demo-raw --output artifacts/demo_raw
```

命令生成约212MB的**合成bin**：6小时连续波形、30个微震片段、5次合成岩爆时间，然后通过与真实数据相同的主流程完成：

```text
bin读取 → 特征提取 → 标签生成 → 模型训练 → 测试 → 输出预测
```

使用`persistence`持值基线，无需Sundial权重。结果是流程验证，不是现场预测性能。

```text
artifacts/demo_raw/
  raw/continuous/       合成连续bin
  raw/microseismic/     合成微震bin
  rockbursts.txt        合成岩爆时间
  result/              训练和预测结果
```

测试命令：

```bash
python -m unittest discover -s tests -v
```

另有更轻量的`python -m rockburst demo`，直接从合成特征开始。

## 4. 真实数据准备

准备以下目录和文件；bin目录支持递归扫描。连续数据可以按“一个事件一个子文件夹”存放，每个子文件夹包含岩爆前后约1小时的连续bin，微震目录可省略：

```text
data/
  continuous/          连续采集的bin根目录（可含多个记录子文件夹）
    event_001/
      *.bin
    event_002/
      *.bin
  microseismic/        （可选）外部程序检出的微震片段bin
  rockbursts.txt       已确认的真实岩爆时间
```

### bin格式已经固定，无需逐文件填写

例如现有文件名：

```text
0008655-fs-eDAS-5000Hz-0041pt-20260621T151630.846.bin
```

| 信息 | 自动获取方式 |
|---|---|
| 采样率 | `5000Hz` |
| 通道数 | `0041pt` |
| 起始时间 | `20260621T151630.846` |
| 数据排列 | 无文件头、小端int32、采样点×通道交错 |
| 时长 | 文件字节数 ÷ 4 ÷ 通道数 ÷ 采样率 |

微震分支按一个bin一个已检出片段计算整段特征；统计的是检出片段数和片段能量代理量。以后有精确事件边界时，再使用分步接口。

### 只提供真实岩爆发生时间

`rockbursts.txt`支持完整日期时间，也支持你目前的“每天时刻范围”格式，每行一个岩爆事件：

```text
19:39:23-19:39:24
3:34:53-3:34:54
00:06:10-00:06-11
13:12:19-13.12:20
```

时间范围的开始时刻作为岩爆发生时刻，结束时刻保留为事件结束时间。像`00:06-11`和`13.12:20`这样的分隔符小错误也会自动规范化。对于只有时分秒、没有日期的记录，程序会根据每个记录子文件夹内bin文件名的日期和采集时间补齐日期，并要求每条记录只能匹配到一个子文件夹；若匹配不到或有歧义会停止并指出行号。每个子文件夹代表一段连续记录，可以包含多个岩爆；每个岩爆在txt中单独列出。bin文件名需含日期时间字段。直接把原始时刻范围txt传给`--labels`即可；不要先用旧的`_make_labels.py`按“最接近的文件夹钟点”生成`rockbursts_full.csv`，那种近似配对在无日期标签时可能错配岩爆日期。

例如在Windows上的目录`G:\1A岩爆预测\岩爆数据集\5.10.8.49`，WSL中通常对应`/mnt/g/1A岩爆预测/岩爆数据集/5.10.8.49`。如果已有微震片段目录，使用下面的WSL命令：

```bash
python -m rockburst run \
  --continuous-dir "/mnt/g/1A岩爆预测/岩爆数据集/5.10.8.49" \
  --microseismic-dir "/mnt/g/1A岩爆预测/微震识别结果" \
  --labels data/rockbursts.txt \
  --model-path models/sundial-base-128m \
  --device cuda \
  --output artifacts/experiment_01
```

程序也会在WSL/Linux中把传入的`G:\\...`盘符路径尝试转换为`/mnt/g/...`。微震文件目录如有则通过`--microseismic-dir`单独提供；微震文件名不能替代人工确认的岩爆标签。

微震片段尚未整理时，直接省略`--microseismic-dir`：

```bash
python -m rockburst run \
  --continuous-dir "G:\\1A岩爆预测\\岩爆数据集\\5.10.8.49" \
  --labels data/rockbursts.txt \
  --model-path models/sundial-base-128m \
  --device cuda \
  --output artifacts/experiment_without_microseismic
```

该实验会在`input_summary.json`标记微震分支已关闭，并以没有微震输入的配置训练模型。后续流式预测也必须省略该目录；训练时启用了微震分支的模型则必须在流式预测时提供目录。两种模式不能混用，模型输出也应分别评估。

默认时区为北京时间`Asia/Shanghai`；也支持`2026-06-21T15:30:00+08:00`。已有包含`onset_time`列的CSV也可以直接作为`--labels`。

## 5. 下载Sundial权重

按[PyTorch官方安装页](https://pytorch.org/get-started/locally/)安装适合本机CPU/CUDA的PyTorch，再执行：

```bash
python -m pip install -r requirements-sundial.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

适配器固定`transformers==4.40.1`。以下命令查询并下载官方模型的固定提交版本：

```bash
python - <<'PY'
import subprocess
import sys
from huggingface_hub import HfApi

revision = HfApi().model_info("thuml/sundial-base-128m").sha
print("固定模型提交:", revision, flush=True)
subprocess.run([
    sys.executable, "download_model.py", "--revision", revision,
    "--output", "models/sundial-base-128m",
], check=True)
PY
```

`vendor/Sundial`是官方源码子模块，不是权重。权重和自定义模型代码来自Hugging Face；下载后从本地加载。也可手工指定模型仓库40位SHA：

```bash
python download_model.py --revision 模型仓库的完整40位提交SHA --output models/sundial-base-128m
```

## 6. 一条命令从真实bin到训练

```bash
python -m rockburst run \
  --continuous-dir data/continuous \
  --microseismic-dir data/microseismic \
  --labels data/rockbursts.txt \
  --model-path models/sundial-base-128m \
  --device cuda \
  --output artifacts/experiment_01
```

默认使用Sundial、30分钟连续历史；提供微震目录时还会使用过去24小时的微震片段。CPU运行把`--device cuda`改成`--device cpu`。

30分钟窗口已经是默认值；指定监测通道时追加`--monitor 15-34`。

不下载权重、先验证自己的bin能否跑通时：

```bash
python -m rockburst run \
  --continuous-dir data/continuous \
  --microseismic-dir data/microseismic \
  --labels data/rockbursts.txt \
  --backend persistence \
  --output artifacts/baseline_01
```

简化主流程默认文件结束即可用，提供的微震目录和岩爆清单为实验的完整记录，不再要求填写覆盖表。假设保存在`input_summary.json`。

## 7. 比较短时瞬态特征是否改善预警

在完整的`run`训练命令末尾加`--transient-ablation`，例如：

```bash
python -m rockburst run \
  --continuous-dir "/mnt/g/1A岩爆预测/岩爆数据集/5.10.8.49" \
  --labels data/rockbursts.txt \
  --model-path models/sundial-base-128m \
  --device cuda \
  --output artifacts/transient_ablation \
  --transient-ablation
```

程序会先照常训练原有基线，再用**同一批样本、同一文件夹分组和同一时间切分**重复训练以下对照：

- `baseline`：当前已有输入特征。
- `plus_peak_crest`：最近5分钟内10秒块峰值、峰均比。
- `plus_stalta`：最近5分钟内STA/LTA。
- `plus_energy_q99`：每个10秒块内100毫秒短窗能量q99，再对最近5分钟取q95。
- `plus_overthreshold_count`：最近5分钟中超过该10秒块背景阈值的100毫秒短窗数。
- `plus_all_transient`：以上瞬态特征全部加入。

如果发现正负概率严重重叠，可以增加`--event-balanced`重新训练。例如：

```bash
python -m rockburst run ... --transient-ablation --event-balanced
```

该选项不会丢弃样本，而是让训练集中每个岩爆事件的正样本总权重相近，避免大量相邻正常窗口主导损失函数。它只改变训练权重，验证集和测试集仍保持原始分布。

可以让程序只用验证集自动选择报警阈值，并要求连续两个bin超过阈值才触发报警：

```bash
python -m rockburst run ... --event-balanced --auto-threshold --min-active-bins 2
```

阈值选择要求验证集30分钟召回率至少达到`--min-validation-recall`（默认0.4），并优先满足孤立误报段数不超过`--max-isolated-false-alarms`（默认2）。如果没有任何阈值同时满足，模型会记录`no_operating_point`，不会把0.90等全静默阈值伪装成可用方案。选出的阈值会写入`model.json`，测试集不会参与选择。

短窗以100毫秒为窗长、50毫秒步长；超阈值按每个10秒块的短窗能量中位数加3倍稳健标准差计算。新增特征只从预测时刻以前已经可用的数据计算，不使用未来波形。要捕捉亚秒变化，需用原始bin重新提取特征；旧版`continuous.csv`没有新增短窗列。

结果在`artifacts/transient_ablation/ablation/`：`comparison.csv`按验证集/测试集列出三个预测时窗的AP、Brier、事件检出率及检出/可检事件数；`summary.json`会按**验证集30分钟AP**给出`recommended_variant`，测试集只作最终一次评估；每个方案的独立子目录保留模型、指标和预测明细。每个`metrics.json`还包含各时窗的最大概率、正负样本最大概率和阈值扫描表。事件检出率按`--threshold`（默认0.5）计算；若AP提高但0.5阈值检出为0，应根据验证集阈值扫描选择现场阈值，再锁定后评估测试集。由于这里只使用少量岩爆事件，多个对照共享测试集的结果属于探索性比较，不能当成独立重复实验或已验证的现场预警性能。

阈值扫描不会修改模型概率，只用于诊断不同报警阈值下的事件检出率、误报次数和报警时长。不能根据测试集的最佳阈值或最佳特征方案反复调参；最终报告应保留验证集选择过程和一次性的测试集结果。AP是排序指标，不是概率百分比；阈值0.5下`0/3`表示没有测试事件触发报警，不表示AP等于零。

评估会使用数据集保存的真实`onset_time`区分告警类型：预测窗口刚好越过 onset 的告警计入`boundary_near_event_episodes`，事件发生后一个刷新间隔内的告警计入`post_event_carryover_episodes`，只有与事件时间无关的告警才计入`isolated_false_alarm_episodes`。严格误报仍保留在`false_alarm_episodes`中。边界容差取预测刷新间隔，而不是把所有情况粗略固定成一个bin。

## 8. 当前概率模型如何工作

当前模型不是“特征超过某个阈值就报警”。流程是：Sundial（或持值基线）先从最近30分钟连续波形提取并预测特征走势；再把连续波形特征、过去24小时微震统计和可选瞬态特征输入一个三时窗离散时间风险模型。模型学习共享特征系数和5、10、30分钟三个区间截距，输出未来各时窗内至少发生一次岩爆的累计概率，并通过累计风险计算保证`P5 <= P10 <= P30`。

最终报警才使用阈值，例如`p_30min >= 0.30`。因此，阈值只是把概率转换成报警状态；如果正负样本概率完全重叠，单纯调阈值无法解决问题，需要改进训练权重、特征或数据量。`--event-balanced`是针对当前类别和相邻窗口失衡问题的受控对照。

## 9. 直接评估Sundial的未来特征预测

训练流程生成 `continuous.csv` 后，可以单独检查 Sundial 对连续特征的预测是否接近后续实测：

```bash
python -m rockburst evaluate-forecast \
  --features artifacts/experiment_01/continuous.csv \
  --model-path models/sundial-base-128m \
  --device cuda \
  --history-minutes 30 \
  --step-seconds 30 \
  --output artifacts/forecast_eval_01
```

该命令不读取岩爆标签，也不训练概率模型。每个预测起点取此前30分钟的10秒 `mean_square` 特征，转换成 `log1p(mean_square)` 后交给Sundial，生成未来30分钟预测；再分别对比未来5、10、30分钟的预测特征均值与真实特征均值。默认每30秒一个起点，需与连续bin时长一致；例如bin为1分钟时传 `--step-seconds 60`。缺失区间或采集配置变化会切断序列，不跨断点拼接预测窗口。

同时计算持值基线：假设未来特征一直等于起点时最近的观测值。查看 `forecast_metrics.json` 中各时窗的 `mae_log_power`、`rmse_log_power`、`bias_log_power`，以及 `mae_improvement_vs_persistence`。改善比例大于0表示Sundial的MAE低于持值基线，小于0表示更差；为0附近表示两者接近。`forecast_comparison.csv` 保留每个起点预测值、真实值和误差，可用于定位具体时段。

这评估的是**能量特征的时窗平均走势**，不是原始高频岩爆波形重建，也不是岩爆概率准确率。每30秒起点的30分钟目标窗口彼此重叠，因此起点数不等于独立样本数；应把它作为模型诊断，不把这些行当成独立事件做显著性结论。

## 10. 自动训练步骤

1. 从文件名和大小生成两路文件索引。
2. 连续数据每10秒提取特征，微震数据每个片段提取特征。
3. 每完成一个bin构造一次预测样本，取最近30分钟连续特征和前24小时微震统计；保留文件时间中的毫秒。
4. Sundial预测连续特征的未来变化，融合为6维特征。
5. 根据真实岩爆时间建立5、10、30分钟标签。
6. 将同一连续记录子文件夹内的正负样本整体分组；时间相邻到输入历史或未来30分钟标签可能跨界的文件夹合并为同一组，再按组的时间顺序约70%/15%/15%划分训练、验证、测试，并在边界剔除跨界标签。
7. 训练3个区间截距和6个共享系数，默认正则强度1，不做大范围参数搜索。
8. 保存模型、分时段报告和最新一次预测。

概率含义为未来时间段内至少发生一次岩爆，满足`P5 <= P10 <= P30`。Sundial负责预测数值走势，后面的小模型负责学习岩爆概率。

保存的数据在岩爆时刻结束，也能使用已确认的岩爆时间生成该事件前的正样本。正常时段仍用于训练负样本；训练用的真实标签与流式预测输入分开。

## 11. 查看结果

```text
artifacts/experiment_01/
  generated/                       自动生成，无需手填
    continuous_manifest.csv
    microseismic_manifest.csv
    rockbursts.csv
    coverage.csv
  input_summary.json               输入规模和简化假设
  continuous.csv                   连续波形特征
  events.csv                       微震片段特征
  dataset.npz                      融合特征及标签
  dataset.npz.report.json           有效样本统计
  cache/                           推理缓存
  run/
    model.json                     训练好的小型概率模型
    metrics.json                   训练/验证/测试指标
    train_predictions.csv
    validation_predictions.csv
    test_predictions.csv
  prediction.json                  最新一次5/10/30分钟概率
  summary.json                     整条流程结果摘要
```

真实数据、权重、缓存和输出已被Git忽略，不会上传。运行时不会把基线自动冒充Sundial。

## 12. 按一个bin一步流式预测

先用上面的`run`得到30分钟窗口模型，再运行：

```bash
python -m rockburst stream \
  --continuous-dir data/continuous \
  --microseismic-dir data/microseismic \
  --model artifacts/experiment_01/run/model.json \
  --model-path models/sundial-base-128m \
  --device cuda \
  --output artifacts/stream_01
```

不需要`--labels`。backend和生成样本数从训练好的模型读取；训练时选了监测通道，这里传入相同的`--monitor`。训练时提供了微震目录就继续提供；训练时省略了微震目录，这里也要省略。

按每个bin为30秒举例：

| 已处理文件 | 已积累连续数据 | 模型输入及动作 |
|---|---|---|
| 第1～59个 | 不足30分钟 | 积累窗口 |
| 第60个 | 30分钟 | 用第1～60个对应的最近30分钟，第一次预测 |
| 第61个 | 30分30秒 | 窗口前移30秒，用最近30分钟，再预测 |
| 第62个 | 31分钟 | 再前移一个bin，再预测 |
| … | … | 每个新bin输出一次 |
| 第120个 | 1小时 | 用最后30分钟预测 |

实际步长来自文件时长，不把文件强制当成30秒。跨天保存的多个1小时片段会各自积累窗口，时间空档不会拼成连续波形。微震只纳入预测时刻之前已完成的片段。

每次立即写入并刷新：

```text
artifacts/stream_01/
  predictions.jsonl    每行一次预测，处理过程中就能读取
  predictions.csv      同步输出的表格
  latest.json          最新一次预测
  summary.json         回放完成后的统计
```

这版是**对已有bin按文件时间加速流式回放**，不会等待真实30秒，也不监听目录里之后新增的文件。原始波形只分块处理，连续特征窗口保留最近30分钟，微震特征保留最近24小时。增加`--quiet`可关闭逐行终端输出，文件仍逐条刷新。

用合成bin验证（先运行第3节的`demo-raw`，它已默认训练30分钟模型）：

```bash
python -m rockburst stream \
  --continuous-dir artifacts/demo_raw/raw/continuous \
  --microseismic-dir artifacts/demo_raw/raw/microseismic \
  --model artifacts/demo_raw/result/run/model.json \
  --output artifacts/demo_stream
```

如果已有的是旧60分钟模型，先重新运行`demo-raw`或`run`，不要直接改模型JSON里的窗口长度。

### 用模型做指定时刻的预测

`run`已自动生成最新一次预测。其他时刻可以用已经生成的文件：

```bash
python -m rockburst predict \
  --continuous artifacts/experiment_01/continuous.csv \
  --events artifacts/experiment_01/events.csv \
  --coverage artifacts/experiment_01/generated/coverage.csv \
  --zone zone_A \
  --model artifacts/experiment_01/run/model.json \
  --backend sundial \
  --model-path models/sundial-base-128m \
  --device cuda \
  --at 2026-06-25T12:00:00+08:00 \
  --output artifacts/experiment_01/prediction_at.json
```

替换为数据中的实际时刻，保持与训练相同的backend、权重、设备、样本数和区域。预测不需要未来岩爆标签。

## 13. P0 概率校准与 P1 平替对照（2026-10）

真实数据小样本实验后，概率排序能力可用（均衡后验证30min AP≈0.499、测试≈0.449），但**30min 概率未独立校准、自动阈值规则失效**。以下两项实验用同一时间划分（train_end=2026-07-25T05:30:44.568+00:00、validation_end=2026-07-29T17:06:38.837500+00:00）和同一协议落地。

### P0：概率标度校准（experiment_07_calibrated）

问题：30min 概率全数据最大只有≈0.52、未独立校准，`no_operating_point` 时自动阈值规则回退到固定 0.5。

方案（`_p0_calibration.py`，产出 `artifacts/experiment_07_calibrated/`）：

- **Platt 校准**（IRLS 拟合）+ **链接函数选择**（logit / cloglog）+ **正则网格选参**（ridge × l1 × link，按验证集 NLL 选参），不触碰测试集。
- 校准后按验证集扫描三步阈值（要求30min召回≥0.4、孤立误报≤2），选不到工作点就显式记录 `no_operating_point`，不再静默回退高阈值。
- `_p0_selftest.py`：校准前后 NLL / 最大概率 / ECE 的数值自检。

结论：校准修复了概率标度（ECE 明显下降），但**造不出排序分离**——30min 概率仍挤在 [0.5, 0.6) 顶部扁平，真正瓶颈是 30min 窗判别信息不足，由此启动 P1。

### P1：Sundial 平替对照（experiment_08_p1）

动机：验证瓶颈是否在 Sundial 30min 预测环节。

方案：三个 Apache-2.0 开源时序模型 **chronos2 / TiRex / Lag-Llama** 替换 Sundial 的 30min 预测，复用 P0 下游校准协议（feature 取前 6 列 baseline + include_transient，ridge×l1×link 按验证 NLL 选参 + Platt 校准 + 三步阈值），跳过索引与特征提取，只做「模型推理建集 → 训练概率模型 → 保存结果」等价流程。缓存 key 与 `P1Forecaster.predict` 一致，批量预填脚本（`_p1_tirex_prefill.py` / `_p1_lagllama_prefill.py`）先行落盘全部预测。

30min 指标（测试 AP / 验证 AP / 验证 recall / 验证孤立误报）：

| 方案 | 验证 NLL | 测试 NLL | 验证 AP | 测试 AP | 验证 recall | 验证孤立误报 |
|---|---|---|---|---|---|---|
| Sundial 基线 | 0.6914 | 0.7129 | **0.4340** | 0.3746 | 0.8 | — |
| chronos2 | 0.6924 | 0.7024 | 0.3905 | 0.4552 | 0.8 | 4 |
| tirex | **0.6918** | 0.7069 | 0.3828 | 0.4909 | **1.0** | 4 |
| lagllama | 0.6965 | **0.6930** | 0.3918 | **0.5562** | 0.6 | 9 |

结论：三方案测试期 AP 全部显著超过 Sundial 基线（lagllama +48%），测试 NLL 也均更优——**平替成立**；但验证期基线 AP 仍最高、lagllama 验证误报偏多，且 maxp 仍压在 ≈0.53，标度问题跨模型一致，属 30min 窗信息瓶颈的系统性现象。测试集仅 3 个可检事件、多方案共享测试集，结果属**探索性**。

脚本：`_p1_comparison.py`（主对照，进度写 `p1_run.log`）、`_p1_backends.py`（三后端适配）、`_p1_tirex_prefill.py` / `_p1_lagllama_prefill.py`（批量预填）、`_p1_smoke.py` / `_p1_smoke_lag.py`（冒烟）；`_p1_lagllama_src/` 是本地 Lag-Llama 官方建模（含 import 修复与自回归 future_time_feat 语义适配，`use_single_pass_sampling=True` 单趟采样提速约20倍）。

### P2：掩码重建自监督预训练（experiment_09_p2，2026-10）

动机：P0/P1 证明 maxp 始终压在 ≈0.53，瓶颈是 30min 窗判别信息不足；用全部无标签波形学表征来抬信息上限。

方案：在 `continuous.csv` 全部 18285 个 10s 特征块（26 段，约 51 小时无标签波形）上训练 3 层 Transformer encoder（59.7 万参数，128 维）做掩码重建（80% 单点 + 20% 连续块掩码，MSE）；对每个下游样本取其预测时刻前 30min 历史窗口，mean-pool 得 128 维表征，在训练集上 PCA 降 8 维并 z-score，拼接进原 x 前 6 列（sundial / lagllama 两个特征源各一套），再以 P0 协议微调 9 参数风险模型。

30min 指标（同一时间划分 + P0 协议）：

| 方案 | 验证 NLL / maxp / AP / 误报段 | 测试 NLL / maxp / AP / 误报段 |
|---|---|---|
| Sundial 基线（exp07） | 0.6914 / 0.544 / 0.4340 / 5 | 0.7129 / 0.557 / 0.3746 / 4 |
| P2 sundial+表征 | 0.6908 / 0.592 / 0.4693 / 16 | 0.6941 / **0.656** / 0.5098 / 6 |
| P2 lagllama+表征 | 0.6958 / 0.623 / 0.4036 / 12 | **0.6701** / 0.654 / **0.5623** / 7 |
| P2 tirex+表征 | 0.6903 / 0.575 / 0.4522 / 10 | 0.6984 / 0.628 / 0.5587 / 6 |

结论：**测试 maxp 首次突破 0.53 瓶颈（0.557→0.656，+18%）**，测试 AP 较基线 +36%（sundial 0.5098）、+49%（tirex 0.5587），P2 lagllama 测试 NLL 0.6701 首次系统性低于 ln2；表征增益跨三个预测器一致（tirex 测试 AP 0.4909→0.5587，且验证召回保持 1.0）；但验证集孤立误报段普遍上升（sundial 5→16、lagllama 9→12、tirex 4→10），阈值规则仍 `no_operating_point`（2 段限制过严），测试 ECE 0.28 仍欠标定。预训练用全部无标签窗口（与报告口径一致），PCA/尺度统计只在训练集拟合；测试集仅 3 个可检事件，属探索性。

脚本：`_p2_pretrain.py`（掩码重建预训练）、`_p2_extract.py`（表征提取 + PCA + 数据集装配）、`_p2_finetune.py`（P0 协议微调，汇总 `p2_summary.json`）。

## 14. 配置与源码

建议8～16核CPU、32～64GB内存、单张8～16GB显存GPU，或先用CPU。工作盘1TB SSD起步，按原始数据量扩充。无需多卡，小概率模型在CPU训练。

5000Hz、42通道、int32下，一小时连续数据约3.02GB，30个30秒片段约0.76GB；分块在CPU提取特征，不把整段原始波形送入显存。上述为估算，尚无本机真实Sundial吞吐实测。25次真实岩爆先用于小样本实验，当前概率未独立校准。

- `rockburst/workflow.py`：自动索引、一键训练、原始bin演示。
- `rockburst/extract.py`：读取和波形特征。
- `rockburst/forecast.py`：冻结Sundial推理。
- `rockburst/data.py`：融合和标签。
- `rockburst/model.py`：小型概率模型。
- [分步流程参考](docs/advanced-workflow.md)：保留原有手工控制接口，主流程不必使用。
- [验证记录](VALIDATION.md)。
- [Sundial官方仓库](https://github.com/thuml/Sundial)与[模型权重](https://huggingface.co/thuml/sundial-base-128m)。

查看参数或更新代码：

```bash
python -m rockburst run --help
git pull --ff-only
git submodule update --init --recursive
```
