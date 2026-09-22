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
- 当前尚未用真实岩爆训练。`persistence`是无权重的流程验证基线，不是Sundial。

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
  continuous/          连续采集的bin根目录（可含多个事件子文件夹）
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

时间范围的开始时刻作为岩爆发生时刻，结束时刻保留为事件结束时间。像`00:06-11`和`13.12:20`这样的分隔符小错误也会自动规范化。对于只有时分秒、没有日期的记录，程序会根据每个事件子文件夹内bin文件名的日期和采集时间补齐日期，并要求每条记录只能匹配到一个子文件夹；若匹配不到或有歧义会停止并指出行号。请确保每个子文件夹对应一个事件时段，且bin文件名含日期时间字段。

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

## 7. 自动训练步骤

1. 从文件名和大小生成两路文件索引。
2. 连续数据每10秒提取特征，微震数据每个片段提取特征。
3. 每完成一个bin构造一次预测样本，取最近30分钟连续特征和前24小时微震统计；保留文件时间中的毫秒。
4. Sundial预测连续特征的未来变化，融合为6维特征。
5. 根据真实岩爆时间建立5、10、30分钟标签。
6. 按时间顺序约70%/15%/15%划分训练、验证、测试，沿用已有的跨界标签剔除逻辑。
7. 训练3个区间截距和6个共享系数，默认正则强度1，不做大范围参数搜索。
8. 保存模型、分时段报告和最新一次预测。

概率含义为未来时间段内至少发生一次岩爆，满足`P5 <= P10 <= P30`。Sundial负责预测数值走势，后面的小模型负责学习岩爆概率。

保存的数据在岩爆时刻结束，也能使用已确认的岩爆时间生成该事件前的正样本。正常时段仍用于训练负样本；训练用的真实标签与流式预测输入分开。

## 8. 查看结果

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

## 9. 按一个bin一步流式预测

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

## 10. 配置与源码

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
