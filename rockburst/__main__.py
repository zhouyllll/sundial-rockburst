import argparse
import json
import sys

import numpy as np

from .data import Inputs, Unavailable, build_dataset
from .extract import extract
from .forecast import Forecaster
from .io import iso, read_json, timestamp, write_json
from .model import probabilities, train


def add_inputs(parser, labels=False):
    parser.add_argument("--continuous", required=True, help="extract continuous 生成的特征 CSV")
    parser.add_argument("--events", required=True, help="extract events 生成的特征 CSV；可只有表头")
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--zone", required=True)
    if labels:
        parser.add_argument("--rockbursts", required=True, help="人工确认岩爆表，非微震识别表")


def add_forecaster(parser, default=None):
    parser.add_argument("--backend", choices=["sundial", "persistence"], default=default,
                        required=default is None, help="persistence 是流程测试/基线，不是 Sundial")
    parser.add_argument("--model-path", help="本地 Hugging Face Sundial 完整权重目录")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--cache", help="Sundial 推理结果缓存目录")


def parser():
    root = argparse.ArgumentParser(description="连续DAS + 24小时微震片段的实验性岩爆预测基础版")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="主入口：两个bin目录+岩爆时间清单，一次完成读取到训练")
    p.add_argument("--continuous-dir", required=True)
    p.add_argument("--microseismic-dir", required=True)
    p.add_argument("--labels", required=True, help="岩爆时间txt，每行一个时间；也支持含onset_time的CSV")
    p.add_argument("--output", default="artifacts/experiment")
    p.add_argument("--zone", default="zone_A")
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.add_argument("--history-minutes", type=int, choices=[30, 60], default=30)
    p.add_argument("--monitor", default="all")
    p.add_argument("--ridge", type=float, default=1.)
    p.add_argument("--threshold", type=float, default=.5)
    add_forecaster(p, default="sundial")
    p = commands.add_parser("stream", help="按bin时间顺序流式回放，每到一个完整bin预测一次")
    p.add_argument("--continuous-dir", required=True)
    p.add_argument("--microseismic-dir", required=True)
    p.add_argument("--model", required=True, help="30分钟窗口训练的run/model.json")
    p.add_argument("--model-path", help="本地Sundial权重目录；backend自动沿用训练模型")
    p.add_argument("--device", default="cpu")
    p.add_argument("--timezone", default="Asia/Shanghai")
    p.add_argument("--monitor", default="all")
    p.add_argument("--output", default="artifacts/stream")
    p.add_argument("--quiet", action="store_true", help="不逐行打印，仍立即写入JSONL和CSV")
    p = commands.add_parser("demo-raw", help="生成合成bin并跑通目录读取到训练的完整主流程")
    p.add_argument("--output", default="artifacts/demo_raw")
    p = commands.add_parser("extract", help="从 bin manifest 流式提取特征")
    p.add_argument("kind", choices=["continuous", "events"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--monitor", default="all", help="1起始通道，如15-34或all")
    p.add_argument("--band-low", type=float, default=50.)
    p.add_argument("--band-high", type=float, default=500.)
    p = commands.add_parser("build", help="构造可用时间约束下的融合特征与标签")
    add_inputs(p, labels=True)
    add_forecaster(p)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="不包含的右端点")
    p.add_argument("--history-minutes", type=int, choices=[30, 60], default=30)
    p.add_argument("--max-lag-seconds", type=int, choices=range(0, 61), default=60,
                   metavar="0..60", help="允许最新完整波形块延迟，默认最多60秒")
    p.add_argument("--output", required=True)
    p = commands.add_parser("train", help="按时间划分并拟合9参数概率模型")
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--validation-end", required=True)
    p.add_argument("--ridge", type=float, nargs="+", default=[.1, 1., 10.])
    p.add_argument("--threshold", type=float, default=.5, help="预先固定的实验报警阈值")
    p.add_argument("--output", required=True)
    p = commands.add_parser("predict", help="在指定时刻输出三个累计概率")
    add_inputs(p)
    add_forecaster(p)
    p.add_argument("--model", required=True, help="train 输出的 model.json")
    p.add_argument("--at", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("demo", help="合成数据+持值基线，无联网/权重下载")
    p.add_argument("--output", default="artifacts/demo")
    return root


def make_forecaster(args):
    return Forecaster(args.backend, args.model_path, args.device, args.samples, args.cache)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            from .workflow import run_workflow
            result = run_workflow(args.continuous_dir, args.microseismic_dir, args.labels,
                                  args.output, args.backend, args.model_path, args.device,
                                  args.samples, args.cache, args.zone, args.timezone,
                                  args.history_minutes, args.monitor, args.ridge, args.threshold)
        elif args.command == "stream":
            from .stream import replay_stream
            result = replay_stream(args.continuous_dir, args.microseismic_dir, args.model,
                                   args.output, args.model_path, args.device, args.timezone,
                                   args.monitor, args.quiet)
        elif args.command == "demo-raw":
            from .workflow import raw_demo
            result = raw_demo(args.output)
        elif args.command == "extract":
            result = extract(args.manifest, args.output, args.kind, args.monitor,
                             args.band_low, args.band_high)
        elif args.command == "demo":
            from .demo import demo
            result = demo(args.output)
        elif args.command == "train":
            report = train(args.dataset, args.output, timestamp(args.train_end),
                           timestamp(args.validation_end), args.ridge, args.threshold)
            result = {"output": args.output, "selected_ridge": report["selected_ridge"],
                      "test": report["splits"]["test"], "calibration": report["calibration"]}
        elif args.command == "build":
            data = Inputs(args.continuous, args.events, args.coverage, args.zone, args.rockbursts)
            result = build_dataset(data, make_forecaster(args), timestamp(args.start), timestamp(args.end),
                                   args.output, args.history_minutes, args.max_lag_seconds)
        else:
            model = read_json(args.model)
            data = Inputs(args.continuous, args.events, args.coverage, args.zone)
            forecast = make_forecaster(args)
            meta = model["metadata"]
            if meta["forecast"] != forecast.identity:
                raise ValueError("预测器与训练时不同（backend/权重/样本数/运行环境）；请用同一配置重建数据并训练")
            if meta["zone_id"] != args.zone or meta["preprocessing_id"] != data.signature:
                raise ValueError("区域或预处理配置与训练时不一致")
            now = timestamp(args.at)
            try:
                x, quality = data.sample(now, forecast, meta["history_minutes"], meta["max_lag_seconds"])
            except Unavailable as exc:
                result = dict(status="insufficient_data", reason=str(exc), time=iso(now),
                              p_5min=None, p_10min=None, p_30min=None)
                write_json(args.output, result)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 2
            p = probabilities(x[None], model)[0]
            result = dict(status="ok", time=iso(now), zone_id=args.zone,
                          p_5min=float(p[0]), p_10min=float(p[1]), p_30min=float(p[2]),
                          backend=forecast.identity, experimental=True,
                          calibration=model["calibration"], quality=quality)
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
