"""P1：三后端（Chronos-2 / TiRex / Lag-Llama）平替 Sundial 的完整对照实验。

复用 experiment_06_auto_thr 的全部中间产物（continuous.csv / events.csv /
coverage / rockbursts / manifest），只把 forecaster 换成 P1 后端重建 dataset，
再用与 P0 完全相同的下游协议（ridge x l1 x link NLL 选参 + Platt 校准 +
三步阈值）训练评估，产出 experiment_08_p1/{backend}/*，汇总 p1_summary.json。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, str(ROOT))
EXP06 = ROOT / "artifacts" / "experiment_06_auto_thr"
OUT = ROOT / "artifacts" / "experiment_08_p1"
CONTINUOUS_DIR = Path(r"G:\1A岩爆预测\岩爆数据集")
ZONE = "zone_A"
TIMEZONE = "Asia/Shanghai"
HISTORY_MINUTES = 30

from rockburst.io import read_csv, timestamp
from rockburst.data import Inputs, build_dataset
from rockburst.workflow import local_time, recording_group_intervals
from rockburst.model import train, evaluate
from _p1_backends import P1Forecaster


def log(msg):
    print(msg, flush=True)
    with (OUT / "p1_run.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"[P1] 复用 exp06 中间产物: {EXP06}")

    rows = read_csv(str(EXP06 / "generated" / "continuous_manifest.csv"),
                    ["file_path", "zone_id", "start_time", "available_time", "fs", "channels"])
    first = min(local_time(r["start_time"], TIMEZONE) for r in rows)
    last = max(local_time(r["available_time"], TIMEZONE) for r in rows)
    start = first + HISTORY_MINUTES * 60
    end = last + 1e-6
    prediction_times = [local_time(r["available_time"], TIMEZONE) for r in rows]
    source_groups = recording_group_intervals(rows, CONTINUOUS_DIR, TIMEZONE, HISTORY_MINUTES)
    log(f"[P1] 时间窗: {start:.0f} ~ {end:.0f}  预测时刻 {len(prediction_times)}  分组 {len(source_groups)}")

    data = Inputs(str(EXP06 / "continuous.csv"), str(EXP06 / "events.csv"),
                  str(EXP06 / "generated" / "coverage.csv"), ZONE,
                  str(EXP06 / "generated" / "rockbursts.csv"), microseismic_enabled=False)

    exp06_model = json.loads((EXP06 / "run" / "model.json").read_text(encoding="utf-8"))
    train_end = timestamp(exp06_model["train_end"])
    validation_end = timestamp(exp06_model["validation_end"])
    log(f"[P1] 时间划分（与 exp06/P0 相同）: train_end={exp06_model['train_end']} "
        f"validation_end={exp06_model['validation_end']}")

    summary = {"sundial_baseline": "artifacts/experiment_07_calibrated/p0_summary.json",
               "splits_window": exp06_model["train_end"]}
    for backend in ("chronos2", "tirex", "lagllama"):
        out = OUT / backend
        out.mkdir(parents=True, exist_ok=True)
        if (out / "dataset.npz").exists() and (out / "model.json").exists():
            log(f"[P1:{backend}] 已有完整产物，跳过 -> {out}")
        else:
            log(f"[P1:{backend}] 构建 dataset（模型推理，预计较久）...")
            fc = P1Forecaster(backend, device="cuda", samples=20,
                              hf_cache=ROOT / "_p1_cache" / backend)
            log(f"[P1:{backend}] 模型就绪: {fc.identity['torch_version']} {fc.identity['device']}")
            report = build_dataset(data, fc, start, end, out / "dataset.npz", HISTORY_MINUTES,
                                   prediction_times=prediction_times, source_groups=source_groups,
                                   include_transient=True)
            log(f"[P1:{backend}] dataset 完成: 样本 {report['samples']} 事件 {report['independent_events']} "
                f"组 {report['independent_groups']}")
            del fc

        log(f"[P1:{backend}] 下游训练（P0 协议）...")
        tr = train(str(out / "dataset.npz"), str(out), train_end, validation_end,
                   ridges=(0.3, 1., 3., 10., 30.), threshold=0.5,
                   feature_indices=list(range(6)), feature_variant="baseline",
                   event_balanced=True, auto_threshold=True,
                   max_isolated_false_alarms=2, min_validation_recall=0.4,
                   min_active_bins=2, links=("logit", "cloglog"),
                   l1_ratios=(0., 0.5), selection="nll", calibrate=True)
        entry = dict(selected=dict(link=tr["selected_link"], ridge=tr["selected_ridge"],
                                   l1_ratio=tr["selected_l1_ratio"]),
                     threshold_selection=tr["threshold_selection"],
                     threshold=json.loads((out / "model.json").read_text(encoding="utf-8"))["threshold"])
        log(f"[P1:{backend}] 选定 link={tr['selected_link']} ridge={tr['selected_ridge']} "
            f"l1={tr['selected_l1_ratio']} 阈值={entry['threshold']}")

        model = json.loads((out / "model.json").read_text(encoding="utf-8"))
        with np.load(out / "dataset.npz", allow_pickle=False) as pack:
            x, y, mask, times, event_ids, groups = (pack[k] for k in
                                                    ("x", "y", "mask", "times", "event_ids", "groups"))
            event_onsets = (pack["event_onsets"] if "event_onsets" in pack.files
                            else np.full(len(times), np.nan))
            metadata = json.loads(str(pack["metadata"]))
        x = x[:, :, :6]
        refresh = metadata.get("refresh_seconds", 60.)
        t_end, v_end = timestamp(model["train_end"]), timestamp(model["validation_end"])
        entry["splits"] = {}
        for name, sel in (("validation", (times >= t_end) & (times + 1800 <= v_end)),
                          ("test", times >= v_end)):
            metrics, _ = evaluate(x[sel], y[sel], mask[sel], times[sel], event_ids[sel],
                                  model, 0.5, refresh, event_onsets[sel], 2)
            h = metrics["horizons"]["30min"]
            entry["splits"][name] = dict(nll=h["log_loss"], maxp=h["max_probability"],
                                         ece=h["ece"], ap=h["average_precision"],
                                         recall=h["event_recall"],
                                         isolated_fa=h["isolated_false_alarm_episodes"],
                                         time_frac=h["alarm_time_fraction"],
                                         eligible=h["eligible_events"], detected=h["detected_events"])
            log(f"[P1:{backend}:{name}] nll={h['log_loss']:.4f} maxp={h['max_probability']:.4f} "
                f"ece={h['ece']:.4f} ap={h['average_precision']:.4f} recall={h['event_recall']} "
                f"fa={h['isolated_false_alarm_episodes']} time_frac={h['alarm_time_fraction']:.4f}")
        summary[backend] = entry
        (OUT / "p1_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[P1:{backend}] 完成 -> {out}")

    (OUT / "p1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[P1] 全部完成。")


if __name__ == "__main__":
    main()
