"""P0：下游校准 + NLL 选参 + cloglog A/B + 阈值三步协议。

复用 experiment_06_auto_thr/dataset.npz（不重跑 Sundial 推理），
按调研报告 P0 配置重训并输出 experiment_07_calibrated。
"""
import json
from pathlib import Path

import numpy as np

from rockburst.io import timestamp
from rockburst.model import train, evaluate, probabilities, _raw_probabilities

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
DATASET = ROOT / "artifacts" / "experiment_06_auto_thr" / "dataset.npz"
EXP06_MODEL = ROOT / "artifacts" / "experiment_06_auto_thr" / "run" / "model.json"
OUT = ROOT / "artifacts" / "experiment_07_calibrated"
FEATURE_SLICE = slice(0, 6)  # exp06 dataset 前 6 列为 baseline 特征


def load_pack():
    with np.load(DATASET, allow_pickle=False) as pack:
        x, y, mask, times = (pack[k] for k in ("x", "y", "mask", "times"))
        event_ids, groups = pack["event_ids"], pack["groups"]
        event_onsets = (pack["event_onsets"] if "event_onsets" in pack.files
                        else np.full(len(times), np.nan))
        metadata = json.loads(str(pack["metadata"]))
    return x, y, mask, times, event_ids, groups, event_onsets, metadata


def main():
    exp06 = json.loads(EXP06_MODEL.read_text(encoding="utf-8"))
    train_end = timestamp(exp06["train_end"])
    validation_end = timestamp(exp06["validation_end"])
    print(f"复用 exp06 时间划分: train_end={exp06['train_end']} validation_end={exp06['validation_end']}",
          flush=True)

    report = train(str(DATASET), str(OUT), train_end, validation_end,
                   ridges=(0.3, 1., 3., 10., 30.), threshold=0.5,
                   feature_indices=list(range(6)), feature_variant="baseline",
                   event_balanced=True, auto_threshold=True,
                   max_isolated_false_alarms=2, min_validation_recall=0.4,
                   min_active_bins=2, links=("logit", "cloglog"),
                   l1_ratios=(0., 0.5), selection="nll", calibrate=True)
    print(f"[P0] 选定: link={report['selected_link']} ridge={report['selected_ridge']} "
          f"l1_ratio={report['selected_l1_ratio']}", flush=True)
    print(f"[P0] 阈值选择: {json.dumps(report['threshold_selection'], ensure_ascii=False)}", flush=True)

    x, y, mask, times, event_ids, groups, event_onsets, metadata = load_pack()
    x = x[:, :, FEATURE_SLICE]
    refresh = metadata.get("refresh_seconds", 60.)
    model = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
    model_no_cal = dict(model)
    model_no_cal.pop("calibration", None)

    summary = dict(
        selected=dict(link=report["selected_link"], ridge=report["selected_ridge"],
                      l1_ratio=report["selected_l1_ratio"], selection=report["selection"]),
        calibration=report["calibration"],
        threshold_selection=report["threshold_selection"],
        threshold=model["threshold"],
        candidates=report["candidates"],
        splits={}, uncalibrated_splits={}, reliability={})

    # 校准后（同一冻结模型与阈值 0.5 诊断；阈值选择始终基于校准后验证集）
    t_end = timestamp(model["train_end"])
    v_end = timestamp(model["validation_end"])
    sel_masks = dict(
        validation=(times >= t_end) & (times + 1800 <= v_end),
        test=times >= v_end)
    for name, sel in sel_masks.items():
        metrics, p = evaluate(x[sel], y[sel], mask[sel], times[sel], event_ids[sel],
                              model, 0.5, refresh, event_onsets[sel], 2)
        # 未校准对照：同一模型去掉校准层，同一阈值 0.5（仅诊断；阈值选择始终基于校准后验证集）
        metrics_u, p_u = evaluate(x[sel], y[sel], mask[sel], times[sel], event_ids[sel],
                                  model_no_cal, 0.5, refresh, event_onsets[sel], 2)
        summary["splits"][name] = metrics
        summary["uncalibrated_splits"][name] = metrics_u
        if name == "test":
            # reliability（10 bins）：校准后 vs 校准前，30min
            target = (np.cumsum(y[sel], axis=1) > 0)[:, 2].astype(float)
            for tag, pr in (("calibrated", p[:, 2]), ("uncalibrated", p_u[:, 2])):
                edges = np.linspace(0., 1., 11)
                bins = []
                for k in range(10):
                    lo, hi = edges[k], edges[k + 1]
                    m = (pr >= lo) & (pr < hi) if k < 9 else (pr >= lo) & (pr <= hi)
                    bins.append(dict(low=float(lo), high=float(hi), n=int(m.sum()),
                                     mean_pred=float(pr[m].mean()) if m.any() else None,
                                     mean_actual=float(target[m].mean()) if m.any() else None))
                summary["reliability"][tag] = bins
            summary["test_max_events"] = dict(
                eligible=metrics["horizons"]["30min"]["eligible_events"],
                detected=metrics["horizons"]["30min"]["detected_events"],
                recall=metrics["horizons"]["30min"]["event_recall"])
        print(f"[P0:{name}] 校准后 30min nll={metrics['horizons']['30min']['log_loss']:.4f} "
              f"maxp={metrics['horizons']['30min']['max_probability']:.4f} "
              f"ece={metrics['horizons']['30min']['ece']:.4f} "
              f"ap={metrics['horizons']['30min']['average_precision']:.4f} "
              f"recall={metrics['horizons']['30min']['event_recall']} "
              f"isolated_fa={metrics['horizons']['30min']['isolated_false_alarm_episodes']} "
              f"time_frac={metrics['horizons']['30min']['alarm_time_fraction']:.4f}", flush=True)
        print(f"[P0:{name}] 校准前 30min nll={metrics_u['horizons']['30min']['log_loss']:.4f} "
              f"maxp={metrics_u['horizons']['30min']['max_probability']:.4f} "
              f"ece={metrics_u['horizons']['30min']['ece']:.4f}", flush=True)

    (OUT / "p0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[P0] 完成。产物: {OUT}", flush=True)


if __name__ == "__main__":
    main()
