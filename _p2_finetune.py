"""P2：预训练表征 + 下游 9 参数风险模型微调（严格复用 P0 协议）。

对 sundial / lagllama 两个特征源，各跑一遍与 P1 完全相同的下游协议：
ridge×l1×link 按验证 NLL 选参 + Platt 校准 + 三步阈值，评估验证/测试，
汇总 p2_summary.json，并引用 exp07 基线数值。

用法: python _p2_finetune.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, str(ROOT))
from rockburst.io import read_csv, timestamp  # noqa: E402
from rockburst.model import train, evaluate  # noqa: E402

EXP06 = ROOT / "artifacts" / "experiment_06_auto_thr"
P1 = ROOT / "artifacts" / "experiment_08_p1"
P2 = ROOT / "artifacts" / "experiment_09_p2"
N_FEAT = 14  # 6 原特征 + 8 PCA 表征


def log(msg):
    print(msg, flush=True)
    with (P2 / "p2_run.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def evaluate_split(x, y, mask, times, event_ids, model, refresh, event_onsets):
    metrics, _ = evaluate(x, y, mask, times, event_ids, model, 0.5, refresh,
                          event_onsets, 2)
    h = metrics["horizons"]["30min"]
    return dict(nll=h["log_loss"], maxp=h["max_probability"], ece=h["ece"],
                ap=h["average_precision"], recall=h["event_recall"],
                isolated_fa=h["isolated_false_alarm_episodes"],
                time_frac=h["alarm_time_fraction"],
                eligible=h["eligible_events"], detected=h["detected_events"])


def main():
    exp06_model = json.loads((EXP06 / "run" / "model.json").read_text(encoding="utf-8"))
    train_end = timestamp(exp06_model["train_end"])
    validation_end = timestamp(exp06_model["validation_end"])
    summary = dict(splits_window=exp06_model["train_end"],
                   protocol="P0（ridge x l1 x link NLL 选参 + Platt + 三步阈值）",
                   baseline_sundial="artifacts/experiment_07_calibrated/p0_summary.json",
                   baseline_lagllama="artifacts/experiment_08_p1/lagllama/model.json")

    for tag in ("sundial", "lagllama", "tirex"):
        out = P2 / tag
        ds = out / "dataset.npz"
        log(f"[P2:{tag}] 训练（{N_FEAT} 维特征）...")
        tr = train(str(ds), str(out), train_end, validation_end,
                   ridges=(0.3, 1., 3., 10., 30.), threshold=0.5,
                   feature_indices=list(range(N_FEAT)), feature_variant="baseline",
                   event_balanced=True, auto_threshold=True,
                   max_isolated_false_alarms=2, min_validation_recall=0.4,
                   min_active_bins=2, links=("logit", "cloglog"),
                   l1_ratios=(0., 0.5), selection="nll", calibrate=True)
        entry = dict(selected=dict(link=tr["selected_link"], ridge=tr["selected_ridge"],
                                   l1_ratio=tr["selected_l1_ratio"]),
                     threshold_selection=tr["threshold_selection"],
                     threshold=json.loads((out / "model.json").read_text(encoding="utf-8"))["threshold"])
        log(f"[P2:{tag}] 选定 link={tr['selected_link']} ridge={tr['selected_ridge']} "
            f"l1={tr['selected_l1_ratio']} 阈值={entry['threshold']}")

        model = json.loads((out / "model.json").read_text(encoding="utf-8"))
        with np.load(ds, allow_pickle=False) as pack:
            x, y, mask = pack["x"], pack["y"], pack["mask"]
            times, event_ids = pack["times"], pack["event_ids"]
            groups = pack["groups"]
            event_onsets = (pack["event_onsets"] if "event_onsets" in pack.files
                            else np.full(len(times), np.nan))
            metadata = json.loads(str(pack["metadata"]))
        x = x[:, :, :N_FEAT]
        refresh = metadata.get("refresh_seconds", 60.)
        t_end, v_end = timestamp(model["train_end"]), timestamp(model["validation_end"])
        entry["splits"] = {}
        for name, sel in (("validation", (times >= t_end) & (times + 1800 <= v_end)),
                          ("test", times >= v_end)):
            entry["splits"][name] = evaluate_split(x[sel], y[sel], mask[sel],
                                                   times[sel], event_ids[sel],
                                                   model, refresh, event_onsets[sel])
            h = entry["splits"][name]
            log(f"[P2:{tag}:{name}] nll={h['nll']:.4f} maxp={h['maxp']:.4f} "
                f"ece={h['ece']:.4f} ap={h['ap']:.4f} recall={h['recall']} "
                f"fa={h['isolated_fa']} time_frac={h['time_frac']:.4f}")
        summary[tag] = entry
        (P2 / "p2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        log(f"[P2:{tag}] 完成 -> {out}")

    (P2 / "p2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    log("[P2] 全部完成。")


if __name__ == "__main__":
    main()
