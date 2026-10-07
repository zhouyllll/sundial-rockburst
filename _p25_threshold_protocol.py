# -*- coding: utf-8 -*-
"""P2.5：阈值协议固化验证。

在 model.py 新协议（细网格 + criterion）下对 P2 三源重训：
  legacy          对照：应复现 P2 的 no_operating_point + 0.5 回退（向后兼容）。
  cost            最小化 5*漏检事件 + 误报段数（预警语义）。
  cost_tfa0.3     cost 准则叠加"验证集报警时间占比 <= 0.3"约束。

模型参数/校准与 P2 完全一致（同一 dataset、ridge/l1/link 网格、Platt），
仅决策阈值的选择准则不同。输出到 artifacts/experiment_10_p25/。
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rockburst.model import train

REPO = Path(r"G:\1A岩爆预测\sundial-rockburst")
EXP9 = REPO / "artifacts" / "experiment_09_p2"
EXP25 = REPO / "artifacts" / "experiment_10_p25"
TAGS = ("sundial", "lagllama", "tirex")
CRITERIA = (("legacy", "legacy", {}),
            ("cost", "cost", dict(threshold_grid=np.arange(.02, .98, .01), miss_penalty=5.)),
            ("cost_tfa0.3", "cost", dict(threshold_grid=np.arange(.02, .98, .01),
                                         miss_penalty=5., max_alarm_time_fraction=0.3)),
            ("f1", "f1", dict(threshold_grid=np.arange(.02, .98, .01))))


def _ts(s):
    return np.datetime64(s).astype("datetime64[s]").astype(float)


def main():
    exp06_model = json.loads((REPO / "artifacts" / "experiment_06_auto_thr" / "run" / "model.json")
                             .read_text(encoding="utf-8"))
    train_end = _ts(exp06_model["train_end"])
    validation_end = _ts(exp06_model["validation_end"])
    summary = dict(splits_window=train_end,
                   protocol="P0（ridge x l1 x link NLL 选参 + Platt + 三步阈值），阈值准则版本化（P2.5）",
                   baseline="artifacts/experiment_09_p2/p2_summary.json（0.5 回退）")
    for tag in TAGS:
        ds = EXP9 / tag / "dataset.npz"
        summary[tag] = {}
        for crit, criterion, kw in CRITERIA:
            out = EXP25 / tag / crit
            print(f"[P2.5:{tag}:{crit}] 训练 ...", flush=True)
            rpt = train(str(ds), str(out), train_end, validation_end,
                        ridges=(0.3, 1., 3., 10., 30.), threshold=0.5,
                        feature_indices=list(range(14)), feature_variant="baseline",
                        event_balanced=True, auto_threshold=True,
                        max_isolated_false_alarms=2, min_validation_recall=0.4,
                        min_active_bins=2, links=("logit", "cloglog"),
                        l1_ratios=(0., 0.5), selection="nll", calibrate=True,
                        threshold_criterion=criterion, **kw)
            sel = rpt["threshold_selection"] or {}
            entry = dict(threshold=float(json.loads((out / "model.json")
                                                    .read_text(encoding="utf-8"))["threshold"]),
                         threshold_selection=sel)
            for name in ("validation", "test"):
                h = rpt["splits"][name]["horizons"]["30min"]
                entry[name] = dict(nll=h["log_loss"], maxp=h["max_probability"],
                                   ece=h["ece"], ap=h["average_precision"],
                                   recall=h["event_recall"],
                                   isolated_fa=h["isolated_false_alarm_episodes"],
                                   fa=h["false_alarm_episodes"],
                                   time_frac=h["alarm_time_fraction"],
                                   detected=h["detected_events"], eligible=h["eligible_events"])
            summary[tag][crit] = entry
            v, t = entry["validation"], entry["test"]
            print(f"  t={entry['threshold']:.2f} | 验证 recall={v['recall']:.2f} "
                  f"isolFA={v['isolated_fa']} time={v['time_frac']:.3f} | "
                  f"测试 recall={t['recall']:.2f} isolFA={t['isolated_fa']} "
                  f"time={t['time_frac']:.3f} AP={t['ap']:.4f}", flush=True)
    EXP25.mkdir(parents=True, exist_ok=True)
    (EXP25 / "p25_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print("\n[P2.5] 完成 -> artifacts/experiment_10_p25/p25_summary.json")


if __name__ == "__main__":
    main()
