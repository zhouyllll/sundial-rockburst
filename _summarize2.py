# -*- coding: utf-8 -*-
import json

base = r"G:\1A岩爆预测\sundial-rockburst\artifacts"
abl = json.load(open(base + r"\experiment_03_transient_abl\ablation\summary.json", encoding="utf-8"))

print("===== 消融对照表（AP 平均精度 / Brier） =====")
print("variant | split | 5min AP | 10min AP | 30min AP | 30min Brier")
by = {}
for c in abl["comparisons"]:
    key = (c["variant"], c["split"])
    by.setdefault(key, {})[c["horizon"]] = c
for (variant, split) in sorted(by):
    h = by[(variant, split)]
    a5 = h.get("5min", {}).get("average_precision")
    a10 = h.get("10min", {}).get("average_precision")
    a30 = h.get("30min", {}).get("average_precision")
    b30 = h.get("30min", {}).get("brier")
    print("  %-22s %-10s %.4f | %.4f | %.4f | %.4f" % (variant, split, a5, a10, a30, b30))

print()
ev = json.load(open(base + r"\forecast_eval_30\forecast_metrics.json", encoding="utf-8"))
print("===== C evaluate-forecast（Sundial 预测未来特征质量） =====")
for k in ["5min", "10min", "30min"]:
    d = ev["horizons"][k]
    print("  %s: origins=%d mae=%.4f rmse=%.4f persistence_mae=%.4f improvement=%.1f%%" % (
        k, d["origins"], d["mae_log_power"], d["rmse_log_power"],
        d["persistence_mae_log_power"], d["mae_improvement_vs_persistence"] * 100))
