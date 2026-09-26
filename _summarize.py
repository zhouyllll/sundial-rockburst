# -*- coding: utf-8 -*-
import json

base = r"G:\1A岩爆预测\sundial-rockburst\artifacts"
m = json.load(open(base + r"\experiment_02_causal_fix\run\metrics.json", encoding="utf-8"))
print("===== A 基线 (experiment_02) =====")
for split in ["train", "validation", "test"]:
    h = m["splits"][split]["horizons"]
    print("===", split, "samples", m["splits"][split]["samples"])
    for k in ["5min", "10min", "30min"]:
        d = h[k]
        print("  %s: brier=%.4f logloss=%.4f ap=%.4f pos=%d ev=%d det=%d recall=%.2f alarm=%.4f meanp=%.4f" % (
            k, d["brier"], d["log_loss"], d["average_precision"], d["positive_samples"],
            d["eligible_events"], d["detected_events"], d["event_recall"],
            d["alarm_time_fraction"], d["mean_probability"]))
p = json.load(open(base + r"\experiment_02_causal_fix\prediction.json", encoding="utf-8"))
print("最新预测: time=%s p5=%.1f%% p10=%.1f%% p30=%.1f%%" % (
    p["time"], p["p_5min"] * 100, p["p_10min"] * 100, p["p_30min"] * 100))

print()
print("===== B 消融 (experiment_03) =====")
try:
    abl = json.load(open(base + r"\experiment_03_transient_abl\ablation\summary.json", encoding="utf-8"))
    print(json.dumps(abl, ensure_ascii=False, indent=1)[:4000])
except Exception as e:
    print("ablation summary 读取失败:", e)

print()
print("===== C evaluate-forecast =====")
try:
    ev = json.load(open(base + r"\forecast_eval_30\report.json", encoding="utf-8"))
    print(json.dumps(ev, ensure_ascii=False, indent=1)[:3000])
except Exception as e:
    print("forecast eval 读取失败:", e)
