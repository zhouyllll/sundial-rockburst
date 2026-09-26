# -*- coding: utf-8 -*-
import json

base = r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_05_balanced_abl"
abl = json.load(open(base + r"\ablation\summary.json", encoding="utf-8"))
print("===== 消融对照（event-balanced） =====")
print("recommended_variant:", abl.get("recommended_variant"))
print("validation_30min_ap:", {k: round(v, 4) for k, v in abl.get("validation_30min_ap", {}).items()})
by = {}
for c in abl["comparisons"]:
    by.setdefault((c["variant"], c["split"]), {})[c["horizon"]] = c
print("variant | val30AP | test30AP | val30Brier | test30Brier")
for (v, s) in sorted(by):
    if s == "validation":
        h, t = by[(v, s)], by.get((v, "test"), {})
        print("  %-24s %.4f | %.4f | %.4f | %.4f" % (v, h["30min"]["average_precision"],
            t.get("30min", {}).get("average_precision", float("nan")),
            h["30min"]["brier"], t.get("30min", {}).get("brier", float("nan"))))

print()
print("===== 阈值扫描（含三分类误报） =====")
for variant in ["baseline", "plus_overthreshold_count"]:
    m = json.load(open(base + r"\ablation\%s\metrics.json" % variant, encoding="utf-8"))
    print("---", variant, "---")
    for split in ["validation", "test"]:
        h = m["splits"][split]["horizons"]["30min"]
        print("[%s] maxp=%.4f pos_maxp=%.4f neg_maxp=%.4f" % (split,
            h.get("max_probability", 0), h.get("positive_max_probability", 0),
            h.get("negative_max_probability", 0)))
        print("  阈值 | 召回 | 检出 | 误报段 | 边界 | 事件后 | 孤立 | 告警占比")
        for r in h.get("threshold_sweep", []):
            rec = r["event_recall"]
            print("  %.2f | %.2f | %d/%d | %d | %d | %d | %d | %.3f" % (
                r["threshold"], rec if rec is not None else -1, r["detected_events"],
                r["eligible_events"], r["false_alarm_episodes"],
                r.get("boundary_near_event_episodes", -1), r.get("post_event_carryover_episodes", -1),
                r.get("isolated_false_alarm_episodes", -1), r["alarm_time_fraction"]))
