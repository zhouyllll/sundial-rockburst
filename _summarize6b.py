# -*- coding: utf-8 -*-
import json

base = r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_06_auto_thr"
variants = ["baseline", "plus_overthreshold_count", "plus_all_transient", "plus_stalta"]
print("===== 自动阈值（验证集选择）与测试集表现 =====")
for variant in variants:
    path = base + r"\run\metrics.json" if variant == "baseline" else base + r"\ablation\%s\metrics.json" % variant
    m = json.load(open(path, encoding="utf-8"))
    ts = m.get("threshold_selection") or {}
    sel = ts.get("selected", {})
    print("--- %s ---" % variant)
    print("  自动阈值=%.2f  规则=%s  可行=%s" % (ts.get("threshold", float("nan")),
          ts.get("rule", "?"), ts.get("feasible", "?")))
    print("  选中档: 召回=%.2f 检出=%s/%s 误报段=%d (边界%d/事件后%d/孤立%d) 告警占比=%.3f" % (
        sel.get("event_recall") if sel.get("event_recall") is not None else -1,
        sel.get("detected_events"), sel.get("eligible_events"),
        sel.get("false_alarm_episodes"), sel.get("boundary_near_event_episodes", -1),
        sel.get("post_event_carryover_episodes", -1), sel.get("isolated_false_alarm_episodes", -1),
        sel.get("alarm_time_fraction", -1)))
    t = m["splits"]["test"]["horizons"]["30min"]
    print("  测试集30min: ap=%.4f" % t["average_precision"])
    # 测试集在自动阈值下的操作点
    for r in t.get("threshold_sweep", []):
        if abs(r["threshold"] - ts.get("threshold", -9)) < 1e-9:
            print("  测试集@%.2f: 召回=%.2f 检出=%d/%d 误报段=%d (边界%d/事件后%d/孤立%d) 告警占比=%.3f" % (
                r["threshold"], r["event_recall"] if r["event_recall"] is not None else -1,
                r["detected_events"], r["eligible_events"], r["false_alarm_episodes"],
                r.get("boundary_near_event_episodes", -1), r.get("post_event_carryover_episodes", -1),
                r.get("isolated_false_alarm_episodes", -1), r["alarm_time_fraction"]))
