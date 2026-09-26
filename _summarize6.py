# -*- coding: utf-8 -*-
import json

base = r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_06_auto_thr"
abl = json.load(open(base + r"\ablation\summary.json", encoding="utf-8"))
print("===== 消融对照（event-balanced + auto-threshold） =====")
print("recommended_variant:", abl.get("recommended_variant"))
print("validation_30min_ap:", {k: round(v, 4) for k, v in abl.get("validation_30min_ap", {}).items()})
by = {}
for c in abl["comparisons"]:
    by.setdefault((c["variant"], c["split"]), {})[c["horizon"]] = c
print("variant | val30AP | test30AP")
for (v, s) in sorted(by):
    if s == "validation":
        t = by.get((v, "test"), {})
        print("  %-24s %.4f | %.4f" % (v, by[(v, s)]["30min"]["average_precision"],
            t.get("30min", {}).get("average_precision", float("nan"))))

print()
print("===== 自动选中阈值（验证集）与测试集表现 =====")
for variant in ["baseline", "plus_overthreshold_count", "plus_all_transient"]:
    path = base + r"\run\metrics.json" if variant == "baseline" else base + r"\ablation\%s\metrics.json" % variant
    m = json.load(open(path, encoding="utf-8"))
    thr_info = m.get("selected_threshold") or m.get("operating_threshold")
    v = m["splits"]["validation"]["horizons"]["30min"]
    t = m["splits"]["test"]["horizons"]["30min"]
    print("--- %s ---" % variant)
    print("  自动阈值信息:", json.dumps(thr_info, ensure_ascii=False)[:300] if thr_info else "未找到")
    # 找阈值扫描里 selected 对应行
    sel = None
    if thr_info:
        sel = thr_info.get("selected") or thr_info.get("threshold_sweep_selected")
    print("  验证集: ap=%.4f | 测试集: ap=%.4f" % (v["average_precision"], t["average_precision"]))
