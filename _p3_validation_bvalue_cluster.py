# -*- coding: utf-8 -*-
"""
3.3 + 3.4 验证脚本：微震大小分布（b 值）与簇级统计效力
=============================================================
评审意见出处（P3 微震分支候选指标）：
  3.3 缺大小分布类特征：补各窗口 b 值、最大能量、P99/P50 能量比
      （b 值下降是岩爆前兆经典指标；能量和被单条大事件主导，不敏感）
  3.4 统计效力：比例改报簇级 bootstrap 区间（4 独立簇下 79% 是虚假精度）；
      多重比较需说明（几十次比较中的 p=0.003，Bonferroni 后不显著）

用法
----
  python _p3_validation_bvalue_cluster.py --catalog events.csv [--windows 0,3,6,24] \
      [--e_min 1e5] [--cluster_gap_h 24] [--n_boot 5000] [--n_comparisons 30] [--out out.json]

  python _p3_validation_bvalue_cluster.py --demo          # 内置模拟目录自检（含真实 27 条岩爆时间）

输入格式（--catalog）
--------------------
  CSV 表头：time,energy[,group]
    time   : ISO 时间串（如 2026-05-14 07:16:15）或 unix 秒
    energy : 事件能量 E（单位 J；若为相对能量，结果标注"相对量级"）
    group  : 可选，簇标识（如 folder-xxx）；缺省时按 cluster_gap_h 时间间隔聚类

输出
----
  1) 控制台：各窗口 b 值（Aki MLE + Utsu n<50 修正）、最大能量、P99/P50、
     簇数、簇级 bootstrap 95% CI、多重比较校正说明
  2) --out JSON：结构化结果
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ---------------- 常量与公式 ----------------
LOG10E = math.log10(math.e)          # 0.4343
# Gutenberg-Richter 能量-震级：log10 E = 1.5*M + 4.8  (E 单位 J)
ENERGY_M_OFFSET = 4.8
M_PER_LOG10E = 1.5 / 1.0             # dM/d(log10E) = 1/1.5
M_OF_E = lambda log10e: (2.0 / 3.0) * (log10e - ENERGY_M_OFFSET)


def aki_b(m_vals, mc):
    """Aki (1965) MLE：b = log10(e)/(mean(M) - Mc)。返回 (b, n)。"""
    n = len(m_vals)
    if n == 0:
        return float("nan"), 0
    mean_m = float(np.mean(m_vals))
    denom = mean_m - mc
    if denom <= 0:
        return float("nan"), n
    b = LOG10E / denom
    return b, n


def utsu_b(b, n):
    """Utsu (1965) 小样本修正：b_c = b * (N-1)/N（n<50 时用）。"""
    if n < 2:
        return float("nan")
    return b * (n - 1.0) / n


def shi_bolt_sigma(b, m_vals, mc):
    """Shi & Bolt (1982) b 值标准差近似。"""
    n = len(m_vals)
    if n < 2:
        return float("nan")
    dm = m_vals - mc
    ss = np.sum((dm - np.mean(dm)) ** 2)
    return b * 2.30 * math.sqrt(ss / (n * (n - 1.0)))


# ---------------- 输入读取 ----------------
def parse_time(txt):
    txt = txt.strip()
    try:
        return datetime.fromtimestamp(float(txt))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(txt, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {txt}")


def load_catalog(path):
    evts = []
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        fn = rd.fieldnames or []
        missing = [c for c in ("time", "energy") if c not in fn]
        if missing:
            raise ValueError(
                f"事件目录缺少必需列 {missing}（需 time,energy[,group]）：{path}\n"
                f"实际列：{fn}"
            )
        for row in rd:
            t = parse_time(row["time"])
            e = float(row["energy"])
            g = row.get("group") or row.get("group_id") or None
            evts.append({"time": t, "energy": e, "group": g})
    return evts


# ---------------- 簇聚类 ----------------
def cluster_events(evts, gap_hours):
    """按时间间隔聚类：与上一事件间隔 < gap_hours 归同簇。返回簇列表（每簇为事件列表）。"""
    if not evts:
        return []
    order = sorted(evts, key=lambda x: x["time"])
    clusters = [[order[0]]]
    for e in order[1:]:
        if (e["time"] - clusters[-1][-1]["time"]) <= timedelta(hours=gap_hours):
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return clusters


# ---------------- 窗口统计（3.3） ----------------
def window_stats(events, e_min):
    """对一组事件算：n、b 值（Aki+Utsu+ShiBolt）、最大能量、P99/P50、能量和。"""
    if not events:
        return None
    e = np.array([x["energy"] for x in events], dtype=float)
    # 只计入完整性下限以上的事件：低于 E_min 的事件会拉低 mean(M)、抬高 b 估计
    e = e[e >= e_min]
    n = len(e)
    if n == 0:
        return None
    log10e = np.log10(e)
    mc = M_OF_E(np.log10(e_min))
    m_vals = M_OF_E(log10e)
    b, n_used = aki_b(m_vals, mc)
    b_uts = utsu_b(b, n_used)
    sig = shi_bolt_sigma(b, m_vals, mc) if n_used >= 2 else float("nan")
    p99 = float(np.percentile(e, 99))
    p50 = float(np.percentile(e, 50))
    return {
        "n": int(n),
        "b_aki": None if math.isnan(b) else round(b, 4),
        "b_uts_u50": None if math.isnan(b_uts) else round(b_uts, 4),
        "b_std": None if math.isnan(sig) else round(sig, 4),
        "b_used_correction": bool(n < 50),
        "max_energy": float(np.max(e)),
        "p99": p99,
        "p50": p50,
        "p99_p50_ratio": round(p99 / p50, 2) if p50 > 0 else None,
        "energy_sum": float(np.sum(e)),
    }


# ---------------- 簇级 bootstrap（3.4） ----------------
def cluster_bootstrap_ci(clusters, metric_fn, n_boot, seed=12345):
    """以簇为单位有放回重采样 B 次，对 metric_fn(重采样簇) 求 2.5%/97.5% 分位。
    metric_fn 输入簇列表，返回标量（如命中比例）。簇数 < 5 时输出区间并提示样本不足。"""
    n_cl = len(clusters)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_cl, size=n_cl)
        sample = [clusters[i] for i in idx]
        vals.append(metric_fn(sample))
    vals = np.array(vals, dtype=float)
    return {
        "lo": round(float(np.percentile(vals, 2.5)), 4),
        "hi": round(float(np.percentile(vals, 97.5)), 4),
        "median": round(float(np.median(vals)), 4),
        "n_clusters": n_cl,
        "note": "簇数<=4：区间极宽，建议按'无法估计'处理" if n_cl <= 4 else "",
    }


def bonferroni(p_value, n_comparisons):
    """Bonferroni 校正。返回 (校正阈值, 是否显著)。"""
    threshold = 0.05 / n_comparisons
    p_adj = min(1.0, p_value * n_comparisons)
    return threshold, p_adj <= 0.05


# ---------------- Demo 模拟数据 ----------------
def build_demo_catalog():
    """用真实 27 条岩爆时间 + 模拟微震事件（幂律能量, b≈1.05）构造目录。"""
    rb_csv = Path(r"G:\1A岩爆预测\rockbursts_full.csv")
    rb_times = []
    if rb_csv.exists():
        with open(rb_csv, encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            for row in rd:
                rb_times.append(parse_time(row["onset_time"]))
    else:
        rb_times = [datetime(2026, 5, 12, 19, 39, 23)]
    rng = np.random.default_rng(42)
    b_true = 1.05
    e_min = 1e5
    mc = M_OF_E(np.log10(e_min))
    evts = []
    for rb_t in rb_times:
        # 爆前 6h 窗口内平均 40 个事件
        k = rng.poisson(40)
        for _ in range(k):
            dt = timedelta(seconds=rng.uniform(0, 6 * 3600))
            # 震级截断指数抽样（M >= Mc，指数无记忆：M = Mc + Exp(1/(b*ln10))）
            M = mc + rng.exponential(1.0 / (b_true * math.log(10)))
            E = 10 ** (1.5 * M + 4.8)
            evts.append({"time": rb_t - dt, "energy": float(E), "group": None})
    evts.sort(key=lambda x: x["time"])
    return evts, {"b_true": b_true, "e_min": e_min, "n_rb": len(rb_times), "rb_times": rb_times}


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", help="事件目录 CSV（time,energy[,group]）")
    ap.add_argument("--windows", default="0,3,6,24", help="距岩爆前窗口小时数（逗号分隔）")
    ap.add_argument("--e_min", type=float, default=1e5, help="目录完整性能量下限（J）")
    ap.add_argument("--cluster_gap_h", type=float, default=24.0, help="簇聚类间隔（小时）")
    ap.add_argument("--n_boot", type=int, default=5000, help="bootstrap 次数")
    ap.add_argument("--n_comparisons", type=int, default=30, help="多重比较次数（Bonferroni）")
    ap.add_argument("--demo", action="store_true", help="跑内置模拟自检")
    ap.add_argument("--out", help="结果 JSON 输出路径")
    ap.add_argument("--rb_times", help="岩爆时刻 CSV（可选，用于窗口对齐；列 onset_time）")
    args = ap.parse_args()

    if args.demo:
        evts, meta = build_demo_catalog()
        rb_times = meta["rb_times"]
        print(f"[demo] 模拟目录：{len(evts)} 事件，能量幂律 b_true={meta['b_true']}，"
              f"基于 {meta['n_rb']} 条真实岩爆时间")
    else:
        if not args.catalog:
            print("错误：需 --catalog 或 --demo", file=sys.stderr)
            sys.exit(2)
        evts = load_catalog(args.catalog)
        rb_times = None
        if args.rb_times:
            rb_times = []
            with open(args.rb_times, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    rb_times.append(parse_time(row["onset_time"]))
        print(f"[input] 事件目录：{len(evts)} 事件")

    # 簇
    clusters = cluster_events(evts, args.cluster_gap_h)
    print(f"[cluster] 簇数：{len(clusters)}（gap={args.cluster_gap_h}h）")
    sizes = [len(c) for c in clusters]
    print(f"[cluster] 各簇大小：{sizes}")

    # 3.3 窗口 b 值（有岩爆时间时按窗口；否则全目录单窗）
    wins_h = [float(x) for x in args.windows.split(",")]
    stats_all = window_stats(evts, args.e_min)
    win_stats = []
    if rb_times is not None and rb_times:
        for wh in wins_h:
            cut = datetime.max
            win_evts = []
            for e in evts:
                d = min((rb_t - e["time"] for rb_t in rb_times if rb_t >= e["time"]),
                        default=None)
                if d is not None and timedelta(0) <= d <= timedelta(hours=wh):
                    win_evts.append(e)
            st = window_stats(win_evts, args.e_min)
            if st:
                st["window_h"] = wh
                st["n_rb_aligned"] = len(rb_times)
                win_stats.append(st)
    else:
        st = window_stats(evts, args.e_min)
        if st:
            st["window_h"] = None
            st["scope"] = "whole_catalog"
            win_stats.append(st)

    # 3.4 簇级 bootstrap：示例指标 = 命中比例（单簇内 max energy > 1e6 的簇占比）
    def hit_ratio(cls):
        hits = 0
        for c in cls:
            if any(x["energy"] > 1e6 for x in c):
                hits += 1
        return hits / len(cls) if cls else float("nan")

    obs_hit = hit_ratio(clusters)
    ci = cluster_bootstrap_ci(clusters, hit_ratio, args.n_boot)
    # 多重比较：示例 p=0.003（评审意见案例）
    thr, sig = bonferroni(0.003, args.n_comparisons)

    result = {
        "meta": {"mode": "demo" if args.demo else "catalog",
                 "catalog": args.catalog,
                 "e_min": args.e_min,
                 "cluster_gap_h": args.cluster_gap_h,
                 "n_boot": args.n_boot,
                 "n_comparisons": args.n_comparisons},
        "catalog_stats": stats_all,
        "window_stats": win_stats,
        "clusters": {"n_clusters": len(clusters), "sizes": sizes},
        "hit_ratio_1e6_per_cluster": {
            "observed": round(obs_hit, 4),
            "bootstrap_ci": ci,
        },
        "multiple_comparison": {
            "example_p": 0.003,
            "bonferroni_threshold": round(thr, 6),
            "significant_after_correction": bool(sig),
            "note": f"p=0.003 在 {args.n_comparisons} 次比较下阈值 {thr:.4f}，"
                    f"{'显著' if sig else '不显著'}（示例演示）",
        },
    }

    # 控制台输出
    print("\n========== 3.3 大小分布（b 值） ==========")
    if stats_all:
        s = stats_all
        print(f"全目录/全簇：n={s['n']}, b_aki={s['b_aki']}, "
              f"b_uts{'*' if s['b_used_correction'] else ''}={s['b_uts_u50']}, "
              f"±{s['b_std']}, maxE={s['max_energy']:.3g}, P99/P50={s['p99_p50_ratio']}")
        if s["b_used_correction"]:
            print("  * n<50，已用 Utsu (N-1)/N 修正")
    for w in win_stats:
        print(f"窗口 {w['window_h']}h: n={w['n']}, b={w['b_aki']}, "
              f"b_uts={w['b_uts_u50']}, maxE={w['max_energy']:.3g}, "
              f"P99/P50={w['p99_p50_ratio']}")

    print("\n========== 3.4 簇级统计效力 ==========")
    print(f"簇数：{len(clusters)}（独立样本以簇计，非事件数）")
    print(f"簇内最大能量>1e6J 的簇占比：{obs_hit:.3f}")
    print(f"簇级 bootstrap 95% CI：{ci['lo']}–{ci['hi']}（中位 {ci['median']}）{ci['note']}")
    print(f"多重比较：p=0.003 × {args.n_comparisons} 比较 → Bonferroni 阈值 {thr:.4f} "
          f"→ {'显著' if sig else '不显著'}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写：{out_p}")


if __name__ == "__main__":
    main()
