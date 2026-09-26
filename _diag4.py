# -*- coding: utf-8 -*-
"""统计所有预测时刻的 history 窗口缺失分布，定位系统性原因。"""
import sys, io
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")
ROOT = r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_01"

def main():
    from rockburst.data import Inputs
    from rockburst.workflow import local_time, scan_bins
    from rockburst.io import iso

    manifest = scan_bins(r"G:\1A岩爆预测\岩爆数据集", "zone_A", "Asia/Shanghai")
    data = Inputs(ROOT + "/continuous.csv", ROOT + "/events.csv",
                  ROOT + "/generated/coverage.csv", "zone_A",
                  ROOT + "/generated/rockbursts.csv", microseismic_enabled=False)
    times = sorted({float(local_time(r["available_time"], "Asia/Shanghai")) for r in manifest})
    print("预测时刻:", len(times))

    by_day = Counter()
    missing_stats = Counter()   # 缺失 block 数分桶
    cutoff_none = 0
    ok = 0
    sample_list = []
    for now in times:
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%m-%d")
        latest = int(now // 10 * 10)
        cutoff = None
        for end in range(latest, int(((now - 60) / 10) * 10) - 1, -10):
            blk = data.blocks.get(end)
            if blk and blk["available"] <= now:
                cutoff = end
                break
        if cutoff is None:
            cutoff_none += 1
            by_day[day + "(cutoffNone)"] += 1
            continue
        first = int(((now - 1800) / 10) * 10) + 10
        missing = sum(1 for e in range(first, cutoff + 1, 10)
                      if e not in data.blocks or data.blocks[e]["available"] > now)
        if missing == 0:
            ok += 1
            if len(sample_list) < 3:
                sample_list.append(iso(now))
        missing_stats[(missing // 10) * 10] += 1
        by_day[day] += 1
    print("history 完整(将成功):", ok, "示例:", sample_list)
    print("cutoff=None:", cutoff_none)
    print("缺失数分桶(缺失//10*10): 计数")
    for k in sorted(missing_stats):
        print(f"  {k}: {missing_stats[k]}")
    print("按日期分布:")
    for k in sorted(by_day):
        print(f"  {k}: {by_day[k]}")

if __name__ == "__main__":
    main()
