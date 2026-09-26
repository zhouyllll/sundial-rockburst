# -*- coding: utf-8 -*-
"""诊断 [3/5] 无有效样本：重建 Inputs，对最早/代表性预测时刻调 label/sample。"""
import io, sys, csv
from datetime import datetime, timezone
from collections import Counter

sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")

ROOT = r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_01"

def main():
    from rockburst.data import Inputs, Unavailable
    from rockburst.workflow import local_time, scan_bins, read_burst_times
    from rockburst.io import iso

    continuous_dir = r"G:\1A岩爆预测\岩爆数据集"
    labels_path = r"G:\1A岩爆预测\rockbursts_full.csv"
    zone = "zone_A"

    manifest = scan_bins(continuous_dir, zone, "Asia/Shanghai")
    print("manifest 行数:", len(manifest))
    labels = read_burst_times(labels_path, zone, "Asia/Shanghai")
    print("标签:", len(labels), "范围", labels[0]["onset_time"], "->", labels[-1]["onset_time"])

    data = Inputs(ROOT + "/continuous.csv", ROOT + "/events.csv",
                  ROOT + "/generated/coverage.csv", zone,
                  ROOT + "/generated/rockbursts.csv", microseismic_enabled=False)
    print("Inputs OK, blocks 数:", len(data.blocks), "签名:", data.signature)
    print("blocks 时间范围:", iso(min(data.blocks)), "->", iso(max(data.blocks)))

    times = sorted({float(local_time(r["available_time"], "Asia/Shanghai")) for r in manifest})
    print("预测时刻数(去重):", len(times), "范围", iso(times[0]), "->", iso(times[-1]))

    # 统计每个失败原因
    skipped = Counter()
    first_fail = None
    for now in times:
        try:
            data.label(now)
            data.sample(now, None, 30, 60)
        except Unavailable as exc:
            skipped[str(exc)] += 1
            if first_fail is None:
                first_fail = (now, str(exc))
            continue
    print("跳过统计:", dict(skipped))
    if first_fail:
        now, why = first_fail
        print("最早失败时刻:", iso(now), "原因:", why)
        # 分析该时刻的 blocks 覆盖
        latest = int(now // 10 * 10)
        cutoff = None
        for end in range(latest, int(((now - 60) / 10) * 10) - 1, -10):
            blk = data.blocks.get(end)
            if blk and blk["available"] <= now:
                cutoff = end
                break
        print("cutoff:", cutoff)
        first = int(((now - 1800) / 10) * 10) + 10
        missing = [e for e in range(first, cutoff + 1, 10) if e not in data.blocks]
        late = [e for e in range(first, cutoff + 1, 10) if e in data.blocks and data.blocks[e]["available"] > now]
        print("缺失 block 数:", len(missing), "示例:", [iso(e) for e in missing[:5]])
        print("available>now block 数:", len(late), "示例:", [iso(e) for e in late[:5]])
        print("窗口 [now-1800, now]:", iso(first - 10), "->", iso(now))

if __name__ == "__main__":
    main()
