"""按用户口径统计：每个事件文件夹的 bin 时段内落有几个岩爆标签（用于确认补全日期正确）。"""
import re
import csv
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")
root = Path(r"G:\1A岩爆预测\岩爆数据集")

# 1. 文件夹 bin 时段
folder_bounds = {}
for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue
    starts, ends = [], []
    for p in folder.glob("*.bin"):
        m = re.search(r"(\d{8})T(\d{6})(?:\.(\d+))?", p.name)
        if not m:
            continue
        fs = int(re.search(r"(\d+)Hz", p.name, re.I).group(1))
        ch = int(re.search(r"(\d+)pt", p.name, re.I).group(1))
        size = p.stat().st_size
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        frac = (m.group(3) or "").ljust(6, "0")[:6]
        s = dt.replace(microsecond=int(frac), tzinfo=TZ).timestamp()
        e = s + size / (4 * ch * fs)
        starts.append(s); ends.append(e)
    if starts:
        folder_bounds[folder.name] = (min(starts), max(ends))

# 2. 读补全后的标签（rockbursts_full.csv 由最近配对生成）
rows = list(csv.DictReader(open(r"G:\1A岩爆预测\rockbursts_full.csv", encoding="utf-8-sig")))
labels = []
for r in rows:
    t = datetime.fromisoformat(r["onset_time"].replace(" ", "T")).replace(tzinfo=TZ)
    labels.append((t, r["event_id"]))

# 3. 每个文件夹时段内落有几个标签（含边界容差 60 分钟）
print(f"{'文件夹':14s} {'bin时段':28s} {'时段内标签数':>8s}  标签时刻")
print("-" * 120)
total = 0
for name, (lo, hi) in sorted(folder_bounds.items()):
    lo_dt, hi_dt = datetime.fromtimestamp(lo, TZ), datetime.fromtimestamp(hi, TZ)
    inside = [(t, eid) for t, eid in labels if lo - 3600 <= t.timestamp() <= hi + 3600]
    if inside:
        total += len(inside)
        times = ", ".join(f"{t:%m-%d %H:%M:%S}({eid})" for t, eid in sorted(inside))
        print(f"{name:14s} {lo_dt:%m-%d %H:%M}~{hi_dt:%m-%d %H:%M} {len(inside):>8d}  {times}")
print("-" * 120)
print(f"落入文件夹时段的标签总数: {total} / {len(labels)}")
missing = [f"{t:%m-%d %H:%M:%S}({eid})" for t, eid in labels if not any(lo - 3600 <= t.timestamp() <= hi + 3600 for lo, hi in folder_bounds.values())]
if missing:
    print("未落入任何文件夹时段的标签:", missing)
else:
    print("全部标签均落入某文件夹时段（±1小时容差）")
