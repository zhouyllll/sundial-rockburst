"""统计所有事件文件夹内相邻 bin 的时间重叠/间隙分布。"""
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter

TZ = ZoneInfo("Asia/Shanghai")
root = Path(r"G:\1A岩爆预测\岩爆数据集")

def parse(path):
    m = re.search(r"(\d{8})T(\d{6})(?:\.(\d+))?", path.name)
    fs = int(re.search(r"(\d+)Hz", path.name, re.I).group(1))
    ch = int(re.search(r"(\d+)pt", path.name, re.I).group(1))
    size = path.stat().st_size
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    frac = (m.group(3) or "").ljust(6, "0")[:6]
    start = dt.replace(microsecond=int(frac), tzinfo=TZ).timestamp()
    end = start + size / (4 * ch * fs)
    return start, end

overlap_ms = []   # 后文件开始 - 前文件结束，负数=重叠
gap_ms = []
for folder in sorted(root.iterdir()):
    bins = sorted(root.glob(f"{folder.name}/*.bin"))
    rows = []
    for p in bins:
        try:
            rows.append((parse(p), p.name))
        except Exception:
            continue
    rows.sort(key=lambda x: x[0][0])
    for (l_start, l_end, lname), (r_start, r_end, rname) in zip(
            [(a, b, c) for (a, b), c in rows], [(a, b, c) for (a, b), c in rows][1:]):
        diff_ms = (r_start - l_end) * 1000  # 正=间隙，负=重叠
        if diff_ms < 0:
            overlap_ms.append((diff_ms, folder.name, lname, rname))
        elif diff_ms > 1:
            gap_ms.append((diff_ms, folder.name, lname, rname))

print(f"总文件对数统计完成")
print(f"重叠对数: {len(overlap_ms)}, 间隙对数(>1ms): {len(gap_ms)}")
if overlap_ms:
    neg = sorted(overlap_ms)
    print(f"最大重叠: {neg[0][0]:.1f} ms ({neg[0][1]})")
    print(f"最小重叠: {neg[-1][0]:.1f} ms")
    # 分布
    dist = Counter()
    for ms, *_ in overlap_ms:
        if ms >= -10: dist["0-10ms"] += 1
        elif ms >= -100: dist["10-100ms"] += 1
        elif ms >= -1000: dist["0.1-1s"] += 1
        else: dist[">1s"] += 1
    print("重叠分布:", dict(dist))
    print("重叠样本(前3):")
    for ms, f, ln, rn in overlap_ms[:3]:
        print(f"  {ms:.1f}ms | {f} | {ln.split('-')[-1]} -> {rn.split('-')[-1]}")
if gap_ms:
    print(f"最大间隙: {max(gap_ms)[0]:.0f} ms")
