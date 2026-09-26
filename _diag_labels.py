"""诊断标签与事件文件夹的匹配关系，定位多匹配/零匹配。"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")

from rockburst.workflow import scan_bins, _folder_windows, _parse_clock_range, _resolve_clock_range

TZ = "Asia/Shanghai"
root = r"G:\1A岩爆预测\岩爆数据集"
labels_file = r"G:\1A岩爆预测\label.txt"

rows = scan_bins(root, "zone_A", TZ)
print(f"连续bin总数: {len(rows)}")
windows = _folder_windows(rows, root, TZ)
print(f"事件文件夹数: {len(windows)}")
for folder, (lo, hi) in sorted(windows.items()):
    print(f"  {folder}: {datetime.fromtimestamp(lo, ZoneInfo(TZ))} ~ {datetime.fromtimestamp(hi, ZoneInfo(TZ))}")

lines = [l.strip() for l in open(labels_file, encoding="utf-8-sig").read().splitlines() if l.strip()]
print(f"\n标签行数: {len(lines)}")
print("=" * 100)
for i, line in enumerate(lines):
    parsed = _parse_clock_range(line)
    if not parsed:
        print(f"第{i+1}行 {line!r}: 不是时钟范围格式")
        continue
    s_clock, e_clock = parsed[0], parsed[1]
    cands = []
    for folder, bounds in windows.items():
        for start, end in _resolve_clock_range(s_clock, e_clock, bounds, TZ):
            cands.append((folder, datetime.fromtimestamp(start, ZoneInfo(TZ)), datetime.fromtimestamp(end, ZoneInfo(TZ))))
    if len(cands) == 1:
        print(f"第{i+1}行 {line!r} -> OK: {cands[0][0]} ({cands[0][1]}~{cands[0][2]})")
    else:
        print(f"第{i+1}行 {line!r} -> 匹配数={len(cands)}: " + "; ".join(f"{c[0]}({c[1]})" for c in cands))

# 反向：哪些文件夹没有被任何标签匹配
print("=" * 100)
matched_folders = set()
for i, line in enumerate(lines):
    parsed = _parse_clock_range(line)
    if not parsed:
        continue
    s_clock, e_clock = parsed[0], parsed[1]
    for folder, bounds in windows.items():
        for start, end in _resolve_clock_range(s_clock, e_clock, bounds, TZ):
            matched_folders.add(folder)
print("未匹配到任何标签的文件夹:")
for folder in sorted(windows):
    if folder not in matched_folders:
        print(f"  {folder}")
