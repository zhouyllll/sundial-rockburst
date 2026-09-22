"""为 label.txt 补全日期：对每个标签找钟点最接近的事件文件夹（文件夹名=M.D.HH.MM），取该文件夹日期。"""
import re
from datetime import datetime, timedelta
from pathlib import Path

data_root = Path(r"G:\1A岩爆预测\岩爆数据集")
labels_file = Path(r"G:\1A岩爆预测\label.txt")
out_txt = Path(r"G:\1A岩爆预测\rockbursts_full.txt")
out_csv = Path(r"G:\1A岩爆预测\rockbursts_full.csv")

YEAR = 2026

# 1. 解析所有文件夹（含空文件夹），名字格式 M.D.HH.MM
folders = []
for p in sorted(data_root.iterdir()):
    if not p.is_dir():
        continue
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{1,2})\.(\d{1,2})", p.name)
    if not m:
        print(f"跳过无法解析的文件夹名: {p.name}")
        continue
    mo, d, h, mi = map(int, m.groups())
    try:
        t = datetime(YEAR, mo, d, h, mi)
    except ValueError:
        print(f"跳过非法日期文件夹: {p.name}")
        continue
    nbin = len(list(p.glob("*.bin")))
    folders.append({"name": p.name, "time": t, "nbin": nbin})
folders.sort(key=lambda x: x["time"])
print(f"解析到 {len(folders)} 个文件夹（含空文件夹）")

# 2. 解析标签行
lines = [l.strip() for l in labels_file.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
print(f"标签行数: {len(lines)}")

def parse_label(line):
    m = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})[:.](\d{2})\s*-\s*(\d{1,2})[:.](\d{2})[:.-](\d{2})\s*", line)
    if not m:
        return None
    h1, m1, s1, h2, m2, s2 = map(int, m.groups())
    return h1, m1, s1, h2, m2, s2

# 3. 对每个标签找钟点最接近的文件夹（按秒差）
rows = []
used = set()
for i, line in enumerate(lines):
    p = parse_label(line)
    if not p:
        print(f"第{i+1}行无法解析: {line!r}")
        continue
    h1, m1, s1, h2, m2, s2 = p
    label_seconds = h1 * 3600 + m1 * 60 + s1
    best = None
    best_diff = None
    for f in folders:
        ft = f["time"]
        f_seconds = ft.hour * 3600 + ft.minute * 60
        # 时钟差（考虑跨午夜）
        diff = abs(label_seconds - f_seconds)
        diff = min(diff, 86400 - diff)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = f
    # 用该文件夹的日期 + 标签的时分秒构造完整时间戳
    event_dt = datetime(YEAR, best["time"].month, best["time"].day, h1, m1, s1)
    end_dt = event_dt.replace(hour=h2, minute=m2, second=s2)
    if end_dt < event_dt:
        end_dt += timedelta(days=1)
    rows.append({"idx": i + 1, "label": line, "folder": best["name"], "folder_time": best["time"],
                 "diff_s": best_diff, "nbin": best["nbin"], "start": event_dt, "end": end_dt,
                 "group": "folder-" + __import__("hashlib").sha1(best["name"].encode("utf-8")).hexdigest()[:12]})

print("=" * 110)
print("配对结果（按标签行号）：")
for r in rows:
    flag = " [!空文件夹!]" if r["nbin"] == 0 else ""
    print(f"第{r['idx']:2d}行 {r['label']!r} -> {r['folder']} ({r['folder_time']:%m-%d %H:%M}) 差{r['diff_s']}s | {r['start']:%Y-%m-%d %H:%M:%S}{flag}")

# 冲突检查：同一文件夹被多个标签选中
from collections import Counter
cnt = Counter(r["folder"] for r in rows)
conflicts = {k: v for k, v in cnt.items() if v > 1}
print("=" * 110)
if conflicts:
    print("冲突：以下文件夹被多个标签选中", conflicts)
else:
    print("无冲突：每个文件夹最多对应一个标签")

empty_pairs = [r for r in rows if r["nbin"] == 0]
if empty_pairs:
    print("注意：以下标签配对了空文件夹（无bin数据）:")
    for r in empty_pairs:
        print(f"  第{r['idx']}行 {r['label']} -> {r['folder']}")
else:
    print("所有标签都配对了有数据的文件夹")

# 4. 写出完整时间戳标签（单时间戳 txt + 含起止/分组的 CSV）
import csv as _csv
with open(out_txt, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(f"{r['start']:%Y-%m-%d %H:%M:%S}\n")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=["onset_time", "end_time", "event_id", "group_id"])
    w.writeheader()
    for r in rows:
        w.writerow({"onset_time": f"{r['start']:%Y-%m-%d %H:%M:%S}",
                    "end_time": f"{r['end']:%Y-%m-%d %H:%M:%S}",
                    "event_id": f"rb-{r['idx']:04d}",
                    "group_id": r["group"]})
print(f"\n已写出: {out_txt} 和 {out_csv}")
