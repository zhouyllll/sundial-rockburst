# -*- coding: utf-8 -*-
"""修正版：continuous_rows 是 extract() 的嵌套函数，新代码需保持 4 空格缩进。"""
import io

p = r"G:\1A岩爆预测\sundial-rockburst\rockburst\extract.py"
c = io.open(p, encoding="utf-8").read()

# 先回滚上一次错误替换（从首次 def _cell_features 到 write_csv 行）
i = c.find("def _cell_features(arg):")
j = c.find("    write_csv(output, event_rows() if kind")
assert i >= 0 and j > i, "找不到上次替换区间"
c = c[:i] + "def continuous_rows():\n" + c[j:]

# 现在定位 extract 内的 continuous_rows（带 4 空格缩进）
k = c.find("    def continuous_rows():")
assert k >= 0, "未找到 continuous_rows"
m = c.find("    write_csv(output, event_rows() if kind")
assert m > k
old = c[k:m]

new = '''    def _cell_features(arg):
        """进程池 worker：拼接各段波形并计算特征。arg=(segments, fs, low, high)"""
        segments, fs, low, high = arg
        data = np.concatenate([read_segment(r, a, b) for r, a, b in segments], axis=0)
        if abs(len(data) - 10 * fs) > len(segments):
            raise ValueError("跨文件采样点数异常")
        return waveform_features(data, fs, low, high)

    def continuous_rows():
        nonlocal count
        jobs = []  # (zone, a, b, available, signature, segments, fs, low, high)
        for zone in sorted({r["zone"] for r in records}):
            group = sorted((r for r in records if r["zone"] == zone), key=lambda r: r["start"])
            starts = [r["start"] for r in group]
            # Iterate occupied grid cells, not long gaps between recordings.
            last_cell = None
            for record in group:
                for cell in range(math.floor(record["start"] / 10), math.ceil(record["end"] / 10)):
                    if last_cell is not None and cell <= last_cell:
                        continue
                    last_cell = cell
                    a, b = cell * 10., cell * 10. + 10.
                    idx = bisect_right(starts, a + 1e-6) - 1
                    if idx < 0:
                        continue
                    position, segments, sources = a, [], []
                    while idx < len(group) and position < b - 1e-6:
                        r = group[idx]
                        if r["start"] > position + 1e-6 or r["end"] <= position:
                            break
                        end = min(b, r["end"])
                        sources.append(r)
                        segments.append((r, position, end))
                        position = end
                        idx += 1
                    if position < b - 1e-6:
                        continue
                    if len({r["signature"] for r in sources}) != 1:
                        continue  # A block crossing a device configuration change is missing.
                    r = sources[0]
                    jobs.append((zone, a, b, max(s["available"] for s in sources),
                                 r["signature"], segments, r["fs"], low, high))
        # 多进程并行计算特征，按时间顺序输出（executor.map 保序）
        import os
        from concurrent.futures import ProcessPoolExecutor
        workers = max(1, min(int(os.environ.get("ROCKBURST_WORKERS", "12")), os.cpu_count() or 4))
        args = [(segments, fs, low, high) for _, _, _, _, _, segments, fs, low, high in jobs]
        metas = [(zone, a, b, available, signature)
                 for zone, a, b, available, signature, _, _, _, _ in jobs]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for meta, f in zip(metas, executor.map(_cell_features, args)):
                zone, a, b, available, signature = meta
                count += 1
                yield dict(zone_id=zone, start_time=iso(a), end_time=iso(b),
                           available_time=iso(available), preprocessing_id=signature, **f)

'''

c = c[:k] + new + c[m:]
io.open(p, "w", encoding="utf-8", newline="").write(c)
import ast
ast.parse(io.open(p, encoding="utf-8").read())
print("替换完成，语法 OK")
