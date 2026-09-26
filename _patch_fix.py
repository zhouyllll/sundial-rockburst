# -*- coding: utf-8 -*-
"""修复 extract.py（毫秒间隙容差+零填充）与 data.py（available 容差）。"""
import io, ast

p = r"G:\1A岩爆预测\sundial-rockburst\rockburst\extract.py"
c = io.open(p, encoding="utf-8").read()

# 1) 替换串行收集的 while 循环与覆盖检查（注意 20 空格缩进）
old_loop = '''                    position, segments, sources = a, [], []
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
                        continue'''
new_loop = '''                    position, segments, sources = a, [], []
                    while idx < len(group) and position < b - 1e-6:
                        r = group[idx]
                        if r["end"] <= position:
                            idx += 1
                            continue
                        if r["start"] > position + GAP_TOL:
                            break  # 真实长间隙，本 cell 无法覆盖
                        start_off = max(position, r["start"])
                        end = min(b, r["end"])
                        sources.append(r)
                        segments.append((r, start_off, end))
                        position = end
                        idx += 1
                    if b - position > GAP_TOL:
                        continue'''
assert old_loop in c, "extract while 循环未匹配"
c = c.replace(old_loop, new_loop)

# 2) 在 continuous_rows 顶部加 GAP_TOL 常量
old_def = "    def continuous_rows():\n        nonlocal count"
new_def = "    def continuous_rows():\n        nonlocal count\n        GAP_TOL = 0.2  # bin 之间毫秒级时钟漂移容差（秒）"
assert old_def in c, "continuous_rows 定义未匹配"
c = c.replace(old_def, new_def)

# 3) _cell_features 支持段间零填充 + 末尾补零
old_feat = '''def _cell_features(arg):
    """进程池 worker：拼接各段波形并计算特征。arg=(segments, fs, low, high)"""
    segments, fs, low, high = arg
    data = np.concatenate([read_segment(r, a, b) for r, a, b in segments], axis=0)
    if abs(len(data) - 10 * fs) > len(segments):
        raise ValueError("跨文件采样点数异常")
    return waveform_features(data, fs, low, high)'''
new_feat = '''def _cell_features(arg):
    """进程池 worker：拼接各段波形并计算特征。arg=(segments, fs, low, high)
    段与段之间若存在毫秒级间隙（bin 时钟漂移），以零填充补齐。"""
    segments, fs, low, high = arg
    chunks, prev_end = [], None
    for r, a, b in segments:
        seg = read_segment(r, a, b)
        if prev_end is not None:
            gap = a - prev_end
            if gap > 1e-6:
                chunks.append(np.zeros((int(round(gap * fs)), seg.shape[1]), dtype=seg.dtype))
        chunks.append(seg)
        prev_end = b
    data = np.concatenate(chunks, axis=0)
    target = int(10 * fs)
    if len(data) < target:
        data = np.vstack([data, np.zeros((target - len(data), data.shape[1]), dtype=data.dtype)])
    if abs(len(data) - target) > len(segments):
        raise ValueError("跨文件采样点数异常")
    return waveform_features(data, fs, low, high)'''
assert old_feat in c, "_cell_features 未匹配"
c = c.replace(old_feat, new_feat)

io.open(p, "w", encoding="utf-8", newline="").write(c)
ast.parse(io.open(p, encoding="utf-8").read())
print("extract.py 修复完成，语法 OK")

# ---------- data.py ----------
p2 = r"G:\1A岩爆预测\sundial-rockburst\rockburst\data.py"
d = io.open(p2, encoding="utf-8").read()
old_avail = 'if not blocks or any(r is None or r["available"] > now for r in blocks):'
new_avail = 'if not blocks or any(r is None or r["available"] > now + max_lag_seconds for r in blocks):'
assert old_avail in d, "data.py available 检查未匹配"
d = d.replace(old_avail, new_avail)
io.open(p2, "w", encoding="utf-8", newline="").write(d)
ast.parse(io.open(p2, encoding="utf-8").read())
print("data.py 修复完成，语法 OK")
