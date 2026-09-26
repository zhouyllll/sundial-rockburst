# -*- coding: utf-8 -*-
import io
c = io.open(r"G:\1A岩爆预测\sundial-rockburst\rockburst\extract.py", encoding="utf-8").read()
probes = [
    'position, segments, sources = a, [], []',
    'while idx < len(group) and position < b - 1e-6:',
    'if r["start"] > position + 1e-6 or r["end"] <= position:',
    'segments.append((r, position, end))',
    'if position < b - 1e-6:',
    'continue  # A block crossing',
]
for p in probes:
    print(repr(p), '->', c.find(p))
i = c.find('while idx < len(group)')
print('=== while 附近原文 ===')
print(repr(c[i-130:i+360]))
