# -*- coding: utf-8 -*-
"""修复 extract.py continuous_rows：移除重叠前置检查，保留 starts 行。"""
import io

p = r"G:\1A岩爆预测\sundial-rockburst\rockburst\extract.py"
c = io.open(p, encoding="utf-8").read()

lines = c.splitlines(keepends=True)
print("---- 修改前 142-156 ----")
for i in range(141, min(157, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()}")

# 定位 continuous_rows 内的 group 行
gi = None
for i, line in enumerate(lines):
    if 'group = sorted((r for r in records if r["zone"] == zone)' in line:
        gi = i
        break
assert gi is not None, "group 行未找到"

# 从 group 行后开始，找到 starts 定义或注释行
# 期望结构: group行 / [注释行...] / starts行 /  # Iterate...
# 直接重建: group行 后面必须是 starts 行
new_lines = lines[: gi + 1]
new_lines.append('            starts = [r["start"] for r in group]\n')
# 跳过原 group 行之后的旧内容，直到 "# Iterate occupied grid cells" 注释行
for j in range(gi + 1, len(lines)):
    if "Iterate occupied grid cells" in lines[j]:
        new_lines.extend(lines[j:])
        break
else:
    raise SystemExit("未找到 Iterate 注释行")

out = "".join(new_lines)
io.open(p, "w", encoding="utf-8", newline="").write(out)
print("---- 修改后 142-156 ----")
for i, line in enumerate(out.splitlines()):
    if 141 <= i <= 156:
        print(f"{i+1}: {line}")
print("OK")
