# -*- coding: utf-8 -*-
"""把 _cell_features 从 extract() 内部提升为模块级函数（spawn 可 pickle）。"""
import io, ast

p = r"G:\1A岩爆预测\sundial-rockburst\rockburst\extract.py"
c = io.open(p, encoding="utf-8").read()

start_marker = "    def _cell_features(arg):"
end_marker = "        return waveform_features(data, fs, low, high)"
i = c.find(start_marker)
j = c.find(end_marker)
assert i >= 0 and j > i, "未找到 _cell_features"
j += len(end_marker)

fn = c[i:j]
# 去掉 4 空格基础缩进，提升为模块级
lines = fn.splitlines()
assert all(ln.startswith("    ") for ln in lines), "函数行缩进不一致"
module_fn = "\n".join(ln[4:] if ln else ln for ln in lines) + "\n"

# 从 extract 内部移除
c = c[:i] + c[j:]
# 插到模块顶层 def extract( 之前
pos = c.find("\ndef extract(")
assert pos >= 0
c = c[:pos] + "\n" + module_fn + c[pos:]

io.open(p, "w", encoding="utf-8", newline="").write(c)
ast.parse(io.open(p, encoding="utf-8").read())
print("提升为模块级完成，语法 OK")
