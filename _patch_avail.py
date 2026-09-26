# -*- coding: utf-8 -*-
import io, ast
p = r"G:\1A岩爆预测\sundial-rockburst\rockburst\extract.py"
c = io.open(p, encoding="utf-8").read()
old = '''                    r = sources[0]
                    jobs.append((zone, a, b, max(s["available"] for s in sources),
                                 r["signature"], segments, r["fs"], low, high))'''
new = '''                    r = sources[0]
                    avail = max(max(s["available"] for s in sources), b)
                    jobs.append((zone, a, b, avail,
                                 r["signature"], segments, r["fs"], low, high))'''
assert old in c, "未匹配 jobs.append"
c = c.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(c)
ast.parse(io.open(p, encoding="utf-8").read())
print("available 修复完成，语法 OK")
