"""探测 exp06 dataset.npz 结构与特征定义来源。"""
import json
import numpy as np

ROOT = r"G:\1A岩爆预测\sundial-rockburst"
with np.load(ROOT + r"\artifacts\experiment_06_auto_thr\dataset.npz", allow_pickle=False) as pack:
    for k in pack.files:
        a = pack[k]
        print(k, getattr(a, "shape", None), getattr(a, "dtype", None))
    md = json.loads(str(pack["metadata"]))
print("metadata:", json.dumps(md, ensure_ascii=False)[:800])
