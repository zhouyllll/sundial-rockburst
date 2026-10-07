import numpy as np, json, sys
p = r'G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_06_auto_thr\dataset.npz'
d = np.load(p, allow_pickle=True)
print('keys:', d.files, flush=True)
for k in d.files:
    a = d[k]
    print(k, a.shape, a.dtype, flush=True)
if 'metadata' in d.files:
    m = d['metadata']
    obj = m.item() if hasattr(m, 'item') else m
    print('metadata:', json.dumps(obj, indent=1, ensure_ascii=False)[:2000], flush=True)
