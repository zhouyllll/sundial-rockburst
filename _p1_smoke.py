"""P1 冒烟：三个后端各预测一次，打印分布形状与身份信息。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")
from _p1_backends import P1Forecaster

CACHE = Path(r"G:\1A岩爆预测\sundial-rockburst\_p1_cache")
CACHE.mkdir(exist_ok=True)

rng = np.random.default_rng(7)
seq = np.log1p(rng.lognormal(0, 1.0, size=180).clip(1e-3))

for backend, model_id in [
    ("chronos2", None),
    ("tirex", None),
    ("lagllama", None),
]:
    try:
        fc = P1Forecaster(backend, model_id=model_id, device="cuda", samples=20, hf_cache=CACHE)
        paths = fc.predict(seq, horizon=180)
        print(f"[{backend}] OK shape={paths.shape} finite={np.isfinite(paths).all()} "
              f"range=({paths.min():.3f},{paths.max():.3f}) identity={fc.identity}", flush=True)
        del fc
    except Exception as exc:
        print(f"[{backend}] FAIL {type(exc).__name__}: {exc}", flush=True)
