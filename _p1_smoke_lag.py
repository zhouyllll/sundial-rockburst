import sys
sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")
import numpy as np
from _p1_backends import P1Forecaster

fc = P1Forecaster("lagllama", device="cuda", samples=20)
for n, h in [(180, 180), (178, 182), (174, 186), (179, 181)]:
    seq = np.expm1(np.linspace(0.0, 3.0, n)).astype(np.float32)  # 已是 log1p 域
    out = fc.predict(seq, horizon=h)
    ok = out.shape == (20, h) and np.isfinite(out).all()
    print(f"len={n} h={h} -> {out.shape} finite={bool(np.isfinite(out).all())} ok={ok}", flush=True)
print("done", flush=True)
