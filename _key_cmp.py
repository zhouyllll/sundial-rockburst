import hashlib, json, sys
sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")
import numpy as np
from pathlib import Path
from _p1_backends import _identity, P1Forecaster

ident = _identity("tirex", None, 20, "cuda")
ident_json = json.dumps(ident, sort_keys=True).encode()
seq = np.array([0.1, 0.2, 0.3], dtype=np.float64)
seq32 = np.asarray(seq, dtype=np.float32)
k1 = hashlib.sha256(ident_json + seq32.tobytes() + str(180).encode()).hexdigest()
fc = P1Forecaster("tirex", device="cuda", samples=20, hf_cache=Path(r"G:\1A岩爆预测\sundial-rockburst\_p1_cache\tirex"))
k2 = hashlib.sha256(json.dumps(fc.identity, sort_keys=True).encode() + seq32.tobytes() + str(180).encode()).hexdigest()
print("ident1:", json.dumps(ident, sort_keys=True))
print("ident2:", json.dumps(fc.identity, sort_keys=True))
print("match:", k1 == k2, k1[:12], k2[:12])
