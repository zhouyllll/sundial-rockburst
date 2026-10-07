import sys
sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst\_p1_lagllama_src")
import numpy as np
import torch
from lag_llama.model.module import LagLlamaModel

orig = LagLlamaModel.prepare_input
def patched(self, past_target, past_observed_values, past_time_feat=None,
           future_time_feat=None, future_target=None):
    print("past_target", tuple(past_target.shape), "past_time_feat", None if past_time_feat is None else tuple(past_time_feat.shape),
          "future_time_feat", None if future_time_feat is None else tuple(future_time_feat.shape),
          "max_lag", self.lags_seq and max(self.lags_seq), flush=True)
    return orig(self, past_target, past_observed_values, past_time_feat, future_time_feat, future_target)
LagLlamaModel.prepare_input = patched

from _p1_backends import P1Forecaster
fc = P1Forecaster("lagllama", device="cuda", samples=20)
seq = np.linspace(0.0, 3.0, 180).astype(np.float32)
try:
    out = fc.predict(seq, horizon=180)
    print("OK", out.shape, flush=True)
except Exception as e:
    print("ERR", str(e)[:200], flush=True)
