import sys
sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst\_p1_lagllama_src")
import numpy as np
import torch
from lag_llama.gluon.lightning_module import LagLlamaLightningModule

import types
_dummy = types.ModuleType("gluonts.torch.modules.loss")
class _DistributionLoss:
    pass
class _NegativeLogLikelihood(_DistributionLoss):
    pass
_dummy.DistributionLoss = _DistributionLoss
_dummy.NegativeLogLikelihood = _NegativeLogLikelihood
sys.modules.setdefault("gluonts.torch.modules.loss", _dummy)

from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("time-series-foundation-models/Lag-Llama", "lag-llama.ckpt")
raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
mk = dict(raw["hyper_parameters"]["model_kwargs"])
mk["context_length"] = 180
mk["num_parallel_samples"] = 20
module = LagLlamaLightningModule(
    model_kwargs=mk, context_length=180,
    prediction_length=187, use_kv_cache=True)
module.load_state_dict(raw["state_dict"])
module = module.to("cuda")
module.eval()
model = module.model
print("model ready", model.device if hasattr(model, "device") else "?", flush=True)

import pandas as pd
from gluonts.time_feature import time_features_from_frequency_str
lag_pad = 1092
ctx = 180
for n, h in [(180, 180), (178, 182), (174, 186), (179, 181)]:
    past_len = lag_pad + ctx
    dates = pd.date_range("2024-01-01 00:00:00", periods=past_len + h, freq="s")
    past_dates, future_dates = dates[:past_len], dates[past_len:]
    past_tf = np.stack([f(past_dates) for f in time_features_from_frequency_str("s")], axis=-1).astype(np.float32)
    fut_tf = np.stack([f(future_dates) for f in time_features_from_frequency_str("s")], axis=-1).astype(np.float32)
    past = torch.zeros(1, past_len).float().cuda()
    seq = torch.from_numpy(np.linspace(0.0, 3.0, n).astype(np.float32)).cuda()
    past[0, lag_pad:lag_pad + n] = seq
    obs = torch.zeros(1, past_len).float().cuda()
    obs[0, lag_pad:lag_pad + n] = 1.0
    try:
        with torch.no_grad():
            out = module(past_target=past, past_observed_values=obs,
                         past_time_feat=torch.from_numpy(past_tf)[None].cuda(),
                         future_time_feat=torch.from_numpy(fut_tf)[None].cuda())
        print(f"len={n} h={h} out={tuple(out.shape)}", flush=True)
        samps = out[0].cpu().numpy()
        print(f"len={n} h={h} samps={samps.shape} finite={bool(np.isfinite(samps).all())} ok", flush=True)
    except Exception as e:
        print(f"len={n} h={h} ERR {str(e)[:160]}", flush=True)
print("done", flush=True)
