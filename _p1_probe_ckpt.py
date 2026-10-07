"""探测 Lag-Llama ckpt 配置与官方 time features 数量（供修复 time_feat 维度）。"""
import glob
import sys
import types

from gluonts.time_feature import time_features_from_frequency_str

# py3.13 无 gluonts.torch.modules.loss wheel：注册占位模块供 ckpt 反序列化
_dummy = types.ModuleType("gluonts.torch.modules.loss")

class _DistributionLoss:
    pass

class _NegativeLogLikelihood(_DistributionLoss):
    pass

_dummy.DistributionLoss = _DistributionLoss
_dummy.NegativeLogLikelihood = _NegativeLogLikelihood
sys.modules.setdefault("gluonts.torch.modules.loss", _dummy)

tfs = time_features_from_frequency_str("S")
print("n_tf_S", len(tfs), [type(t).__name__ for t in tfs])

import torch
p = glob.glob(r"G:\hf_cache\huggingface\hub\models--time-series-foundation-models--Lag-Llama\snapshots\*\lag-llama.ckpt")[0]
raw = torch.load(p, map_location="cpu", weights_only=False)
mk = raw["hyper_parameters"]["model_kwargs"]
print("mk_time_feat", mk.get("time_feat"))
print("mk_lags", mk.get("lags_seq"))
print("mk_keys", [k for k in mk.keys()])
sd = raw["state_dict"]
print("sd_keys_head", [k for k in sd.keys() if "proj" in k or "embed" in k][:12])
