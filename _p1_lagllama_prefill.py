"""Lag-Llama 批量预填：single-pass 自回归按 batch 并行，一次写全部预测缓存。

缓存 key 与 _p1_backends.P1Forecaster.predict 完全一致
（sha256(identity_json + seq.tobytes() + str(horizon))），
主实验跑 lagllama 时逐样本命中缓存，不再串行推理。

single-pass 分支（use_single_pass_sampling=True）原生支持 batch>1：
past_target (B, past_len) -> 每步 forward batch=B 的 greedy 预测，
再 repeat_interleave(num_parallel_samples) 并行采样。
CPU prepare_input 每步只做一次（B 序列共享），是相对串行约 B 倍的提速点。
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "_p1_lagllama_src"))
EXP06 = ROOT / "artifacts" / "experiment_06_auto_thr"
CACHE = ROOT / "_p1_cache" / "lagllama"
ZONE = "zone_A"
TIMEZONE = "Asia/Shanghai"
HISTORY_MINUTES = 30
MAX_LAG_SECONDS = 60
BATCH = 8

from rockburst.io import read_csv
from rockburst.data import Inputs
from rockburst.workflow import local_time
from _p1_backends import _identity


def build_sequences():
    rows = read_csv(str(EXP06 / "generated" / "continuous_manifest.csv"),
                    ["file_path", "zone_id", "start_time", "available_time", "fs", "channels"])
    data = Inputs(str(EXP06 / "continuous.csv"), str(EXP06 / "events.csv"),
                  str(EXP06 / "generated" / "coverage.csv"), ZONE,
                  str(EXP06 / "generated" / "rockbursts.csv"), microseismic_enabled=False)
    out = []
    for r in rows:
        now = local_time(r["available_time"], TIMEZONE)
        latest = math.floor(now / 10) * 10
        cutoff = None
        for end in range(latest, math.ceil((now - MAX_LAG_SECONDS) / 10) * 10 - 1, -10):
            block = data.blocks.get(end)
            if block and block["available"] <= now:
                cutoff = end
                break
        if cutoff is None:
            continue
        first = math.ceil((now - HISTORY_MINUTES * 60) / 10) * 10 + 10
        blocks = [data.blocks.get(end) for end in range(first, cutoff + 1, 10)]
        if not blocks or any(r is None or r["available"] > now for r in blocks):
            continue
        if len(blocks) < HISTORY_MINUTES * 6 - math.ceil(MAX_LAG_SECONDS / 10) - 1:
            continue
        power = np.array([r["mean_square"] for r in blocks], dtype=np.float64)
        seq = np.log1p(power).astype(np.float32)
        lag_steps = math.ceil((now - cutoff) / 10)
        horizon = 180 + lag_steps
        out.append((seq, horizon, now))
    return out


def main():
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
        prediction_length=187, use_single_pass_sampling=True, use_kv_cache=True)
    module.load_state_dict(raw["state_dict"])
    module = module.to("cuda")
    module.eval()
    print("module ready", flush=True)

    ident = _identity("lagllama", None, 20, "cuda")
    ident_json = json.dumps(ident, sort_keys=True).encode()

    items = build_sequences()
    print(f"有效序列 {len(items)}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = []
    for seq, horizon, now in items:
        key = hashlib.sha256(ident_json + seq.tobytes() + str(horizon).encode()).hexdigest()
        if (CACHE / f"{key}.npy").exists():
            continue
        todo.append((seq, horizon, key, now))
    print(f"待推理 {len(todo)}", flush=True)

    import pandas as pd
    from gluonts.time_feature import time_features_from_frequency_str
    past_len = max(mk["lags_seq"]) + 180
    pred_len = 187
    dates = pd.date_range("2024-01-01 00:00:00", periods=past_len + pred_len, freq="s")
    past_dates = dates[:past_len]
    future_dates = dates[past_len:]
    past_time_feat = np.stack([f(past_dates) for f in
                               time_features_from_frequency_str("s")],
                              axis=-1).astype(np.float32)  # (past_len, 6)
    future_time_feat = np.stack([f(future_dates) for f in
                                 time_features_from_frequency_str("s")],
                                axis=-1).astype(np.float32)  # (pred_len, 6)

    done = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        n = len(chunk)
        past_target = torch.zeros(n, past_len, dtype=torch.float32, device="cuda")
        past_observed = torch.zeros(n, past_len, dtype=torch.float32, device="cuda")
        for j, (seq, _, _, _) in enumerate(chunk):
            k0 = max(mk["lags_seq"])  # 尾部对齐：lag_pad 之后放序列
            past_target[j, k0:k0 + len(seq)] = torch.from_numpy(seq).to("cuda")
            past_observed[j, k0:k0 + len(seq)] = 1.0
        pt = torch.from_numpy(past_time_feat)[None].expand(n, -1, -1).to("cuda")
        ft = torch.from_numpy(future_time_feat)[None].expand(n, -1, -1).to("cuda")
        with torch.inference_mode():
            out = module.forward(past_target=past_target,
                                 past_observed_values=past_observed,
                                 past_time_feat=pt,
                                 future_time_feat=ft)
        # (n, 20, 187)
        samps = out.detach().cpu().numpy()
        for j, (seq, horizon, key, now) in enumerate(chunk):
            np.save(CACHE / f"{key}.npy", samps[j, :, :horizon].astype(np.float32))
            done += 1
        if done % 800 < BATCH:
            print(f"已写 {done}/{len(todo)}", flush=True)
    print(f"预填完成 {done}", flush=True)


if __name__ == "__main__":
    main()
