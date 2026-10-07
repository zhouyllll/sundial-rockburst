"""P1：Sundial 平替后端（Chronos-2 / TiRex / Lag-Llama）。

与 rockburst/forecast.py 的 Forecaster 保持相同 predict 契约：
predict(sequence: 1-D float) -> (samples, horizon) float32 有限数组。
各模型输出分布统一整理为 (分布支路, horizon) 的路径矩阵，供 data.py 的
窗口均值/中位数特征复用（transform="log1p-mean-square-v1" 同口径）。

Lag-Llama 走本地官方 modeling（_p1_lagllama_src），加载 hf 仓库
time-series-foundation-models/Lag-Llama 的 lag-llama.ckpt（lightning 权重）。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

LAGLLAMA_SRC = Path(r"G:\1A岩爆预测\sundial-rockburst\_p1_lagllama_src")


def _identity(backend, model_id, samples, device):
    import torch
    import transformers
    return {
        "backend": backend, "samples": samples, "model_id": model_id,
        "transform": "log1p-mean-square-v1",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device, "dtype": "float32",
    }


class P1Forecaster:
    def __init__(self, backend, model_id=None, device="cuda", samples=20,
                 hf_cache=None, context_length=180):
        if backend not in {"chronos2", "tirex", "lagllama"}:
            raise ValueError(f"未知 P1 backend: {backend}")
        self.backend = backend
        self.model_id = model_id
        self.device = device
        self.samples = samples
        self.context_length = context_length
        self.cache = Path(hf_cache) if hf_cache else None
        self.model = None
        self._load()
        self.identity = _identity(backend, model_id, self._branch_count(), device)

    # ---------- 加载 ----------
    def _load(self):
        import torch
        if self.backend == "chronos2":
            from chronos import Chronos2Pipeline
            pipe = Chronos2Pipeline.from_pretrained(self.model_id or "amazon/chronos-2",
                                                    torch_dtype=torch.float32)
            pipe.model = pipe.model.to(self.device).eval()
            pipe.model.requires_grad_(False)
            self.pipe = pipe
        elif self.backend == "tirex":
            import tirex
            self.model = tirex.load_model(self.model_id or "NX-AI/TiRex",
                                          device=self.device, backend="torch")
            self.model.eval()
        elif self.backend == "lagllama":
            if str(LAGLLAMA_SRC) not in sys.path:
                sys.path.insert(0, str(LAGLLAMA_SRC))
            from lag_llama.gluon.lightning_module import (LagLlamaLightningModule,
                                                         NegativeLogLikelihood)
            ckpt_path = self.model_id or "time-series-foundation-models/Lag-Llama"
            if not Path(ckpt_path).is_file():
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download("time-series-foundation-models/Lag-Llama",
                                            "lag-llama.ckpt")
            # ckpt 里 pickled 了 gluonts loss 实例（py3.13 无该模块 wheel）：
            # 注册占位模块供 pickle 反序列化，推理路径不使用 loss。
            import types
            _dummy = types.ModuleType("gluonts.torch.modules.loss")

            class _DistributionLoss:
                pass

            class _NegativeLogLikelihood(_DistributionLoss):
                pass

            _dummy.DistributionLoss = _DistributionLoss
            _dummy.NegativeLogLikelihood = _NegativeLogLikelihood
            sys.modules.setdefault("gluonts.torch.modules.loss", _dummy)
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mk = dict(raw["hyper_parameters"]["model_kwargs"])
            mk["context_length"] = self.context_length
            # 并行样本数与 Sundial（20）口径一致，控制推理成本（官方默认 100）
            mk["num_parallel_samples"] = self.samples
            # 预测路径不依赖 loss；不传 loss 可跳过 LightningModule 构造时的
            # validator 校验 forward（该路径的 future_time_feat 长度语义不同）。
            # use_single_pass_sampling=True：每步只 forward batch=1 的 greedy 预测，
            # 再一次性并行采样 num_parallel_samples 条（官方加速推理路径，速度快约 20 倍）。
            module = LagLlamaLightningModule(
                model_kwargs=mk, context_length=self.context_length,
                prediction_length=self.context_length + 7,
                use_single_pass_sampling=True,
                use_kv_cache=True)
            module.load_state_dict(raw["state_dict"])
            module = module.to(self.device)
            module.eval()
            self.model = module
            self._lag_pad = max(mk["lags_seq"])

    def _branch_count(self):
        if self.backend == "chronos2":
            return len(getattr(self.pipe, "quantiles", [])) or 9
        if self.backend == "tirex":
            return len(self.model.config.quantiles)
        return int(getattr(self.model.model, "num_parallel_samples", self.samples))

    # ---------- 预测 ----------
    def predict(self, sequence, horizon=180):
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 1 or not len(sequence) or not np.isfinite(sequence).all():
            raise ValueError("预测输入必须是非空、有限的一维序列")
        if horizon <= 0:
            raise ValueError("预测长度必须为正")
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode() +
                             sequence.tobytes() + str(horizon).encode()).hexdigest()
        cache_file = self.cache / f"{key}.npy" if self.cache else None
        if cache_file and cache_file.exists():
            return np.load(cache_file, allow_pickle=False)
        paths = self._predict_raw(sequence, horizon)
        paths = np.asarray(paths, dtype=np.float32)
        if paths.shape != (self._branch_count(), horizon) or not np.isfinite(paths).all():
            raise ValueError(f"预测输出形状/数值异常: {paths.shape}")
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.save(stream, paths, allow_pickle=False)
            tmp.replace(cache_file)
        return paths

    def _predict_raw(self, sequence, horizon):
        import torch
        if self.backend == "chronos2":
            # 输入保持 CPU tensor：pipeline 内部负责搬运到 self.device；
            # 若传入 cuda tensor，pin_memory 会报 only dense CPU tensors can be pinned。
            out = self.pipe.predict(torch.from_numpy(sequence)[None, None],
                                    prediction_length=horizon, batch_size=1)
            q = out[0]  # (n_variates, n_quantiles, pred_len)
            return q.squeeze(0).detach().cpu().numpy()
        if self.backend == "tirex":
            quantiles, _ = self.model._forecast_quantiles(
                torch.from_numpy(sequence)[None].to(self.device), prediction_length=horizon)
            return quantiles[0].detach().cpu().numpy().T  # (n_quantiles, pred_len)
        # lagllama：官方自回归预测路径（module.forward 逐 t 生成，
        # 循环到 prediction_length 步；future_time_feat 必须给满
        # prediction_length 步，内部按 [:t+1] 逐步切片）。
        import pandas as pd
        from gluonts.time_feature import time_features_from_frequency_str
        past_len = self._lag_pad + self.context_length
        pred_len = self.context_length + 7  # 与 module.prediction_length 一致
        horizon_dates = past_len + pred_len
        dates = pd.date_range("2024-01-01 00:00:00", periods=horizon_dates, freq="s")
        past_dates = dates[:past_len]
        future_dates = dates[past_len:]
        past_time_feat = np.stack([f(past_dates) for f in
                                   time_features_from_frequency_str("s")],
                                  axis=-1).astype(np.float32)
        future_time_feat = np.stack([f(future_dates) for f in
                                     time_features_from_frequency_str("s")],
                                    axis=-1).astype(np.float32)
        past_target = torch.zeros(1, past_len, dtype=torch.float32,
                                  device=self.device)
        past_target[0, self._lag_pad:self._lag_pad + len(sequence)] = \
            torch.from_numpy(sequence).to(self.device)
        past_observed = torch.zeros(1, past_len, dtype=torch.float32,
                                    device=self.device)
        past_observed[0, self._lag_pad:self._lag_pad + len(sequence)] = 1.0
        with torch.inference_mode():
            out = self.model.forward(past_target=past_target,
                                     past_observed_values=past_observed,
                                     past_time_feat=torch.from_numpy(
                                         past_time_feat)[None].to(self.device),
                                     future_time_feat=torch.from_numpy(
                                         future_time_feat)[None].to(self.device))
        # (1, num_parallel_samples, prediction_length) -> (num_samples, horizon)
        return out[0].detach().cpu().numpy()[:, :horizon]
