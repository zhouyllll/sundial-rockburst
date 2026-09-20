"""Frozen Sundial adapter; persistence is explicitly a separate demo baseline."""

import hashlib
from pathlib import Path

import numpy as np


class Forecaster:
    def __init__(self, backend="persistence", model_path=None, device="cpu",
                 samples=20, cache=None):
        if backend not in {"persistence", "sundial"} or samples < 1:
            raise ValueError("backend 或 samples 无效")
        self.backend, self.samples = backend, samples
        self.cache = Path(cache) if cache else None
        self.model = None
        self.device = device
        identity = {"backend": backend, "samples": samples, "transform": "log1p-mean-square-v1"}
        if backend == "sundial":
            if not model_path or not Path(model_path).is_dir():
                raise ValueError("Sundial 模式需要已下载的本地 --model-path；不会自动联网下载")
            folder = Path(model_path)
            # Hash actual code/config/weights, not a mutable directory name.
            files = sorted(p for p in folder.rglob("*") if p.is_file() and
                           p.suffix in {".json", ".py", ".safetensors", ".bin"}
                           and ".cache" not in p.parts)
            if not any(p.suffix in {".safetensors", ".bin"} for p in files):
                raise ValueError("本地模型目录没有权重文件")
            digest = hashlib.sha256()
            for p in files:
                digest.update(str(p.relative_to(folder)).encode())
                with p.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            identity["model_sha256"] = digest.hexdigest()
            try:
                import torch
                import transformers
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise ValueError("请先安装 requirements-sundial.txt 和适合本机的 PyTorch") from exc
            if transformers.__version__ != "4.40.1":
                raise ValueError("基础版 Sundial 适配器要求 transformers==4.40.1")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(folder), trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.float32).to(device).eval()
            self.model.requires_grad_(False)
            identity.update(torch_version=torch.__version__, transformers_version=transformers.__version__,
                            device=device, dtype="float32")
        self.identity = identity

    def predict(self, sequence, horizon=180):
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 1 or not len(sequence) or not np.isfinite(sequence).all():
            raise ValueError("预测输入必须是非空、有限的一维序列")
        if horizon <= 0:
            raise ValueError("预测长度必须为正")
        import json
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode() +
                             sequence.tobytes() + str(horizon).encode()).hexdigest()
        cache_file = self.cache / f"{key}.npy" if self.cache else None
        if cache_file and cache_file.exists():
            result = np.load(cache_file, allow_pickle=False)
        elif self.backend == "persistence":
            result = np.full((self.samples, horizon), sequence[-1], dtype=np.float32)
        else:
            import torch
            torch.manual_seed(int(key[:8], 16))
            with torch.inference_mode():
                result = self.model.generate(
                    torch.from_numpy(sequence[None]).to(self.device),
                    max_new_tokens=horizon, num_samples=self.samples,
                ).detach().cpu().numpy()[0]
        if result.shape != (self.samples, horizon) or not np.isfinite(result).all():
            raise ValueError(f"预测输出形状/数值异常: {result.shape}")
        if cache_file and not cache_file.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            temp = cache_file.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.save(stream, result, allow_pickle=False)
            temp.replace(cache_file)
        return result
