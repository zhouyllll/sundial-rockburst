"""Tirex 批量预填：用官方 batch API 一次生成全部预测缓存。

缓存 key 与 _p1_backends.P1Forecaster.predict 完全一致
（sha256(identity_json + seq.tobytes() + str(horizon))），
主实验跑 tirex 时逐样本命中缓存，不再串行推理。

sequence 构造复刻 rockburst/data.py Inputs.sample() 的 blocks 选择逻辑：
now -> latest/cutoff/first -> mean_square blocks -> log1p(power)，
horizon = 180 + lag_steps（lag_steps = ceil((now-cutoff)/10)）。
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, str(ROOT))
EXP06 = ROOT / "artifacts" / "experiment_06_auto_thr"
CACHE = ROOT / "_p1_cache" / "tirex"
ZONE = "zone_A"
TIMEZONE = "Asia/Shanghai"
HISTORY_MINUTES = 30
MAX_LAG_SECONDS = 60

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
    import tirex
    import torch
    model = tirex.load_model("NX-AI/TiRex", device="cuda", backend="torch")
    model.eval()
    # identity.samples 字段 = 分支数（与 P1Forecaster 一致），不是用户 samples 参数
    ident = _identity("tirex", None, len(model.config.quantiles), "cuda")
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

    # 按 horizon 分桶（context 长度一致才可 batch），桶内一次前向
    by_h = {}
    for seq, horizon, key, now in todo:
        by_h.setdefault(horizon, []).append((seq, key, now))
    done = 0
    for horizon in sorted(by_h):
        bucket = by_h[horizon]
        seqs = [s for s, _, _ in bucket]
        context = np.stack(seqs).astype(np.float32)  # (B, T)
        print(f"horizon={horizon} 条数={len(bucket)} 推理中...", flush=True)
        for i in range(0, len(context), 512):
            chunk = context[i:i + 512]
            with torch.inference_mode():
                fc = model.forecast(chunk, prediction_length=horizon,
                                    batch_size=len(chunk), output_type="torch")
            q = fc[0].detach().cpu().numpy()  # (B, pred_len, Q)
            q = np.transpose(q, (0, 2, 1))  # (B, Q, pred_len)
            for j, (_, key, _) in enumerate(bucket[i:i + 512]):
                np.save(CACHE / f"{key}.npy", q[j].astype(np.float32))
                done += 1
            if done % 500 < 512:
                print(f"已写 {done}/{len(todo)}", flush=True)
    print(f"预填完成 {done}", flush=True)


if __name__ == "__main__":
    main()
