"""P2：掩码重建自监督预训练（报告 9.1 P2 数据范式，C7 类别二）。

在全部无标签连续波形特征（continuous.csv，约 1.8 万 10s 块）上，
用轻量 Transformer encoder 做掩码重建，学习特征时序分布；
产物 encoder 权重 + 归一化统计，供 _p2_extract.py 对每个预测样本的
30min 历史窗口提取表征，再接入 9 参数风险模型微调（_p2_finetune.py）。

用法:
    python _p2_pretrain.py                # 全量预训练并保存
    python _p2_pretrain.py --smoke        # 冒烟：3 步训练，验证链路
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, str(ROOT))
CSV = ROOT / "artifacts" / "experiment_06_auto_thr" / "continuous.csv"
OUT = ROOT / "artifacts" / "experiment_09_p2"
OUT.mkdir(parents=True, exist_ok=True)

# 预训练特征：8 维核心连续特征（不含短窗瞬态列，exp06 起均有）
FEAT_COLS = ["mean_square", "rms", "peak", "energy_proxy",
             "stalta", "coherence", "active_fraction", "low_band_fraction"]
WINDOW = 180          # 30min（10s 块）
STRIDE = 10           # ~1.7min 步长，增加重叠窗口样本量
MASK_RATIO = 0.25
D_MODEL, N_LAYERS, N_HEADS, FFN = 128, 3, 4, 512
EPOCHS = 50
BATCH = 64
LR = 1e-3
SEED = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MaskedEncoder(nn.Module):
    """轻量 Transformer encoder，掩码重建任务。"""

    def __init__(self, n_features=len(FEAT_COLS), d_model=D_MODEL,
                 n_layers=N_LAYERS, n_heads=N_HEADS, ffn=FFN):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_features)

    def forward(self, x, mask):
        # x: (B, T, F)；mask: (B, T) bool，True=被掩码（输入置 0）
        z = self.proj(x * (~mask).unsqueeze(-1).float())
        z = self.encoder(self.pos(z))
        return self.head(z)          # (B, T, F) 各位置重建


def load_data():
    import csv
    import datetime as dt
    rows = []
    with open(CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["zone_id"] != "zone_A":
                continue
            t = dt.datetime.fromisoformat(row["available_time"])
            rows.append((t.timestamp(), [float(row[c]) for c in FEAT_COLS]))
    rows.sort(key=lambda x: x[0])
    times = np.array([t for t, _ in rows])
    feats = np.array([v for _, v in rows], dtype=np.float64)   # (N, F)
    # 按时间连续性切段：相邻块间隔 > 90s 视为断点（10s 块，60s 容差）
    cuts = np.where(np.diff(times) > 90.0)[0] + 1
    segs = np.split(feats, cuts)
    segs = [s for s in segs if len(s) >= WINDOW]
    return times, feats, segs


def build_windows(segs):
    xs = []
    for s in segs:
        for i in range(0, len(s) - WINDOW + 1, STRIDE):
            xs.append(s[i:i + WINDOW])
    return np.stack(xs).astype(np.float32)   # (W, T, F)


def make_mask(B, T, ratio=MASK_RATIO, rng=None):
    # 80% 单点掩码 + 20% 连续块掩码（模拟"预测一段未来"）
    n_mask = max(1, int(T * ratio))
    mask = np.zeros((B, T), dtype=bool)
    for b in range(B):
        if rng.random() < 0.8:
            idx = rng.choice(T, size=n_mask, replace=False)
            mask[b, idx] = True
        else:
            start = int(rng.integers(0, T - n_mask + 1))
            mask[b, start:start + n_mask] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    times, feats, segs = load_data()
    logp = np.log1p(feats)
    # 全局 z-score（无标签数据，含全部时段；与报告"全部无标签窗口"一致）
    mu, sd = logp.mean(0), logp.std(0) + 1e-8
    norm = (logp - mu) / sd
    # 按切段边界把标准化后的序列切回各段
    seg_lens = [len(s) for s in segs]
    segs_n = []
    off = 0
    for L in seg_lens:
        segs_n.append(norm[off:off + L])
        off += L

    X = build_windows(segs_n)
    n_w = len(X)
    print(f"[P2] 数据: {len(feats)} 块 / {len(segs)} 段(>=180块) / 窗口 {n_w}", flush=True)

    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskedEncoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P2] device={device} params={n_params}", flush=True)

    Xt = torch.from_numpy(X)
    epochs = 3 if args.smoke else EPOCHS
    N = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            x = Xt[idx].to(device)
            B, T, F = x.shape
            mask = torch.from_numpy(make_mask(B, T, rng=rng)).to(device)
            out = model(x, mask)
            # 只对掩码位置算 MSE
            loss = ((out - x) ** 2 * mask.unsqueeze(-1).float()).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"[P2] epoch {ep + 1}/{epochs}  loss={tot / nb:.5f}", flush=True)

    if not args.smoke:
        torch.save({"model": model.state_dict(),
                    "mu": mu, "sd": sd,
                    "feat_cols": FEAT_COLS,
                    "d_model": D_MODEL, "window": WINDOW,
                    "params": n_params, "n_windows": n_w,
                    "epochs": EPOCHS, "device": device},
                   OUT / "encoder.pt")
        (OUT / "p2_pretrain.json").write_text(json.dumps({
            "status": "done", "blocks": int(len(feats)), "segments": len(segs),
            "windows": n_w, "epochs": EPOCHS, "params": n_params,
            "device": device, "final_loss": tot / nb,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[P2] 保存 encoder -> {OUT / 'encoder.pt'}", flush=True)


if __name__ == "__main__":
    main()
