"""P2：预训练表征提取 + 特征装配。

对每个下游样本（dataset.times 的预测时刻 t），取 t 前 30min（<=t 且 >t-30min）
的 8 维连续特征序列（log1p + encoder 的 z-score 统计），过 MaskedEncoder
mean-pool 得 128 维表征；在训练集上 PCA→8 维并 z-score 标准化；
拼接进下游 x 的前 6 列（Sundial/lagllama 特征）生成 P2 数据集。

用法: python _p2_extract.py
"""
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, str(ROOT))
from _p2_pretrain import MaskedEncoder, FEAT_COLS, WINDOW  # noqa: E402

EXP06 = ROOT / "artifacts" / "experiment_06_auto_thr"
P1 = ROOT / "artifacts" / "experiment_08_p1"
P2 = ROOT / "artifacts" / "experiment_09_p2"
CSV = EXP06 / "continuous.csv"
ENC = P2 / "encoder.pt"
N_PC = 8


def load_rows():
    rows = []
    with open(CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["zone_id"] != "zone_A":
                continue
            t = dt.datetime.fromisoformat(row["available_time"]).timestamp()
            rows.append((t, [float(row[c]) for c in FEAT_COLS]))
    rows.sort(key=lambda x: x[0])
    return np.array([t for t, _ in rows]), np.array([v for _, v in rows], dtype=np.float64)


def extract_repr(model, times, feats, sample_times, mu, sd, device):
    """对每个样本预测时刻取前 30min 窗口，mean-pool 出表征。"""
    T_hist = 1800.0
    model.eval()
    out = np.zeros((len(sample_times), model.proj.out_features), dtype=np.float32)
    # 样本时刻升序；窗口滑动
    lo = 0
    with torch.no_grad():
        for i, t in enumerate(sample_times):
            while lo < len(times) and times[lo] <= t - T_hist:
                lo += 1
            hi = lo
            while hi < len(times) and times[hi] <= t:
                hi += 1
            seg = feats[lo:hi]
            if len(seg) == 0:
                out[i] = 0.0
                continue
            lp = np.log1p(seg)
            nrm = (lp - mu) / sd
            n = len(nrm)
            if n < WINDOW:
                pad = WINDOW - n
                nrm = np.concatenate([nrm, np.repeat(nrm[-1:], pad, axis=0)], axis=0)
            elif n > WINDOW:
                nrm = nrm[-WINDOW:]
            x = torch.from_numpy(nrm[None].astype(np.float32)).to(device)
            z = model.encoder(model.pos(model.proj(x)))
            out[i] = z.mean(1).cpu().numpy()[0]
    return out


def pca_fit_transform(Z, train_mask, n_pc=N_PC):
    Zc = Z[train_mask]
    mean = Zc.mean(0)
    Zc0 = Zc - mean
    U, S, Vt = np.linalg.svd(Zc0, full_matrices=False)
    W = Vt[:n_pc]                     # (n_pc, d)
    proj = (Z - mean) @ W.T           # (N, n_pc)
    mu = proj[train_mask].mean(0)
    sd = proj[train_mask].std(0) + 1e-8
    return (proj - mu) / sd, dict(mean=mean.tolist(), W=W.tolist(),
                                  z_mu=mu.tolist(), z_sd=sd.tolist())


def main():
    ck = torch.load(ENC, map_location="cpu", weights_only=False)
    mu, sd = ck["mu"], ck["sd"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskedEncoder().to(device)
    model.load_state_dict(ck["model"])
    print(f"[P2-extract] encoder 加载完成 device={device}", flush=True)

    times, feats = load_rows()
    print(f"[P2-extract] continuous {len(times)} 块", flush=True)

    exp06_model = json.loads((EXP06 / "run" / "model.json").read_text(encoding="utf-8"))
    from rockburst.io import timestamp
    train_end = timestamp(exp06_model["train_end"])
    validation_end = timestamp(exp06_model["validation_end"])
    print(f"[P2-extract] 划分 train_end={exp06_model['train_end']} "
          f"validation_end={exp06_model['validation_end']}", flush=True)

    for tag, src in (("sundial", EXP06), ("lagllama", P1 / "lagllama")):
        out_dir = P2 / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = out_dir / "dataset.npz"
        if ds.exists() and (out_dir / "repr_pca.npy").exists():
            print(f"[P2-extract:{tag}] 已有产物，跳过 -> {out_dir}", flush=True)
            continue
        with np.load(src / "dataset.npz", allow_pickle=True) as pack:
            x0 = pack["x"]
            y, mask = pack["y"], pack["mask"]
            times_s, event_ids = pack["times"], pack["event_ids"]
            event_onsets = (pack["event_onsets"] if "event_onsets" in pack.files
                            else np.full(len(times_s), np.nan))
            groups = pack["groups"]
            metadata = json.loads(str(pack["metadata"]))
        print(f"[P2-extract:{tag}] 源样本 {len(times_s)}", flush=True)

        repr_128 = extract_repr(model, times, feats, times_s, mu, sd, device)
        train_mask = times_s < train_end
        repr_z, pca_scale = pca_fit_transform(repr_128, train_mask)
        np.save(out_dir / "repr_128.npy", repr_128)
        np.save(out_dir / "repr_pca.npy", repr_z)
        (out_dir / "pca_scale.json").write_text(json.dumps(pca_scale, ensure_ascii=False),
                                                encoding="utf-8")

        x_new = np.concatenate([x0[:, :, :6],
                                np.repeat(repr_z[:, None, :], 3, axis=1)], axis=2)
        names = list(metadata["feature_names"])[:6] + [f"repr_pc{i + 1}" for i in range(N_PC)]
        meta2 = dict(metadata, feature_names=names)
        meta2["forecast"] = dict(metadata.get("forecast", {}), p2_representation="masked-encoder-v1")
        np.savez_compressed(ds, x=x_new, y=y, mask=mask, times=times_s,
                            event_ids=event_ids, event_onsets=event_onsets,
                            groups=groups, metadata=np.str_(json.dumps(meta2)))
        print(f"[P2-extract:{tag}] 数据集 -> {ds} 特征 {len(names)} 维", flush=True)

    print("[P2-extract] 完成。", flush=True)


if __name__ == "__main__":
    main()
