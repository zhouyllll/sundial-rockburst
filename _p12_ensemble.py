"""P12：Sundial 与通用时序大模型（Chronos-2 / TiRex / Lag-Llama）预测集成对照。

对同一批下游样本，把各源 forecast 特征（dataset.npz 第6列，即 5/10/30min
窗口的预测均值）拼接，并可叠加分歧特征（各源预测的极差 spread），
再用与 P0/P1 完全相同的下游协议（ridge x l1 x link NLL 选参 + Platt 校准 +
阈值选择）训练评估，产出 experiment_12_ensemble/{variant}/* 与 p12_summary.json。

变体：
  sundial_only  仅 Sundial（同协议基线，特征6维）
  ens3          Sundial + TiRex + Lag-Llama（8维）
  ens3_spread   同上 + 分歧极差（11维）
  ens4          Sundial + Chronos-2 + TiRex + Lag-Llama（9维）
  ens4_spread   同上 + 分歧极差（12维）
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"G:\1A岩爆预测\sundial-rockburst")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "experiment_12_ensemble"

SOURCES = {
    "sundial": ROOT / "artifacts" / "experiment_06_auto_thr" / "dataset.npz",
    "chronos2": ROOT / "artifacts" / "experiment_08_p1" / "chronos2" / "dataset.npz",
    "tirex": ROOT / "artifacts" / "experiment_08_p1" / "tirex" / "dataset.npz",
    "lagllama": ROOT / "artifacts" / "experiment_08_p1" / "lagllama" / "dataset.npz",
}


def log(msg):
    print(msg, flush=True)
    with (OUT / "p12_run.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load(tag):
    path = SOURCES[tag]
    if not path.exists():
        raise FileNotFoundError(f"缺少源数据集: {path}")
    with np.load(path, allow_pickle=False) as pack:
        x = pack["x"]
        y, mask = pack["y"], pack["mask"]
        times = pack["times"]
        event_ids = pack["event_ids"]
        event_onsets = (pack["event_onsets"] if "event_onsets" in pack.files
                        else np.full(len(times), np.nan))
        groups = pack["groups"]
        meta = json.loads(str(pack["metadata"]))
    if x.ndim != 3 or x.shape[1] != 3 or x.shape[2] < 6:
        raise ValueError(f"{tag} x 形状异常: {x.shape}")
    return dict(x=x, y=y, mask=mask, times=times, event_ids=event_ids,
                event_onsets=event_onsets, groups=groups, meta=meta)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    packs = {t: load(t) for t in SOURCES}
    ref = packs["sundial"]
    for t, p in packs.items():
        if not np.array_equal(p["times"], ref["times"]):
            raise ValueError(f"{t} 样本时刻与 sundial 不对齐")
        if not np.array_equal(p["y"], ref["y"]) or not np.array_equal(p["mask"], ref["mask"]):
            raise ValueError(f"{t} 标签/掩码与 sundial 不一致")
    log(f"[P12] 四源样本对齐: N={len(ref['times'])}")

    exp06_model = json.loads((ROOT / "artifacts" / "experiment_06_auto_thr" / "run" /
                              "model.json").read_text(encoding="utf-8"))
    from rockburst.io import timestamp
    train_end = timestamp(exp06_model["train_end"])
    validation_end = timestamp(exp06_model["validation_end"])
    log(f"[P12] 时间划分（与 P0/P1 相同）: train_end={exp06_model['train_end']} "
        f"validation_end={exp06_model['validation_end']}")

    y, mask = ref["y"], ref["mask"]
    times, event_ids = ref["times"], ref["event_ids"]
    event_onsets, groups = ref["event_onsets"], ref["groups"]
    base5 = ref["x"][:, :, :5]
    fc = {t: packs[t]["x"][:, :, 5] for t in SOURCES}  # 每源 3 时窗预测特征

    variants = {
        "sundial_only": (["sundial"], None),
        "ens3": (["sundial", "tirex", "lagllama"], None),
        "ens3_spread": (["sundial", "tirex", "lagllama"], "spread"),
        "ens4": (["sundial", "chronos2", "tirex", "lagllama"], None),
        "ens4_spread": (["sundial", "chronos2", "tirex", "lagllama"], "spread"),
    }

    from rockburst.model import train

    summary = {}
    for name, (srcs, spread) in variants.items():
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        parts = [base5]
        names = list(ref["meta"]["feature_names"])[:5]
        for s in srcs:
            parts.append(fc[s][:, :, None])
            names.append(f"forecast_{s}")
        if spread is not None:
            arr = np.stack([fc[s] for s in srcs], axis=-1)  # [N,3,K]
            parts.append((arr.max(-1) - arr.min(-1))[:, :, None])
            names.append(f"forecast_{spread}")
        x = np.concatenate(parts, axis=2)
        meta2 = dict(ref["meta"], feature_names=names)
        meta2["forecast"] = dict(meta2.get("forecast", {}),
                                 ensemble=dict(sources=srcs, spread=spread))
        ds = out / "dataset.npz"
        np.savez_compressed(ds, x=x, y=y, mask=mask, times=times, event_ids=event_ids,
                            event_onsets=event_onsets, groups=groups,
                            metadata=np.str_(json.dumps(meta2)))
        log(f"[P12:{name}] 特征 {x.shape[2]} 维 -> {ds}")

        tr = train(str(ds), str(out), train_end, validation_end,
                   ridges=(0.3, 1., 3., 10., 30.), threshold=0.5,
                   feature_indices=list(range(x.shape[2])), feature_variant=name,
                   event_balanced=True, auto_threshold=True,
                   max_isolated_false_alarms=2, min_validation_recall=0.4,
                   min_active_bins=2, links=("logit", "cloglog"),
                   l1_ratios=(0., 0.5), selection="nll", calibrate=True)
        model = json.loads((out / "model.json").read_text(encoding="utf-8"))
        entry = dict(features=x.shape[2],
                     selected=dict(link=tr["selected_link"], ridge=tr["selected_ridge"],
                                   l1_ratio=tr["selected_l1_ratio"]),
                     threshold=model["threshold"],
                     threshold_selection=tr["threshold_selection"],
                     splits={})
        for split_name in ("validation", "test"):
            h = tr["splits"][split_name]["horizons"]["30min"]
            entry["splits"][split_name] = dict(
                nll=h["log_loss"], maxp=h["max_probability"], ece=h["ece"],
                ap=h["average_precision"], recall=h["event_recall"],
                isolated_fa=h["isolated_false_alarm_episodes"],
                time_frac=h["alarm_time_fraction"],
                eligible=h["eligible_events"], detected=h["detected_events"])
            log(f"[P12:{name}:{split_name}] nll={h['log_loss']:.4f} maxp={h['max_probability']:.4f} "
                f"ece={h['ece']:.4f} ap={h['average_precision']:.4f} recall={h['event_recall']} "
                f"fa={h['isolated_false_alarm_episodes']} time_frac={h['alarm_time_fraction']:.4f}")
        summary[name] = entry
        (OUT / "p12_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[P12:{name}] 完成 -> {out}")

    (OUT / "p12_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[P12] 全部完成。")


if __name__ == "__main__":
    main()
