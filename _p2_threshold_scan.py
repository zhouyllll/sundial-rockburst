# -*- coding: utf-8 -*-
"""P2 三源 30min 阈值工作点扫描。

只对已冻结的拟合+校准重扫决策阈值（不重训、不重拟合校准器）：
在验证集上按「预警语义」的多种准则选工作点，冻结后报测试集指标，
与当前 0.5 回退值对比，回答"阈值 0.5 如何调整"。

准则：
  R60   召回 >= 0.6 下孤立误报段最少（并列取召回更高）
  R80   召回 >= 0.8 下孤立误报段最少
  F1    验证集事件级 F1 最大（precision = 检出事件/报警段）
  cost5 最小化 5 x 漏检事件 + 误报段数（漏报代价 5 倍于误报）
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rockburst.model import probabilities, threshold_sweep

REPO = Path(r"G:\1A岩爆预测\sundial-rockburst")
EXP9 = REPO / "artifacts" / "experiment_09_p2"
TAGS = ("sundial", "lagllama", "tirex")
CRITERIA = ("base05", "R60", "R80", "F1", "cost5")
GRID = np.arange(0.02, 0.99, 0.01)  # 98 个候选阈值


def timestamp(s):
    return np.datetime64(s).astype("datetime64[s]").astype(float)


def pick(rows, crit):
    if not rows:
        return None
    if crit == "R60":
        cand = [r for r in rows if r["event_recall"] is not None and r["event_recall"] >= 0.6]
        return min(cand, key=lambda r: (r["isolated_false_alarm_episodes"], -r["event_recall"])) if cand else None
    if crit == "R80":
        cand = [r for r in rows if r["event_recall"] is not None and r["event_recall"] >= 0.8]
        return min(cand, key=lambda r: (r["isolated_false_alarm_episodes"], -r["event_recall"])) if cand else None
    if crit == "F1":
        def f1(r):
            alm, elig = r["alarm_episodes"], r["eligible_events"]
            if not alm or not elig or r["event_recall"] is None:
                return -1.0
            prec = r["detected_events"] / alm
            rec = r["event_recall"]
            return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else -1.0
        return max(rows, key=f1)
    if crit == "cost5":
        def cost(r):
            miss = r["eligible_events"] - r["detected_events"] if r["eligible_events"] else 0
            return 5 * miss + r["false_alarm_episodes"]
        return min(rows, key=cost)
    if crit == "base05":  # 对照：当前回退值 0.5
        return min(rows, key=lambda r: abs(r["threshold"] - 0.5))
    raise ValueError(crit)


def report(mask, target, scores, times, event_ids, refresh, event_onsets, t):
    r = threshold_sweep(target[mask], scores[mask], times[mask], event_ids[mask], refresh,
                        thresholds=[t], horizon_seconds=1800.,
                        event_onsets=event_onsets[mask], min_active_bins=2)[0]
    return dict(threshold=float(t), event_recall=r["event_recall"],
                detected=r["detected_events"], eligible=r["eligible_events"],
                alarm_episodes=r["alarm_episodes"], false_alarm_episodes=r["false_alarm_episodes"],
                isolated_false_alarm_episodes=r["isolated_false_alarm_episodes"],
                alarm_time_fraction=r["alarm_time_fraction"])


def main():
    out = {}
    for tag in TAGS:
        d = EXP9 / tag
        model = json.loads((d / "model.json").read_text(encoding="utf-8"))
        with np.load(d / "dataset.npz", allow_pickle=False) as pack:
            x, y, mask, times = pack["x"], pack["y"], pack["mask"], pack["times"]
            event_ids = pack["event_ids"]
            event_onsets = (pack["event_onsets"] if "event_onsets" in pack.files
                            else np.full(len(times), np.nan))
            metadata = json.loads(str(pack["metadata"]))
        refresh = metadata.get("refresh_seconds", 60.)
        t_end, v_end = timestamp(model["train_end"]), timestamp(model["validation_end"])
        va = (times >= t_end) & (times + 1800 <= v_end)
        te = times >= v_end
        p = probabilities(x, model)
        target = np.cumsum(y, axis=1) > 0
        target, scores = target[:, 2], p[:, 2]
        rows = threshold_sweep(target[va], scores[va], times[va], event_ids[va], refresh,
                               thresholds=GRID, horizon_seconds=1800.,
                               event_onsets=event_onsets[va], min_active_bins=2)
        tag_out = {"validation_events": int(rows[0]["eligible_events"]) if rows else None}
        print(f"\n=== {tag} (验证 {tag_out['validation_events']} 可检事件) ===")
        for crit in CRITERIA:
            chosen = pick(rows, crit)
            if chosen is None:
                print(f"  {crit}: 无可行工作点")
                tag_out[crit] = None
                continue
            t = chosen["threshold"]
            val = report(va, target, scores, times, event_ids, refresh, event_onsets, t)
            tes = report(te, target, scores, times, event_ids, refresh, event_onsets, t)
            tag_out[crit] = dict(validation=val, test=tes)
            print(f"  {crit}: t={t:.2f} | 验证 recall={val['event_recall']:.2f} "
                  f"isolFA={val['isolated_false_alarm_episodes']} FA={val['false_alarm_episodes']} "
                  f"time={val['alarm_time_fraction']:.3f} | 测试 recall={tes['event_recall']:.2f} "
                  f"isolFA={tes['isolated_false_alarm_episodes']} FA={tes['false_alarm_episodes']} "
                  f"time={tes['alarm_time_fraction']:.3f}")
        out[tag] = tag_out
    (EXP9 / "p2_threshold_scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print("\n[p2-scan] 完成 -> artifacts/experiment_09_p2/p2_threshold_scan.json")


if __name__ == "__main__":
    main()
