"""Availability-aware dual-branch samples and conditional interval labels."""

from collections import Counter
from bisect import bisect_left, bisect_right
import json
import math

import numpy as np

from .extract import BASE, FEATURES
from .io import read_csv, timestamp, iso, write_json


FEATURE_NAMES = ["short_log_power_change", "short_coherence", "log_event_count_6h",
                 "log_event_energy_24h", "log_minutes_since_event",
                 "forecast_log_power_change"]
TRANSIENT_FEATURE_NAMES = ["recent_peak_log", "recent_crest_factor", "recent_stalta_log",
                           "recent_short_energy_q99_log", "recent_overthreshold_count_log"]
FEATURE_GROUPS = {
    "baseline": [],
    "peak_crest": ["recent_peak_log", "recent_crest_factor"],
    "stalta": ["recent_stalta_log"],
    "energy_q99": ["recent_short_energy_q99_log"],
    "overthreshold_count": ["recent_overthreshold_count_log"],
    "all_transient": TRANSIENT_FEATURE_NAMES,
}
ALL_FEATURE_NAMES = FEATURE_NAMES + TRANSIENT_FEATURE_NAMES
HORIZONS = np.array([300., 600., 1800.])


def select_model_features(x, feature_names):
    """Slice a full baseline+transient sample in the model's saved feature order."""
    feature_names = list(feature_names or FEATURE_NAMES)
    if len(set(feature_names)) != len(feature_names) or any(name not in ALL_FEATURE_NAMES for name in feature_names):
        raise ValueError("模型包含未知或重复的特征名")
    values = np.asarray(x)
    if values.ndim != 3 or values.shape[-1] != len(ALL_FEATURE_NAMES):
        raise ValueError("全量预测特征应为[N,3,11]")
    indices = [ALL_FEATURE_NAMES.index(name) for name in feature_names]
    return values[:, :, indices]


class Unavailable(ValueError):
    """An expected missing-data condition; never silently replace it with zero risk."""


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class Inputs:
    @classmethod
    def from_stream(cls, continuous_rows, event_rows, zone, now, microseismic_enabled=True):
        """Build an inference view of the bounded buffers; no future labels needed."""
        obj = cls.__new__(cls)
        obj.zone = zone
        obj.microseismic_enabled = microseismic_enabled
        signatures = {r["preprocessing_id"] for r in continuous_rows}
        if len(signatures) != 1:
            raise Unavailable("continuous_history_incomplete")
        obj.signature = next(iter(signatures))
        obj.blocks = {round(timestamp(r["end_time"])): cls._feature(r) for r in continuous_rows}
        obj.transient_features_available = all(r.get("_transient_present", False) for r in obj.blocks.values())
        if any(r["preprocessing_id"] != obj.signature for r in event_rows):
            raise ValueError("微震与连续特征的采集配置不一致")
        obj.events = sorted((cls._feature(r) for r in event_rows), key=lambda r: r["start"])
        obj.event_starts = [r["start"] for r in obj.events]
        obj.ends = sorted(obj.blocks)
        obj.coverage = {"event_archive_complete": [(now - 86400, now)] if microseismic_enabled else [],
                        "rockburst_record_complete": []}
        obj.bursts, obj.onsets = [], []
        return obj

    def __init__(self, continuous, events, coverage, zone, rockbursts=None,
                 microseismic_enabled=True):
        self.zone = zone
        self.microseismic_enabled = microseismic_enabled
        self.blocks = {}
        signatures = set()
        for row in read_csv(continuous, BASE + FEATURES):
            if row["zone_id"] != zone:
                continue
            r = self._feature(row)
            if abs(r["end"] - r["start"] - 10) > 1e-5 or abs(r["end"] / 10 - round(r["end"] / 10)) > 1e-5:
                raise ValueError("连续特征必须是 UTC 网格上完整的10秒块")
            key = round(r["end"])
            if key in self.blocks:
                raise ValueError("连续特征有重复时间块")
            signatures.add(row["preprocessing_id"])
            self.blocks[key] = r
        if not self.blocks or len(signatures) != 1:
            raise ValueError("所选区域无连续数据或包含多种采集配置；请先按一致物理通道配置分组")
        self.signature = next(iter(signatures))
        self.transient_features_available = all(r.get("_transient_present", False) for r in self.blocks.values())
        self.ends = sorted(self.blocks)
        self.events, seen = [], set()
        for row in read_csv(events, BASE + ["event_id"] + FEATURES):
            if row["zone_id"] != zone:
                continue
            if row["event_id"] in seen or not row["event_id"]:
                raise ValueError("微震 event_id 为空或重复")
            seen.add(row["event_id"])
            r = self._feature(row)
            if row["preprocessing_id"] != self.signature:
                raise ValueError("微震与连续特征的采集/通道/滤波配置不一致；不能直接混用能量代理量")
            self.events.append(r)
        self.events.sort(key=lambda r: r["start"])
        self.event_starts = [r["start"] for r in self.events]
        self.coverage = {}
        rows = read_csv(coverage, ["zone_id", "start_time", "end_time",
                                   "event_archive_complete", "rockburst_record_complete"])
        zone_intervals = sorted((timestamp(r["start_time"]), timestamp(r["end_time"]))
                                for r in rows if r["zone_id"] == zone)
        if any(right[0] < left[1] for left, right in zip(zone_intervals, zone_intervals[1:])):
            raise ValueError("coverage 记录重叠；请拆成不重叠区间以免掩盖缺失")
        for flag in ["event_archive_complete", "rockburst_record_complete"]:
            intervals = []
            for row in rows:
                if row["zone_id"] != zone:
                    continue
                a, b = timestamp(row["start_time"]), timestamp(row["end_time"])
                if a >= b or row[flag] not in {"0", "1"}:
                    raise ValueError("coverage 时间区间或 0/1 标记无效")
                if row[flag] == "1":
                    intervals.append((a, b))
            self.coverage[flag] = merge_intervals(intervals)
        self.bursts = []
        if rockbursts:
            seen = set()
            for row in read_csv(rockbursts, ["event_id", "zone_id", "onset_time", "end_time", "group_id"]):
                if row["zone_id"] != zone:
                    continue
                a, b = timestamp(row["onset_time"]), timestamp(row["end_time"])
                if a > b or not row["event_id"] or row["event_id"] in seen or not row["group_id"]:
                    raise ValueError("岩爆事件时间、event_id 或 group_id 无效")
                seen.add(row["event_id"])
                self.bursts.append(dict(onset=a, end=b, event_id=row["event_id"], group_id=row["group_id"]))
            self.bursts.sort(key=lambda r: r["onset"])
        self.onsets = [r["onset"] for r in self.bursts]

    @staticmethod
    def _feature(row):
        r = {key: float(row[key]) for key in FEATURES}
        transient_fields = ["short_window_energy_q99", "short_window_overthreshold_count"]
        present = all(row.get(key) not in (None, "") for key in transient_fields)
        for key in transient_fields:
            r[key] = float(row[key]) if row.get(key) not in (None, "") else 0.
        r["_transient_present"] = present
        if not all(math.isfinite(v) and v >= 0 for k, v in r.items() if not k.startswith("_")):
            raise ValueError("特征中有负数、NaN 或无穷值")
        if any(r[k] > 1 + 1e-6 for k in ["coherence", "active_fraction", "low_band_fraction"]):
            raise ValueError("比例特征超出[0,1]")
        r.update(start=timestamp(row["start_time"]), end=timestamp(row["end_time"]),
                 available=timestamp(row["available_time"]))
        if r["start"] >= r["end"] or r["available"] + 1e-6 < r["end"]:
            raise ValueError("特征结束/可用时间无效")
        return r

    def covered(self, start, end, flag):
        return any(a <= start + 1e-6 and b >= end - 1e-6 for a, b in self.coverage[flag])

    def sample(self, now, forecaster, history_minutes=60, max_lag_seconds=60,
               include_transient=False):
        if history_minutes not in {30, 60}:
            raise ValueError("基础版 history_minutes 只支持30或60")
        if not 0 <= max_lag_seconds <= 60:
            raise ValueError("基础版最多允许60秒输入延迟")
        if self.microseismic_enabled and not self.covered(now - 86400, now, "event_archive_complete"):
            raise Unavailable("event_history_incomplete")
        cutoff = None
        latest = math.floor(now / 10) * 10
        # File completion can lag the last waveform block by up to max_lag_seconds.
        # Historical observations stay entirely in [now-history, now].
        for end in range(latest, math.ceil((now - max_lag_seconds) / 10) * 10 - 1, -10):
            block = self.blocks.get(end)
            if block and block["available"] <= now:
                cutoff = end
                break
        if cutoff is None:
            raise Unavailable("continuous_too_old_or_unavailable")
        first = math.ceil((now - history_minutes * 60) / 10) * 10 + 10
        blocks = [self.blocks.get(end) for end in range(first, cutoff + 1, 10)]
        if not blocks or any(r is None or r["available"] > now for r in blocks):
            raise Unavailable("continuous_history_incomplete")
        if len(blocks) < history_minutes * 6 - math.ceil(max_lag_seconds / 10) - 1:
            raise Unavailable("continuous_history_too_short")
        power = np.array([r["mean_square"] for r in blocks], dtype=np.float64)
        coherence = np.array([r["coherence"] for r in blocks])
        left = bisect_left(self.event_starts, now - 86400)
        right = bisect_right(self.event_starts, now)
        past = [r for r in self.events[left:right] if r["end"] <= now and r["available"] <= now]
        count = sum(r["start"] > now - 21600 for r in past)
        energy = sum(r["energy_proxy"] for r in past)
        gap = min(1440., (now - max(r["start"] for r in past)) / 60) if past else 1440.
        short_change = np.log1p(power[-30:].mean()) - np.log1p(power[:-30].mean())
        z = np.array([short_change, coherence[-30:].mean(), np.log1p(count),
                      np.log1p(energy), np.log1p(gap), 0.])
        # Forecast begins at cutoff; skip already elapsed steps before summarizing.
        lag_steps = math.ceil((now - cutoff) / 10)
        paths = forecaster.predict(np.log1p(power), horizon=180 + lag_steps)
        paths = paths[:, lag_steps:lag_steps + 180]
        x = np.tile(z, (3, 1))
        for j, (a, b) in enumerate([(0, 30), (30, 60), (60, 180)]):
            x[j, len(FEATURE_NAMES) - 1] = float(np.median(paths[:, a:b].mean(axis=1))) - np.log1p(power[-30:].mean())
        if include_transient:
            if not self.transient_features_available:
                raise Unavailable("transient_features_missing; 请重新从原始bin提取特征")
            recent = blocks[-30:]
            peaks = np.asarray([r["peak"] for r in recent], dtype=np.float64)
            rms = np.asarray([r["rms"] for r in recent], dtype=np.float64)
            stalta = np.asarray([r["stalta"] for r in recent], dtype=np.float64)
            q95 = np.asarray([r["short_window_energy_q99"] for r in recent], dtype=np.float64)
            counts = np.asarray([r["short_window_overthreshold_count"] for r in recent], dtype=np.float64)
            transient = np.asarray([
                np.log1p(float(np.max(peaks))),
                np.log1p(float(np.mean(peaks / np.maximum(rms, 1e-12)))),
                np.log1p(float(np.max(stalta))),
                np.log1p(float(np.quantile(q95, .95))),
                np.log1p(float(np.sum(counts))),
            ])
            x = np.concatenate([x, np.tile(transient, (3, 1))], axis=1)
        if not np.isfinite(x).all():
            raise ValueError("融合特征包含非法值")
        return x, {"continuous_cutoff": iso(cutoff), "continuous_lag_seconds": now - cutoff,
                   "continuous_blocks": len(blocks), "microseismic_count_24h": len(past)}

    def label(self, now):
        if any(r["onset"] <= now <= r["end"] for r in self.bursts):
            raise Unavailable("rockburst_in_progress")
        idx = bisect_right(self.onsets, now)
        y, mask = np.zeros(3), np.ones(3, dtype=bool)
        event_id, group_id = "", ""
        if idx < len(self.bursts):
            burst = self.bursts[idx]
            delay = burst["onset"] - now
            if delay <= 1800:
                # A known first event determines all cumulative targets, even
                # when the saved waveform stops at that event.
                if not self.covered(now, burst["onset"], "rockburst_record_complete"):
                    raise Unavailable("future_labels_incomplete")
                j = int(np.searchsorted(HORIZONS, delay, side="left"))
                y[j] = 1
                mask[j + 1:] = False
                event_id, group_id = burst["event_id"], burst["group_id"]
                return y, mask, event_id, group_id
        if not self.covered(now, now + 1800, "rockburst_record_complete"):
            raise Unavailable("future_labels_incomplete")
        return y, mask, event_id, group_id


def build_dataset(inputs, forecaster, start, end, output, history_minutes=60, max_lag_seconds=60,
                  prediction_times=None, source_groups=None, include_transient=False):
    if start >= end:
        raise ValueError("start 必须早于 end")
    if prediction_times is None and (start % 60 or end % 60):
        raise ValueError("start/end 必须递增且对齐整分钟；end为不包含的右端点")
    by_file = prediction_times is not None
    schedule = (sorted({float(t) for t in prediction_times if start <= t < end}) if by_file
                else list(range(round(start), round(end), 60)))
    xs, ys, masks, times, event_ids, groups = [], [], [], [], [], []
    skipped = Counter()
    for i, now in enumerate(schedule):
        try:
            y, mask, event_id, group = inputs.label(now)
            if include_transient:
                x, _ = inputs.sample(now, forecaster, history_minutes, max_lag_seconds,
                                     include_transient=True)
            else:
                x, _ = inputs.sample(now, forecaster, history_minutes, max_lag_seconds)
        except Unavailable as exc:
            skipped[str(exc)] += 1
            continue
        source_matches = [group_id for a, b, group_id in (source_groups or [])
                          if a - 1e-6 <= now <= b + 1e-6]
        if len(source_matches) > 1:
            raise ValueError(f"预测时刻{iso(now)}同时归属多个连续记录分组")
        if source_groups and not source_matches:
            raise ValueError(f"预测时刻{iso(now)}未匹配到连续记录分组")
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        times.append(now)
        event_ids.append(event_id)
        groups.append(source_matches[0] if source_matches else group)
        if (i + 1) % 100 == 0:
            print(f"已扫描 {i + 1} 个预测时刻，有效 {len(xs)}", flush=True)
    if not xs:
        raise ValueError(f"没有有效样本: {dict(skipped)}")
    feature_names = FEATURE_NAMES + (TRANSIENT_FEATURE_NAMES if include_transient else [])
    metadata = dict(schema=1, feature_names=feature_names, zone_id=inputs.zone,
                    preprocessing_id=inputs.signature, history_minutes=history_minutes,
                    max_lag_seconds=max_lag_seconds, forecast=forecaster.identity,
                    microseismic_enabled=getattr(inputs, "microseismic_enabled", True),
                    start=iso(start), end=iso(end), sample_step="bin_file" if by_file else "minute",
                    grouping_unit="recording_folder_component" if source_groups else "event_group_or_time",
                    refresh_seconds=float(np.median(np.diff(schedule))) if len(schedule) > 1 else 30.)
    from pathlib import Path
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, x=np.stack(xs), y=np.stack(ys), mask=np.stack(masks),
                            times=np.array(times, dtype=np.float64), event_ids=np.array(event_ids),
                            groups=np.array(groups), metadata=json.dumps(metadata))
    report = dict(samples=len(xs), skipped=dict(skipped),
                  independent_events=len(set(event_ids) - {""}),
                  independent_groups=len(set(groups) - {""}), metadata=metadata)
    write_json(str(path) + ".report.json", report)
    return report
