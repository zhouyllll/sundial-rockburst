"""Read headerless interleaved int32 DAS files without loading an hour at once.

Continuous output is aligned to UTC 10-second boundaries. Filtering uses only
the completed block; its availability is no earlier than all contributing files.
Event boundaries come from the user's external detector, not a new detector.
"""

from bisect import bisect_right
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch

from .io import iso, read_csv, timestamp, write_csv


FEATURES = ["mean_square", "rms", "peak", "energy_proxy", "stalta",
            "coherence", "active_fraction", "low_band_fraction"]
TRANSIENT_RAW_FEATURES = ["short_window_energy_q99", "short_window_overthreshold_count"]
BASE = ["zone_id", "start_time", "end_time", "available_time", "preprocessing_id"]
CONT_COLUMNS = BASE + FEATURES + TRANSIENT_RAW_FEATURES
EVENT_COLUMNS = BASE + ["event_id"] + FEATURES + TRANSIENT_RAW_FEATURES


def channels(spec, count):
    if spec == "all":
        # 数据集内通道数不一致（35/36/41/42pt），统一取交集前35通道，保证特征可比、签名一致。
        return list(range(min(count, 35)))
    result = set()
    for part in spec.split(","):
        ends = [int(v) for v in part.split("-")]
        if len(ends) == 1:
            ends *= 2
        if len(ends) != 2 or not 1 <= ends[0] <= ends[1] <= count:
            raise ValueError(f"无效监测通道: {spec}; 文件通道数={count}")
        result.update(range(ends[0] - 1, ends[1]))
    if not result:
        raise ValueError("监测通道不能为空")
    return sorted(result)


def load_manifest(path, kind, monitor, low, high):
    required = ["file_path", "zone_id", "start_time", "available_time", "fs", "channels"]
    if kind == "events":
        required += ["event_id", "event_start_time", "event_end_time"]
    records, ids = [], set()
    for row in read_csv(path, required):
        p = Path(row["file_path"])
        if not p.is_absolute():
            p = Path(path).resolve().parent / p
        fs, count = int(row["fs"]), int(row["channels"])
        if fs <= 0 or count <= 0 or not 0 < low < high < fs / 2:
            raise ValueError(f"采样率/通道/带通频率无效: {p}")
        size = p.stat().st_size
        if size == 0 or size % (4 * count):
            raise ValueError(f"bin 大小不能按 int32 × {count} 通道整除: {p}")
        start = timestamp(row["start_time"])
        end = start + size / (4 * count * fs)
        available = timestamp(row["available_time"])
        if available + 1e-6 < end:
            raise ValueError(f"完整文件 available_time 早于采集结束: {p}")
        selected = channels(monitor, count)
        config = dict(version=1, fs=fs, channels=len(selected), selected=selected,
                      low=low, high=high, dtype="<i4")
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        r = dict(path=p, zone=row["zone_id"], start=start, end=end,
                 available=available, fs=fs, count=count, selected=selected,
                 signature=signature, low=low, high=high)
        if not r["zone"]:
            raise ValueError("zone_id 不能为空")
        if kind == "events":
            event_id = row["event_id"]
            if not event_id or (r["zone"], event_id) in ids:
                raise ValueError(f"event_id 为空或重复；请先在外部去重: {event_id}")
            ids.add((r["zone"], event_id))
            a, b = timestamp(row["event_start_time"]), timestamp(row["event_end_time"])
            if not start <= a < b <= end + 1e-6:
                raise ValueError(f"事件边界超出 bin: {event_id}")
            r.update(event_id=event_id, event_start=a, event_end=min(b, end))
        records.append(r)
    if not records and kind != "events":
        raise ValueError("manifest 没有数据行")
    return records


def read_segment(record, start, end):
    raw = np.memmap(record["path"], mode="r", dtype="<i4").reshape(-1, record["count"])
    a = round((start - record["start"]) * record["fs"])
    b = round((end - record["start"]) * record["fs"])
    return np.asarray(raw[a:b, record["selected"]], dtype=np.float64)


def waveform_features(data, fs, low=50.0, high=500.0):
    if len(data) < 32 or data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("片段过短或包含非法数值，无法提取特征")
    x = data - data.mean(axis=0, keepdims=True)
    x = sosfiltfilt(butter(4, [low, high], fs=fs, btype="bandpass", output="sos"), x, axis=0)
    power = np.mean(x * x, axis=1)
    mean_square = float(power.mean())
    channel_power = np.mean(x * x, axis=0)
    if mean_square > 0:
        envelope = np.sqrt(power)
        sta = uniform_filter1d(envelope, max(1, round(.05 * fs)))
        lta = uniform_filter1d(envelope, max(1, round(.3 * fs)))
        stalta = float(np.max(sta / np.maximum(lta, 1e-12)))
        active = float(np.mean(channel_power > .5 * channel_power.max()))
    else:
        envelope = np.zeros_like(power)
        stalta, active = 0., 0.
    # Capture sub-second concentration inside each 10-second feature cell.
    # 100 ms windows overlap by 50%, retaining brief energy bursts without
    # storing or exposing raw waveform samples to the classifier.
    window = max(32, round(.1 * fs))
    hop = max(1, round(.05 * fs))
    if len(power) >= window:
        frames = np.lib.stride_tricks.sliding_window_view(power, window)[::hop]
        short_power = np.asarray(frames.mean(axis=1), dtype=np.float64)
    else:
        short_power = np.asarray([mean_square], dtype=np.float64)
    short_median = float(np.median(short_power))
    short_mad = float(np.median(np.abs(short_power - short_median)))
    threshold = short_median + 3. * 1.4826 * short_mad
    overthreshold = int(np.sum(short_power > max(threshold, short_median + 1e-12)))
    short_q99 = float(np.quantile(short_power, .99))
    if x.shape[1] > 1:
        numerator = np.mean(x[:, :-1] * x[:, 1:], axis=0)
        denominator = np.sqrt(channel_power[:-1] * channel_power[1:])
        coherence = float(np.mean(np.abs(numerator / np.maximum(denominator, 1e-12))))
    else:
        coherence = 0.
    freq, psd = welch(x, fs=fs, axis=0, nperseg=min(2048, len(x)))
    psd = psd.mean(axis=1)
    band = (freq >= low) & (freq <= high)
    low_band = band & (freq < 250)
    fraction = float(psd[low_band].sum() / max(float(psd[band].sum()), 1e-12))
    peak = float(np.max(np.abs(x)))
    return dict(mean_square=mean_square, rms=math.sqrt(mean_square),
                peak=peak, energy_proxy=mean_square * len(x) / fs,
                stalta=stalta, coherence=min(coherence, 1.),
                active_fraction=active, low_band_fraction=fraction,
                short_window_energy_q99=short_q99,
                short_window_overthreshold_count=overthreshold)


def _cell_features(arg):
    """进程池 worker：拼接各段波形并计算特征。arg=(segments, fs, low, high)
    段与段之间若存在毫秒级间隙（bin 时钟漂移），以零填充补齐。"""
    segments, fs, low, high = arg
    chunks, prev_end = [], None
    for r, a, b in segments:
        seg = read_segment(r, a, b)
        if prev_end is not None:
            gap = a - prev_end
            if gap > 1e-6:
                chunks.append(np.zeros((int(round(gap * fs)), seg.shape[1]), dtype=seg.dtype))
        chunks.append(seg)
        prev_end = b
    data = np.concatenate(chunks, axis=0)
    target = int(10 * fs)
    if len(data) < target:
        data = np.vstack([data, np.zeros((target - len(data), data.shape[1]), dtype=data.dtype)])
    if abs(len(data) - target) > len(segments):
        raise ValueError("跨文件采样点数异常")
    return waveform_features(data, fs, low, high)

def extract(manifest, output, kind, monitor="all", low=50., high=500.):
    records = load_manifest(manifest, kind, monitor, low, high)
    count = 0

    def event_rows():
        nonlocal count
        for r in records:
            a, b = r["event_start"], r["event_end"]
            f = waveform_features(read_segment(r, a, b), r["fs"], low, high)
            count += 1
            yield dict(zone_id=r["zone"], start_time=iso(a), end_time=iso(b),
                       available_time=iso(r["available"]), preprocessing_id=r["signature"],
                       event_id=r["event_id"], **f)



    def continuous_rows():
        nonlocal count
        GAP_TOL = 0.2  # bin 之间毫秒级时钟漂移容差（秒）
        jobs = []  # (zone, a, b, available, signature, segments, fs, low, high)
        for zone in sorted({r["zone"] for r in records}):
            group = sorted((r for r in records if r["zone"] == zone), key=lambda r: r["start"])
            starts = [r["start"] for r in group]
            # Iterate occupied grid cells, not long gaps between recordings.
            last_cell = None
            for record in group:
                for cell in range(math.floor(record["start"] / 10), math.ceil(record["end"] / 10)):
                    if last_cell is not None and cell <= last_cell:
                        continue
                    last_cell = cell
                    a, b = cell * 10., cell * 10. + 10.
                    idx = bisect_right(starts, a + 1e-6) - 1
                    if idx < 0:
                        continue
                    position, segments, sources = a, [], []
                    while idx < len(group) and position < b - 1e-6:
                        r = group[idx]
                        if r["end"] <= position:
                            idx += 1
                            continue
                        if r["start"] > position + GAP_TOL:
                            break  # 真实长间隙，本 cell 无法覆盖
                        start_off = max(position, r["start"])
                        end = min(b, r["end"])
                        sources.append(r)
                        segments.append((r, start_off, end))
                        position = end
                        idx += 1
                    if b - position > GAP_TOL:
                        continue
                    if len({r["signature"] for r in sources}) != 1:
                        continue  # A block crossing a device configuration change is missing.
                    r = sources[0]
                    avail = max(max(s["available"] for s in sources), b)
                    jobs.append((zone, a, b, avail,
                                 r["signature"], segments, r["fs"], low, high))
        # 多进程并行计算特征，按时间顺序输出（executor.map 保序）
        import os
        from concurrent.futures import ProcessPoolExecutor
        workers = max(1, min(int(os.environ.get("ROCKBURST_WORKERS", "12")), os.cpu_count() or 4))
        args = [(segments, fs, low, high) for _, _, _, _, _, segments, fs, low, high in jobs]
        metas = [(zone, a, b, available, signature)
                 for zone, a, b, available, signature, _, _, _, _ in jobs]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for meta, f in zip(metas, executor.map(_cell_features, args)):
                zone, a, b, available, signature = meta
                count += 1
                yield dict(zone_id=zone, start_time=iso(a), end_time=iso(b),
                           available_time=iso(available), preprocessing_id=signature, **f)

    write_csv(output, event_rows() if kind == "events" else continuous_rows(),
              EVENT_COLUMNS if kind == "events" else CONT_COLUMNS)
    if not count and kind != "events":
        raise ValueError("没有可用特征，检查记录长度、连续性及事件边界")
    return {"kind": kind, "rows": count, "output": str(output)}
