"""File-time streaming replay with 30-minute waveform and 24-hour event buffers."""

from collections import Counter
import csv
import json
from pathlib import Path

from .data import (FEATURE_NAMES, Inputs, TRANSIENT_FEATURE_NAMES, Unavailable,
                   merge_intervals, select_model_features)
from .extract import CONT_COLUMNS, extract, load_manifest, read_segment, waveform_features
from .forecast import Forecaster
from .io import iso, timestamp, read_csv, read_json, write_csv, write_json
from .model import probabilities
from .workflow import scan_bins, scan_optional_bins, MANIFEST_COLUMNS, EVENT_MANIFEST_COLUMNS


def replay_stream(continuous_dir, microseismic_dir, model_file, output, model_path=None,
                  device="cpu", timezone="Asia/Shanghai", monitor="all", quiet=False):
    model = read_json(model_file)
    meta = model["metadata"]
    if meta["history_minutes"] != 30:
        raise ValueError("stream 使用30分钟窗口，请先用 --history-minutes 30 训练模型")
    feature_names = meta.get("feature_names", FEATURE_NAMES)
    use_transient = any(name in TRANSIENT_FEATURE_NAMES for name in feature_names)
    zone = meta["zone_id"]
    forecast = Forecaster(meta["forecast"]["backend"], model_path, device,
                          meta["forecast"]["samples"])
    if forecast.identity != meta["forecast"]:
        raise ValueError("Sundial权重或运行环境与训练时不一致")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    continuous = scan_bins(continuous_dir, zone, timezone)
    microseismic_enabled = bool(meta.get("microseismic_enabled", True))
    if microseismic_enabled != bool(microseismic_dir):
        raise ValueError("流式预测的微震目录配置必须与训练模型一致（训练时有目录就提供，没有就省略）")
    micro = scan_optional_bins(microseismic_dir, zone, timezone, event_clips=True)
    if not continuous:
        raise ValueError("连续bin目录为空")
    write_csv(root / "microseismic_manifest.csv", micro, EVENT_MANIFEST_COLUMNS)
    event_records = sorted(load_manifest(root / "microseismic_manifest.csv", "events", monitor, 50., 500.),
                           key=lambda r: r["available"])
    continuous.sort(key=lambda r: timestamp(r["available_time"]))
    sessions = merge_intervals([(timestamp(r["start_time"]), timestamp(r["available_time"]))
                                for r in continuous])
    # One prediction per completed bin, preserving sub-second filename timestamps.
    ticks = sorted({timestamp(r["available_time"]) for r in continuous
                    if any(a + 1800 <= timestamp(r["available_time"]) <= b for a, b in sessions)})
    buffer, events = {}, []
    ci = ei = emitted = valid = max_blocks = max_events = 0
    previous = None
    skipped = Counter()
    columns = ["time", "status", "p_5min", "p_10min", "p_30min", "reason"]
    with (root / "predictions.jsonl").open("w", encoding="utf-8") as json_file, \
            (root / "predictions.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()
        for now in ticks:
            # Read only files that would have completed by this prediction time.
            while ci < len(continuous) and timestamp(continuous[ci]["available_time"]) <= now:
                current = continuous[ci]
                recent = [current]
                if previous and timestamp(previous["available_time"]) >= timestamp(current["start_time"]) - 1e-6:
                    recent.insert(0, previous)
                write_csv(root / "current_manifest.csv", recent, MANIFEST_COLUMNS)
                extract(root / "current_manifest.csv", root / "current_features.csv", "continuous", monitor)
                for row in read_csv(root / "current_features.csv", CONT_COLUMNS):
                    if row["preprocessing_id"] != meta["preprocessing_id"]:
                        raise ValueError("流式波形预处理配置与训练时不一致")
                    if timestamp(row["end_time"]) > now - 1800:
                        buffer[round(timestamp(row["end_time"]))] = row
                previous = current
                ci += 1
            while ei < len(event_records) and event_records[ei]["available"] <= now:
                r = event_records[ei]
                if r["event_start"] >= now - 86400:
                    a, b = r["event_start"], r["event_end"]
                    features = waveform_features(read_segment(r, a, b), r["fs"], r["low"], r["high"])
                    events.append(dict(zone_id=zone, event_id=r["event_id"], start_time=iso(a),
                                       end_time=iso(b), available_time=iso(r["available"]),
                                       preprocessing_id=r["signature"], **features))
                ei += 1
            buffer = {end: row for end, row in buffer.items() if end > now - 1800}
            events = [row for row in events if timestamp(row["start_time"]) >= now - 86400]
            max_blocks, max_events = max(max_blocks, len(buffer)), max(max_events, len(events))
            result = dict(time=iso(now), zone_id=zone, status="ok", backend=meta["forecast"]["backend"],
                          experimental=True, calibration=model["calibration"])
            try:
                view = Inputs.from_stream(list(buffer.values()), events, zone, now, microseismic_enabled)
                x, quality = view.sample(now, forecast, 30, meta["max_lag_seconds"],
                                         include_transient=use_transient)
                if use_transient:
                    x = select_model_features(x[None], feature_names)[0]
                p = probabilities(x[None], model)[0]
                result.update(p_5min=float(p[0]), p_10min=float(p[1]), p_30min=float(p[2]), quality=quality)
                valid += 1
            except Unavailable as exc:
                result.update(status="insufficient_data", reason=str(exc), p_5min=None, p_10min=None, p_30min=None)
                skipped[str(exc)] += 1
            line = json.dumps(result, ensure_ascii=False, allow_nan=False)
            json_file.write(line + "\n")
            json_file.flush()
            writer.writerow(result)
            csv_file.flush()
            write_json(root / "latest.json", result)
            if not quiet:
                print(line, flush=True)
            emitted += 1
    summary = dict(mode="file_time_stream_replay", predictions=emitted, valid_predictions=valid,
                   skipped=dict(skipped), history_minutes=30, step="one_completed_bin",
                   max_buffered_10s_blocks=max_blocks, max_buffered_event_clips=max_events,
                   output=str(root), backend=meta["forecast"]["backend"],
                   assumptions="文件结束即可用；微震归档完整。按已有文件时间回放，不监听新增文件。")
    write_json(root / "summary.json", summary)
    return summary
