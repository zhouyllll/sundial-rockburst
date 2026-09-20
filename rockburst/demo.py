"""Synthetic feature-level smoke test; never presents fake data as field results."""

from pathlib import Path

import numpy as np

from .data import Inputs, build_dataset
from .extract import CONT_COLUMNS, EVENT_COLUMNS
from .forecast import Forecaster
from .io import iso, timestamp, write_csv, write_json
from .model import train


def demo(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260920)
    origin = timestamp("2026-01-01T00:00:00+00:00")
    stop = origin + 4 * 86400
    burst_times = [origin + hour * 3600 for hour in [28, 33, 39, 44, 52, 58, 64, 69, 77, 82, 88, 93]]
    rows = []
    for end in range(int(origin + 10), int(stop + 1), 10):
        delay = min([b - end for b in burst_times if b > end] or [1e9])
        rise = max(0., 1 - delay / 2400)
        power = 10 + float(rng.uniform(0, 3)) + 90 * rise
        rows.append(dict(zone_id="demo", start_time=iso(end - 10), end_time=iso(end),
                         available_time=iso(end), preprocessing_id="synthetic-v1",
                         mean_square=power, rms=float(np.sqrt(power)), peak=float(np.sqrt(power) * 3),
                         energy_proxy=power * 10, stalta=1 + rise, coherence=.2 + .6 * rise,
                         active_fraction=.5, low_band_fraction=.7))
    write_csv(root / "continuous.csv", rows, CONT_COLUMNS)
    events = []
    for idx, start in enumerate(np.arange(origin + 1800, stop - 30, 7200)):
        power = float(rng.uniform(10, 100))
        events.append(dict(zone_id="demo", start_time=iso(start), end_time=iso(start + 1),
                           available_time=iso(start + 30), preprocessing_id="synthetic-v1",
                           event_id=f"micro-{idx}", mean_square=power, rms=float(np.sqrt(power)),
                           peak=float(np.sqrt(power) * 3), energy_proxy=power,
                           stalta=2., coherence=.5, active_fraction=.5, low_band_fraction=.7))
    write_csv(root / "events.csv", events, EVENT_COLUMNS)
    write_csv(root / "coverage.csv", [dict(zone_id="demo", start_time=iso(origin), end_time=iso(stop),
                                          event_archive_complete=1, rockburst_record_complete=1)],
              ["zone_id", "start_time", "end_time", "event_archive_complete", "rockburst_record_complete"])
    bursts = [dict(event_id=f"burst-{i}", zone_id="demo", onset_time=iso(t), end_time=iso(t + 60),
                   group_id=f"group-{i}") for i, t in enumerate(burst_times)]
    write_csv(root / "rockbursts.csv", bursts,
              ["event_id", "zone_id", "onset_time", "end_time", "group_id"])
    inputs = Inputs(root / "continuous.csv", root / "events.csv", root / "coverage.csv",
                    "demo", root / "rockbursts.csv")
    build_dataset(inputs, Forecaster("persistence"), origin + 86400, stop - 1800,
                  root / "dataset.npz")
    report = train(root / "dataset.npz", root / "run", origin + 48 * 3600, origin + 72 * 3600)
    write_json(root / "SYNTHETIC_DEMO_ONLY.json", {
        "warning": "仅合成数据+持值基线的流程验证；未运行Sundial，不代表真实岩爆性能。",
        "backend": "persistence", "seed": 20260920})
    return {"output": str(root), "synthetic": True, "backend": "persistence",
            "test_samples": report["splits"]["test"]["samples"]}
