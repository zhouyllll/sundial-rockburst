"""Evaluate Sundial's future log-power feature forecasts against observed values."""

from collections import defaultdict

import numpy as np

from .extract import CONT_COLUMNS
from .io import iso, read_csv, timestamp, write_csv, write_json

HORIZONS = (("5min", 30), ("10min", 60), ("30min", 180))
DETAIL_COLUMNS = [
    "origin_time", "horizon", "target_start", "target_end", "observed_log_power_mean",
    "sundial_log_power_mean", "persistence_log_power_mean", "sundial_abs_error",
    "persistence_abs_error", "sundial_squared_error", "persistence_squared_error",
    "sundial_bias", "persistence_bias",
]


def _metrics(rows, model_name):
    errors = np.asarray([r[f"{model_name}_log_power_mean"] - r["observed_log_power_mean"] for r in rows])
    absolute = np.abs(errors)
    persistence_errors = np.asarray([
        r["persistence_log_power_mean"] - r["observed_log_power_mean"] for r in rows
    ])
    persistence_absolute = np.abs(persistence_errors)
    mae = float(absolute.mean())
    baseline_mae = float(persistence_absolute.mean())
    return {
        "origins": len(rows),
        "mae_log_power": mae,
        "rmse_log_power": float(np.sqrt(np.mean(errors ** 2))),
        "bias_log_power": float(errors.mean()),
        "persistence_mae_log_power": baseline_mae,
        "persistence_rmse_log_power": float(np.sqrt(np.mean(persistence_errors ** 2))),
        "mae_improvement_vs_persistence": (
            float(1. - mae / baseline_mae) if baseline_mae > 0 else None
        ),
    }


def evaluate_forecasts(features_csv, forecaster, output_dir, history_minutes=30,
                       step_seconds=30, minimum_origins=1):
    """Compare forecast and actual horizon-mean log1p(mean_square), without label use.

    Forecast origin is the end of a completed 10-second feature block. The target
    consists only of subsequent blocks. Gaps, unavailable blocks, and preprocessing
    changes split recording runs so no history/target window crosses them.
    """
    if history_minutes not in {30, 60}:
        raise ValueError("history_minutes 只支持30或60")
    if step_seconds < 10 or step_seconds % 10:
        raise ValueError("step_seconds 必须是10秒的正整数倍")
    history_count = history_minutes * 6
    stride = step_seconds // 10
    rows = read_csv(features_csv, CONT_COLUMNS)
    if not rows:
        raise ValueError("连续特征CSV为空")
    by_zone = defaultdict(list)
    for row in rows:
        by_zone[row["zone_id"]].append(row)

    detail = []
    skipped = defaultdict(int)
    for zone, zone_rows in sorted(by_zone.items()):
        zone_rows.sort(key=lambda r: timestamp(r["end_time"]))
        # A preprocessing change or time gap starts a new independent run.
        runs, run = [], []
        previous_end = None
        previous_signature = None
        for row in zone_rows:
            end = timestamp(row["end_time"])
            available = timestamp(row["available_time"])
            if abs(end / 10 - round(end / 10)) > 1e-5:
                raise ValueError(f"特征时间未对齐10秒网格: {row['end_time']}")
            if abs(end - timestamp(row["start_time"]) - 10.) > 1e-5:
                raise ValueError(f"连续特征必须是完整10秒块: {row['start_time']} ~ {row['end_time']}")
            signature = row["preprocessing_id"]
            if previous_end is not None and (abs(end - previous_end - 10.) > 1e-5 or
                                              signature != previous_signature):
                if run:
                    runs.append(run)
                run = []
            run.append(row)
            previous_end, previous_signature = end, signature
        if run:
            runs.append(run)

        for sequence in runs:
            values = np.asarray([float(r["mean_square"]) for r in sequence], dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("mean_square 必须是有限的非负数")
            log_values = np.log1p(values)
            for origin_idx in range(history_count - 1, len(sequence) - 180, stride):
                origin = timestamp(sequence[origin_idx]["end_time"])
                history = log_values[origin_idx - history_count + 1:origin_idx + 1]
                actual_future = log_values[origin_idx + 1:origin_idx + 181]
                if len(history) != history_count or len(actual_future) != 180:
                    skipped["incomplete_history_or_target"] += 1
                    continue
                if any(timestamp(r["available_time"]) > origin + 1e-5
                       for r in sequence[origin_idx - history_count + 1:origin_idx + 1]):
                    skipped["history_not_available_at_origin"] += 1
                    continue
                paths = np.asarray(forecaster.predict(history, horizon=180), dtype=np.float64)
                if paths.ndim != 2 or paths.shape[1] != 180 or not np.isfinite(paths).all():
                    raise ValueError(f"预测路径形状或数值异常: {paths.shape}")
                baseline = float(history[-1])
                for horizon_name, steps in HORIZONS:
                    observed_mean = float(actual_future[:steps].mean())
                    sundial_mean = float(np.median(paths[:, :steps].mean(axis=1)))
                    baseline_mean = baseline
                    sundial_error = sundial_mean - observed_mean
                    baseline_error = baseline_mean - observed_mean
                    target_start = timestamp(sequence[origin_idx + 1]["start_time"])
                    target_end = timestamp(sequence[origin_idx + steps]["end_time"])
                    detail.append({
                        "origin_time": iso(origin), "zone_id": zone, "horizon": horizon_name,
                        "target_start": iso(target_start), "target_end": iso(target_end),
                        "observed_log_power_mean": observed_mean,
                        "sundial_log_power_mean": sundial_mean,
                        "persistence_log_power_mean": baseline_mean,
                        "sundial_abs_error": abs(sundial_error),
                        "persistence_abs_error": abs(baseline_error),
                        "sundial_squared_error": sundial_error ** 2,
                        "persistence_squared_error": baseline_error ** 2,
                        "sundial_bias": sundial_error,
                        "persistence_bias": baseline_error,
                    })
    if not detail:
        raise ValueError("没有可评估的时刻；需至少有历史窗口 + 30分钟连续未来特征")

    summary = {
        "task": "future_log1p_mean_square_horizon_mean",
        "forecast_backend": forecaster.identity,
        "history_minutes": history_minutes,
        "step_seconds": step_seconds,
        "feature_interval_seconds": 10,
        "horizons": {},
        "skipped": dict(skipped),
        "interpretation": (
            "误差针对log1p(mean_square)的时窗平均值，不是原始高频岩爆波形误差，也不是岩爆概率指标。"
        ),
    }
    for horizon_name, _ in HORIZONS:
        horizon_rows = [r for r in detail if r["horizon"] == horizon_name]
        metrics = _metrics(horizon_rows, "sundial")
        summary["horizons"][horizon_name] = metrics
        if metrics["origins"] < minimum_origins:
            raise ValueError(f"{horizon_name}只有{metrics['origins']}个有效预测起点，少于要求的{minimum_origins}")

    from pathlib import Path
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    columns = DETAIL_COLUMNS[:1] + ["zone_id"] + DETAIL_COLUMNS[1:]
    write_csv(root / "forecast_comparison.csv", detail, columns)
    write_json(root / "forecast_metrics.json", summary)
    return summary
