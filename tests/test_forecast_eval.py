from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rockburst.__main__ import parser
from rockburst.extract import CONT_COLUMNS
from rockburst.forecast_eval import evaluate_forecasts
from rockburst.io import iso, timestamp, write_csv


class KnownForecaster:
    identity = {"backend": "fixture"}

    def predict(self, sequence, horizon):
        return np.full((4, horizon), np.log1p(np.exp(3.) - 1.), dtype=np.float32)


class ForecastEvaluationTests(unittest.TestCase):
    def test_horizon_means_compared_to_future_and_persistence_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "continuous.csv"
            origin = timestamp("2026-01-01T00:00:00Z")
            rows = []
            for i in range(360):
                start = origin + i * 10
                end = start + 10
                log_power = 1. if i < 180 else 3.
                rows.append(dict(zone_id="z", start_time=iso(start), end_time=iso(end),
                                 available_time=iso(end), preprocessing_id="same",
                                 mean_square=np.expm1(log_power), rms=0, peak=0,
                                 energy_proxy=0, stalta=0, coherence=0,
                                 active_fraction=0, low_band_fraction=0))
            write_csv(path, rows, CONT_COLUMNS)
            result = evaluate_forecasts(path, KnownForecaster(), root / "out", step_seconds=30)
            for horizon in ("5min", "10min", "30min"):
                metrics = result["horizons"][horizon]
                self.assertEqual(metrics["origins"], 1)
                self.assertAlmostEqual(metrics["mae_log_power"], 0., places=5)
                self.assertGreater(metrics["persistence_mae_log_power"], 1.9)
                self.assertGreater(metrics["mae_improvement_vs_persistence"], .99)
            with (root / "out/forecast_comparison.csv").open() as stream:
                content = stream.read()
            self.assertIn("observed_log_power_mean", content)
            self.assertIn("persistence_log_power_mean", content)
            self.assertTrue((root / "out/forecast_metrics.json").exists())

    def test_requires_history_plus_thirty_minutes_of_future(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            origin = timestamp("2026-01-01T00:00:00Z")
            for i in range(359):
                start = origin + i * 10
                rows.append(dict(zone_id="z", start_time=iso(start), end_time=iso(start + 10),
                                 available_time=iso(start + 10), preprocessing_id="same",
                                 mean_square=1, rms=0, peak=0, energy_proxy=0, stalta=0,
                                 coherence=0, active_fraction=0, low_band_fraction=0))
            write_csv(root / "short.csv", rows, CONT_COLUMNS)
            with self.assertRaisesRegex(ValueError, r"历史窗口 \+ 30分钟连续未来"):
                evaluate_forecasts(root / "short.csv", KnownForecaster(), root / "out")

    def test_cli_exposes_diagnostic_command(self):
        args = parser().parse_args(["evaluate-forecast", "--features", "continuous.csv",
                                    "--model-path", "weights"])
        self.assertEqual(args.command, "evaluate-forecast")
        self.assertEqual(args.backend, "sundial")
        self.assertEqual(args.step_seconds, 30)


if __name__ == "__main__":
    unittest.main()
