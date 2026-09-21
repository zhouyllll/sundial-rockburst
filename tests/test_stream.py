from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rockburst.data import build_dataset
from rockburst.extract import load_manifest
from rockburst.forecast import Forecaster
from rockburst.io import timestamp, write_csv, write_json
from rockburst.stream import replay_stream
from rockburst.workflow import scan_bins, MANIFEST_COLUMNS


class StreamingTests(unittest.TestCase):
    def test_one_hour_replay_advances_by_each_bin_and_keeps_future_events_out(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            continuous, micro = root / "continuous", root / "micro"
            continuous.mkdir()
            micro.mkdir()
            origin = timestamp("2026-01-01T00:00:00Z")

            def make_file(directory, start):
                stamp = datetime.fromtimestamp(start, timezone.utc).strftime("%Y%m%dT%H%M%S.000")
                p = directory / f"data-1200Hz-0002pt-{stamp}.bin"
                with p.open("wb") as stream:
                    stream.truncate(1200 * 2 * 4 * 30)

            for offset in range(0, 3600, 30):
                make_file(continuous, origin + offset)
            make_file(micro, origin + 32 * 60)
            rows = scan_bins(continuous, "z", "UTC")
            write_csv(root / "manifest.csv", rows, MANIFEST_COLUMNS)
            records = load_manifest(root / "manifest.csv", "continuous", "all", 50., 500.)
            model = dict(mean=[0.] * 6, scale=[1.] * 6, beta=[0., 0., 1., 0., 0., 0.],
                         intercepts=[-3.] * 3, calibration="test_only",
                         metadata=dict(zone_id="z", history_minutes=30, max_lag_seconds=60,
                                       preprocessing_id=records[0]["signature"], forecast=Forecaster().identity))
            write_json(root / "model.json", model)
            report = replay_stream(continuous, micro, root / "model.json", root / "out", timezone="UTC", quiet=True)
            predictions = [json.loads(line) for line in (root / "out/predictions.jsonl").read_text().splitlines()]
            self.assertEqual(report["valid_predictions"], 61)
            self.assertLessEqual(report["max_buffered_10s_blocks"], 180)
            times = [timestamp(r["time"]) for r in predictions]
            self.assertEqual(times[0], origin + 1800)
            self.assertEqual(times[-1], origin + 3600)
            np.testing.assert_array_equal(np.diff(times), np.full(60, 30))
            for row, now in zip(predictions, times):
                self.assertEqual(row["quality"]["microseismic_count_24h"], int(now >= origin + 1950))
                self.assertLessEqual(row["p_5min"], row["p_10min"])
                self.assertLessEqual(row["p_10min"], row["p_30min"])
            self.assertGreater(predictions[-1]["p_5min"], predictions[0]["p_5min"])

    def test_file_step_dataset_keeps_milliseconds(self):
        class Fixture:
            zone, signature = "z", "test"

            def label(self, now):
                return np.zeros(3), np.ones(3, dtype=bool), "", ""

            def sample(self, now, forecast, history_minutes, max_lag):
                return np.zeros((3, 6)), {}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.npz"
            start = timestamp("2026-01-01T00:30:00.846Z")
            schedule = [start, start + 30, start + 60]
            build_dataset(Fixture(), Forecaster(), start, start + 61, path, 30,
                          prediction_times=schedule)
            with np.load(path, allow_pickle=False) as pack:
                np.testing.assert_array_equal(pack["times"], schedule)
                meta = json.loads(str(pack["metadata"]))
            self.assertEqual(meta["sample_step"], "bin_file")
            self.assertEqual(meta["refresh_seconds"], 30)


if __name__ == "__main__":
    unittest.main()
