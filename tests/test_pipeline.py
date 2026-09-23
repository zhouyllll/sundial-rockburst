import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rockburst.data import Inputs, Unavailable, HORIZONS
from rockburst.extract import extract, CONT_COLUMNS, EVENT_COLUMNS
from rockburst.forecast import Forecaster
from rockburst.io import iso, timestamp, write_csv, read_csv
from rockburst.model import fit_hazard, probabilities, average_precision, threshold_sweep, train
from rockburst.data import FEATURE_NAMES, TRANSIENT_FEATURE_NAMES, select_model_features


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = timestamp("2026-06-02T12:00:00+08:00")
        self.rows = []
        for end in range(int(self.now - 3590), int(self.now + 1), 10):
            self.rows.append(dict(zone_id="z", start_time=iso(end - 10), end_time=iso(end),
                                  available_time=iso(end), preprocessing_id="test-v1",
                                  mean_square=10., rms=10. ** .5, peak=10., energy_proxy=100.,
                                  stalta=2., coherence=.5, active_fraction=.5, low_band_fraction=.7,
                                  short_window_energy_q99=15., short_window_overthreshold_count=2))
        self.events = []
        self.bursts = []
        self.coverage = [dict(zone_id="z", start_time=iso(self.now - 86400),
                              end_time=iso(self.now + 3600), event_archive_complete=1,
                              rockburst_record_complete=1)]

    def inputs(self):
        write_csv(self.root / "c.csv", self.rows, CONT_COLUMNS)
        write_csv(self.root / "e.csv", self.events, EVENT_COLUMNS)
        write_csv(self.root / "coverage.csv", self.coverage,
                  ["zone_id", "start_time", "end_time", "event_archive_complete", "rockburst_record_complete"])
        write_csv(self.root / "b.csv", self.bursts,
                  ["event_id", "zone_id", "onset_time", "end_time", "group_id"])
        return Inputs(self.root / "c.csv", self.root / "e.csv", self.root / "coverage.csv",
                      "z", self.root / "b.csv")

    def event(self, start, available):
        row = self.rows[0].copy()
        row.update(event_id=f"event-{len(self.events)}", start_time=iso(start),
                   end_time=iso(start + 1), available_time=iso(available))
        return row

    def test_timestamps_require_timezone(self):
        with self.assertRaises(ValueError):
            timestamp("2026-06-01T12:00:00")
        self.assertEqual(timestamp("2026-06-01T12:00:00+08:00"),
                         timestamp("2026-06-01T04:00:00Z"))

    def test_empty_event_history_is_valid_with_coverage(self):
        x, quality = self.inputs().sample(self.now, Forecaster())
        self.assertEqual(x.shape, (3, 6))
        transient, _ = self.inputs().sample(self.now, Forecaster(), include_transient=True)
        self.assertEqual(transient.shape, (3, 11))
        self.assertGreater(transient[0, -1], 0.)
        self.assertEqual(quality["microseismic_count_24h"], 0)
        self.assertTrue(np.isfinite(x).all())

    def test_future_and_delayed_events_do_not_leak(self):
        baseline, _ = self.inputs().sample(self.now, Forecaster())
        self.events = [self.event(self.now + 10, self.now + 40),
                       self.event(self.now - 10, self.now + 20)]
        # Explicit unique IDs for two independently created fixture rows.
        self.events[1]["event_id"] = "delayed"
        actual, quality = self.inputs().sample(self.now, Forecaster())
        np.testing.assert_array_equal(actual, baseline)
        self.assertEqual(quality["microseismic_count_24h"], 0)

    def test_past_event_enters_features(self):
        self.events = [self.event(self.now - 100, self.now - 70)]
        x, quality = self.inputs().sample(self.now, Forecaster())
        self.assertEqual(quality["microseismic_count_24h"], 1)
        self.assertAlmostEqual(x[0, 2], np.log(2))

    def test_future_waveform_does_not_leak(self):
        baseline, _ = self.inputs().sample(self.now, Forecaster())
        future = self.rows[-1].copy()
        future.update(start_time=iso(self.now), end_time=iso(self.now + 10),
                      available_time=iso(self.now + 10), mean_square=1e15)
        self.rows.append(future)
        actual, _ = self.inputs().sample(self.now, Forecaster())
        np.testing.assert_array_equal(actual, baseline)

    def test_unavailable_waveform_block_is_not_used(self):
        self.rows[-1]["available_time"] = iso(self.now + 10)
        self.rows[-1]["mean_square"] = 1e15
        x, quality = self.inputs().sample(self.now, Forecaster())
        self.assertEqual(quality["continuous_lag_seconds"], 10)
        self.assertAlmostEqual(x[0, 0], 0.)

    def test_historical_unavailable_waveform_block_is_rejected(self):
        # The newest block may be lagged and is handled by the cutoff above,
        # but an unavailable block inside the selected history cannot be used.
        historical = self.rows[-20]
        historical["available_time"] = iso(self.now + 10)
        with self.assertRaisesRegex(Unavailable, "continuous_history_incomplete"):
            self.inputs().sample(self.now, Forecaster())

    def test_missing_continuous_is_not_zero_filled(self):
        self.rows.pop(200)
        with self.assertRaisesRegex(Unavailable, "continuous_history_incomplete"):
            self.inputs().sample(self.now, Forecaster())

    def test_missing_event_coverage_rejected(self):
        self.coverage[0]["event_archive_complete"] = 0
        with self.assertRaisesRegex(Unavailable, "event_history_incomplete"):
            self.inputs().sample(self.now, Forecaster())

    def test_disabled_microseismic_branch_does_not_require_archive_coverage(self):
        self.coverage[0]["event_archive_complete"] = 0
        data = self.inputs()
        data.microseismic_enabled = False
        x, quality = data.sample(self.now, Forecaster())
        self.assertEqual(quality["microseismic_count_24h"], 0)
        self.assertEqual(x.shape, (3, 6))

    def test_incomplete_future_is_not_negative(self):
        self.coverage[0]["rockburst_record_complete"] = 0
        with self.assertRaisesRegex(Unavailable, "future_labels_incomplete"):
            self.inputs().label(self.now)

    def test_interval_labels_boundaries_and_masks(self):
        for delay, expected in [(1, 0), (300, 0), (301, 1), (600, 1), (601, 2), (1800, 2)]:
            with self.subTest(delay=delay):
                self.bursts = [dict(event_id="b", zone_id="z", onset_time=iso(self.now + delay),
                                    end_time=iso(self.now + delay + 1), group_id="g")]
                y, mask, eid, group = self.inputs().label(self.now)
                self.assertEqual(int(np.argmax(y)), expected)
                np.testing.assert_array_equal(mask, np.arange(3) <= expected)
                self.assertEqual((eid, group), ("b", "g"))

    def test_ongoing_burst_excluded(self):
        self.bursts = [dict(event_id="b", zone_id="z", onset_time=iso(self.now - 10),
                            end_time=iso(self.now + 10), group_id="g")]
        with self.assertRaisesRegex(Unavailable, "rockburst_in_progress"):
            self.inputs().label(self.now)

    def test_known_burst_at_recording_end_keeps_pre_event_labels(self):
        onset = self.now + 480
        self.coverage[0]["end_time"] = iso(onset)
        self.bursts = [dict(event_id="b", zone_id="z", onset_time=iso(onset),
                            end_time=iso(onset), group_id="g")]
        y, mask, _, _ = self.inputs().label(self.now)
        np.testing.assert_array_equal(y, [0, 1, 0])
        np.testing.assert_array_equal(mask, [True, True, False])

    def test_duplicate_events_rejected(self):
        event = self.event(self.now - 100, self.now - 70)
        self.events = [event, event.copy()]
        with self.assertRaisesRegex(ValueError, "event_id"):
            self.inputs()

    def test_conflicting_coverage_rejected(self):
        self.coverage.append(dict(self.coverage[0], event_archive_complete=0))
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.inputs()

    def test_preprocessing_mismatch_rejected(self):
        self.events = [dict(self.event(self.now - 100, self.now - 70), preprocessing_id="other")]
        with self.assertRaisesRegex(ValueError, "配置不一致"):
            self.inputs()

    def test_persistence_cache_roundtrip(self):
        forecast = Forecaster(cache=self.root / "cache")
        seq = np.arange(360, dtype=np.float32)
        a = forecast.predict(seq)
        b = forecast.predict(seq)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (20, 180))
        self.assertTrue(np.all(a == 359))

    def test_sundial_does_not_fall_back_when_missing(self):
        with self.assertRaisesRegex(ValueError, "model-path"):
            Forecaster("sundial")

    def test_hazard_fit_monotonic_and_roundtrip(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(240, 3, 6))
        y, mask = np.zeros((240, 3)), np.ones((240, 3), dtype=bool)
        for i in range(120):
            j = i % 3
            y[i, j], mask[i, j + 1:] = 1, False
        model = fit_hazard(x, y, mask, 1.)
        p = probabilities(x, json.loads(json.dumps(model)))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.all(np.diff(p, axis=1) >= 0))
        self.assertTrue(np.all((0 <= p) & (p <= 1)))
        np.testing.assert_allclose(p, probabilities(x, model))

    def test_no_positive_training_rejected(self):
        with self.assertRaisesRegex(ValueError, "缺少正例"):
            fit_hazard(np.zeros((10, 3, 6)), np.zeros((10, 3)), np.ones((10, 3), bool), 1.)

    def test_average_precision_ties(self):
        self.assertAlmostEqual(average_precision(np.array([1, 0, 1, 0]), np.ones(4)), .5)
        self.assertIsNone(average_precision(np.zeros(3), np.ones(3)))


class BinTests(unittest.TestCase):
    def test_cross_file_grid_and_availability(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fs, count = 2000, 2
            t = np.arange(20 * fs) / fs
            wave = np.column_stack([1000 * np.sin(2 * np.pi * 100 * t)] * count).astype("<i4")
            wave[:15000].tofile(root / "a.bin")
            wave[15000:].tofile(root / "b.bin")
            origin = timestamp("2026-06-01T00:00:00Z")
            rows = [dict(file_path="a.bin", zone_id="z", start_time=iso(origin),
                         available_time=iso(origin + 7.5), fs=fs, channels=count),
                    dict(file_path="b.bin", zone_id="z", start_time=iso(origin + 7.5),
                         available_time=iso(origin + 20), fs=fs, channels=count)]
            write_csv(root / "manifest.csv", rows, list(rows[0]))
            result = extract(root / "manifest.csv", root / "features.csv", "continuous")
            self.assertEqual(result["rows"], 2)
            features = read_csv(root / "features.csv", CONT_COLUMNS)
            self.assertEqual(timestamp(features[0]["available_time"]), origin + 20)
            self.assertGreater(float(features[0]["rms"]), 600)
            self.assertAlmostEqual(float(features[0]["coherence"]), 1., places=5)

    def test_subsecond_transient_features_detect_local_energy_burst(self):
        from rockburst.extract import waveform_features
        rng = np.random.default_rng(19)
        fs = 2000
        t = np.arange(10 * fs) / fs
        base = .5 * np.sin(2 * np.pi * 120 * t)
        wave = np.column_stack([base + rng.normal(0, .05, len(t))] * 2)
        pulse = (t >= 5.) & (t < 5.2)
        wave[pulse] += 15. * np.sin(2 * np.pi * 120 * t[pulse, None])
        features = waveform_features(wave, fs)
        self.assertGreater(features["short_window_energy_q99"], features["mean_square"])
        self.assertGreater(features["short_window_overthreshold_count"], 0)
        self.assertGreater(features["peak"] / features["rms"], 1.)

    def test_truncated_bin_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "bad.bin").write_bytes(b"abc")
            row = dict(file_path="bad.bin", zone_id="z", start_time="2026-01-01T00:00:00Z",
                       available_time="2026-01-01T00:00:30Z", fs=5000, channels=42)
            write_csv(root / "m.csv", [row], list(row))
            with self.assertRaisesRegex(ValueError, "整除"):
                extract(root / "m.csv", root / "f.csv", "continuous")

    def test_empty_microseismic_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            columns = ["file_path", "zone_id", "start_time", "available_time", "fs", "channels",
                       "event_id", "event_start_time", "event_end_time"]
            write_csv(root / "m.csv", [], columns)
            result = extract(root / "m.csv", root / "e.csv", "events")
            self.assertEqual(result["rows"], 0)
            self.assertEqual(read_csv(root / "e.csv", EVENT_COLUMNS), [])


class SplitTests(unittest.TestCase):
    def make_dataset(self, root, conflicting_groups=False):
        origin = timestamp("2026-06-01T00:00:00Z")
        times = origin + np.arange(300) * 60
        burst_times = origin + np.array([45, 145, 245]) * 60
        y = np.zeros((300, 3))
        mask = np.ones((300, 3), dtype=bool)
        ids, groups = [], []
        for i, now in enumerate(times):
            future = np.flatnonzero(burst_times > now)
            eid = group = ""
            if len(future):
                k = future[0]
                delay = burst_times[k] - now
                if delay <= 1800:
                    j = int(np.searchsorted(HORIZONS, delay))
                    y[i, j], mask[i, j + 1:] = 1, False
                    eid = f"event-{k}"
                    group = "same-group" if conflicting_groups else f"group-{k}"
            ids.append(eid)
            groups.append(group)
        x = np.random.default_rng(7).normal(size=(300, 3, 6))
        path = root / "dataset.npz"
        meta = dict(feature_names=FEATURE_NAMES, forecast={"backend": "persistence"})
        np.savez_compressed(path, x=x, y=y, mask=mask, times=times,
                            event_ids=np.array(ids), groups=np.array(groups), metadata=json.dumps(meta))
        return path, origin, x, y, mask, times, ids, groups, meta

    def test_temporal_purge_and_test_does_not_affect_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, origin, x, y, mask, times, ids, groups, meta = self.make_dataset(root)
            report = train(path, root / "run1", origin + 100 * 60, origin + 200 * 60, ridges=[1.])
            self.assertEqual(report["purged_samples"], 58)
            model1 = json.loads((root / "run1/model.json").read_text())
            x[times >= origin + 200 * 60] += 1e6
            np.savez_compressed(path, x=x, y=y, mask=mask, times=times,
                                event_ids=np.array(ids), groups=np.array(groups), metadata=json.dumps(meta))
            train(path, root / "run2", origin + 100 * 60, origin + 200 * 60, ridges=[1.])
            model2 = json.loads((root / "run2/model.json").read_text())
            self.assertEqual(model1, model2)

    def test_inference_selects_the_saved_ablation_feature_order(self):
        full = np.arange(3 * 11, dtype=float).reshape(1, 3, 11)
        selected_names = FEATURE_NAMES + TRANSIENT_FEATURE_NAMES[:2]
        actual = select_model_features(full, selected_names)
        self.assertEqual(actual.shape, (1, 3, 8))
        np.testing.assert_array_equal(actual, full[:, :, :8])

    def test_train_supports_selected_transient_feature_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, origin, x, y, mask, times, ids, groups, _ = self.make_dataset(root)
            names = FEATURE_NAMES + ["recent_peak_log", "recent_crest_factor",
                                     "recent_stalta_log", "recent_short_energy_q99_log",
                                     "recent_overthreshold_count_log"]
            expanded = np.concatenate([x, np.random.default_rng(31).normal(size=(len(x), 3, 5))], axis=2)
            with np.load(path, allow_pickle=False) as pack:
                metadata = json.loads(str(pack["metadata"]))
            metadata["feature_names"] = names
            np.savez_compressed(path, x=expanded, y=y, mask=mask, times=times,
                                event_ids=np.array(ids), groups=np.array(groups),
                                metadata=json.dumps(metadata))
            selected = list(range(8))
            report = train(path, root / "run", origin + 100 * 60, origin + 200 * 60,
                           ridges=[1.], feature_indices=selected, feature_variant="plus_peak_crest")
            model = json.loads((root / "run/model.json").read_text())
            self.assertEqual(len(model["beta"]), 8)
            self.assertEqual(model["metadata"]["feature_names"], names[:8])
            self.assertEqual(report["feature_variant"], "plus_peak_crest")
            self.assertEqual(set(report["splits"]), {"train", "validation", "test"})

    def test_same_event_group_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, origin, *_ = self.make_dataset(root, conflicting_groups=True)
            with self.assertRaisesRegex(ValueError, "同一岩爆群"):
                train(path, root / "run", origin + 100 * 60, origin + 200 * 60)

    def test_threshold_sweep_reports_event_and_alarm_diagnostics(self):
        rows = threshold_sweep(
            np.array([False, True, True, False]),
            np.array([.1, .4, .7, .2]),
            np.array([0., 30., 60., 90.]),
            np.array(["", "event-1", "event-1", ""]),
            refresh_seconds=30., thresholds=[.2, .5])
        self.assertEqual([r["threshold"] for r in rows], [.2, .5])
        self.assertEqual(rows[0]["eligible_events"], 1)
        self.assertEqual(rows[1]["detected_events"], 1)
        self.assertIn("false_alarms_per_24h_evaluated", rows[0])


if __name__ == "__main__":
    unittest.main()
