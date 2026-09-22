from pathlib import Path
import tempfile
import unittest

from rockburst.__main__ import parser
from rockburst.io import iso, local_path, timestamp
from rockburst.workflow import (grouped_time_boundaries, read_burst_times,
                                 recording_group_intervals, scan_bins)


class WorkflowTests(unittest.TestCase):
    def test_windows_drive_path_maps_to_wsl_mount(self):
        self.assertEqual(str(local_path(r"G:\1A岩爆预测\岩爆数据集\5.10.8.49")),
                         "/mnt/g/1A岩爆预测/岩爆数据集/5.10.8.49")

    def test_scan_reads_existing_filename_format(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "0008655-fs-eDAS-5000Hz-0041pt-20260621T151630.846.bin"
            with path.open("wb") as stream:
                stream.truncate(5000 * 41 * 4 * 30)
            row = scan_bins(root, "zone_A")[0]
            self.assertEqual((row["fs"], row["channels"]), (5000, 41))
            self.assertEqual(timestamp(row["start_time"]), timestamp("2026-06-21T15:16:30.846+08:00"))
            self.assertAlmostEqual(timestamp(row["available_time"]) - timestamp(row["start_time"]), 30.)

    def test_microseismic_file_becomes_one_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested = root / "nested"
            nested.mkdir()
            path = nested / "micro-2000Hz-0002pt-20260621T151630.bin"
            with path.open("wb") as stream:
                stream.truncate(2000 * 2 * 4 * 30)
            row = scan_bins(root, "zone_A", event_clips=True)[0]
            self.assertEqual(row["event_start_time"], row["start_time"])
            self.assertEqual(row["event_end_time"], row["available_time"])
            self.assertEqual(row["event_id"], "clip-000001")

    def test_recording_groups_keep_folder_samples_together_and_merge_overlapping_histories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            rows = []
            cases = [("event_a", "2026-06-01T00:00:00+00:00", 7200),
                     ("event_b", "2026-06-01T02:10:00+00:00", 3600),
                     ("event_c", "2026-06-01T05:00:00+00:00", 3600)]
            for name, start_text, duration in cases:
                directory = root / name
                directory.mkdir(parents=True)
                path = directory / "placeholder.bin"
                path.touch()
                start = timestamp(start_text)
                rows.append(dict(file_path=str(path), start_time=iso(start),
                                 available_time=iso(start + duration)))
            intervals = recording_group_intervals(rows, root, "UTC", 30)
            self.assertEqual(len(intervals), 2)
            self.assertTrue(intervals[0][2].startswith("recording-"))
            self.assertTrue(intervals[1][2].startswith("folder-"))

    def test_flat_continuous_directory_has_no_folder_holdout_group(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "data.bin"
            path.touch()
            row = dict(file_path=str(path), start_time="2026-06-01T00:00:00Z",
                       available_time="2026-06-01T00:00:30Z")
            self.assertEqual(recording_group_intervals([row], root, "UTC", 30), [])

    def test_grouped_time_boundaries_split_whole_chronological_groups(self):
        origin = timestamp("2026-06-01T00:00:00Z")
        times, groups = [], []
        for i in range(10):
            start = origin + i * 3 * 3600
            times.extend(start + j * 60 for j in range(100))
            groups.extend([f"folder-{i}"] * 100)
        train_end, validation_end = grouped_time_boundaries(times, groups)
        labels = []
        for now in times:
            labels.append(0 if now + 1800 <= train_end else
                          1 if now >= train_end and now + 1800 <= validation_end else
                          2 if now >= validation_end else -1)
        for group in set(groups):
            assigned = {labels[i] for i, value in enumerate(groups) if value == group}
            assigned.discard(-1)
            self.assertLessEqual(len(assigned), 1, group)
        self.assertTrue(any(v == 0 for v in labels))
        self.assertTrue(any(v == 1 for v in labels))
        self.assertTrue(any(v == 2 for v in labels))

    def test_plain_timestamp_list_needs_no_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rockbursts.txt"
            path.write_text("# confirmed events\n2026-06-21 15:30:00\n2026-06-23T18:40:00+08:00\n", encoding="utf-8")
            rows = read_burst_times(path, "zone_A", "Asia/Shanghai")
            self.assertEqual(len(rows), 2)
            self.assertEqual(timestamp(rows[0]["onset_time"]), timestamp("2026-06-21T15:30:00+08:00"))
            self.assertEqual(rows[0]["end_time"], rows[0]["onset_time"])

    def test_clock_ranges_match_event_subfolder_dates_and_accept_separator_typos(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            event_a = root / "event_a"
            event_b = root / "event_b"
            event_a.mkdir(parents=True)
            event_b.mkdir()
            for event_dir, stamp, duration in [(event_a, "20260621T100000", 4 * 3600),
                                               (event_b, "20260622T000000", 2 * 3600)]:
                path = event_dir / f"data-1Hz-0001pt-{stamp}.bin"
                with path.open("wb") as stream:
                    stream.truncate(4 * duration)
            continuous = scan_bins(root, "zone_A")
            labels = Path(folder) / "rockbursts.txt"
            labels.write_text("11:59:59-12:00:01\n00:06:10-00:06-11\n"
                              "13:12:19-13.12:20\n", encoding="utf-8")
            rows = read_burst_times(labels, "zone_A", "Asia/Shanghai", continuous, root)
            by_id = {row["event_id"]: row for row in rows}
            self.assertNotEqual(by_id["rb-0001"]["group_id"], by_id["rb-0002"]["group_id"])
            self.assertEqual(by_id["rb-0001"]["group_id"], by_id["rb-0003"]["group_id"])
            self.assertEqual(timestamp(by_id["rb-0002"]["end_time"]),
                             timestamp("2026-06-22T00:06:11+08:00"))
            self.assertEqual(timestamp(by_id["rb-0003"]["onset_time"]),
                             timestamp("2026-06-21T13:12:19+08:00"))

    def test_run_requires_directories_and_labels_not_manifests(self):
        args = parser().parse_args(["run", "--continuous-dir", "continuous",
                                    "--microseismic-dir", "micro", "--labels", "rockbursts.txt"])
        self.assertEqual(args.backend, "sundial")
        self.assertEqual(args.history_minutes, 30)
        self.assertFalse(hasattr(args, "coverage"))

    def test_run_accepts_missing_microseismic_directory(self):
        args = parser().parse_args(["run", "--continuous-dir", "continuous",
                                    "--labels", "rockbursts.txt"])
        self.assertIsNone(args.microseismic_dir)


if __name__ == "__main__":
    unittest.main()
