from pathlib import Path
import tempfile
import unittest

from rockburst.__main__ import parser
from rockburst.io import local_path, timestamp
from rockburst.workflow import read_burst_times, scan_bins


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
