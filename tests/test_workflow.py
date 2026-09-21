from pathlib import Path
import tempfile
import unittest

from rockburst.__main__ import parser
from rockburst.io import timestamp
from rockburst.workflow import read_burst_times, scan_bins


class WorkflowTests(unittest.TestCase):
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

    def test_run_requires_directories_and_labels_not_manifests(self):
        args = parser().parse_args(["run", "--continuous-dir", "continuous",
                                    "--microseismic-dir", "micro", "--labels", "rockbursts.txt"])
        self.assertEqual(args.backend, "sundial")
        self.assertEqual(args.history_minutes, 30)
        self.assertFalse(hasattr(args, "coverage"))


if __name__ == "__main__":
    unittest.main()
