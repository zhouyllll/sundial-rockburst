"""Main path: continuous bins + optional microseismic bins + rockburst times.

The convenience workflow assumes the supplied microseismic archive and labels
are complete. CSVs are internal artifacts, not forms for the user to fill in.
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import re

import numpy as np

from .data import Inputs, build_dataset
from .extract import extract
from .forecast import Forecaster
from .io import iso, local_path, read_csv, write_csv, write_json
from .model import probabilities, train


MANIFEST_COLUMNS = ["file_path", "zone_id", "start_time", "available_time", "fs", "channels"]
EVENT_MANIFEST_COLUMNS = MANIFEST_COLUMNS + ["event_id", "event_start_time", "event_end_time"]
BURST_COLUMNS = ["event_id", "zone_id", "onset_time", "end_time", "group_id"]
COVERAGE_COLUMNS = ["zone_id", "start_time", "end_time", "event_archive_complete", "rockburst_record_complete"]


def local_time(value, timezone):
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone))
    return dt.timestamp()


def scan_bins(directory, zone, timezone="Asia/Shanghai", event_clips=False):
    root = local_path(directory)
    if not root.is_dir():
        raise ValueError(f"bin目录不存在: {root}")
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".bin"):
        fs_match = re.search(r"(\d+)Hz", path.name, re.I)
        ch_match = re.search(r"(\d+)pt", path.name, re.I)
        time_match = re.search(r"(\d{8})T(\d{6})(?:\.(\d+))?", path.name)
        if not (fs_match and ch_match and time_match):
            raise ValueError(f"文件名需要包含5000Hz、0041pt、20260621T151630.846这样的字段: {path.name}")
        fs, count = int(fs_match[1]), int(ch_match[1])
        size = path.stat().st_size
        if fs <= 0 or count <= 0 or not size or size % (4 * count):
            raise ValueError(f"文件大小或采样参数与int32格式不符: {path.name}")
        dt = datetime.strptime(time_match[1] + time_match[2], "%Y%m%d%H%M%S")
        fraction = (time_match[3] or "").ljust(6, "0")[:6]
        start = dt.replace(microsecond=int(fraction), tzinfo=ZoneInfo(timezone)).timestamp()
        end = start + size / (4 * count * fs)
        row = dict(file_path=str(path.resolve()), zone_id=zone, start_time=iso(start),
                   available_time=iso(end), fs=fs, channels=count)
        if event_clips:
            row.update(event_id=f"clip-{len(rows)+1:06d}", event_start_time=iso(start), event_end_time=iso(end))
        rows.append(row)
    rows.sort(key=lambda row: row["start_time"])
    return rows


def scan_optional_bins(directory, zone, timezone="Asia/Shanghai", event_clips=False):
    """A missing optional archive means the corresponding input branch is disabled."""
    return scan_bins(directory, zone, timezone, event_clips) if directory else []


def _parse_clock_range(line):
    """Parse HH:MM:SS-HH:MM:SS, including dot/minor separator typos."""
    match = re.fullmatch(
        r"\s*(\d{1,2})[:.](\d{2})[:.](\d{2})\s*-\s*"
        r"(\d{1,2})[:.](\d{2})[:.-](\d{2})\s*", line)
    if not match:
        return None
    h1, m1, s1, h2, m2, s2 = map(int, match.groups())
    try:
        return time(h1, m1, s1), time(h2, m2, s2)
    except ValueError as exc:
        raise ValueError(f"无效的岩爆时刻范围: {line}") from exc


def _folder_windows(continuous_rows, continuous_root, timezone):
    root = local_path(continuous_root).resolve()
    grouped = {}
    for row in continuous_rows:
        filepath = Path(row["file_path"]).resolve()
        relative = filepath.relative_to(root)
        folder = relative.parts[0] if len(relative.parts) > 1 else "."
        start = local_time(row["start_time"], timezone)
        end = local_time(row["available_time"], timezone)
        bounds = grouped.setdefault(folder, [start, end])
        bounds[0] = min(bounds[0], start)
        bounds[1] = max(bounds[1], end)
    return grouped


def _resolve_clock_range(start_clock, end_clock, bounds, timezone):
    tz = ZoneInfo(timezone)
    first_day = datetime.fromtimestamp(bounds[0], tz).date() - timedelta(days=1)
    last_day = datetime.fromtimestamp(bounds[1], tz).date() + timedelta(days=1)
    matches = []
    day = first_day
    while day <= last_day:
        start_dt = datetime.combine(day, start_clock, tzinfo=tz)
        end_day = day + timedelta(days=1) if end_clock < start_clock else day
        end_dt = datetime.combine(end_day, end_clock, tzinfo=tz)
        start, end = start_dt.timestamp(), end_dt.timestamp()
        if bounds[0] <= start <= bounds[1] and start <= end <= bounds[1] + 86400:
            matches.append((start, end))
        day += timedelta(days=1)
    return matches


def read_burst_times(path, zone, timezone, continuous_rows=None, continuous_root=None):
    """Read full timestamps or time-of-day ranges matched to event-folder bin dates."""
    path = local_path(path)
    if path.suffix.lower() == ".csv":
        entries = read_csv(path, ["onset_time"])
        values = [(r["onset_time"], r.get("end_time") or r["onset_time"],
                   r.get("event_id") or f"rb-{i+1:04d}", r.get("group_id") or f"group-{i+1:04d}")
                  for i, r in enumerate(entries)]
        rows = [dict(event_id=eid, zone_id=zone, onset_time=iso(local_time(start, timezone)),
                     end_time=iso(local_time(end, timezone)), group_id=group)
                for start, end, eid, group in values]
    else:
        lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        rows = []
        clock_entries = []
        for i, line in enumerate(lines):
            parsed = _parse_clock_range(line)
            if parsed:
                clock_entries.append((i, line, *parsed))
            else:
                rows.append(dict(event_id=f"rb-{i+1:04d}", zone_id=zone,
                                 onset_time=iso(local_time(line, timezone)),
                                 end_time=iso(local_time(line, timezone)), group_id=f"group-{i+1:04d}"))
        if clock_entries:
            if not continuous_rows or not continuous_root:
                raise ValueError("HH:MM:SS-HH:MM:SS格式需要连续bin目录，以便从子文件夹bin时间补全日期")
            windows = _folder_windows(continuous_rows, continuous_root, timezone)
            for i, line, start_clock, end_clock in clock_entries:
                candidates = []
                for folder, bounds in windows.items():
                    for start, end in _resolve_clock_range(start_clock, end_clock, bounds, timezone):
                        candidates.append((folder, start, end))
                if len(candidates) != 1:
                    reason = "没有匹配的事件子文件夹" if not candidates else "匹配到多个事件子文件夹"
                    raise ValueError(f"第{i+1}行 {line!r} {reason}；请确认每个子文件夹的bin文件名日期覆盖该时刻，且每个时刻只对应一个子文件夹")
                folder, start, end = candidates[0]
                group = "folder-" + hashlib.sha1(folder.encode("utf-8")).hexdigest()[:12]
                rows.append(dict(event_id=f"rb-{i+1:04d}", zone_id=zone, onset_time=iso(start),
                                 end_time=iso(end), group_id=group))
    if not rows:
        raise ValueError("岩爆时间清单为空，请提供真实岩爆发生时间")
    return sorted(rows, key=lambda row: row["onset_time"])


def run_workflow(continuous_dir, microseismic_dir, labels, output, backend="sundial",
                 model_path=None, device="cpu", samples=20, cache=None,
                 zone="zone_A", timezone="Asia/Shanghai", history_minutes=30,
                 monitor="all", ridge=1., threshold=.5):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    print("[1/5] 扫描bin文件名，自动生成内部索引和岩爆标签", flush=True)
    continuous = scan_bins(continuous_dir, zone, timezone)
    microseismic_enabled = bool(microseismic_dir)
    events = scan_optional_bins(microseismic_dir, zone, timezone, event_clips=True)
    if not continuous:
        raise ValueError("连续bin目录为空")
    bursts = read_burst_times(labels, zone, timezone, continuous, continuous_dir)
    generated = root / "generated"
    write_csv(generated / "continuous_manifest.csv", continuous, MANIFEST_COLUMNS)
    write_csv(generated / "microseismic_manifest.csv", events, EVENT_MANIFEST_COLUMNS)
    write_csv(generated / "rockbursts.csv", bursts, BURST_COLUMNS)
    first = min(local_time(r["start_time"], timezone) for r in continuous)
    last = max(local_time(r["available_time"], timezone) for r in continuous)
    # Explicit happy-path assumption: supplied archive/labels cover this experiment.
    write_csv(generated / "coverage.csv", [dict(zone_id=zone, start_time=iso(first - 86400),
              end_time=iso(last), event_archive_complete=int(microseismic_enabled),
              rockburst_record_complete=1)], COVERAGE_COLUMNS)
    assumptions = dict(mode="simple_directory_workflow", timezone=timezone,
                       bin_format="headerless little-endian int32, sample-major interleaved channels",
                       availability="file_end_time_assumed",
                       microseismic_unit="whole_detected_clip" if microseismic_enabled else None,
                       microseismic_enabled=microseismic_enabled,
                       microseismic_archive_assumed_complete=microseismic_enabled,
                       disabled_microseismic_behavior="event features omitted; zero clips represented" if not microseismic_enabled else None,
                       rockburst_labels_assumed_complete=True,
                       continuous_files=len(continuous), microseismic_files=len(events),
                       labeled_rockbursts=len(bursts))
    write_json(root / "input_summary.json", assumptions)

    print("[2/5] 提取连续波形和微震片段特征", flush=True)
    extract(generated / "continuous_manifest.csv", root / "continuous.csv", "continuous", monitor)
    extract(generated / "microseismic_manifest.csv", root / "events.csv", "events", monitor)
    data = Inputs(root / "continuous.csv", root / "events.csv", generated / "coverage.csv",
                  zone, generated / "rockbursts.csv", microseismic_enabled=microseismic_enabled)
    forecast = Forecaster(backend, model_path, device, samples, cache or root / "cache")
    start = first + history_minutes * 60
    end = last + 1e-6
    print(f"[3/5] {backend}推理，融合两路特征并建立5/10/30分钟标签", flush=True)
    dataset_report = build_dataset(data, forecast, start, end, root / "dataset.npz", history_minutes,
                                   prediction_times=[local_time(r["available_time"], timezone) for r in continuous])
    with np.load(root / "dataset.npz", allow_pickle=False) as pack:
        times = pack["times"]
    train_end = float(times[min(int(len(times) * .7), len(times) - 1)])
    validation_end = float(times[min(int(len(times) * .85), len(times) - 1)])
    print("[4/5] 按时间70%/15%/15%划分，训练小型概率模型并测试", flush=True)
    report = train(root / "dataset.npz", root / "run", train_end, validation_end,
                   ridges=[ridge], threshold=threshold)

    print("[5/5] 保存模型、测试报告和最新一次预测", flush=True)
    from .io import read_json
    model = read_json(root / "run/model.json")
    now = last
    x, quality = data.sample(now, forecast, history_minutes)
    p = probabilities(x[None], model)[0]
    prediction = dict(status="ok", time=iso(now), zone_id=zone, p_5min=float(p[0]),
                      p_10min=float(p[1]), p_30min=float(p[2]), backend=forecast.identity,
                      experimental=True, calibration=model["calibration"], quality=quality)
    write_json(root / "prediction.json", prediction)
    summary = dict(output=str(root), backend=backend, input=assumptions,
                   samples=dataset_report["samples"], skipped=dataset_report["skipped"],
                   train_end=iso(train_end), validation_end=iso(validation_end),
                   model=str(root / "run/model.json"), metrics=str(root / "run/metrics.json"),
                   prediction=str(root / "prediction.json"),
                   split_samples={name: value["samples"] for name, value in report["splits"].items()})
    write_json(root / "summary.json", summary)
    return summary


def raw_demo(output):
    """Generate actual small DAS bins, then exercise the identical directory workflow."""
    root = Path(output)
    continuous_dir, event_dir = root / "raw/continuous", root / "raw/microseismic"
    continuous_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    fs, count = 1200, 2
    origin = local_time("2026-06-01T00:00:00", "Asia/Shanghai")
    burst_times = [origin + minute * 60 for minute in [100, 160, 220, 270, 320]]
    t = np.arange(30 * fs) / fs
    rng = np.random.default_rng(20260921)

    def save(folder, start, amplitude):
        wave = amplitude * np.sin(2 * np.pi * 120 * t)
        data = np.column_stack([wave + rng.normal(0, 10, len(t)),
                                .8 * wave + rng.normal(0, 10, len(t))]).astype("<i4")
        stamp = datetime.fromtimestamp(start, ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S.000")
        data.tofile(folder / f"demo-fs-eDAS-{fs}Hz-{count:04d}pt-{stamp}.bin")

    print("生成6小时合成连续bin及30个微震片段（约212MB），不是真实监测数据", flush=True)
    for start in range(round(origin), round(origin + 6 * 3600), 30):
        delay = min([b - start for b in burst_times if b > start] or [1e9])
        amplitude = 100 + 500 * max(0., 1 - delay / 2400)
        save(continuous_dir, start, amplitude)
    for i in range(30):
        save(event_dir, origin - 86400 + i * 3600, 150 + 10 * i)
    labels = root / "rockbursts.txt"
    labels.write_text("\n".join(iso(t) for t in burst_times) + "\n", encoding="utf-8")
    summary = run_workflow(continuous_dir, event_dir, labels, root / "result", backend="persistence")
    write_json(root / "SYNTHETIC_DEMO_ONLY.json", dict(synthetic=True, backend="persistence",
               note="实际bin读取到模型训练的主流程验证；未使用真实Sundial权重。"))
    return dict(summary, synthetic=True)
