"""Explicit UTC timestamps and validated, portable CSV interfaces."""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re


def local_path(value):
    """Accept native paths and translate Windows drive paths inside WSL."""
    text = str(value)
    match = re.match(r"^([A-Za-z]):[\\/]?(.*)$", text)
    if match and os.name != "nt":
        tail = match[2].replace("\\", "/")
        return Path("/mnt") / match[1].lower() / tail
    return Path(text)


def timestamp(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"时间必须带时区，例如 2026-06-01T12:00:00+08:00: {value}")
    return dt.timestamp()


def iso(value):
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()


def read_csv(path, required):
    with local_path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")
        return list(reader)


def write_csv(path, rows, columns):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a failed run from leaving a half-written table.
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))
