# -*- coding: utf-8 -*-
"""并行特征提取功能验证：合成 4 个 bin（35/36/41/42pt 混合）跑 extract，检查行数、签名、特征。"""
import io, sys, json, hashlib, tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

sys.path.insert(0, r"G:\1A岩爆预测\sundial-rockburst")

def main():
    from rockburst.extract import extract
    work = Path(tempfile.mkdtemp(prefix="rb_verify_"))
    fs = 5000
    # 每个文件 30 秒，不同通道数
    for ch, seq in [(35, 100), (36, 101), (41, 102), (42, 103)]:
        t0 = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc).timestamp() + seq * 30
        for i in range(2):
            start = t0 + i * 30
            n = fs * 30
            data = (np.random.default_rng(i + ch).normal(0, 100, (n, ch))).astype("<i4")
            stamp = datetime.fromtimestamp(start, timezone.utc).strftime("%Y%m%dT%H%M%S.000")
            (work / f"0000{seq}-fs-eDAS-{fs}Hz-{ch:04d}pt-{stamp}.bin").write_bytes(data.tobytes())
    # 手写 manifest
    manifest = work / "manifest.csv"
    rows = ["file_path,zone_id,start_time,available_time,fs,channels"]
    for p in sorted(work.glob("*.bin")):
        size = p.stat().st_size
        ch = int(p.name.split("-")[4].replace("pt", ""))
        stamp = p.name.split("-")[-1].replace(".bin", "")
        dt = datetime.strptime(stamp, "%Y%m%dT%H%M%S.%f").replace(tzinfo=timezone.utc)
        dur = size / (4 * ch * fs)
        rows.append(f"{p.as_posix()},zone_A,{dt.isoformat()},{datetime.fromtimestamp(dt.timestamp()+dur,timezone.utc).isoformat()},{fs},{ch}")
    manifest.write_text("\n".join(rows), encoding="utf-8")
    out = work / "continuous.csv"
    report = extract(manifest, out, "continuous")
    print("报告:", report)
    import csv
    with open(out, encoding="utf-8-sig") as f:
        recs = list(csv.DictReader(f))
    print("行数:", len(recs), "（预期 8：4文件×2×3cell）")
    sigs = {r["preprocessing_id"] for r in recs}
    print("签名数:", len(sigs), "签名:", sigs)
    print("样例行:", {k: recs[0][k] for k in ["zone_id", "start_time", "end_time", "preprocessing_id", "rms"]})
    assert len(recs) == 8, "行数不对"
    assert len(sigs) == 1, "签名未统一"
    assert all(float(r["rms"]) > 0 for r in recs), "特征异常"
    print("并行特征提取验证通过 ✓")

if __name__ == "__main__":
    main()
