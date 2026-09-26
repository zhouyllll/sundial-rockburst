# -*- coding: utf-8 -*-
"""守护训练进程：30秒心跳记录，退出时捕获退出码与日志尾部。"""
import subprocess, sys, time, os
from pathlib import Path

REPO = Path(r"G:\1A岩爆预测\sundial-rockburst")
OUT = Path(r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_01")
GUARD = REPO / "guard.log"
RUN_LOG = REPO / "train_run3.log"
ERR_LOG = REPO / "train_err3.log"

cmd = [
    sys.executable, "-u", "-m", "rockburst", "run",
    "--continuous-dir", r"G:\1A岩爆预测\岩爆数据集",
    "--labels", r"G:\1A岩爆预测\rockbursts_full.csv",
    "--model-path", str(REPO / "models" / "sundial-base-128m"),
    "--device", "cuda",
    "--output", str(OUT),
]

def log(msg):
    with GUARD.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

log(f"守护启动，训练命令: {' '.join(cmd)}")
stdout = open(RUN_LOG, "wb", buffering=0)
stderr = open(ERR_LOG, "wb", buffering=0)
proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=stdout, stderr=stderr)

import psutil
last_tmp = -1
while True:
    time.sleep(30)
    alive = proc.poll() is None
    try:
        p = psutil.Process(proc.pid)
        cpu = p.cpu_percent(interval=None)
        mem = p.memory_info().rss / 1024 / 1024
        cpu_total = p.cpu_times().user + p.cpu_times().system
    except Exception:
        cpu = mem = cpu_total = -1
    tmp = OUT / "continuous.csv.tmp"
    tmp_size = tmp.stat().st_size if tmp.exists() else -1
    delta = tmp_size - last_tmp
    last_tmp = tmp_size
    log(f"alive={alive} cpu%={cpu:.0f} cpu_total={cpu_total:.1f}s mem={mem:.0f}MB tmp={tmp_size}B delta30s={delta}")
    if not alive:
        break

log(f"训练进程退出，exit_code={proc.returncode}，时间={time.strftime('%Y-%m-%d %H:%M:%S')}")
log("--- RUN_LOG 尾部 ---")
for line in list(RUN_LOG.read_bytes().decode("utf-8", "replace").splitlines())[-15:]:
    log(line)
log("--- ERR_LOG 尾部 ---")
for line in list(ERR_LOG.read_bytes().decode("utf-8", "replace").splitlines())[-15:]:
    log(line)
log("守护结束")
