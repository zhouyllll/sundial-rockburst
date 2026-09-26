# -*- coding: utf-8 -*-
"""重跑消融（event-balanced + 三分类告警），输出 experiment_05。"""
import subprocess, sys, time
from pathlib import Path

REPO = Path(r"G:\1A岩爆预测\sundial-rockburst")
OUT = Path(r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_05_balanced_abl")
GUARD = REPO / "guard_abl5.log"
RUN_LOG = REPO / "train_run8.log"
ERR_LOG = REPO / "train_err8.log"

cmd = [
    sys.executable, "-u", "-m", "rockburst", "run",
    "--continuous-dir", r"G:\1A岩爆预测\岩爆数据集",
    "--labels", r"G:\1A岩爆预测\rockbursts_full.csv",
    "--model-path", str(REPO / "models" / "sundial-base-128m"),
    "--monitor", "16-35",
    "--device", "cuda",
    "--transient-ablation",
    "--event-balanced",
    "--output", str(OUT),
]

def log(msg):
    with GUARD.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

log(f"schtask 守护启动: {' '.join(cmd)}")
stdout = open(RUN_LOG, "wb", buffering=0)
stderr = open(ERR_LOG, "wb", buffering=0)
proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=stdout, stderr=stderr)

last_tmp = -1
while True:
    time.sleep(30)
    rc = proc.poll()
    alive = rc is None
    tmp = OUT / "continuous.csv.tmp"
    tmp_size = tmp.stat().st_size if tmp.exists() else -1
    delta = tmp_size - last_tmp
    last_tmp = tmp_size
    log(f"alive={alive} rc={rc} tmp={tmp_size}B d30s={delta}")
    if not alive:
        break

log(f"训练进程退出 rc={proc.returncode} 时间={time.strftime('%Y-%m-%d %H:%M:%S')}")
log("--- RUN_LOG 尾部 ---")
for line in list(RUN_LOG.read_bytes().decode("utf-8", "replace").splitlines())[-20:]:
    log(line)
log("--- ERR_LOG 尾部 ---")
for line in list(ERR_LOG.read_bytes().decode("utf-8", "replace").splitlines())[-20:]:
    log(line)
log("守护结束")
