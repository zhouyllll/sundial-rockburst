# -*- coding: utf-8 -*-
"""attach 版守护：用 ctypes 监控已有训练进程 PID，30秒心跳，退出时记录退出码与日志尾部。"""
import ctypes, sys, time
from ctypes import wintypes
from pathlib import Path

pid = int(sys.argv[1])
REPO = Path(r"G:\1A岩爆预测\sundial-rockburst")
OUT = Path(r"G:\1A岩爆预测\sundial-rockburst\artifacts\experiment_01")
GUARD = REPO / "guard.log"
RUN_LOG = REPO / "train_run3.log"
ERR_LOG = REPO / "train_err3.log"

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259

def log(msg):
    with GUARD.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def get_exit_code(handle):
    code = wintypes.DWORD()
    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
        return None
    return code.value

def get_cpu_mem():
    try:
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return -1, -1
        t = wintypes.FILETIME(); k = wintypes.FILETIME(); u = wintypes.FILETIME()
        kernel32.GetProcessTimes(h, t, t, k, u)
        cpu = (k.dwHighDateTime << 32 | k.dwLowDateTime) / 1e7 + (u.dwHighDateTime << 32 | u.dwLowDateTime) / 1e7
        class PMC(wintypes._memoize.wintypes if False else ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        ok = ctypes.WinDLL("psapi").GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        mem = pmc.WorkingSetSize / 1024 / 1024 if ok else -1
        kernel32.CloseHandle(h)
        return round(cpu, 1), round(mem, 0)
    except Exception as e:
        return -1, -1

log(f"attach 守护启动，监控 PID={pid}")
h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
if not h:
    log(f"无法打开进程 {pid}（可能已退出）")
    sys.exit(1)

last_tmp = -1
while True:
    time.sleep(30)
    code = get_exit_code(h)
    alive = (code is None) or (code == STILL_ACTIVE)
    cpu, mem = get_cpu_mem()
    tmp = OUT / "continuous.csv.tmp"
    tmp_size = tmp.stat().st_size if tmp.exists() else -1
    delta = tmp_size - last_tmp
    last_tmp = tmp_size
    log(f"alive={alive} exit={code} cpu={cpu}s mem={mem}MB tmp={tmp_size}B d30s={delta}")
    if not alive:
        break

log(f"训练进程退出，exit_code={code}，时间={time.strftime('%Y-%m-%d %H:%M:%S')}")
log("--- RUN_LOG 尾部 ---")
for line in list(RUN_LOG.read_bytes().decode("utf-8", "replace").splitlines())[-15:]:
    log(line)
log("--- ERR_LOG 尾部 ---")
for line in list(ERR_LOG.read_bytes().decode("utf-8", "replace").splitlines())[-15:]:
    log(line)
log("守护结束")
