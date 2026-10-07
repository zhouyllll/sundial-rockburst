"""P0 改动数值自检：cloglog 梯度、Platt 校准、端到端 train() 冒烟测试。"""
import json
import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit

from rockburst.model import (fit_hazard, fit_platt_calibration, probabilities,
                             expected_calibration_error, train, _raw_probabilities)

rng = np.random.default_rng(7)
N, F = 400, 6

# 1) cloglog 梯度 vs 数值梯度
x = rng.normal(size=(N, 3, F))
y = (rng.random((N, 3)) < 0.2).astype(float)
mask = np.ones((N, 3), dtype=bool)
weights = np.ones(N)
model = fit_hazard(x, y, mask, 1.0, weights, link="cloglog", l1_ratio=0.3)

def fun(theta):
    beta = theta[3:]
    logits = (x - model["mean"]) / model["scale"] @ beta + theta[:3]
    u = np.exp(logits)
    h = 1 - np.exp(-u)
    hc = np.clip(h, 1e-12, 1 - 1e-12)
    w = mask.astype(float) * weights[:, None] / np.sum(weights)
    loss = np.sum(w * (-(y * np.log(hc) + (1 - y) * np.log1p(-hc))))
    s = np.sqrt(beta ** 2 + 1e-8)
    return loss + 0.5 * 1.0 * 0.7 * np.sum(beta ** 2) + 1.0 * 0.3 * np.sum(s)

theta0 = np.r_[model["intercepts"], model["beta"]]

def numeric_grad(f, t, h=1e-6):
    g = np.zeros_like(t)
    for k in range(len(t)):
        tp = t.copy(); tp[k] += h
        tm = t.copy(); tm[k] -= h
        g[k] = (f(tp) - f(tm)) / (2 * h)
    return g

def analytic_grad(theta):
    beta = theta[3:]
    z = (x - model["mean"]) / model["scale"]
    logits = z @ beta + theta[:3]
    u = np.exp(logits)
    h = 1 - np.exp(-u)
    dlam = u * np.exp(-u)
    w = mask.astype(float) * weights[:, None] / np.sum(weights)
    residual = w * dlam * ((1 - y) / np.maximum(1 - h, 1e-12) - y / np.maximum(h, 1e-12))
    grad = np.r_[residual.sum(axis=0), np.einsum("nj,njk->k", residual, z)]
    s = np.sqrt(beta ** 2 + 1e-8)
    grad = np.r_[grad[:3], grad[3:] + 1.0 * (0.7 * beta + 0.3 * beta / s)]
    return grad

ag = analytic_grad(theta0)
ng = numeric_grad(fun, theta0)
print("cloglog+l1 grad max abs diff:", float(np.max(np.abs(ag - ng))))

# 2) Platt 校准恢复性
p0 = rng.uniform(0.01, 0.95, 500)
y0 = (rng.random(500) < p0).astype(float)
# 人为制造压缩：把 logit 压成一半
z0 = np.log(p0 / (1 - p0))
p_compressed = expit(0.5 * z0 + 0.3)
cal = fit_platt_calibration(y0, p_compressed)
print("platt A/B:", cal["A"], cal["B"])

# 3) ECE 简单测试
print("ece:", expected_calibration_error(np.array([1, 0, 1, 0]), np.array([0.9, 0.4, 0.3, 0.1])))

# 4) 端到端 train() 冒烟：合成 dataset.npz
times = np.arange(N) * 30.0
groups = np.array([""] * N)
event_ids = np.array([""] * N)
event_ids[10:15] = "evA"
event_ids[60:65] = "evB"
event_ids[150:155] = "evC"
event_ids[300:305] = "evD"
event_onsets = np.full(N, np.nan)
for ev, t0 in [("evA", 12), ("evB", 62), ("evC", 152), ("evD", 302)]:
    event_onsets[t0] = times[t0]
meta = dict(refresh_seconds=30.0, forecast="persistence",
            feature_names=[f"f{i}" for i in range(F)], history_minutes=30, max_lag_seconds=60,
            zone_id="zone_A", preprocessing_id="test")
np.savez("_p0_test_dataset.npz", x=x, y=y, mask=mask, times=times, event_ids=event_ids,
         groups=groups, event_onsets=event_onsets,
         metadata=np.asarray(json.dumps(meta)))
train_end = times[200]
validation_end = times[300]
rep = train("_p0_test_dataset.npz", "_p0_test_out", train_end, validation_end,
            ridges=(0.3, 1.0, 3.0), threshold=0.5, event_balanced=True,
            auto_threshold=True, min_active_bins=2, links=("logit", "cloglog"),
            l1_ratios=(0.0, 0.5), selection="nll", calibrate=True)
print("train OK: link", rep["selected_link"], "ridge", rep["selected_ridge"],
      "l1", rep["selected_l1_ratio"])
print("calibration:", json.dumps(rep["calibration"], default=float))
print("threshold:", rep.get("threshold"), "sel:", rep.get("threshold_selection", {}).get("status"))
for split in ("validation", "test"):
    h = rep["splits"][split]["horizons"]["30min"]
    print(split, "nll", round(h["log_loss"], 4), "maxp", round(h["max_probability"], 4),
          "ece", h["ece"], "recall", h["event_recall"], "isolated_fa",
          h["isolated_false_alarm_episodes"], "time_frac", round(h["alarm_time_fraction"], 3))
