"""Nine-parameter shared-slope discrete-time hazard model and temporal holdout."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .data import FEATURE_NAMES
from .io import iso, write_csv, write_json


def probabilities(x, model):
    x = np.asarray(x, dtype=np.float64)
    beta = np.asarray(model.get("beta", []), dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (3, len(beta)) or not len(beta) or not np.isfinite(x).all():
        raise ValueError("概率模型输入必须为 [N,3,F]，且F须与模型特征数相同")
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = z @ np.asarray(model["beta"]) + np.asarray(model["intercepts"])
    return 1 - np.cumprod(1 - expit(logits), axis=1)


def fit_hazard(x, y, mask, ridge, sample_weights=None):
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge 必须是有限正数")
    for j in range(3):
        eligible = y[mask[:, j], j]
        if not len(eligible) or len(np.unique(eligible)) != 2:
            raise ValueError(f"训练集区间{j+1}缺少正例或负例；请增加独立事件和正常时段")
    mean = x[mask].mean(axis=0)
    scale = x[mask].std(axis=0)
    scale[scale < 1e-8] = 1.
    z = (x - mean) / scale
    initial = np.zeros(3 + x.shape[-1])
    for j in range(3):
        rate = y[mask[:, j], j].mean()
        initial[j] = np.log(rate / (1 - rate))
    if sample_weights is None:
        sample_weights = np.ones(len(x), dtype=np.float64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if sample_weights.shape != (len(x),) or not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
        raise ValueError("sample_weights必须是长度等于样本数的有限正数")
    weight = mask.astype(float) * sample_weights[:, None] / np.sum(sample_weights)

    def objective(theta):
        intercepts, beta = theta[:3], theta[3:]
        logits = z @ beta + intercepts
        loss = np.sum(weight * (np.logaddexp(0., logits) - y * logits)) + .5 * ridge * np.sum(beta ** 2)
        residual = weight * (expit(logits) - y)
        grad = np.r_[residual.sum(axis=0), np.einsum("nj,njk->k", residual, z) + ridge * beta]
        return float(loss), grad

    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": 1000, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"优化未收敛: {result.message}")
    return dict(mean=mean.tolist(), scale=scale.tolist(), intercepts=result.x[:3].tolist(),
                beta=result.x[3:].tolist(), ridge=float(ridge), objective=float(result.fun),
                iterations=int(result.nit))


def event_balanced_weights(y, event_ids):
    """Give each positive event equal total weight while retaining all samples."""
    event_ids = np.asarray(event_ids)
    weights = np.ones(len(y), dtype=np.float64)
    positive = np.any(np.asarray(y, dtype=bool), axis=1)
    events = sorted({str(v) for v in event_ids[positive] if str(v)})
    if not events:
        return weights, 0
    negative_count = max(int(np.sum(~positive)), 1)
    per_event = negative_count / len(events)
    for event in events:
        members = positive & (event_ids.astype(str) == event)
        weights[members] = per_event / max(int(np.sum(members)), 1)
    return weights, len(events)


def average_precision(y, p):
    if not np.any(y):
        return None
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y)[ends]
    recall = tp / y.sum()
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def threshold_sweep(y, scores, times, event_ids, refresh_seconds=60., thresholds=None):
    """Return event-aware alarm diagnostics without changing the fitted probabilities."""
    if thresholds is None:
        thresholds = (.05, .10, .15, .20, .25, .30, .40, .50, .60, .70, .80, .90)
    y = np.asarray(y, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    times = np.asarray(times, dtype=float)
    event_ids = np.asarray(event_ids)
    rows = []
    for threshold in thresholds:
        active = scores >= threshold
        starts = np.flatnonzero(active & np.r_[True, (~active[:-1]) |
                          ~np.isclose(np.diff(times), refresh_seconds, atol=.02)])
        eligible = {str(v) for v in event_ids[y] if v}
        detected = {str(event_ids[k]) for k in starts if y[k] and event_ids[k]}
        false_alarms = sum(not y[k] for k in starts)
        rows.append(dict(threshold=float(threshold), event_recall=(len(detected) / len(eligible)
                    if eligible else None), detected_events=len(detected), eligible_events=len(eligible),
                    alarm_episodes=len(starts), false_alarm_episodes=int(false_alarms),
                    false_alarms_per_24h_evaluated=float(false_alarms /
                        (len(times) * refresh_seconds / 86400)) if len(times) else None,
                    alarm_time_fraction=float(active.mean()) if len(active) else None))
    return rows


def evaluate(x, y, mask, times, event_ids, model, threshold, refresh_seconds=60.):
    p = probabilities(x, model)
    truth = np.cumsum(y, axis=1) > 0
    results = {}
    for j, horizon in enumerate([5, 10, 30]):
        target, scores = truth[:, j], p[:, j]
        active = scores >= threshold
        starts = np.flatnonzero(active & np.r_[True, (~active[:-1]) |
                    ~np.isclose(np.diff(times), refresh_seconds, atol=.02)])
        matched = {str(event_ids[k]) for k in starts if target[k] and event_ids[k]}
        eligible_events = {str(v) for v in event_ids[target] if v}
        false_alarms = sum(not target[k] for k in starts)
        clipped = np.clip(scores, 1e-12, 1 - 1e-12)
        results[f"{horizon}min"] = dict(
            brier=float(np.mean((scores - target) ** 2)),
            log_loss=float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
            average_precision=average_precision(target, scores),
            positive_samples=int(target.sum()), mean_probability=float(scores.mean()),
            alarm_time_fraction=float(active.mean()), alarm_episodes=len(starts),
            false_alarm_episodes=int(false_alarms),
            false_alarms_per_24h_evaluated=float(false_alarms / (len(times) * refresh_seconds / 86400)),
            eligible_events=len(eligible_events), detected_events=len(matched),
            event_recall=len(matched) / len(eligible_events) if eligible_events else None,
            max_probability=float(np.max(scores)) if len(scores) else None,
            positive_max_probability=float(np.max(scores[target])) if np.any(target) else None,
            negative_max_probability=float(np.max(scores[~target])) if np.any(~target) else None,
            threshold_sweep=threshold_sweep(target, scores, times, event_ids, refresh_seconds),
        )
    return dict(samples=len(x), evaluated_hours=len(x) * refresh_seconds / 3600,
                nominal_step_seconds=refresh_seconds,
                threshold=threshold, horizons=results), p


def train(dataset, output, train_end, validation_end, ridges=(.1, 1., 10.), threshold=.5,
          feature_indices=None, feature_variant=None, event_balanced=False):
    if train_end >= validation_end or not 0 < threshold < 1:
        raise ValueError("划分时间或报警阈值无效")
    with np.load(dataset, allow_pickle=False) as pack:
        x, y, mask, times = (pack[k] for k in ["x", "y", "mask", "times"])
        event_ids, groups = pack["event_ids"], pack["groups"]
        metadata = json.loads(str(pack["metadata"]))
    refresh = metadata.get("refresh_seconds", 60.)
    dataset_feature_names = list(metadata["feature_names"])
    if x.ndim != 3 or x.shape[1] != 3 or x.shape[2] != len(dataset_feature_names):
        raise ValueError("数据集特征名与数组维度不匹配")
    if feature_indices is None:
        feature_indices = list(range(len(dataset_feature_names)))
    feature_indices = list(feature_indices)
    if not feature_indices or len(set(feature_indices)) != len(feature_indices) or any(
            not isinstance(i, (int, np.integer)) or i < 0 or i >= len(dataset_feature_names)
            for i in feature_indices):
        raise ValueError("特征选择索引无效")
    selected_features = [dataset_feature_names[i] for i in feature_indices]
    x = x[:, :, feature_indices]
    model_metadata = dict(metadata, feature_names=selected_features)
    if not np.all(np.diff(times) > 0):
        raise ValueError("数据集时刻必须严格递增")
    splits = dict(train=times + 1800 <= train_end,
                  validation=(times >= train_end) & (times + 1800 <= validation_end),
                  test=times >= validation_end)
    if any(not np.any(v) for v in splits.values()):
        raise ValueError("训练/验证/测试至少一个为空；边界前30分钟标签会被剔除")
    group_sets = {k: set(groups[v]) - {""} for k, v in splits.items()}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if group_sets[a] & group_sets[b]:
                raise ValueError(f"同一岩爆群跨越 {a}/{b}；请移动划分边界")
    tr, va, te = (splits[k] for k in ["train", "validation", "test"])
    if not np.any(y[va]):
        raise ValueError("验证集没有岩爆正例，无法选择模型；请调整时间边界")
    sample_weights, balanced_events = event_balanced_weights(y[tr], event_ids[tr]) if event_balanced else (
        np.ones(np.sum(tr), dtype=np.float64), 0)
    candidates = []
    for ridge in ridges:
        model = fit_hazard(x[tr], y[tr], mask[tr], float(ridge), sample_weights)
        validation, _ = evaluate(x[va], y[va], mask[va], times[va], event_ids[va], model, threshold, refresh)
        score = np.mean([v["brier"] for v in validation["horizons"].values()])
        candidates.append((float(score), model))
    _, model = min(candidates, key=lambda item: item[0])
    # Do not refit on validation: the final test evaluates exactly the selected fit.
    model.update(schema=1, metadata=model_metadata, threshold=float(threshold),
                 calibration="not_independently_calibrated", train_end=iso(train_end),
                 validation_end=iso(validation_end),
                 feature_variant=feature_variant,
                 event_balanced=bool(event_balanced), balanced_training_events=int(balanced_events),
                 parameter_count=3 + len(selected_features), experimental=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model.json", model)
    report = dict(backend=metadata["forecast"], calibration=model["calibration"],
                  feature_variant=feature_variant, feature_names=selected_features,
                  event_balanced=bool(event_balanced), balanced_training_events=int(balanced_events),
                  selected_ridge=model["ridge"], candidates=[dict(ridge=m["ridge"], validation_brier=s)
                                                            for s, m in candidates],
                  purged_samples=int(np.sum(~(tr | va | te))), splits={},
                  alarm_definition="threshold crossing; consecutive active samples merge; a new alarm must precede the next event within its horizon",
                  limitation="实验性结果；相邻样本相关。未进行独立概率校准或置信区间估计。测试只评估有完整输入的时刻；时长按名义步长估算。")
    for name, selected in splits.items():
        metrics, p = evaluate(x[selected], y[selected], mask[selected], times[selected],
                              event_ids[selected], model, threshold, refresh)
        metrics.update(independent_events=len(set(event_ids[selected]) - {""}),
                       independent_groups=len(group_sets[name]),
                       start=iso(times[selected][0]), end=iso(times[selected][-1]))
        report["splits"][name] = metrics
        target = np.cumsum(y[selected], axis=1).astype(int)
        rows = [dict(time=iso(t), p_5min=float(pr[0]), p_10min=float(pr[1]), p_30min=float(pr[2]),
                     y_5min=int(truth[0]), y_10min=int(truth[1]), y_30min=int(truth[2]), event_id=str(eid))
                for t, pr, truth, eid in zip(times[selected], p, target, event_ids[selected])]
        write_csv(output / f"{name}_predictions.csv", rows,
                  ["time", "p_5min", "p_10min", "p_30min", "y_5min", "y_10min", "y_30min", "event_id"])
    write_json(output / "metrics.json", report)
    return report
