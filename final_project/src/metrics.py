"""Evaluation metrics. Every metric is defined here once and used identically for all models."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from . import config as C
from .data import stage_from_rul_min


def phm_score_single(act_s, pred_s):
    """Official IEEE PHM 2012 score for ONE bearing (challenge document, Eq. 1-2). Er in percent."""
    er = 100.0 * (act_s - pred_s) / act_s
    return float(np.exp(-np.log(0.5) * (er / 5.0)) if er <= 0 else np.exp(np.log(0.5) * (er / 20.0)))


def phm_score(act_s, pred_s):
    """Official score = mean over the 11 test bearings (Eq. 3). Inputs are per-bearing arrays at the truncation point."""
    return float(np.mean([phm_score_single(a, p) for a, p in zip(act_s, pred_s)]))


def sustained(flag, k=C.ALARM_PERSISTENCE):
    f = pd.Series(np.asarray(flag).astype(int))
    return (f.rolling(k, min_periods=k).sum() == k).values


def regression_metrics(meta, pred_min):
    """MAE/RMSE in minutes against the CAPPED true RUL, pooled, per bearing, and per true stage."""
    err = pred_min - meta.rul_cap_min.values
    out = {"MAE_min": float(np.abs(err).mean()), "RMSE_min": float(np.sqrt((err ** 2).mean()))}
    pb = pd.DataFrame({"b": meta.bearing.values, "ae": np.abs(err)}).groupby("b").ae.mean()
    out["MAE_min_per_bearing_mean"] = float(pb.mean())
    for s, name in enumerate(("Normal", "Warning", "Critical")):
        m = meta.stage.values == s
        out[f"MAE_min_{name}"] = float(np.abs(err[m]).mean()) if m.any() else np.nan
    act = meta.rul_min.values <= C.STAGE_WARNING_MIN                       # actionable region: failure within the warning horizon
    out["MAE_min_actionable"] = float(np.abs(err[act]).mean()) if act.any() else np.nan
    return out


HORIZONS_MIN = [5, 10, 20, 30, 60]


def detection_metrics(meta, score):
    """Binary detection of 'failure within H minutes' (positive = true RUL <= H). `score` is larger for 'more degraded'.
    AUC_detect = AUC at the warning horizon (60 min); AUC_h{H} for the other horizons; AUC_mean_h = mean over horizons."""
    out = {}
    for h in HORIZONS_MIN:
        y = (meta.rul_min.values <= h).astype(int)
        out[f"AUC_h{h}"] = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else np.nan
    out["AUC_detect"] = out[f"AUC_h{int(C.STAGE_WARNING_MIN)}"]
    out["AUC_mean_h"] = float(np.nanmean([out[f"AUC_h{h}"] for h in HORIZONS_MIN]))
    return out


def stage_metrics(meta, pred_stage, prefix="stage"):
    y = meta.stage.values
    p, r, f, _ = precision_recall_fscore_support(y, pred_stage, labels=[0, 1, 2], zero_division=0)
    out = {f"{prefix}_acc": float(accuracy_score(y, pred_stage)), f"{prefix}_macroF1": float(f1_score(y, pred_stage, average="macro", labels=[0, 1, 2], zero_division=0))}
    for i, n in enumerate(("Normal", "Warning", "Critical")):
        out[f"{prefix}_recall_{n}"] = float(r[i]); out[f"{prefix}_precision_{n}"] = float(p[i])
    return out


def alarm_metrics(meta, pred_stage):
    """Alarm = predicted stage >= Warning for ALARM_PERSISTENCE consecutive windows (same rule for every model).
    false_alarm_rate      fraction of truly-Normal windows (RUL > warning horizon) that lie in an alarm.
    detected_bearings     an alarm occurs while the true RUL <= warning horizon (lenient: an alarm already on before the zone counts).
    lead_min_median       RUL at that first in-zone alarm (saturates at the warning horizon when alarms are on early).
    alarm_at_end_bearings the alarm is ACTIVE at the last window before failure (strict).
    lead_sustained_min_median  RUL at the start of the FINAL uninterrupted alarm run that lasts until the end (capped at the RUL cap);
                          NaN when the alarm is not active at the end. A run that started long before the warning zone is
                          partly a false alarm, which is why false_alarm_rate must always be read together with it."""
    fa, det, lead, at_end, lead_s = [], [], [], [], []
    for b in meta.bearing.unique():
        mk = (meta.bearing == b).values
        alarm = sustained(pred_stage[mk] >= 1)
        st = meta.stage.values[mk]; rul = meta.rul_min.values[mk]
        fa.append(alarm[st == 0].mean() if (st == 0).any() else np.nan)
        hit = np.where(alarm & (st >= 1))[0]
        det.append(bool(len(hit))); lead.append(float(rul[hit[0]]) if len(hit) else np.nan)
        at_end.append(bool(alarm[-1]))
        if alarm[-1]:
            off = np.where(~alarm)[0]
            start = (off[-1] + 1) if len(off) else 0
            lead_s.append(min(float(rul[start]), C.RUL_CAP_MIN))
        else:
            lead_s.append(np.nan)
    return {"false_alarm_rate": float(np.nanmean(fa)), "detected_bearings": int(sum(det)), "n_bearings": len(det),
            "lead_min_median": float(np.nanmedian(lead)) if any(det) else np.nan,
            "alarm_at_end_bearings": int(sum(at_end)),
            "lead_sustained_min_median": float(np.nanmedian(lead_s)) if any(at_end) else np.nan}


def per_bearing_table(meta, pred_min, pred_stage):
    rows = []
    for b in meta.bearing.unique():
        m = (meta.bearing == b).values
        mm = meta[m].reset_index(drop=True)
        r = {"bearing": b, "windows": int(m.sum())}
        r.update(regression_metrics(mm, pred_min[m]))
        r.update(stage_metrics(mm, pred_stage[m]))
        r.update(alarm_metrics(mm, pred_stage[m]))
        rows.append(r)
    return pd.DataFrame(rows).set_index("bearing")


def official_phm(meta, pred_min, table_trunc):
    """Official protocol: ONE prediction per test bearing at the truncation snapshot (last window of the truncated Test_set).
    `table_trunc` maps bearing -> truncation snapshot index. Predictions are capped RUL in minutes, converted to seconds."""
    acts, preds, rows = [], [], []
    for b, k in table_trunc.items():
        m = ((meta.bearing == b) & (meta.snapshot_idx == k - 1)).values
        assert m.sum() == 1, (b, k)
        act = C.OFFICIAL_RUL_S[b]; pr = float(pred_min[m][0]) * 60.0
        acts.append(act); preds.append(pr)
        rows.append({"bearing": b, "actual_RUL_s": act, "predicted_RUL_s": round(pr, 1), "pct_error": round(100 * (act - pr) / act, 1), "A_i": phm_score_single(act, pr)})
    return phm_score(acts, preds), pd.DataFrame(rows).set_index("bearing")


def rul_to_stage(pred_min):
    return stage_from_rul_min(pred_min)
