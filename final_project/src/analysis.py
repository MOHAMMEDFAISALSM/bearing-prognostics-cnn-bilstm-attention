"""Descriptive analyses on the LEARNING bearings (health-indicator screening, degradation-phase length).
Nothing here defines a training target; it only characterises the data and is used in explanations."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def combined_rms_hi(g):
    """Descriptive health indicator: log of the combined RMS of both accelerometers."""
    return np.log(np.sqrt(g["horiz_rms"].values ** 2 + g["vert_rms"].values ** 2))


def _smooth(x, w):
    return pd.Series(x).rolling(w, min_periods=1, center=True).median().values


def degradation_onset(g, frac=0.10):
    """DESCRIPTIVE ONLY (uses the whole trajectory = hindsight). Start of the final uninterrupted excursion of the smoothed
    health indicator above baseline + frac*(failure level - baseline). Returns (onset snapshot index, smoothed HI, threshold)."""
    hi = combined_rms_hi(g); h = _smooth(hi, 15); n = len(h)
    base = np.median(np.sort(h)[:int(0.3 * n)]); end = _smooth(hi, 5).max()
    thr = base + frac * (end - base)
    below = np.where(h < thr)[0]
    return int(min(below[-1] + 1 if len(below) else 0, n - 2)), h, thr


def phase_table(table, bearings, fracs=(0.05, 0.10, 0.20)):
    rows = []
    for b in bearings:
        g = table[table.bearing == b].sort_values("snapshot_idx"); n = len(g)
        r = {"bearing": b, "lifetime_min": round((n - 1) * 10 / 60, 1)}
        for f in fracs:
            on, _, _ = degradation_onset(g, f)
            r[f"phase_min@{int(f*100)}%"] = round((n - 1 - on) * 10 / 60, 1)
        rows.append(r)
    return pd.DataFrame(rows).set_index("bearing")


def feature_screening(table, feats, bearings):
    """Per-feature descriptive statistics on the given (learning) bearings:
    trend  = mean over bearings of Spearman(feature, time)
    consistency = fraction of bearings whose |Spearman| > 0.5 with the same sign as the mean trend
    late_shift = median over bearings of z-shift of the feature in the last 60 min vs the first 30 % of life (robust z)."""
    out = []
    for f in feats.columns:
        rhos, shifts = [], []
        for b in bearings:
            m = (table.bearing == b).values
            x = feats.loc[m, f].values; t = table.loc[m, "time_min"].values; rul = table.loc[m, "rul_min"].values
            rhos.append(spearmanr(x, t)[0])
            base = x[: max(10, int(0.3 * len(x)))]; mad = 1.4826 * np.median(np.abs(base - np.median(base))) + 1e-9
            late = x[rul <= 60]
            shifts.append((np.median(late) - np.median(base)) / mad if len(late) else np.nan)
        rhos = np.array(rhos); s = np.sign(np.nanmean(rhos))
        out.append({"feature": f, "trend_spearman": float(np.nanmean(rhos)), "consistency": float(np.mean((np.abs(rhos) > 0.5) & (np.sign(rhos) == s))),
                    "late_shift_z": float(np.nanmedian(shifts))})
    return pd.DataFrame(out).set_index("feature")
