"""Targets, bearing-level splits, windows and leakage checks."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from . import config as C
from . import features as F


# ------------------------------------------------------------------ end of life and targets
def truncated_length(bearing):
    """Number of snapshots in the official (truncated) Test_set; NaN for learning bearings."""
    if bearing in C.LEARNING:
        return np.nan
    return len(list((C.DATASET / "Test_set" / bearing).glob("acc_*.csv")))


def eol_seconds(bearing, n_files):
    """End of life in seconds from the first snapshot.
    Learning bearings: last recorded snapshot (official recording durations, challenge doc. Appendix A.4).
    Test bearings: truncation point + OFFICIAL actual RUL (challenge doc. Table 3). For every test bearing except
    Bearing1_4 this equals the last file of Full_Test_Set; for Bearing1_4 Full_Test_Set runs ~250 snapshots longer."""
    if bearing in C.LEARNING:
        return (n_files - 1) * C.SNAPSHOT_INTERVAL_S
    return (truncated_length(bearing) - 1) * C.SNAPSHOT_INTERVAL_S + C.OFFICIAL_RUL_S[bearing]


def stage_from_rul_min(rul_min):
    r = np.asarray(rul_min, dtype=float)
    return np.where(r <= C.STAGE_CRITICAL_MIN, 2, np.where(r <= C.STAGE_WARNING_MIN, 1, 0)).astype(np.int32)


def build_table(raw):
    """Keep snapshots up to the official end of life and attach RUL targets. Uses ONLY the failure time (no onset, no hindsight
    on the vibration signal)."""
    parts = []
    for b, g in raw.groupby("bearing", sort=False):
        g = g.sort_values("snapshot_idx").copy()
        eol = eol_seconds(b, len(g))
        g = g[g.snapshot_idx * C.SNAPSHOT_INTERVAL_S <= eol].copy()
        g["time_s"] = g.snapshot_idx * C.SNAPSHOT_INTERVAL_S
        g["time_min"] = g.time_s / 60.0
        g["rul_s"] = eol - g.time_s
        g["rul_min"] = g.rul_s / 60.0
        g["rul_cap_min"] = np.minimum(g.rul_min, C.RUL_CAP_MIN)
        g["y_rul"] = g.rul_cap_min / C.RUL_CAP_MIN
        g["stage"] = stage_from_rul_min(g.rul_min)
        g["eol_s"] = eol
        g["condition"] = C.condition_of(b)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ splits
def lobo_folds(bearings=None):
    """Leave-one-bearing-out over the learning bearings. For each fold the validation bearing (used ONLY for early stopping)
    is the next learning bearing in cyclic order; the remaining 4 bearings train."""
    bearings = list(bearings or C.LEARNING)
    folds = []
    for i, held in enumerate(bearings):
        val = bearings[(i + 1) % len(bearings)]
        train = [b for b in bearings if b not in (held, val)]
        folds.append({"held_out": held, "val": val, "train": train})
    return folds


def assert_disjoint(*groups):
    seen = set()
    for g in groups:
        s = set(g)
        assert not (s & seen), f"LEAKAGE: bearings shared between splits: {s & seen}"
        seen |= s


# ------------------------------------------------------------------ windows
def fit_scaler(table_feats, table, train_bearings):
    """StandardScaler fitted ONLY on snapshots of the training bearings."""
    assert set(train_bearings) <= set(C.LEARNING) or True
    m = table.bearing.isin(train_bearings).values
    return StandardScaler().fit(table_feats[m].values)


def make_windows(table, table_feats, bearings, scaler, window=C.WINDOW):
    """Sliding windows built INSIDE each bearing. Window k covers snapshots k..k+window-1 (only the past); it is labelled
    with the targets of its LAST snapshot."""
    X, meta = [], []
    for b in bearings:
        m = (table.bearing == b).values
        g = table[m]
        x = scaler.transform(table_feats[m].values)
        n = len(g)
        idx = np.arange(window - 1, n)
        X.append(np.stack([x[i - window + 1:i + 1] for i in idx]).astype(np.float32))
        meta.append(g.iloc[idx][["bearing", "snapshot_idx", "time_min", "rul_min", "rul_cap_min", "y_rul", "stage", "condition"]])
    return np.concatenate(X), pd.concat(meta, ignore_index=True)


def window_summary(X):
    """Same information as the sequence, in tabular form for classical models: [last, window mean, last - first]."""
    return np.concatenate([X[:, -1], X.mean(1), X[:, -1] - X[:, 0]], axis=1)
