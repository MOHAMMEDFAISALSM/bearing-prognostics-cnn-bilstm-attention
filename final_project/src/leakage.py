"""Automated leakage checks. Every function raises AssertionError on violation, so the notebook stops if leakage is detected."""
import numpy as np
import pandas as pd
from . import config as C
from . import features as F
from . import data as D


def check_splits(folds):
    for f in folds:
        D.assert_disjoint(f["train"], [f["val"]], [f["held_out"]])
        assert set(f["train"]) | {f["val"], f["held_out"]} <= set(C.LEARNING), "a fold touches a non-learning bearing"
        assert not (set(f["train"]) | {f["val"], f["held_out"]}) & set(C.TEST), "TEST bearing inside LOBO fold"
    held = [f["held_out"] for f in folds]
    assert sorted(held) == sorted(C.LEARNING), "LOBO must hold out every learning bearing exactly once"
    return True


def check_scaler_train_only(scaler, feats, table, train_bearings):
    m = table.bearing.isin(train_bearings).values
    assert np.allclose(scaler.mean_, feats.values[m].mean(0), rtol=1e-6, atol=1e-9), "scaler not equal to training-bearing statistics"
    others = ~m
    assert not np.allclose(scaler.mean_, feats.values.mean(0), rtol=1e-6, atol=1e-9) or others.sum() == 0, "scaler equals all-data statistics"
    return True


def check_poison(table, feats, fold, make_windows=D.make_windows):
    """Replace the held-out and validation... held-out bearing's features by garbage: everything derived for TRAINING must be bit-identical."""
    hold = fold["held_out"]
    sc = D.fit_scaler(feats, table, fold["train"])
    X0, m0 = make_windows(table, feats, fold["train"], sc)
    f2 = feats.copy(); mk = (table.bearing == hold).values
    f2.loc[mk, :] = np.random.default_rng(1).normal(size=(mk.sum(), f2.shape[1])) * 1e3
    t2 = table.copy(); t2.loc[mk, ["y_rul", "stage", "rul_min", "rul_cap_min"]] = 0
    sc2 = D.fit_scaler(f2, t2, fold["train"])
    X1, m1 = make_windows(t2, f2, fold["train"], sc2)
    assert np.array_equal(X0, X1) and np.array_equal(sc.mean_, sc2.mean_) and m0.equals(m1), "held-out bearing influences training data"
    return True


def check_window_causality(table, feats, scaler, bearing, cuts):
    """Delete every snapshot after `cut`: windows and targets that exist in both versions must be identical."""
    Xf, mf = D.make_windows(table, feats, [bearing], scaler)
    for cut in cuts:
        keep = ~((table.bearing == bearing) & (table.snapshot_idx >= cut)).values
        Xc, mc = D.make_windows(table[keep].reset_index(drop=True), feats[keep].reset_index(drop=True), [bearing], scaler)
        assert len(Xc) == cut - C.WINDOW + 1 and np.array_equal(Xc, Xf[:len(Xc)]), f"future data changed a past window ({bearing}, cut {cut})"
        assert np.array_equal(mc.snapshot_idx.values, mf.snapshot_idx.values[:len(mc)])
    return True


def check_window_boundaries(X, meta):
    for b, g in meta.groupby("bearing"):
        assert (np.diff(g.snapshot_idx.values) == 1).all(), f"non-contiguous windows in {b}"
        assert g.snapshot_idx.min() == C.WINDOW - 1, "first window must end at snapshot WINDOW-1 (no padding across bearings)"
    return True


def check_features_single_snapshot(raw, n_per_bearing=2, seed=0):
    """Independent re-implementation of three features from the raw CSV of ONE file; must match the stored features, i.e. the
    stored features use no information from any other snapshot."""
    rng = np.random.default_rng(seed)
    for b in C.LEARNING + C.TEST:
        n = (raw.bearing == b).sum()
        for i in rng.choice(n, n_per_bearing, replace=False):
            path = C.DATASET / C.subset_of(b) / b / raw.loc[(raw.bearing == b) & (raw.snapshot_idx == i), "file"].iloc[0]
            arr, _ = F.read_snapshot(path)
            for col, prefix in ((4, "horiz"), (5, "vert")):
                x = arr[:, col]; c = x - x.mean()
                rms = np.sqrt((x ** 2).mean()); kurt = (c ** 4).mean() / ((c ** 2).mean() + 1e-12) ** 2 - 3
                P = (np.abs(np.fft.rfft(c)) / len(x)) ** 2; fr = np.fft.rfftfreq(len(x), 1 / C.FS); cen = (fr * P).sum() / (P.sum() + 1e-12)
                row = raw[(raw.bearing == b) & (raw.snapshot_idx == i)].iloc[0]
                assert np.isclose(rms, row[f"{prefix}_rms"], rtol=1e-6) and np.isclose(kurt, row[f"{prefix}_kurtosis"], rtol=1e-5, atol=1e-6) \
                    and np.isclose(cen, row[f"{prefix}_spec_centroid"], rtol=1e-6), f"feature mismatch {b} {i}"
    return True


def check_targets_only_from_failure_time(raw, table):
    """Targets must not depend on vibration content: rebuild the table from a raw frame whose features are shuffled."""
    shuf = raw.copy()
    cols = [c for c in F.FEATURES] + ["max_abs_h", "max_abs_v"]
    shuf[cols] = shuf[cols].sample(frac=1.0, random_state=0).values
    t2 = D.build_table(shuf)
    for c in ("rul_s", "rul_min", "rul_cap_min", "y_rul", "stage"):
        assert np.array_equal(table[c].values, t2[c].values), f"target {c} depends on vibration features"
    return True


def check_config_frozen(snapshot, current):
    assert snapshot == current, f"configuration changed after freeze: {set(snapshot.items()) ^ set(current.items())}"
    return True


def config_snapshot():
    keys = ["WINDOW", "RUL_CAP_MIN", "STAGE_CRITICAL_MIN", "STAGE_WARNING_MIN", "ALARM_PERSISTENCE", "BATCH", "LR", "MAX_EPOCHS",
            "PATIENCE", "HUBER_DELTA"]
    d = {k: getattr(C, k) for k in keys}; d["LOSS_WEIGHTS"] = tuple(sorted(C.LOSS_WEIGHTS.items()))
    d["SEEDS_FINAL"] = tuple(C.SEEDS_FINAL); d["SEED_LOBO"] = C.SEED_LOBO
    return d
