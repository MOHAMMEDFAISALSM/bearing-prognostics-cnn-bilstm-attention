"""Experiment drivers: LOBO validation and the final train-on-6 / test-on-11 run. Shared by every model so the protocol is identical."""
import time
import numpy as np
import pandas as pd
from . import config as C
from . import data as D
from . import models as M
from . import metrics as Mx

ALL_MODELS = M.CLASSICAL + list(M.NN_KINDS)
LABEL = {**{k: k for k in M.CLASSICAL}, **{k: v[0] for k, v in M.NN_KINDS.items()}}


def _pred_frame(meta, out, model, seed, fold):
    df = meta.copy()
    df["pred_min"] = out["pred_min"]
    for i in range(3):
        df[f"prob{i}"] = out["prob"][:, i] if out["prob"] is not None else np.nan
    df["model"], df["seed"], df["fold"] = model, seed, fold
    return df


def run_lobo(table, feats, model_keys, seeds, keep_models=(), log=print):
    """Leave-one-bearing-out over the 6 learning bearings. Early stopping uses ONLY the fold's validation bearing."""
    folds = D.lobo_folds()
    frames, best_epochs, kept, hists = [], [], {}, {}
    t0 = time.time()
    for fi, f in enumerate(folds):
        sc = D.fit_scaler(feats, table, f["train"])
        Xtr, mtr = D.make_windows(table, feats, f["train"], sc)
        Xva, mva = D.make_windows(table, feats, [f["val"]], sc)
        Xte, mte = D.make_windows(table, feats, [f["held_out"]], sc)
        for mk in model_keys:
            for sd in ([seeds[0]] if mk in ("Constant", "Ridge") else seeds):
                tf0 = time.time()
                if mk in M.CLASSICAL:
                    out = M.Classical(mk, sd).fit(Xtr, mtr).predict(Xte); be = np.nan
                else:
                    net, be, hh = M.fit_nn(mk, Xtr, mtr, sd, Xva, mva)
                    out = M.predict_nn(net, Xte, mk); hists[(mk, sd, fi)] = hh
                    if mk in keep_models:
                        kept[(mk, sd, fi)] = (net, sc)
                frames.append(_pred_frame(mte, out, mk, sd, fi)); best_epochs.append({"model": mk, "seed": sd, "fold": fi, "best_epoch": be, "fit_predict_s": time.time() - tf0})
        log(f"  LOBO fold {fi + 1}/6 (held out {f['held_out']}) done - {time.time() - t0:.0f}s")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(best_epochs), kept, hists


def score_frame(g):
    """All metrics for one prediction frame (one model/seed/fold or one model/seed pooled). Stage & alarms use ONE uniform rule
    (thresholds on the predicted RUL) for every model; the native stage head is scored separately when it exists."""
    g = g.reset_index(drop=True)
    meta = g[["bearing", "snapshot_idx", "time_min", "rul_min", "rul_cap_min", "y_rul", "stage", "condition"]]
    pm = g.pred_min.values
    out = Mx.regression_metrics(meta, pm)
    out.update(Mx.detection_metrics(meta, C.STAGE_WARNING_MIN - pm))
    ps = Mx.rul_to_stage(pm)
    out.update(Mx.stage_metrics(meta, ps, "stage"))
    out.update(Mx.alarm_metrics(meta, ps))
    if g[["prob0", "prob1", "prob2"]].notna().all().all():
        pn = g[["prob0", "prob1", "prob2"]].values.argmax(1)
        out.update(Mx.stage_metrics(meta, pn, "native"))
        an = Mx.alarm_metrics(meta, pn)
        out.update({f"native_{k}": v for k, v in an.items()})
        out["native_AUC_detect"] = Mx.detection_metrics(meta, 1 - g.prob0.values)["AUC_detect"]
    return out


def summarize(pred, by=("model", "seed", "fold")):
    rows = []
    for key, g in pred.groupby(list(by), sort=False):
        r = dict(zip(by, key if isinstance(key, tuple) else (key,))); r.update(score_frame(g)); rows.append(r)
    return pd.DataFrame(rows)


def run_final(table, feats, model_keys, seeds, epochs_by_model, keep=("proposed", "cnn_bilstm_attn"), log=print):
    """Train on ALL 6 learning bearings (scaler on those 6 only) with fixed epochs; predict the 11 test bearings ONCE."""
    sc = D.fit_scaler(feats, table, C.LEARNING)
    Xtr, mtr = D.make_windows(table, feats, C.LEARNING, sc)
    Xte, mte = D.make_windows(table, feats, C.TEST, sc)
    frames, kept, hist = [], {}, {}
    t0 = time.time()
    for mk in model_keys:
        for sd in ([seeds[0]] if mk in ("Constant", "Ridge") else seeds):
            if mk in M.CLASSICAL:
                clf = M.Classical(mk, sd).fit(Xtr, mtr); out = clf.predict(Xte); kept[(mk, sd)] = clf
            else:
                net, _, h = M.fit_nn(mk, Xtr, mtr, sd, epochs=epochs_by_model[mk])
                out = M.predict_nn(net, Xte, mk); hist[(mk, sd)] = h
                if mk in keep:
                    kept[(mk, sd)] = net
            frames.append(_pred_frame(mte, out, mk, sd, -1))
        log(f"  final {mk} done - {time.time() - t0:.0f}s")
    return pd.concat(frames, ignore_index=True), kept, hist, sc, (Xtr, mtr, Xte, mte)
