"""Independent audit of the finished notebook and its artifacts. Uses its own re-implementations (not src.metrics) where possible.
Writes FINAL_AUDIT.md. Exit code != 0 if a hard check fails."""
import json, re, sys, os, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]   # project root (this script lives in scripts/)
NB = ROOT / "final_bearing_prognostics_review.ipynb"
ART, FIG = ROOT / "artifacts", ROOT / "figures"
sys.path.insert(0, str(ROOT))
results, notes = [], []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


nb = json.load(open(NB, encoding="utf8"))
codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
# 1. execution health -------------------------------------------------------------------------------------------------
errs = [(i, o["ename"]) for i, c in enumerate(nb["cells"]) for o in c.get("outputs", []) if o.get("output_type") == "error"]
check("zero error outputs in the executed notebook", not errs, str(errs))
ec = [c["execution_count"] for c in codes]
check("all code cells executed, in order, from a clean kernel (execution_count = 1..N)", ec == list(range(1, len(codes) + 1)), f"{ec[:3]}...{ec[-3:]}")
n_fig_out = sum(1 for c in codes for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
check("figures are embedded as outputs", n_fig_out >= 30, f"{n_fig_out} embedded images")

# 2. artifacts ---------------------------------------------------------------------------------------------------------
need = ["features_raw.csv", "selection.json", "lobo_predictions.csv", "final_predictions_test.csv", "final_metrics.json", "final_metrics_per_model_seed.csv",
        "final_proposed_model.keras", "final_scaler.pkl", "final_feature_config.json", "experiment_config.json", "final_model_comparison_test.csv", "ablation_lobo.csv", "ablation_test.csv"]
check("required artifacts exist", all((ART / f).exists() for f in need), str([f for f in need if not (ART / f).exists()]))
figs = list(FIG.glob("*.png"))
check("figures directory holds >= 30 non-empty PNG files", len(figs) >= 30 and all(f.stat().st_size > 5000 for f in figs), f"{len(figs)} files")

# 3. freeze order: selection written BEFORE the test predictions -------------------------------------------------------
t_sel = (ART / "selection.json").stat().st_mtime; t_test = (ART / "final_predictions_test.csv").stat().st_mtime; t_lobo = (ART / "lobo_predictions.csv").stat().st_mtime
check("selection.json written after LOBO and BEFORE the test predictions", t_lobo <= t_sel < t_test, f"lobo {t_lobo:.0f} <= selection {t_sel:.0f} < test {t_test:.0f}")

# 4. no test bearing in any LOBO prediction; no learning bearing in test predictions --------------------------------------
from src import config as C
lobo = pd.read_csv(ART / "lobo_predictions.csv", usecols=["bearing", "fold", "model", "seed"]); test = pd.read_csv(ART / "final_predictions_test.csv")
check("LOBO predictions contain only learning bearings", set(lobo.bearing) == set(C.LEARNING))
check("test predictions contain exactly the 11 test bearings", set(test.bearing) == set(C.TEST))
check("each learning bearing is held out exactly once per model/seed", (lobo.groupby(["model", "seed"]).fold.nunique() == 6).all())

# 5. independent metric re-implementation ------------------------------------------------------------------------------
def phm_a(act, pred):
    er = 100.0 * (act - pred) / act
    return np.exp(-np.log(0.5) * er / 5.0) if er <= 0 else np.exp(np.log(0.5) * er / 20.0)


cap = C.RUL_CAP_MIN
official = C.OFFICIAL_RUL_S
trunc = {b: len(list((C.DATASET / "Test_set" / b).glob("acc_*.csv"))) for b in C.TEST}
ref = pd.read_csv(ART / "final_metrics_per_model_seed.csv"); bad = []
for (m, s), g in test.groupby(["model", "seed"]):
    row = ref[(ref.model == m) & (ref.seed == s)].iloc[0]
    mae = np.abs(g.pred_min - g.rul_cap_min).mean()
    act = g.rul_min <= 60
    mae_act = np.abs(g.pred_min - g.rul_cap_min)[act].mean()
    aucs = [roc_auc_score(g.rul_min <= h, -g.pred_min) for h in (5, 10, 20, 30, 60)]
    ph = np.mean([phm_a(official[b], float(g[(g.bearing == b) & (g.snapshot_idx == trunc[b] - 1)].pred_min.iloc[0]) * 60) for b in C.TEST])
    for nm, a, b_ in [("MAE_min", mae, row.MAE_min), ("MAE_min_actionable", mae_act, row.MAE_min_actionable), ("AUC_mean_h", np.mean(aucs), row.AUC_mean_h), ("PHM_official", ph, row.PHM_official)]:
        if not np.isclose(a, b_, rtol=1e-6, atol=1e-9):
            bad.append((m, s, nm, a, b_))
check("MAE, actionable MAE, mean AUC and official PHM score re-computed independently match the notebook for every model/seed", not bad, str(bad[:3]))

# 6. official-protocol integrity ----------------------------------------------------------------------------------------
full = {b: len(list((C.DATASET / "Full_Test_Set" / b).glob("acc_*.csv"))) for b in C.TEST}
incons = [b for b in C.TEST if (full[b] - trunc[b]) * 10 != official[b]]
check("official RUL table vs Full_Test_Set: only Bearing1_4 inconsistent (documented and corrected)", incons == ["Bearing1_4"], str(incons))
n14 = int((test.bearing == "Bearing1_4").sum()) // test.model.nunique() // test.seed.nunique() if False else None
last_rul = test[test.bearing == "Bearing1_4"].rul_min.min()
check("Bearing1_4 evaluation stops at the official end of life (last snapshot within one 10 s interval of it)", last_rul * 60 < 10.0, f"min RUL {last_rul*60:.0f} s")
t14 = test[(test.bearing == "Bearing1_4") & (test.snapshot_idx == trunc["Bearing1_4"] - 1)].rul_min.iloc[0]
check("Bearing1_4 RUL at the truncation point equals the official 339 s", abs(t14 * 60 - 339) < 1e-6, f"{t14*60:.1f} s")

# 7. configuration consistency ------------------------------------------------------------------------------------------
cfg = json.load(open(ART / "experiment_config.json"))
check("recorded configuration equals src/config.py", cfg["window"] == C.WINDOW and cfg["rul_cap_min"] == C.RUL_CAP_MIN and cfg["stage_bounds_min"] == [C.STAGE_CRITICAL_MIN, C.STAGE_WARNING_MIN]
      and cfg["seeds"]["final"] == C.SEEDS_FINAL and cfg["max_epochs"] == C.MAX_EPOCHS, "")
check("quick/debug mode is OFF in the recorded run", cfg["max_epochs"] == 40 and cfg["seeds"]["final"] == [42, 43, 44] and cfg["seeds"]["lobo"] == [42, 43])

# 8. reproducibility: retrain the proposed model (seed 42) and compare with the saved test predictions ---------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from src import features as F, data as D, models as M
raw = pd.read_csv(ART / "features_raw.csv"); table = D.build_table(raw); feats = F.transform(table)
sc = D.fit_scaler(feats, table, C.LEARNING); Xtr, mtr = D.make_windows(table, feats, C.LEARNING, sc); Xte, mte = D.make_windows(table, feats, C.TEST, sc)
ep = json.load(open(ART / "selection.json"))["epochs_by_model"]["proposed"]
net, _, _ = M.fit_nn("proposed", Xtr, mtr, 42, epochs=ep)
pred = M.predict_nn(net, Xte, "proposed")["pred_min"]
saved = test[(test.model == "proposed") & (test.seed == 42)].reset_index(drop=True)
same_order = (saved.bearing.values == mte.bearing.values).all() and (saved.snapshot_idx.values == mte.snapshot_idx.values).all()
diff = float(np.abs(pred - saved.pred_min.values).max()) if same_order else np.nan
check("retraining the proposed model (seed 42) reproduces the saved test predictions", same_order and diff < 1e-3, f"max |difference| = {diff:.2e} min")

# 9. static text scan for over-claims (manual review list) --------------------------------------------------------------
words = re.compile(r"\b(best|outperform\w*|state-of-the-art|superior|significantly|proves?|prove[sd]?|guarantee\w*|optimal)\b", re.I)
hits = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown":
        for m_ in words.finditer("".join(c["source"])):
            s = "".join(c["source"]); hits.append((i, s[max(0, m_.start() - 60): m_.end() + 60].replace("\n", " ")))
notes.append(f"{len(hits)} occurrences of strong-claim words in static markdown cells (manual review below).")
outs = []
for i, c in enumerate(nb["cells"]):
    for o in c.get("outputs", []):
        t = "".join(o.get("data", {}).get("text/markdown", [])) if isinstance(o.get("data", {}).get("text/markdown"), list) else o.get("data", {}).get("text/markdown", "")
        for m_ in words.finditer(t or ""):
            outs.append((i, t[max(0, m_.start() - 60): m_.end() + 60].replace("\n", " ")))

# report -----------------------------------------------------------------------------------------------------------------
ok_all = all(r[1] for r in results)
lines = ["# FINAL AUDIT (automated, independent re-implementations)", "", f"Notebook: `{NB.name}`  |  audited at {time.strftime('%Y-%m-%d %H:%M')}", "",
         "| # | Check | Result | Detail |", "|---|---|---|---|"]
for i, (n, ok, d) in enumerate(results, 1):
    lines.append(f"| {i} | {n} | {'PASS' if ok else '**FAIL**'} | {d} |")
lines += ["", f"**Overall: {'ALL AUTOMATED CHECKS PASSED' if ok_all else 'FAILURES PRESENT'}**", "", "## Strong-claim words in static markdown (for manual review)", ""]
lines += [f"* cell {i}: ...{t}..." for i, t in hits] or ["* none"]
lines += ["", "## Strong-claim words in generated (data-driven) markdown output", ""] + ([f"* cell {i}: ...{t}..." for i, t in outs] or ["* none"])
(ROOT / "audits" / "FINAL_AUDIT_automated.md").write_text("\n".join(lines) + "\n", encoding="utf8")
print("\n".join(lines[:30]))
sys.exit(0 if ok_all else 1)
