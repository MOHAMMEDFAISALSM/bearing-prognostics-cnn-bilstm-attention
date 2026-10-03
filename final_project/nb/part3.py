from nbdefs import md, code

# =====================================================================================================================
md(r"""
## 22. Final Unseen Test

**WHAT** – the design is now **frozen**: features, targets, cap and stage boundaries, architecture, training recipe, epochs (from LOBO), and the model-selection file `artifacts/selection.json` were all fixed using the learning bearings only. Now (and only now) the models are trained on **all 6 learning bearings** and each is evaluated **once** on the 11 test bearings (`Full_Test_Set`, truncated at the official end of life).
**GUARDS.** The cell (i) asserts that the configuration equals the snapshot stored at selection time, (ii) asserts that the training data contains only learning bearings and the scoring data only test bearings, and (iii) refuses to run twice in the same session.
**WHY** – the test bearings simulate a *new machine*. Any use of them for decisions would invalidate the result.
**EXPECT** – test performance similar to (or worse than) the LOBO estimates, with the same bearing-to-bearing variability.
""")

code(r'''
assert "TEST_SCORED" not in globals(), "The test bearings were already scored in this session: results must not be regenerated after inspection."
L.check_config_frozen(CFG_AT_SELECTION, L.config_snapshot())
assert (C.ARTIFACTS / "selection.json").exists()
D.assert_disjoint(C.LEARNING, C.TEST)
t0 = time.time()
pred_test, kept_final, hist_final, sc_final, (Xtr_f, mtr_f, Xte_f, mte_f) = E.run_final(table, feats, E.ALL_MODELS, C.SEEDS_FINAL, sel["epochs_by_model"])
assert set(mtr_f.bearing) == set(C.LEARNING) and set(mte_f.bearing) == set(C.TEST)
TEST_SCORED = True
print(f"final training on {len(Xtr_f):,} windows of 6 learning bearings; scored {len(Xte_f):,} windows of 11 test bearings  ({time.time()-t0:.0f}s)")
pred_test.to_csv(C.ARTIFACTS / "final_predictions_test.csv", index=False)

# ---- save the final proposed model and all preprocessing artefacts (used by the demo in Section 23 from disk) ----
import joblib
kept_final[("proposed", C.SEEDS_FINAL[0])].save(C.ARTIFACTS / "final_proposed_model.keras")
joblib.dump(sc_final, C.ARTIFACTS / "final_scaler.pkl")
json.dump({"features": F.FEATURES, "log10_features": F.LOG_FEATURES, "window": C.WINDOW, "rul_cap_min": C.RUL_CAP_MIN, "stage_bounds_min": [C.STAGE_CRITICAL_MIN, C.STAGE_WARNING_MIN]}, open(C.ARTIFACTS / "final_feature_config.json", "w"), indent=1)
''')

code(r'''
trunc = {b: D.truncated_length(b) for b in C.TEST}
metric_cols = ["MAE_min", "MAE_min_actionable", "AUC_h10", "AUC_h60", "AUC_mean_h", "stage_macroF1", "false_alarm_rate", "detected_bearings"]
sm_test = E.summarize(pred_test, by=("model", "seed"))
phm_rows = []
for (m, sd), g in pred_test.groupby(["model", "seed"], sort=False):
    sc_, tab_ = Mx.official_phm(g[["bearing", "snapshot_idx"]].reset_index(drop=True), g.pred_min.values, trunc); phm_rows.append({"model": m, "seed": sd, "PHM_official": sc_})
sm_test = sm_test.merge(pd.DataFrame(phm_rows), on=["model", "seed"])
sm_test.to_csv(C.ARTIFACTS / "final_metrics_per_model_seed.csv", index=False)
gt = sm_test.groupby("model", sort=False)
tab_test = pd.DataFrame({c: gt[c].apply(R.fmt_ms) for c in metric_cols[:-1] + ["PHM_official"]}).loc[E.ALL_MODELS]
tab_test["alarm active at end (of 11)"] = [f"{sm_test[sm_test.model==m].alarm_at_end_bearings.mean():.1f}" for m in E.ALL_MODELS]
tab_test.index = [E.LABEL[m] for m in E.ALL_MODELS]
display(tab_test); tab_test.to_csv(C.ARTIFACTS / "final_model_comparison_test.csv")

R.show("**Official PHM 2012 score, per bearing (proposed model, seed 42).** One prediction per test bearing at the truncation point, percentage error `%Er = 100 (ActRUL - PredRUL)/ActRUL`, `A = exp(-ln(0.5) Er/5)` if `Er <= 0` (late) else `exp(+ln(0.5) Er/20)` (early), mean over the 11 bearings [2]. The predictor is capped at 120 min = 7200 s, so it cannot exceed that; bearings 1_7 and 2_3 have official RUL above 7200 s.")
g = pred_test[(pred_test.model == "proposed") & (pred_test.seed == C.SEEDS_FINAL[0])].reset_index(drop=True)
sc_p, phm_tab = Mx.official_phm(g[["bearing", "snapshot_idx"]], g.pred_min.values, trunc); display(phm_tab.round(3)); print(f"official PHM-2012 score (proposed, seed 42): {sc_p:.3f}")
gc = pred_test[(pred_test.model == "Constant")].reset_index(drop=True); print(f"official PHM-2012 score of the constant predictor: {Mx.official_phm(gc[['bearing','snapshot_idx']], gc.pred_min.values, trunc)[0]:.3f}")
''')

code(r'''
# ---- figures on the test bearings ----
SEED0 = C.SEEDS_FINAL[0]
pt = pred_test[pred_test.seed == SEED0]
fig, axes = plt.subplots(3, 4, figsize=(24, 13)); axes = axes.ravel()
for a_, b in zip(axes, C.TEST):
    for m, col, ls in [("Constant", "#999999", ":"), ("Random Forest", "#8da0cb", "-"), ("proposed", "#d62728", "-")]:
        g_ = pt[(pt.model == m) & (pt.bearing == b)].sort_values("time_min"); a_.plot(g_.time_min, g_.pred_min, color=col, ls=ls, lw=1.1, label=E.LABEL[m])
    g_ = pt[(pt.model == "proposed") & (pt.bearing == b)].sort_values("time_min"); a_.plot(g_.time_min, g_.rul_cap_min, "k--", lw=2, label="true capped RUL")
    a_.set_title(f"{b} (condition {C.condition_of(b)})"); a_.set_xlabel("Operating time (min)"); a_.set_ylabel("RUL (min)")
axes[0].legend(fontsize=8); axes[-1].axis("off"); plt.tight_layout(); R.savefig(fig, "fig_per_bearing_test.png"); plt.show()
R.show("**Figure 24.** Predicted vs true capped RUL for every test bearing (seed 42). This is the direct visual answer to 'does the model warn in time?'.")

fig, ax = plt.subplots(1, 4, figsize=(21, 5), sharex=True, sharey=True)
for a_, m in zip(ax, ["Constant", "Random Forest", "lstm", "proposed"]):
    g_ = pt[pt.model == m]; a_.scatter(g_.rul_cap_min, g_.pred_min, s=3, alpha=0.2, c=[COND_COLOR[c] for c in g_.condition]); a_.plot([0, 120], [0, 120], "k--"); a_.set_title(E.LABEL[m]); a_.set_xlabel("Actual capped RUL (min)")
ax[0].set_ylabel("Predicted RUL (min)"); plt.tight_layout(); R.savefig(fig, "fig_actual_vs_predicted_test.png"); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for a_, h in zip(ax, [60, 10]):
    for m, col in [("Constant", "#999999"), ("Random Forest", "#8da0cb"), ("Gradient Boosting", "#66c2a5"), ("lstm", "#fc8d62"), ("proposed", "#d62728")]:
        g_ = pt[pt.model == m]; y = (g_.rul_min <= h).astype(int); fpr, tpr, _ = roc_curve(y, -g_.pred_min); a_.plot(fpr, tpr, color=col, lw=2, label=f"{E.LABEL[m]} (AUC {roc_auc_score(y, -g_.pred_min):.2f})")
    a_.plot([0, 1], [0, 1], "k:"); a_.set_xlabel("False-positive rate"); a_.set_ylabel("True-positive rate"); a_.set_title(f"ROC (test bearings): failure within {h} min"); a_.legend()
plt.tight_layout(); R.savefig(fig, "fig_roc_test.png"); plt.show()
g_ = pt[pt.model == "proposed"]
cm_n = confusion_matrix(g_.stage, g_[["prob0", "prob1", "prob2"]].values.argmax(1), labels=[0, 1, 2]); cm_u = confusion_matrix(g_.stage, Mx.rul_to_stage(g_.pred_min.values), labels=[0, 1, 2])
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a_, cm, ttl in [(ax[0], cm_n, "stage head (native)"), (ax[1], cm_u, "stage from predicted RUL (uniform rule)")]:
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt="d", cmap="Blues", ax=a_, xticklabels=["Normal", "Warning", "Critical"], yticklabels=["Normal", "Warning", "Critical"], cbar_kws={"label": "row-normalised"}); a_.set_xlabel("Predicted stage"); a_.set_ylabel("True stage"); a_.set_title(f"Test bearings, proposed model, {ttl}")
plt.tight_layout(); R.savefig(fig, "fig_confusion_test.png"); plt.show()
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for m, col in [("Constant", "#999999"), ("Random Forest", "#8da0cb"), ("proposed", "#d62728")]:
    g_ = pt[pt.model == m]; sns.kdeplot(g_.pred_min - g_.rul_cap_min, ax=ax[0], color=col, label=E.LABEL[m], lw=2); sns.kdeplot((g_.pred_min - g_.rul_cap_min)[g_.rul_min <= 60], ax=ax[1], color=col, label=E.LABEL[m], lw=2)
ax[0].set_title("Test error (predicted - true capped RUL), all windows"); ax[1].set_title("Test error, actionable region (true RUL <= 60 min)")
for a_ in ax: a_.axvline(0, color="k", ls=":"); a_.set_xlabel("Error (min); positive = warns too late"); a_.legend()
plt.tight_layout(); R.savefig(fig, "fig_error_distribution_test.png"); plt.show()
R.show("**Figures 25-27.** Actual-vs-predicted scatter, ROC curves and confusion matrices on the 11 unseen test bearings; error distribution (right-hand side positive = the model thinks more life is left than there is).")
''')

code(r'''
# ---- classification metrics of the stage (accuracy, precision, recall, F1 per class), final test, mean over seeds ----
cls_cols = ["stage_acc", "stage_precision_Normal", "stage_precision_Warning", "stage_precision_Critical", "stage_recall_Normal", "stage_recall_Warning", "stage_recall_Critical", "stage_macroF1"]
tb = sm_test.groupby("model", sort=False)[cls_cols].mean().loc[E.ALL_MODELS]; tb.index = [E.LABEL[m] for m in E.ALL_MODELS]
R.show("**Stage classification on the 11 test bearings - uniform rule (stage derived from the predicted RUL with the same thresholds for every model):**")
display(tb.round(3).rename(columns=lambda c: c.replace("stage_", "")))
nat_models = ["Random Forest", "cnn_bilstm_dual", "proposed"]
nat = sm_test[sm_test.model.isin(nat_models)].groupby("model", sort=False)[[c.replace("stage_", "native_") for c in cls_cols]].mean().loc[nat_models]; nat.index = [E.LABEL[m] for m in nat_models]
R.show("**Native stage heads / classifiers (their own class predictions):**")
display(nat.round(3).rename(columns=lambda c: c.replace("native_", "")))
pp_ = tb.loc[E.LABEL["proposed"]]; pn_ = nat.loc[E.LABEL["proposed"]]
R.show(f"**Reading.** Proposed model, uniform rule: accuracy {pp_.stage_acc:.2f}; recall Normal / Warning / Critical = {pp_.stage_recall_Normal:.2f} / {pp_.stage_recall_Warning:.2f} / {pp_.stage_recall_Critical:.2f}; precision = {pp_.stage_precision_Normal:.2f} / {pp_.stage_precision_Warning:.2f} / {pp_.stage_precision_Critical:.2f}. "
       f"Its native stage head: accuracy {pn_.native_acc:.2f}, recall {pn_.native_recall_Normal:.2f} / {pn_.native_recall_Warning:.2f} / {pn_.native_recall_Critical:.2f}. "
       "Accuracy is dominated by the large Normal class; recall of the rare Critical class (the operationally important one) and macro-F1 are the informative numbers.")
''')

code(r'''
# ---- per-bearing test results (proposed) and early warning ----
g = pt[pt.model == "proposed"].reset_index(drop=True)
pb_test = Mx.per_bearing_table(g[["bearing", "snapshot_idx", "time_min", "rul_min", "rul_cap_min", "y_rul", "stage", "condition"]], g.pred_min.values, Mx.rul_to_stage(g.pred_min.values))
pb_test["AUC_h10"] = [roc_auc_score(g[g.bearing == b].rul_min <= 10, -g[g.bearing == b].pred_min) for b in pb_test.index]
pb_test["AUC_h60"] = [roc_auc_score(g[g.bearing == b].rul_min <= 60, -g[g.bearing == b].pred_min) if 0 < (g[g.bearing == b].rul_min <= 60).mean() < 1 else np.nan for b in pb_test.index]
display(pb_test[["windows", "MAE_min", "MAE_min_actionable", "AUC_h10", "AUC_h60", "stage_macroF1", "false_alarm_rate", "alarm_at_end_bearings", "lead_sustained_min_median"]].round(2))
pb_test.to_csv(C.ARTIFACTS / "final_per_bearing_proposed.csv")
ew_t = pd.DataFrame([{"model": E.LABEL[m], "false_alarm_rate": sm_test[sm_test.model == m].false_alarm_rate.mean(), "alarm active at end (of 11)": sm_test[sm_test.model == m].alarm_at_end_bearings.mean(),
                      "median sustained lead (min)": sm_test[sm_test.model == m].lead_sustained_min_median.median(), "in-zone lead, lenient (min)": sm_test[sm_test.model == m].lead_min_median.median()} for m in E.ALL_MODELS]).set_index("model"); display(ew_t.round(2))

REP = ["Bearing1_3", "Bearing2_3", "Bearing3_3"]      # rule fixed in advance: the first test bearing of each operating condition
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
for a_, b in zip(axes, REP):
    g_ = pt[(pt.model == "proposed") & (pt.bearing == b)].sort_values("time_min"); ps = Mx.rul_to_stage(g_.pred_min.values); alarm = Mx.sustained(ps >= 1)
    a_.plot(g_.time_min, g_.rul_cap_min, "k--", lw=1.5, label="true capped RUL"); a_.plot(g_.time_min, g_.pred_min, color="#d62728", lw=1, label="predicted RUL")
    a_.fill_between(g_.time_min, 0, 125, where=g_.stage.values >= 1, color="orange", alpha=0.15, label="true Warning/Critical zone"); a_.fill_between(g_.time_min, 0, 125, where=g_.stage.values == 2, color="red", alpha=0.15, label="true Critical zone")
    a_.scatter(g_.time_min[alarm], np.full(alarm.sum(), 128), marker="|", color="purple", s=40, label="alarm"); a_.set_title(f"{b}: early-warning timeline (unseen test bearing)"); a_.set_ylabel("RUL (min)"); a_.set_ylim(-2, 135)
axes[-1].set_xlabel("Operating time (min)"); axes[0].legend(ncol=5, fontsize=8); plt.tight_layout(); R.savefig(fig, "fig_early_warning_test.png"); plt.show()
R.show("**Figure 28.** Early-warning timelines for the first test bearing of each operating condition (chosen by a rule fixed in advance, not by result).")
''')

code(r'''
# ---- ablation on the test bearings (paired over the 11 test bearings; metrics defined on every bearing) ----
def per_bearing_metric(m, seed, kind):
    g_ = pred_test[(pred_test.model == m) & (pred_test.seed == seed)]
    out = []
    for b in C.TEST:
        gb = g_[g_.bearing == b]; act = gb.rul_min.values <= 60
        out.append(np.abs(gb.pred_min - gb.rul_cap_min).values[act].mean() if kind == "mae" else roc_auc_score(gb.rul_min <= 10, -gb.pred_min))
    return np.array(out)
def seed_avg(m, kind): return np.mean([per_bearing_metric(m, s, kind) for s in ([C.SEEDS_FINAL[0]] if m in ("Constant", "Ridge") else C.SEEDS_FINAL)], axis=0)
rows = []
for a_, b_, lab in chain:
    for kind, hib, nm in [("auc", True, "detection AUC at 10 min"), ("mae", False, "actionable RUL MAE (min)")]:
        r = R.paired_difference(seed_avg(a_, kind), seed_avg(b_, kind), a_, b_, higher_is_better=hib); r.update({"step": lab, "metric": nm}); rows.append(r)
abl_t = pd.DataFrame(rows)[["step", "metric", "mean_diff", "ci_lo", "ci_hi", "units_improved", "verdict"]]; display(abl_t.round(3)); abl_t.to_csv(C.ARTIFACTS / "ablation_test.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for a_, nm in zip(ax, ["detection AUC at 10 min", "actionable RUL MAE (min)"]):
    sub = abl_t[abl_t.metric == nm]; y = np.arange(len(sub)); a_.errorbar(sub.mean_diff, y, xerr=[sub.mean_diff - sub.ci_lo, sub.ci_hi - sub.mean_diff], fmt="o", color="#d62728", capsize=4)
    a_.axvline(0, color="k", ls="--"); a_.set_yticks(y); a_.set_yticklabels(sub.step, fontsize=8); a_.invert_yaxis(); a_.set_title(f"Test bearings: change in {nm}"); a_.set_xlabel("paired difference (mean, 95% bootstrap CI over 11 bearings)")
plt.tight_layout(); R.savefig(fig, "fig_ablation_test.png"); plt.show()
R.show("**Figure 29.** Ablation on the unseen test bearings (same steps as Figure 13). Compare with the LOBO ablation: components that were not supported there should not be supported here either.")
''')

code(r'''
# ---- test-set explainability (DESCRIPTIVE, after the freeze; no decision depends on it) ----
net_t = kept_final[("proposed", SEED0)]
attn_m = KModel(net_t.input, net_t.get_layer("temporal_attention").output[1]); A_t = attn_m.predict(Xte_f, batch_size=512, verbose=0)[:, :, 0]
prof = {nm: A_t[mte_f.stage.values == s_].mean(0) for s_, nm in enumerate(["Normal", "Warning", "Critical"])}
u = KL.GlobalAveragePooling1D()(net_t.get_layer("drop_bilstm").output); u = net_t.get_layer("rul_output")(net_t.get_layer("rul_dense")(net_t.get_layer("shared_dense")(u))); uni_t = KModel(net_t.input, u)
y_a = np.clip(net_t.predict(Xte_f, batch_size=512, verbose=0)["rul_output"].ravel(), 0, 1) * C.RUL_CAP_MIN; y_u = np.clip(uni_t.predict(Xte_f, batch_size=512, verbose=0).ravel(), 0, 1) * C.RUL_CAP_MIN
I_t = perm_importance(lambda X: np.clip(net_t.predict(X, batch_size=1024, verbose=0)["rul_output"].ravel(), 0, 1) * C.RUL_CAP_MIN, Xte_f, mte_f)
I_t.to_csv(C.ARTIFACTS / "perm_importance_proposed_test.csv")
fig, ax = plt.subplots(1, 2, figsize=(17, 5.5))
for nm, col in [("Normal", "#2ca02c"), ("Warning", "#ff7f0e"), ("Critical", "#d62728")]: ax[0].plot(np.arange(1, 17), prof[nm], marker="o", color=col, label=nm)
ax[0].axhline(1 / 16, color="k", ls="--", label="uniform"); ax[0].set_xlabel("Step in window (16 = newest)"); ax[0].set_ylabel("Mean attention weight"); ax[0].set_title("Attention profile by true stage (11 test bearings, descriptive)"); ax[0].legend()
t_ = I_t.sort_values("dAUC", ascending=False).head(12).iloc[::-1]; ax[1].barh(t_.index, t_.dAUC, color=["#1f77b4" if i.startswith("horiz") else "#2ca02c" for i in t_.index], edgecolor="black")
ax[1].set_xlabel("Drop in detection AUC when shuffled"); ax[1].set_title("Permutation importance on test windows (descriptive)")
plt.tight_layout(); R.savefig(fig, "fig_explainability_test.png"); plt.show()
R.show(f"**Figure 30.** Descriptive explainability on the test bearings (no decision uses it). Replacing the learned attention by uniform weights changes predictions by {np.abs(y_a - y_u).mean():.1f} min on average (correlation {np.corrcoef(y_a, y_u)[0,1]:.3f}). "
       f"Top features on test: " + ", ".join(f"`{i}` ({v:+.3f})" for i, v in I_t.dAUC.sort_values(ascending=False).head(4).items()) + f"; overlap of the top-5 with the LOBO top-5: {len(set(I_t.dAUC.nlargest(5).index) & set(I_nn.dAUC.nlargest(5).index))}/5.")

# ---- descriptive: health indicator of ALL 17 bearings, shown only now (after the freeze) ----
allb = table.copy(); allb["hi"] = np.nan
for b in C.LEARNING + C.TEST:
    m = allb.bearing == b; allb.loc[m, "hi"] = A.combined_rms_hi(allb[m])
fig, ax = plt.subplots(1, 2, figsize=(17, 5.5))
for b in C.LEARNING + C.TEST:
    g_ = allb[allb.bearing == b]; ax[0].plot(g_.time_min, g_.hi, lw=0.8, color=COND_COLOR[C.condition_of(b)], alpha=0.9, ls="-" if b in C.LEARNING else "--"); ax[1].plot(np.maximum(g_.rul_min, 0.05), g_.hi, lw=0.8, color=COND_COLOR[C.condition_of(b)], alpha=0.9, ls="-" if b in C.LEARNING else "--")
ax[0].set_xlabel("Operating time (min)"); ax[0].set_ylabel("log combined RMS (g)"); ax[0].set_title("Health indicator, all 17 bearings (solid = learning, dashed = test; colour = condition)")
ax[1].set_xscale("log"); ax[1].invert_xaxis(); ax[1].set_xlabel("RUL (min, log, failure at right)"); ax[1].set_ylabel("log combined RMS (g)"); ax[1].set_title("Aligned at failure")
plt.tight_layout(); R.savefig(fig, "fig_health_indicator_all17.png"); plt.show()
ph_test = A.phase_table(table, C.TEST); ph_test["AUC_h10 (proposed)"] = pb_test.AUC_h10; ph_test["AUC_h60 (proposed)"] = pb_test.AUC_h60; ph_test["MAE_actionable (proposed, min)"] = pb_test.MAE_min_actionable
display(ph_test.round(2)); ph_test.to_csv(C.ARTIFACTS / "test_phase_lengths_descriptive.csv")
R.show("**Figure 31 / table (descriptive, hindsight-based, not used for any decision).** All 17 health indicators; the table lists the visible degradation-phase length of each test bearing and the proposed model's detection quality on it.")
''')

md(r"""
### 22.1 What did we find on the unseen test bearings?
""")

code(r'''
mt = sm_test.groupby("model").mean(numeric_only=True)
sd_t = sm_test.groupby("model").std(numeric_only=True)
ct = mt.loc["Constant"]
b_prop = mt.loc["proposed"]
cand_t = mt.drop(index="Constant")
def tied(col, lower, tol):
    """Models whose mean is within max(tol, seed std of the best model) of the best one: statistically indistinguishable here."""
    best_m = cand_t[col].idxmin() if lower else cand_t[col].idxmax()
    band = max(tol, float(sd_t.loc[best_m, col]) if not np.isnan(sd_t.loc[best_m, col]) else 0.0)
    ok = [m for m in cand_t.index if (cand_t.loc[m, col] - cand_t.loc[best_m, col]) * (1 if lower else -1) <= band]
    return best_m, ok, band
best_det_test, tie_det, band_det = tied("AUC_mean_h", False, 0.02)
best_mae_test, tie_mae, band_mae = tied("MAE_min_actionable", True, 2.0)
rk = cand_t.AUC_mean_h.rank(ascending=False); rk_mae = cand_t.MAE_min_actionable.rank(ascending=True)
lobo_rank = mmean.drop(index="Constant").AUC_mean_h.rank(ascending=False); rho_rank = spearmanr(lobo_rank.loc[rk.index], rk)[0]
lobo_rank_m = mmean.drop(index="Constant").MAE_min_actionable.rank(ascending=True); rho_rank_m = spearmanr(lobo_rank_m.loc[rk_mae.index], rk_mae)[0]
R.show("\n\n".join([
    f"**Detection (mean AUC over horizons 5-60 min; constant = 0.50):** proposed **{b_prop.AUC_mean_h:.3f}** (seed std {sd_t.loc['proposed','AUC_mean_h']:.3f}); highest mean on test: {E.LABEL[best_det_test]} ({mt.loc[best_det_test].AUC_mean_h:.3f}). "
    f"Models within {band_det:.3f} of the best are **statistically indistinguishable** here (seed noise): " + ", ".join(E.LABEL[m] for m in tie_det) + ". "
    f"The LOBO-selected detection model was **{E.LABEL[sel['best_detection_AUC_mean_h']]}**; it ranks #{int(rk[sel['best_detection_AUC_mean_h']])} of 10 on test. Rank correlation LOBO vs test (10 models): {rho_rank:.2f}.",
    f"**RUL in the actionable region (constant {ct.MAE_min_actionable:.1f} min):** proposed {b_prop.MAE_min_actionable:.1f} min; lowest on test {E.LABEL[best_mae_test]} ({mt.loc[best_mae_test].MAE_min_actionable:.1f} min); tied within {band_mae:.1f} min: " + ", ".join(E.LABEL[m] for m in tie_mae) +
    f". The LOBO-selected RUL model was **{E.LABEL[sel['best_actionable_RUL_MAE']]}**; it ranks #{int(rk_mae[sel['best_actionable_RUL_MAE']])} of 10 on test (rank correlation LOBO vs test {rho_rank_m:.2f}). **Overall MAE** (constant {ct.MAE_min:.1f} min): proposed {b_prop.MAE_min:.1f} min.",
    f"**Official PHM-2012 score** (one prediction per bearing at the truncation point; higher = better, 1 = perfect): proposed {b_prop.PHM_official:.3f}, constant {ct.PHM_official:.3f}, Random Forest {mt.loc['Random Forest'].PHM_official:.3f}. With only 11 predictions and a capped predictor this score is very noisy; the per-bearing table shows that at the truncation points of the abrupt bearings the model still predicts a healthy bearing (RUL > 100 min), which is punished as a very late prediction.",
    f"**False alarms** on truly Normal windows: proposed {b_prop.false_alarm_rate:.2f}, Random Forest {mt.loc['Random Forest'].false_alarm_rate:.2f}, Gradient Boosting {mt.loc['Gradient Boosting'].false_alarm_rate:.2f} (uniform rule). **Stage macro-F1:** proposed {b_prop.stage_macroF1:.2f} (uniform rule), constant {ct.stage_macroF1:.2f}.",
    ("**Consistency of validation and test.** The rank correlations between LOBO and test are " + f"{rho_rank:.2f} (detection) and {rho_rank_m:.2f} (actionable RUL error): " +
     ("the validation ranking is only weakly or not at all reproduced on the test bearings, so six learning bearings give an **unstable model ranking**; differences between the top models must not be over-interpreted." if min(rho_rank, rho_rank_m) < 0.7
      else "the validation ranking is largely reproduced on the test bearings."))]))
''')

md(r"""
### 22.2 What does this mean, and does it support the hypotheses?
The numbers above are the **final, one-shot** test result. Compare them with the LOBO estimates: consistent ranking between validation and test increases confidence that the conclusions are not artefacts of the six learning bearings; disagreement means the estimates are unstable. Bearing-to-bearing variability (per-bearing table and phase-length table) is large, and the models detect impending failure well only for bearings with a visible degradation phase. **H1, H2 and H3 are judged in Section 26 from these measurements, not assumed.**
""")

# =====================================================================================================================
md(r"""
## 23. Sample Prediction Demonstration (for the review)

**WHAT** – pick a test bearing and a time; the demo takes **only the data available up to that time** (asserted), rebuilds the 16-snapshot window from the raw table, loads the **saved model and scaler from disk**, and shows the predicted RUL, stage probabilities, the true values for comparison, the health indicator and the raw waveform.
**HOW TO USE.** Change `DEMO_BEARING` and `DEMO_TIME_MIN` in the next cell and run it. `DEMO_TIME_MIN=None` uses the moment when the true RUL is 30 minutes.
**DEFAULT CHOICE.** The default bearing is the test bearing whose proposed-model actionable-region error is the **median** of the 11 (so the demonstration is neither the best nor the worst case). This choice was made *after* the test result, for demonstration only, and is disclosed here.
""")

code(r'''
# ------------------ EDIT THESE TWO LINES TO DEMONSTRATE ANOTHER BEARING / TIME ------------------
med_b = pb_test.MAE_min_actionable.sort_values().index[len(pb_test) // 2]
DEMO_BEARING = med_b            # any of C.TEST, e.g. "Bearing1_3"
DEMO_TIME_MIN = None            # operating time in minutes since start; None -> 30 minutes before failure
# --------------------------------------------------------------------------------------------------
scaler_disk = joblib.load(C.ARTIFACTS / "final_scaler.pkl")
model_disk = tf.keras.models.load_model(C.ARTIFACTS / "final_proposed_model.keras")
cfg_disk = json.load(open(C.ARTIFACTS / "final_feature_config.json"))
STAGES = ["Normal (RUL > 60 min)", "Warning (20-60 min)", "Critical (<= 20 min)"]

def predict_sample(bearing, t_min=None):
    tb = table[table.bearing == bearing].sort_values("snapshot_idx")
    if t_min is None:
        t_min = float(tb.time_min[tb.rul_min <= 30].iloc[0])
    k = int(np.argmin(np.abs(tb.time_min.values - t_min)))
    assert k >= C.WINDOW - 1, "need at least 16 snapshots of history"
    past = tb.iloc[: k + 1]                                   # ONLY data up to the chosen time
    x = scaler_disk.transform(feats.loc[past.index].values)[-C.WINDOW:][None].astype("float32")
    out = model_disk.predict(x, verbose=0)
    pred = float(np.clip(out["rul_output"][0, 0], 0, 1) * cfg_disk["rul_cap_min"]); prob = out["risk_output"][0]
    row = tb.iloc[k]
    return dict(bearing=bearing, snapshot=int(row.snapshot_idx), t_min=float(row.time_min), window=(float(tb.iloc[k - C.WINDOW + 1].time_min), float(row.time_min)), pred_rul=pred, prob=prob,
                stage_native=int(prob.argmax()), stage_uniform=int(D.stage_from_rul_min([pred])[0]), true_rul=float(row.rul_min), true_capped=float(row.rul_cap_min), true_stage=int(row.stage),
                past=past, features_now=feats.loc[past.index[-1]])

def demo(bearing, t_min=None):
    r = predict_sample(bearing, t_min); tb = table[table.bearing == bearing].sort_values("snapshot_idx"); past = r["past"]
    R.show(f"### Sample prediction - {bearing} (condition {C.condition_of(bearing)}: {C.CONDITION[C.condition_of(bearing)][0]} rpm, {C.CONDITION[C.condition_of(bearing)][1]} N)\n"
           f"* **Input sample:** window of 16 snapshots covering {r['window'][0]:.1f}-{r['window'][1]:.1f} min of operation (snapshot {r['snapshot']}); the model sees nothing after {r['t_min']:.1f} min.\n"
           f"* **Predicted RUL (capped at 120 min):** **{r['pred_rul']:.1f} min**   |   **true RUL:** {r['true_rul']:.1f} min (capped {r['true_capped']:.1f})   |   error {r['pred_rul'] - r['true_capped']:+.1f} min\n"
           f"* **Predicted stage (stage head):** **{STAGES[r['stage_native']]}** (probabilities N/W/C = {r['prob'][0]:.2f}/{r['prob'][1]:.2f}/{r['prob'][2]:.2f})   |   from predicted RUL: {STAGES[r['stage_uniform']]}   |   **true stage:** {STAGES[r['true_stage']]}\n"
           f"* **Health indicator (log combined RMS):** {A.combined_rms_hi(past.iloc[[-1]])[0]:.2f}  (median of this bearing's first 10 snapshots: {np.median(A.combined_rms_hi(past.iloc[:10])):.2f})")
    fig = plt.figure(figsize=(17, 9)); gs = fig.add_gridspec(2, 3)
    a0 = fig.add_subplot(gs[0, :2])
    a0.plot(past.time_min, A.combined_rms_hi(past), color="#1f77b4", lw=1.2, label="health indicator: data available to the model"); a0.set_xlim(0, tb.time_min.max()); a0.axvspan(r["window"][0], r["window"][1], color="orange", alpha=0.4, label="16-snapshot input window"); a0.axvline(r["t_min"], color="k", ls="--")
    a0.set_xlabel("Operating time (min)"); a0.set_ylabel("log combined RMS (g)"); a0.set_title("Health indicator"); a0.legend(fontsize=8)
    a1 = fig.add_subplot(gs[0, 2]); a1.bar(["Normal", "Warning", "Critical"], r["prob"], color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="black"); a1.set_ylim(0, 1.05); a1.set_title("Predicted stage probabilities"); a1.axvline(r["true_stage"], color="k", ls=":")
    a2 = fig.add_subplot(gs[1, :2]); ks = np.arange(C.WINDOW - 1, len(past), 3); xs = scaler_disk.transform(feats.loc[past.index].values).astype("float32"); Wn = np.stack([xs[i - C.WINDOW + 1:i + 1] for i in ks])
    traj = np.clip(model_disk.predict(Wn, verbose=0)["rul_output"].ravel(), 0, 1) * C.RUL_CAP_MIN
    a2.plot(tb.time_min, tb.rul_cap_min, "k--", lw=2, label="true capped RUL (evaluation only)"); a2.set_xlim(0, tb.time_min.max()); a2.plot(past.time_min.values[ks], traj, color="#d62728", lw=1.5, label="model prediction so far"); a2.scatter([r["t_min"]], [r["pred_rul"]], s=90, color="#d62728", zorder=5)
    a2.set_xlabel("Operating time (min)"); a2.set_ylabel("RUL (min)"); a2.set_title("Predicted vs true RUL up to the chosen time"); a2.legend()
    a3 = fig.add_subplot(gs[1, 2]); sub = "Learning_set" if bearing in C.LEARNING else "Full_Test_Set"; arr_, _ = F.read_snapshot(C.DATASET / sub / bearing / f"acc_{r['snapshot']+1:05d}.csv")
    a3.plot(np.arange(len(arr_)) / C.FS * 1000, arr_[:, 4], lw=0.6, color="#1f77b4"); a3.set_xlabel("ms"); a3.set_ylabel("Horizontal acceleration (g)"); a3.set_title(f"Raw waveform at snapshot {r['snapshot']} (0.1 s)")
    plt.tight_layout(); R.savefig(fig, f"fig_sample_prediction_{bearing}_t{int(round(r['t_min']))}min.png"); plt.show(); return r

r_demo = demo(DEMO_BEARING, DEMO_TIME_MIN)
''')

code(r'''
# a second, healthy-phase example for the same bearing (t = 25% of life) to show a "Normal" case
_ = demo(DEMO_BEARING, float(table[table.bearing == DEMO_BEARING].time_min.max() * 0.25))
''')

md(r"""
### 23.1 What did we find?
The demonstration shows the complete deployment path on a bearing the model never saw: causal window → saved scaler → saved network → RUL and stage. Use it in the review to explain the prediction *and its uncertainty*: compare predicted with true RUL and note the stage probabilities. The first example is 30 minutes before failure; the second is a healthy phase.
""")

# =====================================================================================================================
md(r"""
## 24. Results Summary and Model Selection

**WHAT** – one objective comparison across the dimensions that matter (accuracy, generalisation, cost, interpretability, stability, false alarms, RUL performance, detection, size). **A model is called best only for the criterion where the numbers say so.**
""")

code(r'''
rows = []
for m in E.ALL_MODELS:
    l = fold_mean.xs(m, level="model"); t_ = sm_test[sm_test.model == m]
    rows.append({"model": E.LABEL[m], "detection AUC LOBO": l.AUC_mean_h.mean(), "detection AUC test": t_.AUC_mean_h.mean(), "actionable MAE LOBO (min)": l.MAE_min_actionable.mean(), "actionable MAE test (min)": t_.MAE_min_actionable.mean(),
                 "overall MAE test (min)": t_.MAE_min.mean(), "false-alarm rate test": t_.false_alarm_rate.mean(), "PHM-2012 score test": t_.PHM_official.mean(),
                 "stability: std of test AUC over seeds": t_.AUC_mean_h.std(), "stability: std of LOBO AUC over folds": l.AUC_mean_h.std(),
                 "generalisation gap AUC (LOBO - test)": l.AUC_mean_h.mean() - t_.AUC_mean_h.mean(), "parameters": n_params.get(m, np.nan),
                 "fit+predict s (LOBO median)": ep_lobo[ep_lobo.model == m].fit_predict_s.median(),
                 "interpretability": {"Constant": "trivial", "Ridge": "coefficients", "Random Forest": "importances", "Gradient Boosting": "importances"}.get(m, "attention + permutation" if m == "proposed" else "black box")})
crit = pd.DataFrame(rows).set_index("model"); display(crit.round(3)); crit.to_csv(C.ARTIFACTS / "final_multicriteria_comparison.csv")
pick = lambda col, lower: (crit.drop(index="Constant")[col].idxmin() if lower else crit.drop(index="Constant")[col].idxmax())
purposes = pd.DataFrame({"purpose": ["best detection of imminent failure (test AUC)", "lowest actionable RUL error (test)", "lowest false-alarm rate (test)", "most stable across seeds (test AUC std)", "cheapest to train", "richest built-in explanation"],
                         "model with the best mean": [pick("detection AUC test", False), pick("actionable MAE test (min)", True), pick("false-alarm rate test", True), pick("stability: std of test AUC over seeds", True),
                                                      crit.drop(index="Constant")["fit+predict s (LOBO median)"].idxmin(), "PROPOSED: CNN + BiLSTM + Attention (dual head)"],
                         "statistically tied with it": [", ".join(E.LABEL[m] for m in tie_det), ", ".join(E.LABEL[m] for m in tie_mae), "see table (no seed-noise band computed)", "-", "-", "design property, not a measurement"]})
display(purposes)
R.show("**Selection rule reminder.** The recommendation for detection and RUL was fixed on **LOBO** results (Section 17) and saved *before* the test; the test columns here confirm or contradict it. 'Richest built-in explanation' is a design property of the proposed model, not a performance claim.")
''')

# =====================================================================================================================
md(r"""
## 25. Limitations

* **Very few independent bearings.** 6 learning bearings (3 conditions; Condition 3 has 2) and 11 test bearings. Every conclusion has wide uncertainty; LOBO has only six folds and bootstrap intervals over six units are wide.
* **Different operating conditions and heterogeneous degradation.** Conditions were not separate in training; degradation patterns differ strongly between bearings [1], [2].
* **Abrupt failures.** For several bearings the vibration is almost healthy until the last minutes. No model can give a long warning there (Figure 8, phase-length tables). This is a property of the data.
* **Label uncertainty.** End of life comes from recording durations and the official RUL table. The 20 g criterion is not visible in every snapshot (only 0.1 s per 10 s is recorded), and Bearing1_4 in `Full_Test_Set` extends beyond its official end of life. Bearing2_1/2_2 may contain early level shifts that are not damage.
* **Design choices.** The cap (120 min) and stage boundaries (20/60 min) were chosen a priori for operational meaning; conclusions were checked for cap sensitivity but the horizons still define what "success" means.
* **Hindsight in offline analysis.** Training uses the failure time of the learning bearings. Descriptive onset/phase-length tables use whole trajectories and are not real-time detectors.
* **Prior exposure to the test bearings.** In earlier experiments (v1/v2) I inspected test-bearing health-indicator curves and results. The final target does not use any vibration-derived label, and all design decisions here were made on learning bearings, but the choice of formulation was motivated by earlier findings. This cannot be undone.
* **Detection is not RUL estimation.** Good AUC at short horizons shows that *imminent failure can be recognised*; it does not show accurate minute-level RUL prediction, which the results do not support.
* **RUL metric.** MAE over all windows is dominated by the healthy phase; the actionable-region MAE is more informative but is computed only on windows within 60 minutes of failure. The official PHM score uses one prediction per bearing and a capped predictor (cannot exceed 120 min), so it is noisy and not comparable with challenge leaderboards.
* **Attention.** Attention weights are not causal explanations [9]; Section 20 tests their functional role.
* **Small models, limited tuning.** No hyper-parameter search was performed (deliberately, to avoid over-fitting six bearings); better settings may exist. Window length (16) was not varied.
* **Computational cost.** Deterministic TensorFlow on CPU: the full notebook takes about an hour; the network is small (~100 k parameters) and inference is fast.
* **Dataset specificity.** One test rig, one bearing type, three conditions: results may not transfer to other machines.
""")

# =====================================================================================================================
md(r"""
## 26. Conclusions
""")

code(r'''
# Hypothesis verdicts computed from the measurements (no verdict is typed by hand)
# ---- H1: family comparison using LOBO-SELECTED representatives (nothing is selected on the test bearings) ----
rep_nn = max(M.NN_KINDS, key=lambda m: mmean.loc[m, "AUC_mean_h"]); rep_tab = max([m for m in M.CLASSICAL if m != "Constant"], key=lambda m: mmean.loc[m, "AUC_mean_h"])
h1_l = R.paired_difference(fold_mean.xs(rep_tab, level="model").AUC_mean_h.values, fold_mean.xs(rep_nn, level="model").AUC_mean_h.values, rep_tab, rep_nn, higher_is_better=True)
h1_t = R.paired_difference(seed_avg(rep_tab, "auc"), seed_avg(rep_nn, "auc"), rep_tab, rep_nn, higher_is_better=True)
n_ok = int(h1_l["verdict"] == "measurable improvement") + int(h1_t["verdict"] == "measurable improvement")
h1_label = {2: "SUPPORTED", 1: "PARTLY SUPPORTED", 0: "NOT SUPPORTED"}[n_ok]

# ---- H2: every ablation contrast that adds the component, LOBO (AUC, actionable MAE) and test (AUC@10, actionable MAE) ----
def collect(steps):
    v = []
    for s in steps:
        v += list(abl[(abl.step == s) & abl.metric.isin(["detection AUC (mean over horizons)", "actionable RUL MAE (min)"])].verdict)
        v += list(abl_t[abl_t.step == s].verdict)
    return v
def status(v):
    imp = sum(x == "measurable improvement" for x in v); deg = sum(x == "measurable degradation" for x in v); no = len(v) - imp - deg
    lab = "SUPPORTED" if (imp >= 0.5 * len(v) and deg == 0) else ("NOT SUPPORTED" if imp == 0 else "MIXED")
    return lab, imp, deg, no
att_lab, ai, ad, an = status(collect(["add attention (single head)", "add attention to the dual-head model"]))
dual_lab, di, dd, dn = status(collect(["add second head (dual head)", "add second head WITHOUT attention"]))

# ---- H3: a USEFUL warning = alarm still active at the end AND sustained lead >= 10 min, per test bearing (proposed, seed 42, uniform rule) ----
abrupt = (ph_test["phase_min@10%"] < 20).values
useful = ((pb_test.alarm_at_end_bearings == 1) & (pb_test.lead_sustained_min_median >= 10)).values
n_useful = int(useful.sum()); n_grad_useful = int(useful[~abrupt].sum()); n_abrupt_useful = int(useful[abrupt].sum())
h3_label = "SUPPORTED" if n_useful == 11 else "NOT SUPPORTED"
R.show("\n\n".join([
    f"* **H1 - sequence models detect impending failure better than tabular baselines: {h1_label}.** Representatives chosen on LOBO: {E.LABEL[rep_nn]} vs {E.LABEL[rep_tab]}. LOBO (6 folds): mean difference in AUC {h1_l['mean_diff']:+.3f} (95% CI {h1_l['ci_lo']:+.3f} to {h1_l['ci_hi']:+.3f}; {h1_l['verdict']}); test (11 bearings, AUC at 10 min): {h1_t['mean_diff']:+.3f} (CI {h1_t['ci_lo']:+.3f} to {h1_t['ci_hi']:+.3f}; {h1_t['verdict']}).",
    f"* **H2 - attention and the dual head improve accuracy.** Attention: **{att_lab}** ({ai} improvements, {ad} degradations, {an} no difference among the ablation contrasts on LOBO and test). Dual head: **{dual_lab}** ({di} improvements, {dd} degradations, {dn} no difference). *MIXED* means the effect changes sign or significance between metrics or between validation and test, i.e. no robust claim can be made.",
    f"* **H3 - a useful warning for every bearing: {h3_label}.** A useful warning (alarm active at the end and sustained lead >= 10 min) exists for {n_useful}/11 test bearings ({n_grad_useful}/{int((~abrupt).sum())} with a visible degradation phase >= 20 min, {n_abrupt_useful}/{int(abrupt.sum())} of the abrupt ones), at a false-alarm rate of {pb_test.false_alarm_rate.mean():.2f} on healthy windows (proposed, seed 42, uniform rule).",
    f"* **RQ3/RQ6:** on unseen bearings the proposed model reaches mean detection AUC {mt.loc['proposed'].AUC_mean_h:.2f} (constant 0.50), but minute-level RUL accuracy is limited: actionable-region MAE {mt.loc['proposed'].MAE_min_actionable:.1f} min vs {mt.loc['Constant'].MAE_min_actionable:.1f} min for a constant, and the official PHM score is {mt.loc['proposed'].PHM_official:.3f} vs {mt.loc['Constant'].PHM_official:.3f} for a constant.",
    f"* **RQ4 - is the proposed model the best?** Highest mean detection AUC on test: **{E.LABEL[best_det_test]}**; models statistically tied with it: {', '.join(E.LABEL[m] for m in tie_det)}. Lowest actionable RUL error: **{E.LABEL[best_mae_test]}**; tied: {', '.join(E.LABEL[m] for m in tie_mae)}. "
    f"The proposed model is {'in' if 'proposed' in tie_det else 'not in'} the detection tie group and {'in' if 'proposed' in tie_mae else 'not in'} the RUL tie group. Its distinctive property is the combination of an alarm head with inspectable attention/feature analysis (functional value measured in Section 20); it is **not** shown to be more accurate than the simplest competitive baselines.",
    "* **RQ5:** the model is *inspectable* (attention, permutation importance), but attention is not a causal explanation and appears only weakly functional (Section 20)."]))
''')

md(r"""
### 26.1 One-paragraph conclusion (for an IEEE-style paper)
The claims that this project **supports** are limited to: (i) a leakage-controlled bearing-level evaluation protocol on PRONOSTIA with automated checks; (ii) on unseen bearings, vibration-based models recognise an *imminent* failure better than chance but far from perfectly (see the test AUC tables), with large differences between bearings, detectability falling as the horizon grows, and poor performance for bearings that fail abruptly; (iii) the proposed CNN–BiLSTM–Attention dual-head network is compared honestly with simple baselines and ablations, and the tables above state where it is or is not better. The claims that this project does **not** support are: accurate minute-level RUL prediction; a long warning for all bearings; causal explanation by attention; comparability with challenge leaderboards.
""")

# =====================================================================================================================
md(r"""
## 27. Reproducibility and Saved Artifacts
""")

code(r'''
config_all = {"seeds": {"global": SEED, "lobo": C.SEEDS_LOBO, "final": C.SEEDS_FINAL}, "window": C.WINDOW, "rul_cap_min": C.RUL_CAP_MIN, "stage_bounds_min": [C.STAGE_CRITICAL_MIN, C.STAGE_WARNING_MIN],
              "alarm_persistence": C.ALARM_PERSISTENCE, "optimizer": "Adam", "learning_rate": C.LR, "batch_size": C.BATCH, "max_epochs": C.MAX_EPOCHS, "early_stopping_patience": C.PATIENCE,
              "loss": {"rul": f"Huber(delta={C.HUBER_DELTA})", "stage": "sparse categorical cross-entropy, balanced class weights"}, "loss_weights": C.LOSS_WEIGHTS,
              "epochs_final_by_model": sel["epochs_by_model"], "scaler": "StandardScaler fitted on training bearings only", "features": len(F.FEATURES), "log10_features": F.LOG_FEATURES,
              "tf_deterministic_ops": True, "parameters": {k: int(v) for k, v in n_params.items()}, "versions": VERSIONS,
              "lobo_folds": D.lobo_folds(), "final_train": C.LEARNING, "final_test": C.TEST, "official_rul_s": C.OFFICIAL_RUL_S}
json.dump(config_all, open(C.ARTIFACTS / "experiment_config.json", "w"), indent=1, default=str)
final_metrics = {"lobo": {E.LABEL[m]: {k: float(v) for k, v in fold_mean.xs(m, level="model").mean(numeric_only=True)[metric_cols[:-1] + ["detected_bearings", "alarm_at_end_bearings"]].items()} for m in E.ALL_MODELS},
                 "test": {E.LABEL[m]: {k: float(v) for k, v in sm_test[sm_test.model == m].mean(numeric_only=True)[metric_cols[:-1] + ["detected_bearings", "alarm_at_end_bearings", "PHM_official"]].items()} for m in E.ALL_MODELS}}
json.dump(final_metrics, open(C.ARTIFACTS / "final_metrics.json", "w"), indent=1)

# dataset audit report (generated from the audit table and anomaly list, not typed)
def md_table(df):
    d = df.reset_index(); head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.values)
(C.PROJECT / "DATASET_AUDIT_REPORT.md").write_text("# Dataset audit report (generated by the notebook)\n\nSource: IEEE PHM 2012 / PRONOSTIA files in the project folder; official facts from the challenge document [2].\n\n" + md_table(audit) +
    "\n\n## Anomalies\n\n" + "\n".join("* " + a.replace("**", "") for a in anom) + "\n", encoding="utf8")

expected = ["features_raw.csv", "dataset_audit_table.csv", "feature_documentation.csv", "feature_names.json", "selection.json", "lobo_predictions.csv", "lobo_metrics_per_fold_seed.csv", "lobo_model_comparison.csv",
            "ablation_lobo.csv", "final_predictions_test.csv", "final_metrics_per_model_seed.csv", "final_model_comparison_test.csv", "final_per_bearing_proposed.csv", "ablation_test.csv",
            "final_multicriteria_comparison.csv", "final_metrics.json", "final_proposed_model.keras", "final_scaler.pkl", "final_feature_config.json", "experiment_config.json"]
missing = [f for f in expected if not (C.ARTIFACTS / f).exists()]; assert not missing, f"missing artifacts: {missing}"
figs = sorted(p.name for p in C.FIGURES.glob("*.png"))
R.show(f"**{len(expected)} artifacts verified in `artifacts/`** and **{len(figs)} figures** in `figures/`. Re-running the notebook top-to-bottom from a clean kernel regenerates everything (features are always re-extracted from the raw files).")
display(pd.DataFrame({"artifact": expected, "size_KB": [round((C.ARTIFACTS / f).stat().st_size / 1024, 1) for f in expected]}).set_index("artifact"))
print(*figs, sep="\n")
''')

# =====================================================================================================================
md(r"""
## How to Explain This Project in a Review
""")

code(r'''
pp, cc, rf_ = mt.loc["proposed"], mt.loc["Constant"], mt.loc["Random Forest"]
R.show(f"""
### 1. 30-second explanation
Bearings fail suddenly and unplanned stops are expensive. I use vibration data from real run-to-failure bearings to estimate how much life is left and to give a Normal / Warning / Critical alarm. The main result: on bearings the model has never seen, on average it recognises an *imminent* failure better than chance (AUC {pp.AUC_h10:.2f} at 10 minutes vs 0.50, with large differences between bearings), but the data do not allow reliable long-horizon prediction, especially for bearings that fail abruptly.

### 2. 1-minute explanation
Data: the public PRONOSTIA / IEEE PHM 2012 benchmark, 17 bearings, two accelerometers, 25.6 kHz. Each 0.1 s recording becomes 34 vibration features; the model looks at the last 16 recordings (160 s). A CNN finds local patterns, a BiLSTM follows the trend, attention weights the moments, and two heads output remaining life (capped at 120 min) and a stage. To avoid leakage I split **by bearing**, use leave-one-bearing-out validation on the 6 learning bearings, and score the 11 test bearings once after freezing the design. I compared the model with a constant predictor, Ridge, Random Forest, Gradient Boosting, LSTM, CNN and ablations.

### 3. 2-minute technical explanation
Formulation: multi-task learning with a capped-RUL regression head (Huber loss) and a 3-class stage head (cross-entropy, balanced weights), targets derived only from the failure time (no vibration-based onset label). The end of life follows the official durations; an audit found that `Full_Test_Set` Bearing1_4 extends past its official end of life, which is corrected. Scaler and all statistics are fitted on training bearings; automated leakage checks (poison, truncation, causality, target-shuffle, scaler) must pass. Evaluation: MAE (minutes) overall and in the actionable region, detection AUC at several horizons, stage macro-F1, false-alarm rate, lead time, and the official PHM-2012 score at the truncation points. Ablation with bootstrap intervals over bearings tests each component. Explainability: attention profile plus a uniform-attention functional test, and permutation importance.

### 4. Dataset explanation
17 run-to-failure bearings from the PRONOSTIA platform (FEMTO-ST): 6 learning + 11 test bearings, 3 operating conditions (1800/1650/1500 rpm; 4000/4200/5000 N), 2560 samples per 0.1 s snapshot every 10 s, lifetimes from {audit.lifetime_min.min():.0f} to {audit.lifetime_min.max():.0f} min.

### 5. Problem statement
Given the recent vibration history of a bearing, estimate its remaining useful life and health stage, for bearings never seen during training.

### 6. Why this dataset?
It is the standard public benchmark for bearing prognostics with real run-to-failure data and an official protocol and score [2].

### 7. Why these features?
Amplitude (RMS, peak) shows energy growth, impulsiveness (kurtosis, crest factor) shows early impacts, spectral shape and band energies show resonance excitation; each is documented (Section 9) and screened on learning bearings only (Section 8).

### 8-11. Why CNN, BiLSTM, attention, dual head?
CNN: local patterns/noise smoothing. BiLSTM: trend inside the window. Attention: learned weighting of the 16 steps, giving inspectable weights. Dual head: an alarm level next to the number, and possible regularisation. **Whether each part helps was measured (Sections 18 and 22):** attention *{att_lab}* ({ai} improvements / {ad} degradations / {an} no difference), dual head *{dual_lab}* ({di} / {dd} / {dn}), from paired comparisons on validation and test bearings.

### 12. How was leakage prevented?
Bearing-level splitting; scaler fitted on training bearings only; targets from failure time only; causal windows; automated poison/truncation/target-shuffle tests; test bearings scored once after saving the selection file; random window splitting was shown to inflate results (Section 13).

### 13. How were models compared?
Same windows, targets, folds and metrics for all models; LOBO for selection; one-shot test; paired bootstrap for ablation.

### 14. What do the final results mean?
Detection AUC (mean over horizons) on test: proposed {pp.AUC_mean_h:.2f}, Random Forest {rf_.AUC_mean_h:.2f}, constant 0.50. Actionable-region RUL error: proposed {pp.MAE_min_actionable:.1f} min, constant {cc.MAE_min_actionable:.1f} min. Highest mean on test for detection: {E.LABEL[best_det_test]} (statistically tied: {', '.join(E.LABEL[m] for m in tie_det)}); lowest RUL error: {E.LABEL[best_mae_test]} (tied: {', '.join(E.LABEL[m] for m in tie_mae)}).

### 15. Limitations
Six independent learning bearings; abrupt failures with almost no visible precursor; label uncertainty; hindsight in offline analysis; prior exposure to test curves in earlier experiments; attention is not causal.

### 16. Future work
Train with leave-one-bearing-out over all 17 bearings for the final deployment; vary window length; per-condition models; uncertainty estimates; physically informed features (bearing fault frequencies) despite the mismatch reported in [2]; validation on another bearing dataset.

### 17. Ten likely faculty questions
1. *Why not random train/test split?* Overlapping windows of the same bearing leak; Section 13 shows the inflation.
2. *Is the proposed model the best?* Only where the tables say so (Section 24); simple models can be as good or better.
3. *Why is RUL error large?* Most of a bearing's life looks healthy and remaining time is unpredictable; some bearings fail abruptly (Figure 8).
4. *What does AUC mean here?* Probability that a window closer to failure gets a higher alarm score than a healthier one (0.5 = chance).
5. *Why cap the RUL at 120 minutes?* Beyond that the data carry no information; the cap is a documented design choice with a sensitivity check.
6. *Does attention explain the failure?* No; it shows where the network looked, and we test how much it matters (Section 20).
7. *How do you know there is no leakage?* Automated tests (Section 12) and a one-shot final test after freezing.
8. *Why only 6 training bearings?* That is what the challenge provides; LOBO uses them efficiently; more data is the main way to improve.
9. *What is the official PHM score and how did you do?* Asymmetric percentage-error score at the truncation points; proposed {pp.PHM_official:.3f}, constant {cc.PHM_official:.3f} (noisy with 11 bearings).
10. *What would you do next?* More data / other datasets, per-condition models, uncertainty estimation.
""")
''')

# =====================================================================================================================
md(r"""
## 28. References

Sources actually opened and read for this project. Items marked with † were verified through the arXiv/publisher abstract page or bibliographic record only.

[1] P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni, C. Varnier, "PRONOSTIA: An experimental platform for bearings accelerated degradation tests," *IEEE International Conference on Prognostics and Health Management (PHM'12)*, Denver, CO, USA, 2012 (author manuscript, HAL hal-00719503).

[2] IEEE Reliability Society and FEMTO-ST Institute, "IEEE PHM 2012 Prognostic Challenge: Outline, Experiments, Scoring of results, Winners," 2012 (file `IEEEPHM2012-Challenge-Details.pdf` distributed with the dataset).

[3] "IEEE PHM 2012 Data Challenge Dataset," repository README (dataset copy used in this project; requests citation of [1]).

[4] F. Huang, A. Sava, K. H. Adjallah, Z. Wang, "Fuzzy model identification based on mixture distribution analysis for bearings remaining useful life estimation using small training data set," arXiv:2012.04589.

[5] Z. Xu, Y. Guo, J. H. Saleh, "Remaining useful life prediction with uncertainty quantification: development of a highly accurate model for rotating machinery," arXiv:2109.11579 (benchmarks on the PHM12 bearing dataset).

[6] † Y. Lei, N. Li, L. Guo, N. Li, T. Yan, J. Lin, "Machinery health prognostics: A systematic review from data acquisition to RUL prediction," *Mechanical Systems and Signal Processing*, vol. 104, pp. 799-834, 2018, doi:10.1016/j.ymssp.2017.11.016.

[7] † X. Li, Q. Ding, J.-Q. Sun, "Remaining useful life estimation in prognostics using deep convolution neural networks," *Reliability Engineering & System Safety*, vol. 172, pp. 1-11, 2018 (time-window sample preparation; C-MAPSS).

[8] L. Basora, A. Viens, M. Arias Chao, X. Olive, "A benchmark on uncertainty quantification for deep learning prognostics," arXiv:2302.04730 (section on piece-wise linear RUL degradation; turbofan data).

[9] † S. Jain, B. C. Wallace, "Attention is not Explanation," *NAACL-HLT 2019*, arXiv:1902.10186.

[10] † M. M. R. Shamim et al., "Leakage-Robust Evaluation and Data-Scale Sensitivity of Attention-Enhanced Multi-Task Learning for Joint Fault Diagnosis and Remaining Useful Life Estimation," arXiv:2607.16493, July 2026 (preprint, not peer-reviewed).

**Software.** Python, NumPy, pandas, SciPy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn (versions in Section 4 and `artifacts/experiment_config.json`).

**Not cited because they could not be verified while preparing this work: [CITATION NEEDED]** — Adam optimiser, Huber loss, LSTM/BiLSTM, additive attention, a peer-reviewed CNN–LSTM–attention PHM-2012 paper, and a paper reporting MAE/RMSE on PHM-2012.
""")
