from nbdefs import md, code

# =====================================================================================================================
md(r"""
## 14. Baseline Models

**WHAT** – build progressively stronger models and evaluate them **with exactly the same protocol**: the same 16×34 windows (tabular models receive the same information as `[last snapshot, window mean, last − first]`), the same targets (capped RUL and 3-class stage), the same scaler rule, the same LOBO folds and the same metrics.
**WHY** – a complex model is only justified if it beats simple ones. If Random Forest wins, we say so.
**EXPECT** – the constant baseline shows what "no skill" looks like; Ridge and trees show what a fast tabular model can do; the neural models show whether temporal structure helps.

| Model | Family | What it can and cannot do |
|---|---|---|
| Constant | none | predicts the training mean / majority stage: the **no-skill reference** |
| Ridge | linear | linear map of the window summary; RUL by Ridge regression, stage by balanced logistic regression |
| Random Forest | trees (200) | non-linear, robust to feature scale; no notion of order inside the window beyond the 3 summaries |
| Gradient Boosting | trees (HistGradientBoosting, no new dependency) | usually the strongest tabular baseline |
| LSTM | recurrent | reads the 34-feature sequence directly |
| CNN | convolutional | local temporal patterns, then pooling |
| CNN + LSTM | conv → recurrent | local patterns then one-direction memory |
| CNN + BiLSTM | conv → bidirectional recurrent | memory in both directions of the past window, average pooling |
| CNN + BiLSTM + Attention (single head) | + attention | learned weighting of the 16 steps; RUL head only |
| CNN + BiLSTM (dual head, no attention) | ablation helper | isolates the effect of the second head from the effect of attention |
| **PROPOSED** | CNN + BiLSTM + Attention, **RUL head + stage head** | the model this project is about |

Classical models are evaluated here; the neural models are trained in Section 16 (same function, same folds).
""")

code(r'''
SEEDS_LOBO = C.SEEDS_LOBO
t0 = time.time()
pred_cls, ep_cls, _, _ = E.run_lobo(table, feats, M.CLASSICAL, SEEDS_LOBO)
print(f"classical models, LOBO: {len(pred_cls):,} held-out window predictions in {time.time()-t0:.0f}s")
''')

# =====================================================================================================================
md(r"""
## 15. Proposed Architecture

```
Input window (16 snapshots x 34 features)
        |
   Conv1D(64, k=3) + BatchNorm + Dropout   x2      <- local patterns / feature combinations
        |
   Bidirectional LSTM (64 + 64)                     <- trend and order inside the 160 s window
        |
   Temporal attention:  alpha_t = softmax(u^T tanh(W h_t + b)),  c = sum_t alpha_t h_t
        |
   Shared Dense(64) + Dropout                        <- shared representation
      /                    \
RUL head                Risk / stage head
Dense(32) -> Dense(1, sigmoid)      Dense(32) -> Dense(3, softmax)
capped RUL / 120 min                Normal / Warning / Critical
```

**In simple words.** The CNN looks at short stretches of the last minutes and mixes the 34 features; the BiLSTM reads the whole 160 s window and remembers the *trend*; attention decides which of the 16 moments matter for this prediction and hands one summary vector to two small heads: one outputs "how many minutes are left" (capped at 120), the other "how worried should we be".

**Why each part (design arguments — Section 18 tests whether they hold).**
* *Why CNN?* Neighbouring snapshots are correlated; a convolution learns local combinations cheaply and smooths noise.
* *Why BiLSTM?* Degradation is a process in time; a recurrent layer models order and trend. "Bidirectional" is legitimate because the whole window lies in the past.
* *Why attention?* It gives the head a learned weighted summary and produces α_t that can be inspected (with the limits of Section 20).
* *Why two heads?* The stage task is coarser and easier; it can regularise the shared representation and gives operators an alarm level. It only helps if the tasks are consistent — tested in Section 18.
* *Why not only a CNN / only an LSTM / a Random Forest?* Not assumed — these are exactly the baselines and ablations that are evaluated with the same protocol.
""")

code(r'''
net = M.build_nn("proposed", len(F.FEATURES), C.WINDOW)
lt = M.layer_table(net)
purpose = {"sequence_input": "16 snapshots x 34 features", "conv1": "local feature combinations over 3 neighbouring snapshots", "bn1": "stabilises training", "drop1": "regularisation",
           "conv2": "second local layer", "bn2": "stabilises training", "drop2": "regularisation", "bilstm": "trend/order in both directions of the past window (2 x 64 units)",
           "drop_bilstm": "regularisation", "temporal_attention": "weights the 16 steps (alpha) and returns the context vector", "shared_dense": "shared representation for both heads",
           "drop_shared": "regularisation", "rul_dense": "RUL head hidden layer", "risk_dense": "stage head hidden layer", "rul_output": "capped RUL / 120 min in [0,1] (sigmoid)", "risk_output": "3 stage probabilities (softmax)"}
lt["purpose"] = lt.layer.map(purpose)
display(lt.set_index("layer"))
n_params = {k: M.build_nn(k, len(F.FEATURES), C.WINDOW).count_params() for k in M.NN_KINDS}
display(pd.Series({M.NN_KINDS[k][0]: v for k, v in n_params.items()}, name="parameters").to_frame())
R.show(f"**Proposed model: {n_params['proposed']:,} parameters, trained on only {len(C.LEARNING)} learning bearings.** The parameter-to-independent-history ratio is a warning sign that is tested in Sections 17-18.")

# Figure: architecture diagram generated from the layer table
from matplotlib.patches import FancyBboxPatch
def shp(name, which="output_shape"): return lt.set_index("layer").loc[name, which].replace("None, ", "")
def par(*names): return int(lt.set_index("layer").loc[list(names), "params"].sum())
fig, ax = plt.subplots(figsize=(11, 11)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 13)
def box(x, y, w, h, txt, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=col, ec="black")); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2, color="black"), zorder=5)
box(2.5, 11.6, 5, 1.0, f"Input window  {shp('sequence_input')}\n34 vibration features x 16 snapshots (160 s)", "#dddddd"); arrow(5, 11.6, 5, 10.9)
box(2.5, 9.6, 5, 1.3, f"2 x [Conv1D 64, k=3 + BatchNorm + Dropout]\n{shp('conv2')}   params {par('conv1','bn1','conv2','bn2'):,}\nlocal patterns", "#c6dbef"); arrow(5, 9.6, 5, 8.9)
box(2.5, 7.6, 5, 1.3, f"Bidirectional LSTM (64 + 64)\n{shp('bilstm')}   params {par('bilstm'):,}\ntrend / order in the window", "#c7e9c0"); arrow(5, 7.6, 5, 6.9)
box(2.5, 5.6, 5, 1.3, f"Temporal attention  alpha_t, context c\noutput {shp('temporal_attention')}   params {par('temporal_attention'):,}\nwhich moments matter", "#fdd0a2"); arrow(5, 5.6, 5, 4.9)
box(2.5, 3.9, 5, 1.0, f"Shared Dense 64 + Dropout   params {par('shared_dense'):,}\nshared representation", "#dadaeb"); arrow(4, 3.9, 2.6, 3.0); arrow(6, 3.9, 7.4, 3.0)
box(0.3, 1.6, 4.5, 1.4, f"RUL head\nDense 32 -> Dense 1 (sigmoid)\nparams {par('rul_dense','rul_output'):,}\ncapped RUL / 120 min", "#fcbba1")
box(5.2, 1.6, 4.5, 1.4, f"Risk / stage head\nDense 32 -> Dense 3 (softmax)\nparams {par('risk_dense','risk_output'):,}\nNormal / Warning / Critical", "#fcbba1")
ax.text(5, 0.6, f"Total trainable + non-trainable parameters: {net.count_params():,}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Proposed dual-head CNN-BiLSTM-Attention architecture (generated from the built model)", fontweight="bold")
R.savefig(fig, "fig_architecture.png"); plt.show()
del net
''')

# =====================================================================================================================
md(r"""
## 16. Training

**WHAT** – train every neural variant in every LOBO fold, with **two random seeds**, using the recipe below. Early stopping monitors the loss on the fold's *validation bearing* only; the held-out bearing is scored after training.
**WHY** – early stopping needs some validation data, but it must not be the bearing we score.
**EXPECT** – neural networks overfit the 4 training bearings quickly (validation loss stops improving after a few epochs).

| Setting | Value |
|---|---|
| Optimiser / learning rate | Adam, 1e-3, reduced ×0.5 on plateau (patience 3) |
| Batch size / max epochs / early stopping | 64 / 40 / patience 6, best weights restored |
| Loss (RUL head) | Huber, δ = 0.1, on capped RUL / 120 min |
| Loss (stage head) | sparse categorical cross-entropy with **balanced class weights** |
| Loss weights | RUL 1.0, stage 0.5 (fixed, **not tuned**) |
| Window / features | 16 snapshots × 34 features, `StandardScaler` fitted on the fold's training bearings only |
| Seeds | 42, 43 (LOBO); 42, 43, 44 (final). TensorFlow deterministic ops enabled |
""")

code(r'''
t0 = time.time()
pred_nn, ep_nn, kept_lobo, hists = E.run_lobo(table, feats, list(M.NN_KINDS), SEEDS_LOBO, keep_models=("proposed",))
pred_lobo = pd.concat([pred_cls, pred_nn], ignore_index=True)
ep_lobo = pd.concat([ep_cls, ep_nn], ignore_index=True)
print(f"neural models, LOBO: done in {time.time()-t0:.0f}s | total held-out predictions: {len(pred_lobo):,}")
sm = E.summarize(pred_lobo)                                                  # metrics per model x seed x fold
fold_mean = sm.groupby(["model", "fold"]).mean(numeric_only=True)            # average seeds inside each fold -> 6 independent folds
pred_lobo.to_csv(C.ARTIFACTS / "lobo_predictions.csv", index=False); sm.to_csv(C.ARTIFACTS / "lobo_metrics_per_fold_seed.csv", index=False)
''')

code(r'''
# Figures: training vs validation loss and metrics (proposed model, seed 42, all six LOBO folds)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for fi in range(6):
    h = hists[("proposed", SEEDS_LOBO[0], fi)]; ep = np.arange(1, len(h["loss"]) + 1)
    ax[0].plot(ep, h["loss"], color="#4c72b0", alpha=0.6); ax[0].plot(ep, h["val_loss"], color="#c44e52", alpha=0.6, ls="--")
    ax[1].plot(ep, h["rul_output_mae"], color="#4c72b0", alpha=0.6); ax[1].plot(ep, h["val_rul_output_mae"], color="#c44e52", alpha=0.6, ls="--")
ax[0].plot([], [], color="#4c72b0", label="training loss (4 bearings)"); ax[0].plot([], [], color="#c44e52", ls="--", label="validation loss (1 unseen bearing)")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Total loss (Huber + 0.5 x cross-entropy)"); ax[0].set_title("Training vs validation loss (6 folds)"); ax[0].legend()
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("RUL MAE (fraction of the 120 min cap)"); ax[1].set_title("Training vs validation RUL error")
ax[1].plot([], [], color="#4c72b0", label="training"); ax[1].plot([], [], color="#c44e52", ls="--", label="validation"); ax[1].legend()
plt.tight_layout(); R.savefig(fig, "fig_training_curves.png"); plt.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for fi in range(6):
    h = hists[("proposed", SEEDS_LOBO[0], fi)]; ep = np.arange(1, len(h["loss"]) + 1)
    ax[0].plot(ep, h["risk_output_accuracy"], color="#4c72b0", alpha=0.6); ax[0].plot(ep, h["val_risk_output_accuracy"], color="#c44e52", alpha=0.6, ls="--")
    ax[1].plot(ep, h["risk_output_loss"], color="#4c72b0", alpha=0.6); ax[1].plot(ep, h["val_risk_output_loss"], color="#c44e52", alpha=0.6, ls="--")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Stage accuracy"); ax[0].set_title("Stage head accuracy: training (blue) vs validation (red, dashed)")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Stage cross-entropy"); ax[1].set_title("Stage head loss")
plt.tight_layout(); R.savefig(fig, "fig_training_metrics.png"); plt.show()
best_ep = ep_nn.groupby("model").best_epoch.median()
gap = np.mean([hists[("proposed", SEEDS_LOBO[0], fi)]["risk_output_accuracy"][-1] - hists[("proposed", SEEDS_LOBO[0], fi)]["val_risk_output_accuracy"][-1] for fi in range(6)])
R.show(f"**Figures 10-11.** Validation loss (red) stops improving after a handful of epochs while training loss keeps falling. Median best epoch of the proposed model: **{best_ep['proposed']:.1f}**. "
       f"At the end of training the stage accuracy on the training bearings exceeds that on the unseen validation bearing by **{100*gap:.0f} percentage points** on average: the model memorises the four training bearings.")
''')

md(r"""
### 16.1 What did we find?
As expected for 4–5 independent histories, **the networks overfit almost immediately**: the best epoch is early, and the gap between training and validation accuracy is large. This is an early, honest indication that model *capacity is not the bottleneck* — data is. It does not yet say whether the architecture helps; the comparison in Section 17 and the ablation in Section 18 answer that on bearings that were never used for training.
""")

# =====================================================================================================================
md(r"""
## 17. Model Comparison (leave-one-bearing-out, learning bearings only)

**WHAT** – compare all models on the six held-out learning bearings. For every model we average the seeds inside each fold and then report **mean ± std over the 6 folds** (the folds, i.e. bearings, are the independent units).
**METRICS (defined once in `src/metrics.py`).**
* *MAE (min)* – mean absolute error of the RUL, in minutes, against the capped true RUL, over all windows. Most windows are healthy (RUL ≥ 120), so a **constant prediction is hard to beat on this number**.
* *MAE actionable (min)* – the same error, only over windows where failure is within the 60-minute warning horizon: this is the region where a prediction is useful.
* *AUC h10 / h60 / mean-h* – ROC-AUC of detecting "failure within H minutes" from the predicted RUL (score = −predicted RUL) at H = 10 or 60 minutes, and the mean over H = 5, 10, 20, 30, 60. 0.5 = chance, 1 = perfect.
* *Macro-F1* – F1 of the three stages averaged over classes (uniform rule: stage from predicted RUL with the same thresholds for every model).
* *False-alarm rate* – fraction of truly Normal windows (RUL > 60 min) that lie in an alarm (predicted stage ≥ Warning for 3 consecutive windows).
* *Alarm active at end* – number of held-out bearings whose alarm is still on at the last window before failure (strict). *Sustained lead* – RUL at the start of that final uninterrupted alarm run (capped at 120 min); a run that started long before the warning zone is partly a false alarm, so read it together with the false-alarm rate.
""")

code(r'''
cols = ["MAE_min", "MAE_min_actionable", "AUC_h10", "AUC_h60", "AUC_mean_h", "stage_macroF1", "false_alarm_rate"]
grp = fold_mean[cols].groupby("model", sort=False)
tab = pd.DataFrame({c: grp[c].apply(R.fmt_ms) for c in cols})
order_models = list(E.ALL_MODELS)
tab = tab.loc[order_models]; tab.index = [E.LABEL[m] for m in order_models]
tab["alarm active at end (of 6)"] = [f"{fold_mean.xs(m, level='model').alarm_at_end_bearings.sum():.1f}" for m in order_models]
tab["median sustained lead (min)"] = [f"{fold_mean.xs(m, level='model').lead_sustained_min_median.median():.1f}" for m in order_models]
tab["params"] = [n_params.get(m, np.nan) for m in order_models]
tab["fit+predict s (median)"] = [ep_lobo[ep_lobo.model == m].fit_predict_s.median() for m in order_models]
tab["best epoch (median)"] = [ep_lobo[ep_lobo.model == m].best_epoch.median() for m in order_models]
display(tab)
tab.to_csv(C.ARTIFACTS / "lobo_model_comparison.csv")
per_fold_auc = fold_mean["AUC_mean_h"].unstack(0)[order_models]; per_fold_mae = fold_mean["MAE_min_actionable"].unstack(0)[order_models]
per_fold_auc.index = [f"{f}: {C.LEARNING[f]}" for f in per_fold_auc.index]; per_fold_mae.index = per_fold_auc.index
per_fold_auc.columns = per_fold_mae.columns = [E.LABEL[m] for m in order_models]
R.show("**Per-fold mean AUC (higher is better)** - which held-out bearing is easy or hard:"); display(per_fold_auc.round(2))
R.show("**Per-fold actionable-region MAE in minutes (lower is better):**"); display(per_fold_mae.round(1))
''')

code(r'''
# Figure: model comparison (LOBO)
mm = fold_mean.groupby("model").agg(["mean", "std"])
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
for a_, (met, ttl, lower) in zip(ax, [("MAE_min_actionable", "RUL MAE in the actionable region (min, lower better)", True), ("AUC_mean_h", "Detection AUC, mean over horizons 5-60 min (higher better)", False), ("false_alarm_rate", "False-alarm rate on Normal windows (lower better)", True)]):
    m_ = mm[(met, "mean")].loc[order_models]; s_ = mm[(met, "std")].loc[order_models]
    cols_ = ["#999999" if k in ("Constant",) else ("#d62728" if k == "proposed" else ("#8da0cb" if k in M.CLASSICAL else "#1f77b4")) for k in order_models]
    a_.barh([E.LABEL[k] for k in order_models], m_, xerr=s_, color=cols_, edgecolor="black"); a_.invert_yaxis(); a_.set_title(ttl, fontsize=10)
    a_.axvline(m_["Constant"], color="k", ls="--", lw=1, label="constant baseline"); a_.legend()
plt.tight_layout(); R.savefig(fig, "fig_model_comparison_lobo.png"); plt.show()
R.show("**Figure 12.** LOBO comparison (mean over 6 held-out bearings, error bars = std across bearings; grey = constant, purple = classical, blue = neural, red = proposed).")
''')

code(r'''
# ---- MODEL SELECTION: rules fixed in advance, applied to LOBO results ONLY; written to disk BEFORE any test bearing is scored ----
mmean = fold_mean.groupby("model").mean(numeric_only=True)
cand = mmean.drop(index="Constant")
sel = {"rule": "selection uses LOBO validation results only (learning bearings); the 11 test bearings are not touched before this file is written",
       "best_actionable_RUL_MAE": cand.MAE_min_actionable.idxmin(), "best_detection_AUC_mean_h": cand.AUC_mean_h.idxmax(),
       "lowest_false_alarm_rate": cand.false_alarm_rate.idxmin(), "interpretable_by_design": "proposed",
       "models_beating_constant_on_overall_MAE": [m for m in cand.index if cand.loc[m, "MAE_min"] < mmean.loc["Constant", "MAE_min"]],
       "models_beating_constant_on_actionable_MAE": [m for m in cand.index if cand.loc[m, "MAE_min_actionable"] < mmean.loc["Constant", "MAE_min_actionable"]],
       "epochs_by_model": {m: int(max(3, round(ep_lobo[ep_lobo.model == m].best_epoch.median()))) for m in M.NN_KINDS}}
json.dump(sel, open(C.ARTIFACTS / "selection.json", "w"), indent=1)
CFG_AT_SELECTION = L.config_snapshot()
R.show("**Selection (LOBO only, saved to `artifacts/selection.json`):**\n\n" + "\n".join(f"* {k}: `{v}`" for k, v in sel.items() if k != "epochs_by_model"))
R.show(f"Epochs used for the final training (median best LOBO epoch, at least 3): `{sel['epochs_by_model']}`")
''')

code(r'''
c0 = mmean.loc["Constant"]
best = mmean.drop(index="Constant")
txt = []
txt.append(f"**Do any models beat the constant baseline?** Overall RUL MAE (constant {c0.MAE_min:.1f} min): " + (", ".join(f"{E.LABEL[m]} {mmean.loc[m,'MAE_min']:.1f}" for m in sel['models_beating_constant_on_overall_MAE']) if sel['models_beating_constant_on_overall_MAE'] else "**no model** beats the constant") +
           f". Actionable-region MAE (constant {c0.MAE_min_actionable:.1f} min): " + (", ".join(f"{E.LABEL[m]} {mmean.loc[m,'MAE_min_actionable']:.1f}" for m in sel['models_beating_constant_on_actionable_MAE']) if sel['models_beating_constant_on_actionable_MAE'] else "**no model** beats the constant") + ".")
txt.append(f"**Detection (mean AUC over horizons, constant 0.50):** best = **{E.LABEL[sel['best_detection_AUC_mean_h']]}** ({best.AUC_mean_h.max():.2f}); proposed = {best.loc['proposed','AUC_mean_h']:.2f}; best tabular = {max(best.loc[m,'AUC_mean_h'] for m in M.CLASSICAL if m!='Constant'):.2f}.")
txt.append(f"**Lowest false-alarm rate:** {E.LABEL[sel['lowest_false_alarm_rate']]} ({best.false_alarm_rate.min():.2f}); proposed = {best.loc['proposed','false_alarm_rate']:.2f}. Note that false alarms of the *uniform rule* derive from the noisy RUL estimate.")
R.show("### 17.1 What did we find?\n\n" + "\n\n".join(txt))
''')

md(r"""
### 17.2 What does this mean, and does it support the hypotheses?
* **H1 (sequence models detect impending failure better than tabular baselines):** judge from the AUC columns and the per-fold table: compare the neural rows with the tabular rows, fold by fold, and note the large differences between held-out bearings. Statistical support is limited by n = 6 folds (tested in Section 18).
* **H3 (a warning several tens of minutes ahead for every bearing):** the per-fold tables show the bearings for which this fails.
* **Important:** the dataset has only six independent learning histories. Differences between models that are smaller than the fold-to-fold spread should not be interpreted as real.
""")

# =====================================================================================================================
md(r"""
## 18. Ablation Study

**WHAT** – remove or add **one component at a time** along the chain CNN → CNN+LSTM → CNN+BiLSTM → +Attention → +Dual head, plus the side branch "dual head without attention", and measure the paired change over the 6 held-out learning bearings with a bootstrap 95 % confidence interval.
**WHY** – to test whether each part of the proposed architecture contributes, rather than assume it. **We do not claim a component helps unless the interval excludes zero.**
**EXPECT** – with only 6 folds, most intervals will be wide; a component is "supported" only if the improvement is consistent across bearings.
""")

code(r'''
chain = [("cnn", "cnn_lstm", "CNN -> CNN+LSTM (add recurrence)"), ("cnn_lstm", "cnn_bilstm", "LSTM -> BiLSTM (make it bidirectional)"),
         ("cnn_bilstm", "cnn_bilstm_attn", "add attention (single head)"), ("cnn_bilstm_attn", "proposed", "add second head (dual head)"),
         ("cnn_bilstm", "cnn_bilstm_dual", "add second head WITHOUT attention"), ("cnn_bilstm_dual", "proposed", "add attention to the dual-head model"),
         ("lstm", "cnn_lstm", "add CNN in front of the LSTM")]
rows = []
for a_, b_, lab in chain:
    for met, hib, nm in [("AUC_mean_h", True, "detection AUC (mean over horizons)"), ("MAE_min_actionable", False, "actionable RUL MAE (min)"), ("stage_macroF1", True, "stage macro-F1")]:
        va = fold_mean.xs(a_, level="model")[met].values; vb = fold_mean.xs(b_, level="model")[met].values
        r = R.paired_difference(va, vb, a_, b_, higher_is_better=hib); r.update({"step": lab, "metric": nm}); rows.append(r)
abl = pd.DataFrame(rows)[["step", "metric", "mean_diff", "ci_lo", "ci_hi", "units_improved", "verdict"]]
display(abl.round(3)); abl.to_csv(C.ARTIFACTS / "ablation_lobo.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
for a_, (met, nm) in zip(ax, [("AUC_mean_h", "Detection AUC (higher better)"), ("MAE_min_actionable", "Actionable RUL MAE, min (lower better)"), ("stage_macroF1", "Stage macro-F1 (higher better)")]):
    sub = abl[abl.metric == {"AUC_mean_h": "detection AUC (mean over horizons)", "MAE_min_actionable": "actionable RUL MAE (min)", "stage_macroF1": "stage macro-F1"}[met]]
    y = np.arange(len(sub)); a_.errorbar(sub.mean_diff, y, xerr=[sub.mean_diff - sub.ci_lo, sub.ci_hi - sub.mean_diff], fmt="o", color="#d62728", capsize=4)
    a_.axvline(0, color="k", ls="--"); a_.set_yticks(y); a_.set_yticklabels(sub.step, fontsize=8); a_.invert_yaxis(); a_.set_title(f"Change in {nm}", fontsize=10); a_.set_xlabel("paired difference (mean and 95% bootstrap CI over 6 bearings)")
plt.tight_layout(); R.savefig(fig, "fig_ablation_lobo.png"); plt.show()
R.show("**Figure 13.** Paired change caused by each architectural step. A component is supported only if its interval excludes zero in the helpful direction.")
''')

code(r'''
def verdicts(metric):
    d = abl[abl.metric == metric].set_index("step").verdict
    return d
rows = []
for step in abl.step.unique():
    s = abl[abl.step == step].set_index("metric").verdict
    rows.append(f"* **{step}** -> detection AUC: *{s['detection AUC (mean over horizons)']}*; actionable RUL MAE: *{s['actionable RUL MAE (min)']}*; stage macro-F1: *{s['stage macro-F1']}*")
att = abl[(abl.step == "add attention (single head)")].set_index("metric").verdict
dual = abl[(abl.step == "add second head (dual head)")].set_index("metric").verdict
R.show("### 18.1 What did we find?\n\n" + "\n".join(rows) +
       f"\n\n**Reading (validation bearings only).** Adding attention to the single-head model: actionable RUL MAE *{att['actionable RUL MAE (min)']}*, detection AUC *{att['detection AUC (mean over horizons)']}*. "
       f"Adding the second head: actionable RUL MAE *{dual['actionable RUL MAE (min)']}*, detection AUC *{dual['detection AUC (mean over horizons)']}*. "
       "No claim of improved accuracy is made from the validation ablation alone: the effects differ between metrics, and the **combined verdict for H2 (validation + test, every contrast) is computed in Section 26**.\n\n"
       "A verdict of *no measurable difference* means the experiment cannot distinguish the step from zero with 6 bearings; it is **not** evidence that the step is harmful or useless, only that it was not shown to help.")
''')

# =====================================================================================================================
md(r"""
## 19. Evaluation of the Validation Predictions

**WHAT** – look at the LOBO predictions in detail: actual vs predicted RUL, per-bearing trajectories, error distribution, ROC curves for detecting "failure within 60 minutes", and confusion matrices of the stage.
**WHY** – aggregated numbers hide *where* a model fails (Figure 14–19).
**HOW TO READ.** Each held-out bearing was predicted by a model that never saw it.
""")

code(r'''
SHOW = ["Constant", "Random Forest", "lstm", "proposed"]
SEED0 = SEEDS_LOBO[0]
pl = pred_lobo[(pred_lobo.seed == SEED0)]
# ---- Figure: actual vs predicted RUL ----
fig, ax = plt.subplots(1, 4, figsize=(21, 5), sharex=True, sharey=True)
for a_, m in zip(ax, SHOW):
    g = pl[pl.model == m]
    a_.scatter(g.rul_cap_min, g.pred_min, s=3, alpha=0.25, c=[COND_COLOR[c] for c in g.condition])
    a_.plot([0, 120], [0, 120], "k--", lw=1); a_.set_title(E.LABEL[m]); a_.set_xlabel("Actual capped RUL (min)")
ax[0].set_ylabel("Predicted RUL (min)"); plt.tight_layout(); R.savefig(fig, "fig_actual_vs_predicted_lobo.png"); plt.show()
R.show("**Figure 14.** Actual vs predicted RUL for held-out bearings (colour = operating condition, dashed = perfect). A model with skill would follow the diagonal; a constant predictor is a horizontal line.")

# ---- Figure: per-bearing trajectories ----
fig, axes = plt.subplots(2, 3, figsize=(21, 9)); axes = axes.ravel()
for a_, b in zip(axes, C.LEARNING):
    for m, col, ls in [("Constant", "#999999", ":"), ("Random Forest", "#8da0cb", "-"), ("proposed", "#d62728", "-")]:
        g = pl[(pl.model == m) & (pl.bearing == b)].sort_values("time_min")
        a_.plot(g.time_min, g.pred_min, color=col, ls=ls, lw=1.2, label=E.LABEL[m])
    g = pl[(pl.model == "proposed") & (pl.bearing == b)].sort_values("time_min"); a_.plot(g.time_min, g.rul_cap_min, "k--", lw=2, label="true capped RUL")
    a_.set_title(f"{b} (held out, condition {C.condition_of(b)})"); a_.set_xlabel("Operating time (min)"); a_.set_ylabel("RUL (min)")
axes[0].legend(fontsize=8); plt.tight_layout(); R.savefig(fig, "fig_per_bearing_lobo.png"); plt.show()
R.show("**Figure 15.** Predicted vs true capped RUL over the whole life of each held-out learning bearing (seed 42). Look at *when* the prediction drops: for gradually degrading bearings it starts before failure; for abrupt bearings it starts too late.")
''')

code(r'''
from sklearn.metrics import roc_curve, confusion_matrix
# ---- ROC for 'failure within 60 min' and 'within 10 min' (pooled over held-out bearings, seed 42) ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for a_, h in zip(ax, [60, 10]):
    for m, col in [("Constant", "#999999"), ("Random Forest", "#8da0cb"), ("Gradient Boosting", "#66c2a5"), ("lstm", "#fc8d62"), ("proposed", "#d62728")]:
        g = pl[pl.model == m]; y = (g.rul_min <= h).astype(int)
        fpr, tpr, _ = roc_curve(y, -g.pred_min); a_.plot(fpr, tpr, color=col, lw=2, label=f"{E.LABEL[m]} (AUC {roc_auc_score(y, -g.pred_min):.2f})")
    a_.plot([0, 1], [0, 1], "k:"); a_.set_xlabel("False-positive rate"); a_.set_ylabel("True-positive rate"); a_.set_title(f"ROC: failure within {h} min (pooled held-out windows)"); a_.legend()
plt.tight_layout(); R.savefig(fig, "fig_roc_lobo.png"); plt.show()
R.show("**Figure 16.** ROC curves for detecting an imminent failure from the predicted RUL. The longer horizon (left) is much harder than the short one (right).")

# ---- confusion matrices for the proposed model: native stage head vs uniform rule ----
g = pl[pl.model == "proposed"]
cm_n = confusion_matrix(g.stage, g[["prob0", "prob1", "prob2"]].values.argmax(1), labels=[0, 1, 2]); cm_u = confusion_matrix(g.stage, Mx.rul_to_stage(g.pred_min.values), labels=[0, 1, 2])
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a_, cm, ttl in [(ax[0], cm_n, "stage head (native)"), (ax[1], cm_u, "stage from predicted RUL (uniform rule)")]:
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt="d", cmap="Blues", ax=a_, xticklabels=["Normal", "Warning", "Critical"], yticklabels=["Normal", "Warning", "Critical"], cbar_kws={"label": "row-normalised"})
    a_.set_xlabel("Predicted stage"); a_.set_ylabel("True stage"); a_.set_title(f"Proposed model, {ttl}\n(counts, colour = row fraction)")
plt.tight_layout(); R.savefig(fig, "fig_confusion_lobo.png"); plt.show()
R.show("**Figure 17.** Confusion matrices of the proposed model on the held-out learning bearings. Colour is the fraction of each *true* class; numbers are window counts.")

# ---- error distribution ----
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for m, col in [("Constant", "#999999"), ("Random Forest", "#8da0cb"), ("proposed", "#d62728")]:
    g = pl[pl.model == m]; e_ = g.pred_min - g.rul_cap_min
    sns.kdeplot(e_, ax=ax[0], color=col, label=E.LABEL[m], lw=2); sns.kdeplot((g.pred_min - g.rul_cap_min)[g.rul_min <= 60], ax=ax[1], color=col, label=E.LABEL[m], lw=2)
ax[0].set_title("Error (predicted - true capped RUL), all windows"); ax[1].set_title("Error, actionable region (true RUL <= 60 min)")
for a_ in ax: a_.axvline(0, color="k", ls=":"); a_.set_xlabel("Error (min); positive = predicted too much life left (late warning)"); a_.legend()
plt.tight_layout(); R.savefig(fig, "fig_error_distribution_lobo.png"); plt.show()
R.show("**Figure 18.** Error distributions. In the actionable region a shift to the right means the model over-estimates the remaining life, i.e. warns too late — the dangerous direction.")
''')

code(r'''
# per-bearing table of the proposed model on the held-out learning bearings (seed 42)
g = pl[pl.model == "proposed"].reset_index(drop=True)
pb_lobo = Mx.per_bearing_table(g[["bearing", "snapshot_idx", "time_min", "rul_min", "rul_cap_min", "y_rul", "stage", "condition"]], g.pred_min.values, Mx.rul_to_stage(g.pred_min.values))
display(pb_lobo[["windows", "MAE_min", "MAE_min_actionable", "stage_macroF1", "false_alarm_rate", "alarm_at_end_bearings", "lead_sustained_min_median"]].round(2))
''')

md(r"""
### 19.1 What did we find?
The scatter plots and trajectories make the numbers of Section 17 concrete. Read them with three questions: (1) does the prediction *start to fall before the true RUL reaches the warning zone*? (2) is the fall smooth or does it jump only at the very end? (3) how large are the false alarms while the bearing is healthy? The per-bearing table shows that **performance is very uneven across held-out bearings**, which is consistent with the horizon analysis of Section 11: the models can only warn as early as the vibration changes.
""")

# =====================================================================================================================
md(r"""
## 20. Explainability

**WHAT** – three complementary analyses, all on the **held-out learning bearings** (no test data): (a) the average **attention profile** per true stage and a *functional* test (replace the learned attention by uniform weights inside the same trained network and see whether predictions change); (b) **permutation feature importance** — shuffle one feature across held-out windows and measure how much the error grows (proposed model and Random Forest, per fold); (c) a representative bearing explained.
**WHY** – "explainable" must be measured, not asserted. Attention weights are a property of the model, not a proof of what is *physically causing* the prediction: attention weights can be uncorrelated with feature importance and very different weights can give the same prediction [9]. We therefore test what attention does functionally and report feature importance separately.
**EXPECT** – if attention matters, replacing it by uniform weights should change predictions noticeably; if the profile is near uniform, it explains little.
""")

code(r'''
from tensorflow.keras import layers as KL, Model as KModel
from scipy.stats import spearmanr
att_rows, abl_rows = [], []
fold_windows = {}
for fi, f in enumerate(D.lobo_folds()):
    net_f, sc_f = kept_lobo[("proposed", SEED0, fi)]
    Xh, mh = D.make_windows(table, feats, [f["held_out"]], sc_f); fold_windows[fi] = (Xh, mh)
    attn_m = KModel(net_f.input, net_f.get_layer("temporal_attention").output[1]); A_ = attn_m.predict(Xh, batch_size=512, verbose=0)[:, :, 0]
    ent = -(A_ * np.log(A_ + 1e-12)).sum(1); last4 = A_[:, -4:].sum(1)
    for s_, nm in enumerate(["Normal", "Warning", "Critical"]):
        m_ = mh.stage.values == s_
        if m_.any(): att_rows.append(dict(fold=fi, stage=nm, last4=last4[m_].mean(), entropy=ent[m_].mean(), profile=A_[m_].mean(0)))
    # functional test: uniform weights inside the same network
    u = KL.GlobalAveragePooling1D()(net_f.get_layer("drop_bilstm").output)
    u = net_f.get_layer("rul_output")(net_f.get_layer("rul_dense")(net_f.get_layer("shared_dense")(u)))
    uni = KModel(net_f.input, u)
    y_att = np.clip(net_f.predict(Xh, batch_size=512, verbose=0)["rul_output"].ravel(), 0, 1) * C.RUL_CAP_MIN
    y_uni = np.clip(uni.predict(Xh, batch_size=512, verbose=0).ravel(), 0, 1) * C.RUL_CAP_MIN
    act = mh.rul_min.values <= C.STAGE_WARNING_MIN
    abl_rows.append(dict(fold=fi, MAE_with_attention=np.abs(y_att - mh.rul_cap_min).mean(), MAE_uniform_weights=np.abs(y_uni - mh.rul_cap_min).mean(),
                         actionable_MAE_att=np.abs(y_att - mh.rul_cap_min)[act].mean(), actionable_MAE_uniform=np.abs(y_uni - mh.rul_cap_min)[act].mean(),
                         corr=np.corrcoef(y_att, y_uni)[0, 1], mean_abs_change_min=np.abs(y_att - y_uni).mean()))
att = pd.DataFrame(att_rows); abl_att = pd.DataFrame(abl_rows)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for nm, col in [("Normal", "#2ca02c"), ("Warning", "#ff7f0e"), ("Critical", "#d62728")]:
    P_ = np.stack(att[att.stage == nm].profile.values); ax[0].errorbar(np.arange(1, 17), P_.mean(0), yerr=P_.std(0), marker="o", color=col, capsize=2, label=f"{nm} (mean +/- std over folds)")
ax[0].axhline(1 / 16, color="k", ls="--", label="uniform (1/16)"); ax[0].set_xlabel("Step in the 160 s window (16 = newest)"); ax[0].set_ylabel("Mean attention weight alpha"); ax[0].set_title("(a) Average attention profile by TRUE stage (held-out bearings)"); ax[0].legend(fontsize=8)
ax[1].scatter(abl_att.MAE_with_attention, abl_att.MAE_uniform_weights, s=70, c="#d62728"); lim = [abl_att[["MAE_with_attention", "MAE_uniform_weights"]].min().min() - 1, abl_att[["MAE_with_attention", "MAE_uniform_weights"]].max().max() + 1]
ax[1].plot(lim, lim, "k--"); ax[1].set_xlabel("MAE with learned attention (min)"); ax[1].set_ylabel("MAE with UNIFORM weights (min)"); ax[1].set_title("(b) Functional test: is the learned attention needed? (one point per fold)")
plt.tight_layout(); R.savefig(fig, "fig_attention_lobo.png"); plt.show()
piv = att.pivot(index="fold", columns="stage", values="last4")
R.show(f"**Figure 20.** (a) Mean attention weight on the newest 4 steps (uniform = 0.25): Normal {piv.Normal.mean():.3f}, Warning {piv.Warning.mean():.3f}, Critical {piv.Critical.mean():.3f}; Critical > Normal in {(piv.Critical > piv.Normal).sum()}/{piv.Critical.notna().sum()} folds. "
       f"Entropy (uniform = {np.log(16):.3f}): {att.entropy.mean():.3f}. (b) Replacing the learned attention by uniform weights changes the RUL prediction by {abl_att.mean_abs_change_min.mean():.1f} min on average (correlation {abl_att['corr'].mean():.3f}); MAE {abl_att.MAE_with_attention.mean():.1f} -> {abl_att.MAE_uniform_weights.mean():.1f} min.")
''')

code(r'''
# ---- (b) permutation importance on held-out windows, per fold (proposed = fold model; Random Forest = refit in the fold) ----
def perm_importance(predict, Xh, mh, feat_axis_last=True, reps=2, seed=SEED):
    rng = np.random.default_rng(seed)
    def scores(pred_min):
        act = mh.rul_min.values <= C.STAGE_WARNING_MIN
        return np.abs(pred_min - mh.rul_cap_min.values)[act].mean(), Mx.detection_metrics(mh, -pred_min)["AUC_mean_h"]
    b_mae, b_auc = scores(predict(Xh)); out = []
    for j, name in enumerate(F.FEATURES):
        d_mae, d_auc = [], []
        for _ in range(reps):
            Xp = Xh.copy(); perm = rng.permutation(len(Xp))
            if Xp.ndim == 3: Xp[:, :, j] = Xp[perm, :, j]
            else: Xp[:, j] = Xp[perm, j]
            m_, a_ = scores(predict(Xp)); d_mae.append(m_ - b_mae); d_auc.append(b_auc - a_)
        out.append((name, np.mean(d_mae), np.mean(d_auc)))
    return pd.DataFrame(out, columns=["feature", "dMAE_actionable", "dAUC"]).set_index("feature")
imp_nn, imp_rf = [], []
for fi, f in enumerate(D.lobo_folds()):
    net_f, sc_f = kept_lobo[("proposed", SEED0, fi)]; Xh, mh = fold_windows[fi]
    imp_nn.append(perm_importance(lambda X: np.clip(net_f.predict(X, batch_size=1024, verbose=0)["rul_output"].ravel(), 0, 1) * C.RUL_CAP_MIN, Xh, mh))
    sc2 = D.fit_scaler(feats, table, f["train"] + [f["val"]]); Xt, mt = D.make_windows(table, feats, f["train"] + [f["val"]], sc2); Xh2, mh2 = D.make_windows(table, feats, [f["held_out"]], sc2)
    rf_f = M.Classical("Random Forest", SEED).fit(Xt, mt)
    imp_rf.append(perm_importance(lambda X: rf_f.predict(X)["pred_min"], Xh2, mh2))
    print(f"  fold {fi + 1}/6 done")
I_nn = pd.concat(imp_nn).groupby(level=0).mean(); I_rf = pd.concat(imp_rf).groupby(level=0).mean()
I_nn.to_csv(C.ARTIFACTS / "perm_importance_proposed_lobo.csv"); I_rf.to_csv(C.ARTIFACTS / "perm_importance_rf_lobo.csv")
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
for a_, I, ttl in [(ax[0], I_nn, "Proposed model"), (ax[1], I_rf, "Random Forest")]:
    t_ = I.sort_values("dAUC", ascending=False).head(12).iloc[::-1]
    a_.barh(t_.index, t_.dAUC, color=["#1f77b4" if i.startswith("horiz") else "#2ca02c" for i in t_.index], edgecolor="black")
    a_.set_xlabel("Drop in detection AUC when the feature is shuffled"); a_.set_title(f"{ttl}: permutation importance, mean over 6 held-out bearings\n(blue = horizontal, green = vertical)")
plt.tight_layout(); R.savefig(fig, "fig_feature_importance_lobo.png"); plt.show()
rho = spearmanr(I_nn.dAUC, I_rf.dAUC)[0]
R.show(f"**Figure 21.** Top features by permutation importance. Proposed: " + ", ".join(f"`{i}` ({v:+.3f})" for i, v in I_nn.dAUC.sort_values(ascending=False).head(4).items()) +
       ". Random Forest: " + ", ".join(f"`{i}` ({v:+.3f})" for i, v in I_rf.dAUC.sort_values(ascending=False).head(4).items()) +
       f". Rank agreement between the two models (Spearman): **{rho:.2f}**. Importance of a feature inside a correlated block (Figure 7) is shared with its partners.")
''')

code(r'''
# ---- (c) representative explanation: one gradual bearing (Bearing1_1, fold 0) ----
fi = 0; b = C.LEARNING[fi]; net_f, sc_f = kept_lobo[("proposed", SEED0, fi)]; Xh, mh = fold_windows[fi]
attn_m = KModel(net_f.input, net_f.get_layer("temporal_attention").output[1]); A_ = attn_m.predict(Xh, batch_size=512, verbose=0)[:, :, 0]
pr = np.clip(net_f.predict(Xh, batch_size=512, verbose=0)["rul_output"].ravel(), 0, 1) * C.RUL_CAP_MIN
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
ax[0].plot(mh.time_min, mh.rul_cap_min, "k--", lw=2, label="true capped RUL"); ax[0].plot(mh.time_min, pr, color="#d62728", label="proposed prediction"); ax[0].set_ylabel("RUL (min)"); ax[0].legend(); ax[0].set_title(f"{b} (held out): prediction, the features that moved, and where the model looked")
top_f = list(I_nn.dAUC.sort_values(ascending=False).head(3).index)
for f_ in top_f:
    m = (table.bearing == b).values; g = table[m]
    ax[1].plot(g.time_min[C.WINDOW - 1:], (feats.loc[m, f_].values - feats.loc[m, f_].mean())[C.WINDOW - 1:] / (feats.loc[m, f_].std() + 1e-9), lw=0.9, label=f_)
ax[1].set_ylabel("standardised feature"); ax[1].legend(fontsize=8)
im = ax[2].imshow(A_.T, aspect="auto", origin="lower", extent=[mh.time_min.min(), mh.time_min.max(), 0.5, 16.5], cmap="magma"); ax[2].set_ylabel("step in window (16 = newest)"); ax[2].set_xlabel("Operating time (min)")
plt.colorbar(im, ax=ax[2], label="attention weight", pad=0.01); plt.tight_layout(); R.savefig(fig, "fig_explanation_example_lobo.png"); plt.show()
R.show("**Figure 22.** Example explanation for a held-out gradual bearing: prediction, the three features with the highest permutation importance, and the attention heat-map. Attention is shown as *what the model attended to*, not as the cause of the failure.")
''')

code(r'''
p_uni = abl_att.mean_abs_change_min.mean(); c_uni = abl_att["corr"].mean()
verdict_att = ("**functionally important**: removing it changes predictions substantially" if p_uni > 10 or c_uni < 0.9 else "**functionally weak**: replacing it by uniform weights changes predictions only slightly")
R.show(f"### 20.1 What did we find?\n\n"
       f"* **Attention:** the attention layer is {verdict_att} (mean absolute change {p_uni:.1f} min, correlation {c_uni:.3f}). The stage-wise profiles differ only modestly from uniform weights (Figure 20a). "
       f"Attention therefore tells us *where in the 160 s window the network put more weight*, not *why the bearing is failing*.\n"
       f"* **Features:** permutation importance identifies which vibration features the models rely on (Figure 21); rank agreement between the proposed model and Random Forest is {rho:.2f}, so the two models rely on partly different cues, and correlated features share importance.\n"
       f"* **Does this support our hypothesis that the model is explainable?** Only in a limited sense: the model exposes inspectable weights and feature dependencies, but attention is **not** a causal explanation [9], and the importance analysis is descriptive. We claim *inspectability*, not *causal explanation*.")
''')

# =====================================================================================================================
md(r"""
## 21. Early Warning Analysis (validation bearings)

**WHAT** – turn predictions into alarms and look at them as a maintenance engineer would: *when* does the model raise an alarm, *how much time is left*, and *how often does it cry wolf?*
**RULE (identical for all models).** An alarm is raised when the predicted stage is at least Warning (predicted RUL ≤ 60 min) for **3 consecutive windows**. Lead time = true remaining minutes at the first alarm inside the warning zone; false-alarm rate = fraction of truly Normal windows in alarm.
**EXPECT** – gradual bearings give tens of minutes of warning; abrupt bearings give little or none.
""")

code(r'''
rows = []
for m in E.ALL_MODELS:
    g = sm[sm.model == m]
    rows.append({"model": E.LABEL[m], "false_alarm_rate": g.false_alarm_rate.mean(), "alarm active at end (of 6)": g.alarm_at_end_bearings.mean() * 6 / 6 * 1.0,
                 "median sustained lead (min)": g.lead_sustained_min_median.median(), "median in-zone lead (min, lenient)": g.lead_min_median.median()})
ew = pd.DataFrame(rows).set_index("model"); ew["alarm active at end (of 6)"] = [sm[sm.model == m].groupby("fold").alarm_at_end_bearings.mean().sum() for m in E.ALL_MODELS]; display(ew.round(2))
# per bearing for the proposed model
g = sm[(sm.model == "proposed") & (sm.seed == SEED0)]
ewb = pd.DataFrame({"held-out bearing": [C.LEARNING[f] for f in g.fold], "alarm active at end": g.alarm_at_end_bearings.astype(bool).values, "sustained lead (min)": g.lead_sustained_min_median.values,
                    "false-alarm rate": g.false_alarm_rate.values, "AUC h10": g.AUC_h10.values, "AUC h60": g.AUC_h60.values}).set_index("held-out bearing")
display(ewb.round(2))

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
for a_, b in zip(axes, ["Bearing1_1", "Bearing2_2", "Bearing3_2"]):
    g = pl[(pl.model == "proposed") & (pl.bearing == b)].sort_values("time_min")
    ps = Mx.rul_to_stage(g.pred_min.values); alarm = Mx.sustained(ps >= 1)
    a_.plot(g.time_min, g.rul_cap_min, "k--", lw=1.5, label="true capped RUL"); a_.plot(g.time_min, g.pred_min, color="#d62728", lw=1, label="predicted RUL")
    a_.fill_between(g.time_min, 0, 125, where=g.stage.values >= 1, color="orange", alpha=0.15, label="true Warning/Critical zone"); a_.fill_between(g.time_min, 0, 125, where=g.stage.values == 2, color="red", alpha=0.15, label="true Critical zone")
    a_.scatter(g.time_min[alarm], np.full(alarm.sum(), 128), marker="|", color="purple", s=40, label="alarm")
    a_.set_title(f"{b}: early-warning timeline (held out)"); a_.set_ylabel("RUL (min)"); a_.set_ylim(-2, 135)
axes[-1].set_xlabel("Operating time (min)"); axes[0].legend(ncol=5, fontsize=8); plt.tight_layout(); R.savefig(fig, "fig_early_warning_lobo.png"); plt.show()
R.show("**Figure 23.** Early-warning timelines of three held-out learning bearings (gradual, hump, abrupt). Shaded = true warning/critical zones; purple ticks = alarms of the proposed model under the uniform rule.")
''')

md(r"""
### 21.1 What did we find?
""")

code(r'''
pp_ = sm[sm.model == "proposed"]
far = pp_.false_alarm_rate.mean(); atend = int(pp_.groupby("fold").alarm_at_end_bearings.mean().sum())
weak = ewb[(~ewb["alarm active at end"]) | (ewb["sustained lead (min)"] < 10)]
R.show(f"* Under the uniform alarm rule the proposed model has a **false-alarm rate of {far:.2f}** on truly healthy windows in the validation bearings (Random Forest {sm[sm.model=='Random Forest'].false_alarm_rate.mean():.2f}, Gradient Boosting {sm[sm.model=='Gradient Boosting'].false_alarm_rate.mean():.2f}). A rate this high means the alarm is often already on before the warning zone; the *lenient* lead time therefore saturates at 60 min for almost every model and is **not** a meaningful measure of early warning.\n\n"
       f"* The strict view: the alarm is still active at the last window for {atend} of 6 held-out bearings (seed-mean); the median sustained lead is {ew.loc[E.LABEL['proposed'], 'median sustained lead (min)']:.1f} min. Bearings without a useful warning (alarm not active at the end or sustained lead < 10 min): " + (", ".join(weak.index) if len(weak) else "none") + ".\n\n"
       "* **H3 (a useful warning for every bearing)** cannot be concluded from this validation view alone; it is evaluated on the test bearings in Section 26. The false-alarm price of alarming early is the dominant limitation of the alarm layer.")
''')
