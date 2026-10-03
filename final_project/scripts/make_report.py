"""Builds FINAL_RESEARCH_REPORT.md from the executed notebook outputs and the saved artifacts (single source of truth)."""
import json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # project root (this script lives in scripts/)
ART = ROOT / "artifacts"
nb = json.load(open(ROOT / "final_bearing_prognostics_review.ipynb", encoding="utf8"))


def md_out(marker):
    """Return the markdown output of the first cell output that contains `marker`."""
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            t = o.get("data", {}).get("text/markdown")
            if t:
                t = "".join(t) if isinstance(t, list) else t
                if marker in t:
                    return t
    raise KeyError(marker)


def table(df, floatfmt="{:.3f}"):
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = []
    for r in d.values:
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return head + "\n".join(rows)


sel = json.load(open(ART / "selection.json"))
test = pd.read_csv(ART / "final_model_comparison_test.csv", index_col=0)
lobo = pd.read_csv(ART / "lobo_model_comparison.csv", index_col=0)
abl_l = pd.read_csv(ART / "ablation_lobo.csv"); abl_t = pd.read_csv(ART / "ablation_test.csv")
pb = pd.read_csv(ART / "final_per_bearing_proposed.csv", index_col=0)
crit = pd.read_csv(ART / "final_multicriteria_comparison.csv", index_col=0)
audit = pd.read_csv(ART / "dataset_audit_table.csv", index_col=0)
cfg = json.load(open(ART / "experiment_config.json"))

keep_test = ["MAE_min", "MAE_min_actionable", "AUC_h10", "AUC_h60", "AUC_mean_h", "stage_macroF1", "false_alarm_rate", "PHM_official", "alarm active at end (of 11)"]
keep_lobo = ["MAE_min", "MAE_min_actionable", "AUC_h10", "AUC_h60", "AUC_mean_h", "stage_macroF1", "false_alarm_rate", "alarm active at end (of 6)", "params"]

h_block = md_out("**H1 - ")
find_test = md_out("**Detection (mean AUC over horizons 5-60 min")
find_lobo = md_out("### 17.1 What did we find?")
attn = md_out("### 20.1 What did we find?")
random_leak = md_out("**Figure 9 / table.**")
form_find = md_out("**Direct / capped RUL in minutes (A, A')")
anoms = md_out("### Anomalies found by the audit")

rep = f"""# FINAL RESEARCH REPORT
**Explainable Dual-Head CNN-BiLSTM-Attention Prognostics for Industrial Bearing Health Monitoring - PRONOSTIA / IEEE PHM 2012**

Primary deliverable: `final_bearing_prognostics_review.ipynb` (executed from a clean kernel). This report is generated from that notebook's outputs and artifacts; every number below is copied from them. Source code: `src/`. Sources: `REFERENCES.md`. Audit: `FINAL_AUDIT_automated.md`, `FINAL_AUDIT.md`. Dataset audit: `DATASET_AUDIT_REPORT.md`.

## 1. Research question
Given the recent vibration history of a bearing, can we estimate its remaining useful life (RUL) and health stage for a bearing that was **never seen** during training, and does each component of a CNN-BiLSTM-Attention dual-head network measurably help compared with simple baselines?

## 2. Dataset
PRONOSTIA / IEEE PHM 2012 (FEMTO-ST): 17 run-to-failure bearings (6 learning, 11 test), 3 operating conditions, two accelerometers, 25.6 kHz, 2560 samples per 0.1 s snapshot every 10 s [1,2]. Lifetimes {audit.lifetime_min.min():.0f}-{audit.lifetime_min.max():.0f} min. All {int(audit.acc_files.sum()):,} files were audited (format, delimiters, NaN, clock stamps, 20 g behaviour, consistency with the official RUL table). Findings:

{anoms.split(chr(10), 1)[1].strip()}

## 3. Methodology
* **Unit of independence = the bearing.** Method development and model selection use only the 6 learning bearings with leave-one-bearing-out (LOBO) validation (a separate validation bearing is used only for early stopping). The 11 test bearings are scored once after the selection file `selection.json` is written.
* **Target (formulation E, multi-task).** Capped RUL in minutes (cap {cfg['rul_cap_min']:.0f}) and a 3-class stage (Normal > 60 min, Warning 20-60 min, Critical <= 20 min) derived **only from the failure time** (official durations; test bearings: truncation point + official RUL, which corrects Bearing1_4). No vibration-based onset label is used for training. Horizons were fixed a priori and checked for sensitivity.
* **Formulation study (learning bearings only):** {form_find.replace(chr(10)+chr(10), chr(10))}
* **Features.** 34 documented time/frequency features from one snapshot each (causal); fixed log/signed-log transforms; `StandardScaler` fitted on training bearings only. Input = last {cfg['window']} snapshots.
* **Leakage control.** {random_leak.strip()} Automated checks (splits, scaler, poison test, truncation/causality test, window boundaries, independent feature recomputation, target-shuffle test, frozen configuration) stop the notebook on violation.
* **Models (same windows, targets, folds, metrics).** Constant, Ridge, Random Forest, Gradient Boosting, CNN, LSTM, CNN+LSTM, CNN+BiLSTM, CNN+BiLSTM+Attention (single head), CNN+BiLSTM (dual head, no attention), and the proposed CNN+BiLSTM+Attention dual-head model ({cfg['parameters']['proposed']:,} parameters). Training: Adam {cfg['learning_rate']}, batch {cfg['batch_size']}, Huber({0.1}) + 0.5 x cross-entropy, early stopping on the validation bearing, deterministic TensorFlow, seeds {cfg['seeds']['lobo']} (LOBO) and {cfg['seeds']['final']} (final).
* **Metrics.** MAE (minutes) overall and in the *actionable region* (true RUL <= 60 min); ROC-AUC of detecting "failure within H minutes" for H = 5, 10, 20, 30, 60; stage macro-F1; false-alarm rate; alarm active at the end of life; official PHM-2012 score at the truncation points [2].

## 4. Experiments and results
### 4.1 Leave-one-bearing-out (6 held-out learning bearings; mean +/- std over bearings)
{table(lobo[[c for c in keep_lobo if c in lobo.columns]])}

{find_lobo.split(chr(10), 2)[2].strip()}

### 4.2 Ablation (LOBO, paired over 6 bearings, 95% bootstrap CI)
{table(abl_l)}

### 4.3 Final unseen test (11 test bearings; mean +/- std over 3 seeds)
{table(test[[c for c in keep_test if c in test.columns]])}

{find_test.strip()}

Ablation on the test bearings (paired over 11 bearings):

{table(abl_t)}

### 4.4 Hypotheses (computed from the measurements)
{h_block.strip()}

### 4.5 Per-bearing results of the proposed model on the test bearings
{table(pb[[c for c in ['windows', 'MAE_min', 'MAE_min_actionable', 'AUC_h10', 'AUC_h60', 'stage_macroF1', 'false_alarm_rate', 'alarm_at_end_bearings', 'lead_sustained_min_median'] if c in pb.columns]])}

### 4.6 Objective multi-criteria comparison
{table(crit[[c for c in ['detection AUC LOBO', 'detection AUC test', 'actionable MAE LOBO (min)', 'actionable MAE test (min)', 'overall MAE test (min)', 'false-alarm rate test', 'PHM-2012 score test', 'stability: std of test AUC over seeds', 'parameters', 'fit+predict s (LOBO median)', 'interpretability'] if c in crit.columns]])}

Selection made on LOBO only (saved before the test): `{json.dumps({k: v for k, v in sel.items() if k in ('best_actionable_RUL_MAE', 'best_detection_AUC_mean_h', 'lowest_false_alarm_rate', 'interpretable_by_design')})}`.

## 5. Explainability
{attn.split(chr(10), 2)[2].strip()}

## 6. Limitations
Six independent learning bearings; abrupt failures with 1-2 minutes of visible degradation for several Condition-2/3 bearings; label uncertainty (20 g not observable in every snapshot; Bearing1_4 extends past its official end of life); hindsight in offline (descriptive) analyses; **prior exposure to test-bearing curves in earlier experiments (v1/v2)** - the final target avoids vibration-derived labels, but the choice of formulation was motivated by earlier results; detection is not RUL estimation; official PHM score is noisy with 11 capped predictions; attention is not a causal explanation; no hyper-parameter search; window length not varied; single test rig. See notebook Section 25.

## 7. Conclusion
The supported claims are those printed in Sections 4.4-4.6 and in notebook Section 26. In short: on unseen bearings the vibration-based models recognise *imminent* failure better than a constant predictor (but far from perfectly, with large differences between bearings), but minute-level RUL is not reliably predicted, warnings are unreliable for bearings that fail abruptly, alarms carry a high false-alarm cost, and the proposed architecture is **not shown to be more accurate than simple competitive baselines**; its distinctive property is inspectability (with the limits stated in Section 5).

## References
See `REFERENCES.md`.
"""
(ROOT / "reports" / "FINAL_RESEARCH_REPORT.md").write_text(rep, encoding="utf8")
print("report written:", len(rep), "characters")
