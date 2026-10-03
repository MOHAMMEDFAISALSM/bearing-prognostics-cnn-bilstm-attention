# FINAL RESEARCH REPORT
**Explainable Dual-Head CNN-BiLSTM-Attention Prognostics for Industrial Bearing Health Monitoring - PRONOSTIA / IEEE PHM 2012**

Primary deliverable: `final_bearing_prognostics_review.ipynb` (executed from a clean kernel). This report is generated from that notebook's outputs and artifacts; every number below is copied from them. Source code: `src/`. Sources: `REFERENCES.md`. Audit: `FINAL_AUDIT_automated.md`, `FINAL_AUDIT.md`. Dataset audit: `DATASET_AUDIT_REPORT.md`.

## 1. Research question
Given the recent vibration history of a bearing, can we estimate its remaining useful life (RUL) and health stage for a bearing that was **never seen** during training, and does each component of a CNN-BiLSTM-Attention dual-head network measurably help compared with simple baselines?

## 2. Dataset
PRONOSTIA / IEEE PHM 2012 (FEMTO-ST): 17 run-to-failure bearings (6 learning, 11 test), 3 operating conditions, two accelerometers, 25.6 kHz, 2560 samples per 0.1 s snapshot every 10 s [1,2]. Lifetimes 38-467 min. All 24,889 files were audited (format, delimiters, NaN, clock stamps, 20 g behaviour, consistency with the official RUL table). Findings:

* **Bearing1_4**: Full_Test_Set has 289 snapshots after the truncation point (= 2890 s) but the official actual RUL is 339 s -> the extra 2551 s of data lie **after the official end of life**
* **Bearing1_1**: 3 irregular clock steps (logging glitch in the time stamps only; file order and the 10 s interval from [2] are used instead)
* **Bearing1_1**: first snapshot above 20 g already at 46.3% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration
* **Bearing1_2**: first snapshot above 20 g already at 14.6% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration
* **Bearing1_3**: first snapshot above 20 g already at 68.9% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration
* **Bearing1_4**: first snapshot above 20 g already at 78.8% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration
* **Bearing1_5**: no recorded 0.1 s snapshot ever exceeds 20 g (peak 14.1 g) -> the 20 g end-of-life criterion is **not observable in the snapshots** for this bearing
* **Bearing1_6**: first snapshot above 20 g already at 75.7% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration
* **Bearing2_3**: first snapshot above 20 g already at 13.6% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration
* **Bearing2_4**: no recorded 0.1 s snapshot ever exceeds 20 g (peak 8.8 g) -> the 20 g end-of-life criterion is **not observable in the snapshots** for this bearing
* **Bearing2_6**: no recorded 0.1 s snapshot ever exceeds 20 g (peak 11.5 g) -> the 20 g end-of-life criterion is **not observable in the snapshots** for this bearing
* **Bearing2_7**: first snapshot above 20 g already at 70.4% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration
* **Bearing3_3**: no recorded 0.1 s snapshot ever exceeds 20 g (peak 16.0 g) -> the 20 g end-of-life criterion is **not observable in the snapshots** for this bearing

## 3. Methodology
* **Unit of independence = the bearing.** Method development and model selection use only the 6 learning bearings with leave-one-bearing-out (LOBO) validation (a separate validation bearing is used only for early stopping). The 11 test bearings are scored once after the selection file `selection.json` is written.
* **Target (formulation E, multi-task).** Capped RUL in minutes (cap 120) and a 3-class stage (Normal > 60 min, Warning 20-60 min, Critical <= 20 min) derived **only from the failure time** (official durations; test bearings: truncation point + official RUL, which corrects Bearing1_4). No vibration-based onset label is used for training. Horizons were fixed a priori and checked for sensitivity.
* **Formulation study (learning bearings only):** * **Direct / capped RUL in minutes (A, A'):** the Random-Forest probe does **not** clearly beat the constant prediction (A: 103.3 vs 97.2 min, better in 3/6 folds; A': 42.2 vs 38.8 min, 2/6 folds). Minute-level RUL across bearings is therefore **not supported** by this data.
* **Life fraction (B)** gives lower error (0.20 vs 0.25, 5/6 folds) because the target itself is normalised by the bearing's total lifetime - information that is unknown for a running machine. **Not deployable; not used.**
* **Stage classification (C)** beats the majority-class predictor in 5/6 folds (macro-F1 0.43 vs 0.25), i.e. there is real but moderate signal about *how close to failure* a bearing is.
* **Detectability depends strongly on the horizon** (Figure 8): the mean AUC of 'failure within H min' is 0.89 at 5 min, 0.64 at 20 min and 0.70 at 60 min, and it is close to chance for the abrupt bearings at long horizons.
**Frozen decision.** Keep formulation **E** (capped RUL + stage) with the *a priori* horizons (cap 120 min, Warning <= 60 min, Critical <= 20 min); they are **not tuned** to obtain better scores. Because horizon matters so much, every model is also evaluated at several horizons (AUC at 5, 10, 20, 30, 60 min). We record explicitly that **the dataset does not support reliable early RUL prediction for bearings that fail abruptly**. No vibration-based onset label is used for training; the onset-style analysis (D) is descriptive only.
**Remaining hindsight assumptions.** (1) Training uses the failure time of the learning bearings (standard supervised learning). (2) The cap and stage boundaries are design choices fixed a priori; a sensitivity analysis follows. (3) The descriptive onset/phase-length tables use the whole trajectory (hindsight) and are never used as labels.
* **Features.** 34 documented time/frequency features from one snapshot each (causal); fixed log/signed-log transforms; `StandardScaler` fitted on training bearings only. Input = last 16 snapshots.
* **Leakage control.** **Figure 9 / table.** Same model, same bearings: the random window split reports an MAE of **1.5 min** and AUC **1.00**, the correct bearing-level protocol **42.2 min** and **0.70**. The difference is pure leakage of near-duplicate windows. All results in this notebook use the bearing-level protocol. Automated checks (splits, scaler, poison test, truncation/causality test, window boundaries, independent feature recomputation, target-shuffle test, frozen configuration) stop the notebook on violation.
* **Models (same windows, targets, folds, metrics).** Constant, Ridge, Random Forest, Gradient Boosting, CNN, LSTM, CNN+LSTM, CNN+BiLSTM, CNN+BiLSTM+Attention (single head), CNN+BiLSTM (dual head, no attention), and the proposed CNN+BiLSTM+Attention dual-head model (114,692 parameters). Training: Adam 0.001, batch 64, Huber(0.1) + 0.5 x cross-entropy, early stopping on the validation bearing, deterministic TensorFlow, seeds [42, 43] (LOBO) and [42, 43, 44] (final).
* **Metrics.** MAE (minutes) overall and in the *actionable region* (true RUL <= 60 min); ROC-AUC of detecting "failure within H minutes" for H = 5, 10, 20, 30, 60; stage macro-F1; false-alarm rate; alarm active at the end of life; official PHM-2012 score at the truncation points [2].

## 4. Experiments and results
### 4.1 Leave-one-bearing-out (6 held-out learning bearings; mean +/- std over bearings)
| index | MAE_min | MAE_min_actionable | AUC_h10 | AUC_h60 | AUC_mean_h | stage_macroF1 | false_alarm_rate | alarm active at end (of 6) | params |
|---|---|---|---|---|---|---|---|---|---|
| Constant | 40.310 +/- 4.179 | 53.373 +/- 11.107 | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.246 +/- 0.058 | 0.000 +/- 0.000 | 0.000 | nan |
| Ridge | 55.531 +/- 23.253 | 31.076 +/- 18.026 | 0.603 +/- 0.174 | 0.703 +/- 0.221 | 0.615 +/- 0.198 | 0.267 +/- 0.214 | 0.535 +/- 0.427 | 6.000 | nan |
| Random Forest | 44.125 +/- 17.331 | 38.902 +/- 21.387 | 0.776 +/- 0.162 | 0.662 +/- 0.168 | 0.717 +/- 0.159 | 0.342 +/- 0.165 | 0.365 +/- 0.372 | 4.000 | nan |
| Gradient Boosting | 43.007 +/- 16.689 | 39.680 +/- 20.811 | 0.781 +/- 0.161 | 0.670 +/- 0.142 | 0.725 +/- 0.091 | 0.335 +/- 0.141 | 0.362 +/- 0.363 | 4.000 | nan |
| CNN | 46.152 +/- 19.683 | 28.946 +/- 21.476 | 0.803 +/- 0.183 | 0.835 +/- 0.195 | 0.803 +/- 0.172 | 0.294 +/- 0.163 | 0.563 +/- 0.356 | 5.500 | 25729.000 |
| LSTM | 42.477 +/- 21.791 | 27.978 +/- 26.595 | 0.940 +/- 0.049 | 0.822 +/- 0.128 | 0.886 +/- 0.104 | 0.373 +/- 0.160 | 0.454 +/- 0.341 | 5.500 | 31617.000 |
| CNN + LSTM | 43.131 +/- 18.977 | 24.150 +/- 16.352 | 0.883 +/- 0.106 | 0.811 +/- 0.173 | 0.852 +/- 0.128 | 0.325 +/- 0.182 | 0.517 +/- 0.339 | 6.000 | 58753.000 |
| CNN + BiLSTM | 46.738 +/- 20.546 | 28.407 +/- 22.156 | 0.893 +/- 0.077 | 0.810 +/- 0.175 | 0.873 +/- 0.116 | 0.314 +/- 0.179 | 0.537 +/- 0.346 | 5.500 | 95873.000 |
| CNN + BiLSTM + Attention (single head) | 44.922 +/- 18.441 | 23.340 +/- 17.162 | 0.848 +/- 0.144 | 0.789 +/- 0.193 | 0.830 +/- 0.149 | 0.315 +/- 0.156 | 0.586 +/- 0.345 | 6.000 | 112513.000 |
| CNN + BiLSTM (dual head, no attention) | 47.172 +/- 24.217 | 31.326 +/- 25.701 | 0.879 +/- 0.107 | 0.747 +/- 0.202 | 0.827 +/- 0.126 | 0.317 +/- 0.197 | 0.506 +/- 0.339 | 6.000 | 98052.000 |
| PROPOSED: CNN + BiLSTM + Attention (dual head) | 45.510 +/- 21.669 | 28.752 +/- 25.005 | 0.870 +/- 0.133 | 0.809 +/- 0.113 | 0.847 +/- 0.118 | 0.351 +/- 0.183 | 0.523 +/- 0.324 | 6.000 | 114692.000 |

**Do any models beat the constant baseline?** Overall RUL MAE (constant 40.3 min): **no model** beats the constant. Actionable-region MAE (constant 53.4 min): Gradient Boosting 39.7, Random Forest 38.9, Ridge 31.1, CNN 28.9, CNN + BiLSTM 28.4, CNN + BiLSTM + Attention (single head) 23.3, CNN + BiLSTM (dual head, no attention) 31.3, CNN + LSTM 24.2, LSTM 28.0, PROPOSED: CNN + BiLSTM + Attention (dual head) 28.8.

**Detection (mean AUC over horizons, constant 0.50):** best = **LSTM** (0.89); proposed = 0.85; best tabular = 0.72.

**Lowest false-alarm rate:** Gradient Boosting (0.36); proposed = 0.52. Note that false alarms of the *uniform rule* derive from the noisy RUL estimate.

### 4.2 Ablation (LOBO, paired over 6 bearings, 95% bootstrap CI)
| step | metric | mean_diff | ci_lo | ci_hi | units_improved | verdict |
|---|---|---|---|---|---|---|
| CNN -> CNN+LSTM (add recurrence) | detection AUC (mean over horizons) | 0.050 | -0.024 | 0.134 | 4/6 | no measurable difference (CI includes 0) |
| CNN -> CNN+LSTM (add recurrence) | actionable RUL MAE (min) | -4.796 | -9.675 | -0.772 | 4/6 | measurable improvement |
| CNN -> CNN+LSTM (add recurrence) | stage macro-F1 | 0.031 | -0.020 | 0.078 | 5/6 | no measurable difference (CI includes 0) |
| LSTM -> BiLSTM (make it bidirectional) | detection AUC (mean over horizons) | 0.020 | -0.016 | 0.060 | 3/6 | no measurable difference (CI includes 0) |
| LSTM -> BiLSTM (make it bidirectional) | actionable RUL MAE (min) | 4.257 | 0.422 | 9.578 | 2/6 | measurable degradation |
| LSTM -> BiLSTM (make it bidirectional) | stage macro-F1 | -0.011 | -0.038 | 0.017 | 2/6 | no measurable difference (CI includes 0) |
| add attention (single head) | detection AUC (mean over horizons) | -0.042 | -0.103 | 0.003 | 2/6 | no measurable difference (CI includes 0) |
| add attention (single head) | actionable RUL MAE (min) | -5.067 | -9.621 | -1.860 | 6/6 | measurable improvement |
| add attention (single head) | stage macro-F1 | 0.001 | -0.021 | 0.024 | 2/6 | no measurable difference (CI includes 0) |
| add second head (dual head) | detection AUC (mean over horizons) | 0.017 | -0.031 | 0.071 | 4/6 | no measurable difference (CI includes 0) |
| add second head (dual head) | actionable RUL MAE (min) | 5.412 | 0.067 | 12.647 | 1/6 | measurable degradation |
| add second head (dual head) | stage macro-F1 | 0.036 | -0.003 | 0.078 | 3/6 | no measurable difference (CI includes 0) |
| add second head WITHOUT attention | detection AUC (mean over horizons) | -0.046 | -0.152 | 0.038 | 2/6 | no measurable difference (CI includes 0) |
| add second head WITHOUT attention | actionable RUL MAE (min) | 2.918 | -1.572 | 7.657 | 3/6 | no measurable difference (CI includes 0) |
| add second head WITHOUT attention | stage macro-F1 | 0.003 | -0.083 | 0.063 | 4/6 | no measurable difference (CI includes 0) |
| add attention to the dual-head model | detection AUC (mean over horizons) | 0.021 | -0.048 | 0.103 | 2/6 | no measurable difference (CI includes 0) |
| add attention to the dual-head model | actionable RUL MAE (min) | -2.573 | -6.990 | 0.953 | 3/6 | no measurable difference (CI includes 0) |
| add attention to the dual-head model | stage macro-F1 | 0.034 | -0.003 | 0.087 | 3/6 | no measurable difference (CI includes 0) |
| add CNN in front of the LSTM | detection AUC (mean over horizons) | -0.034 | -0.122 | 0.029 | 3/6 | no measurable difference (CI includes 0) |
| add CNN in front of the LSTM | actionable RUL MAE (min) | -3.828 | -13.134 | 2.424 | 2/6 | no measurable difference (CI includes 0) |
| add CNN in front of the LSTM | stage macro-F1 | -0.048 | -0.098 | -0.000 | 2/6 | measurable degradation |

### 4.3 Final unseen test (11 test bearings; mean +/- std over 3 seeds)
| index | MAE_min | MAE_min_actionable | AUC_h10 | AUC_h60 | AUC_mean_h | stage_macroF1 | false_alarm_rate | PHM_official | alarm active at end (of 11) |
|---|---|---|---|---|---|---|---|---|---|
| Constant | 35.623 | 56.354 | 0.500 | 0.500 | 0.500 | 0.291 | 0.000 | 0.124 | 0.000 |
| Ridge | 39.066 | 45.216 | 0.664 | 0.585 | 0.640 | 0.422 | 0.198 | 0.040 | 7.000 |
| Random Forest | 32.640 +/- 0.101 | 32.713 +/- 0.124 | 0.654 +/- 0.002 | 0.768 +/- 0.001 | 0.696 +/- 0.002 | 0.427 +/- 0.001 | 0.296 +/- 0.001 | 0.078 +/- 0.005 | 9.700 |
| Gradient Boosting | 30.146 +/- 0.000 | 32.651 +/- 0.000 | 0.694 +/- 0.000 | 0.778 +/- 0.000 | 0.725 +/- 0.000 | 0.471 +/- 0.000 | 0.237 +/- 0.000 | 0.037 +/- 0.000 | 10.000 |
| CNN | 27.806 +/- 0.889 | 42.575 +/- 2.302 | 0.751 +/- 0.030 | 0.694 +/- 0.035 | 0.732 +/- 0.032 | 0.454 +/- 0.019 | 0.146 +/- 0.011 | 0.181 +/- 0.022 | 10.000 |
| LSTM | 32.324 +/- 0.773 | 36.779 +/- 2.906 | 0.739 +/- 0.005 | 0.710 +/- 0.024 | 0.728 +/- 0.011 | 0.450 +/- 0.027 | 0.208 +/- 0.022 | 0.143 +/- 0.023 | 10.300 |
| CNN + LSTM | 30.610 +/- 3.932 | 42.055 +/- 1.535 | 0.719 +/- 0.010 | 0.708 +/- 0.034 | 0.715 +/- 0.009 | 0.423 +/- 0.065 | 0.182 +/- 0.059 | 0.221 +/- 0.027 | 10.000 |
| CNN + BiLSTM | 34.205 +/- 3.342 | 40.092 +/- 7.291 | 0.737 +/- 0.061 | 0.703 +/- 0.060 | 0.725 +/- 0.055 | 0.410 +/- 0.041 | 0.225 +/- 0.070 | 0.106 +/- 0.026 | 9.700 |
| CNN + BiLSTM + Attention (single head) | 32.336 +/- 5.166 | 41.184 +/- 4.669 | 0.726 +/- 0.041 | 0.706 +/- 0.025 | 0.715 +/- 0.034 | 0.433 +/- 0.050 | 0.202 +/- 0.071 | 0.179 +/- 0.023 | 9.000 |
| CNN + BiLSTM (dual head, no attention) | 35.047 +/- 3.243 | 34.110 +/- 5.958 | 0.741 +/- 0.029 | 0.769 +/- 0.038 | 0.748 +/- 0.026 | 0.399 +/- 0.045 | 0.277 +/- 0.055 | 0.093 +/- 0.031 | 8.700 |
| PROPOSED: CNN + BiLSTM + Attention (dual head) | 29.592 +/- 5.308 | 38.072 +/- 6.187 | 0.759 +/- 0.024 | 0.749 +/- 0.012 | 0.749 +/- 0.017 | 0.468 +/- 0.045 | 0.172 +/- 0.109 | 0.146 +/- 0.088 | 8.300 |

**Detection (mean AUC over horizons 5-60 min; constant = 0.50):** proposed **0.749** (seed std 0.017); highest mean on test: PROPOSED: CNN + BiLSTM + Attention (dual head) (0.749). Models within 0.020 of the best are **statistically indistinguishable** here (seed noise): CNN, CNN + BiLSTM (dual head, no attention), PROPOSED: CNN + BiLSTM + Attention (dual head). The LOBO-selected detection model was **LSTM**; it ranks #4 of 10 on test. Rank correlation LOBO vs test (10 models): 0.47.

**RUL in the actionable region (constant 56.4 min):** proposed 38.1 min; lowest on test Gradient Boosting (32.7 min); tied within 2.0 min: Gradient Boosting, Random Forest, CNN + BiLSTM (dual head, no attention). The LOBO-selected RUL model was **CNN + BiLSTM + Attention (single head)**; it ranks #7 of 10 on test (rank correlation LOBO vs test -0.52). **Overall MAE** (constant 35.6 min): proposed 29.6 min.

**Official PHM-2012 score** (one prediction per bearing at the truncation point; higher = better, 1 = perfect): proposed 0.146, constant 0.124, Random Forest 0.078. With only 11 predictions and a capped predictor this score is very noisy; the per-bearing table shows that at the truncation points of the abrupt bearings the model still predicts a healthy bearing (RUL > 100 min), which is punished as a very late prediction.

**False alarms** on truly Normal windows: proposed 0.17, Random Forest 0.30, Gradient Boosting 0.24 (uniform rule). **Stage macro-F1:** proposed 0.47 (uniform rule), constant 0.29.

**Consistency of validation and test.** The rank correlations between LOBO and test are 0.47 (detection) and -0.52 (actionable RUL error): the validation ranking is only weakly or not at all reproduced on the test bearings, so six learning bearings give an **unstable model ranking**; differences between the top models must not be over-interpreted.

Ablation on the test bearings (paired over 11 bearings):

| step | metric | mean_diff | ci_lo | ci_hi | units_improved | verdict |
|---|---|---|---|---|---|---|
| CNN -> CNN+LSTM (add recurrence) | detection AUC at 10 min | -0.055 | -0.110 | -0.002 | 3/11 | measurable degradation |
| CNN -> CNN+LSTM (add recurrence) | actionable RUL MAE (min) | -0.396 | -6.187 | 3.527 | 4/11 | no measurable difference (CI includes 0) |
| LSTM -> BiLSTM (make it bidirectional) | detection AUC at 10 min | 0.020 | -0.026 | 0.074 | 6/11 | no measurable difference (CI includes 0) |
| LSTM -> BiLSTM (make it bidirectional) | actionable RUL MAE (min) | -1.976 | -7.567 | 3.523 | 7/11 | no measurable difference (CI includes 0) |
| add attention (single head) | detection AUC at 10 min | -0.024 | -0.072 | 0.022 | 4/11 | no measurable difference (CI includes 0) |
| add attention (single head) | actionable RUL MAE (min) | 1.099 | -4.000 | 6.421 | 6/11 | no measurable difference (CI includes 0) |
| add second head (dual head) | detection AUC at 10 min | 0.065 | 0.002 | 0.138 | 8/11 | measurable improvement |
| add second head (dual head) | actionable RUL MAE (min) | -2.816 | -11.652 | 3.416 | 4/11 | no measurable difference (CI includes 0) |
| add second head WITHOUT attention | detection AUC at 10 min | -0.009 | -0.070 | 0.054 | 5/11 | no measurable difference (CI includes 0) |
| add second head WITHOUT attention | actionable RUL MAE (min) | -5.567 | -14.464 | 1.556 | 5/11 | no measurable difference (CI includes 0) |
| add attention to the dual-head model | detection AUC at 10 min | 0.050 | 0.012 | 0.087 | 9/11 | measurable improvement |
| add attention to the dual-head model | actionable RUL MAE (min) | 3.851 | -2.758 | 10.961 | 4/11 | no measurable difference (CI includes 0) |
| add CNN in front of the LSTM | detection AUC at 10 min | -0.026 | -0.080 | 0.021 | 4/11 | no measurable difference (CI includes 0) |
| add CNN in front of the LSTM | actionable RUL MAE (min) | 5.198 | 2.707 | 7.703 | 2/11 | measurable degradation |

### 4.4 Hypotheses (computed from the measurements)
* **H1 - sequence models detect impending failure better than tabular baselines: PARTLY SUPPORTED.** Representatives chosen on LOBO: LSTM vs Gradient Boosting. LOBO (6 folds): mean difference in AUC +0.162 (95% CI +0.092 to +0.238; measurable improvement); test (11 bearings, AUC at 10 min): +0.048 (CI -0.097 to +0.187; no measurable difference (CI includes 0)).

* **H2 - attention and the dual head improve accuracy.** Attention: **MIXED** (2 improvements, 0 degradations, 6 no difference among the ablation contrasts on LOBO and test). Dual head: **MIXED** (1 improvements, 1 degradations, 6 no difference). *MIXED* means the effect changes sign or significance between metrics or between validation and test, i.e. no robust claim can be made.

* **H3 - a useful warning for every bearing: NOT SUPPORTED.** A useful warning (alarm active at the end and sustained lead >= 10 min) exists for 3/11 test bearings (2/3 with a visible degradation phase >= 20 min, 1/8 of the abrupt ones), at a false-alarm rate of 0.05 on healthy windows (proposed, seed 42, uniform rule).

* **RQ3/RQ6:** on unseen bearings the proposed model reaches mean detection AUC 0.75 (constant 0.50), but minute-level RUL accuracy is limited: actionable-region MAE 38.1 min vs 56.4 min for a constant, and the official PHM score is 0.146 vs 0.124 for a constant.

* **RQ4 - is the proposed model the best?** Highest mean detection AUC on test: **PROPOSED: CNN + BiLSTM + Attention (dual head)**; models statistically tied with it: CNN, CNN + BiLSTM (dual head, no attention), PROPOSED: CNN + BiLSTM + Attention (dual head). Lowest actionable RUL error: **Gradient Boosting**; tied: Gradient Boosting, Random Forest, CNN + BiLSTM (dual head, no attention). The proposed model is in the detection tie group and not in the RUL tie group. Its distinctive property is the combination of an alarm head with inspectable attention/feature analysis (functional value measured in Section 20); it is **not** shown to be more accurate than the simplest competitive baselines.

* **RQ5:** the model is *inspectable* (attention, permutation importance), but attention is not a causal explanation and appears only weakly functional (Section 20).

### 4.5 Per-bearing results of the proposed model on the test bearings
| bearing | windows | MAE_min | MAE_min_actionable | AUC_h10 | AUC_h60 | stage_macroF1 | false_alarm_rate | alarm_at_end_bearings | lead_sustained_min_median |
|---|---|---|---|---|---|---|---|---|---|
| Bearing1_3 | 2360 | 5.950 | 17.807 | 0.994 | 0.991 | 0.844 | 0.006 | 1 | 19.667 |
| Bearing1_4 | 1157 | 13.359 | 13.911 | 0.989 | 0.999 | 0.828 | 0.000 | 1 | 41.317 |
| Bearing1_5 | 2448 | 14.592 | 58.919 | 0.913 | 0.741 | 0.503 | 0.001 | 1 | 7.667 |
| Bearing1_6 | 2433 | 33.582 | 23.076 | 0.958 | 0.859 | 0.583 | 0.086 | 1 | 4.167 |
| Bearing1_7 | 2244 | 42.196 | 35.389 | 0.901 | 0.726 | 0.385 | 0.410 | 1 | 8.000 |
| Bearing2_3 | 1940 | 25.392 | 85.356 | 0.650 | 0.698 | 0.294 | 0.025 | 0 | nan |
| Bearing2_4 | 736 | 49.722 | 84.079 | 0.443 | 0.096 | 0.227 | 0.000 | 1 | 1.167 |
| Bearing2_5 | 2296 | 13.938 | 58.774 | 0.837 | 0.962 | 0.359 | 0.002 | 1 | 0.333 |
| Bearing2_6 | 686 | 19.543 | 16.106 | 0.877 | 0.991 | 0.634 | 0.000 | 1 | 6.500 |
| Bearing2_7 | 215 | 99.546 | 99.546 | 0.879 | nan | 0.000 | nan | 0 | nan |
| Bearing3_3 | 419 | 22.287 | 17.946 | 0.498 | 1.000 | 0.466 | 0.000 | 1 | 44.167 |

### 4.6 Objective multi-criteria comparison
| model | detection AUC LOBO | detection AUC test | actionable MAE LOBO (min) | actionable MAE test (min) | overall MAE test (min) | false-alarm rate test | PHM-2012 score test | stability: std of test AUC over seeds | parameters | fit+predict s (LOBO median) | interpretability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Constant | 0.500 | 0.500 | 53.373 | 56.354 | 35.623 | 0.000 | 0.124 | nan | nan | 0.010 | trivial |
| Ridge | 0.615 | 0.640 | 31.076 | 45.216 | 39.066 | 0.198 | 0.040 | nan | nan | 0.407 | coefficients |
| Random Forest | 0.717 | 0.696 | 38.902 | 32.713 | 32.640 | 0.296 | 0.078 | 0.002 | nan | 13.371 | importances |
| Gradient Boosting | 0.725 | 0.725 | 39.680 | 32.651 | 30.146 | 0.237 | 0.037 | 0.000 | nan | 4.315 | importances |
| CNN | 0.803 | 0.732 | 28.946 | 42.575 | 27.806 | 0.146 | 0.181 | 0.032 | 25729.000 | 7.662 | black box |
| LSTM | 0.886 | 0.728 | 27.978 | 36.779 | 32.324 | 0.208 | 0.143 | 0.011 | 31617.000 | 8.162 | black box |
| CNN + LSTM | 0.852 | 0.715 | 24.150 | 42.055 | 30.610 | 0.182 | 0.221 | 0.009 | 58753.000 | 17.088 | black box |
| CNN + BiLSTM | 0.873 | 0.725 | 28.407 | 40.092 | 34.205 | 0.225 | 0.106 | 0.055 | 95873.000 | 19.830 | black box |
| CNN + BiLSTM + Attention (single head) | 0.830 | 0.715 | 23.340 | 41.184 | 32.336 | 0.202 | 0.179 | 0.034 | 112513.000 | 28.628 | black box |
| CNN + BiLSTM (dual head, no attention) | 0.827 | 0.748 | 31.326 | 34.110 | 35.047 | 0.277 | 0.093 | 0.026 | 98052.000 | 20.904 | black box |
| PROPOSED: CNN + BiLSTM + Attention (dual head) | 0.847 | 0.749 | 28.752 | 38.072 | 29.592 | 0.172 | 0.146 | 0.017 | 114692.000 | 22.075 | attention + permutation |

Selection made on LOBO only (saved before the test): `{"best_actionable_RUL_MAE": "cnn_bilstm_attn", "best_detection_AUC_mean_h": "lstm", "lowest_false_alarm_rate": "Gradient Boosting", "interpretable_by_design": "proposed"}`.

## 5. Explainability
* **Attention:** the attention layer is **functionally weak**: replacing it by uniform weights changes predictions only slightly (mean absolute change 1.4 min, correlation 0.996). The stage-wise profiles differ only modestly from uniform weights (Figure 20a). Attention therefore tells us *where in the 160 s window the network put more weight*, not *why the bearing is failing*.
* **Features:** permutation importance identifies which vibration features the models rely on (Figure 21); rank agreement between the proposed model and Random Forest is -0.05, so the two models rely on partly different cues, and correlated features share importance.
* **Does this support our hypothesis that the model is explainable?** Only in a limited sense: the model exposes inspectable weights and feature dependencies, but attention is **not** a causal explanation [9], and the importance analysis is descriptive. We claim *inspectability*, not *causal explanation*.

## 6. Limitations
Six independent learning bearings; abrupt failures with 1-2 minutes of visible degradation for several Condition-2/3 bearings; label uncertainty (20 g not observable in every snapshot; Bearing1_4 extends past its official end of life); hindsight in offline (descriptive) analyses; **prior exposure to test-bearing curves in earlier experiments (v1/v2)** - the final target avoids vibration-derived labels, but the choice of formulation was motivated by earlier results; detection is not RUL estimation; official PHM score is noisy with 11 capped predictions; attention is not a causal explanation; no hyper-parameter search; window length not varied; single test rig. See notebook Section 25.

## 7. Conclusion
The supported claims are those printed in Sections 4.4-4.6 and in notebook Section 26. In short: on unseen bearings the vibration-based models recognise *imminent* failure better than a constant predictor (but far from perfectly, with large differences between bearings), but minute-level RUL is not reliably predicted, warnings are unreliable for bearings that fail abruptly, alarms carry a high false-alarm cost, and the proposed architecture is **not shown to be more accurate than simple competitive baselines**; its distinctive property is inspectability (with the limits stated in Section 5).

## References
See `REFERENCES.md`.
