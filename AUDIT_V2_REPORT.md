# Audit of the v2 Experiment (LOBO + degradation-timeline target)

Scope: `explainable_dual_head_v2_lobo_onset.ipynb` (cells A2–A10 are the verification cells). Paper untouched. All numbers below come from that notebook's deterministic run (`results/v2_*.csv`, `results/v2_run_config.json`).

## Headline (read this first)
1. **The vibration data does detect the start of the final degradation phase** (test AUC 0.92–0.97 for RF/CNN/proposed vs 0.50 constant). This is real and survives all leakage checks.
2. **v2 does NOT show skill at grading remaining life *inside* the degradation phase.** Degradation-phase MAE: constant 0.313, RF 0.322 (worse), CNN 0.293, proposed 0.275. Almost the whole "MAE 0.054 vs 0.259" gap is on healthy windows (pre-onset MAE 0.010 vs 0.251), i.e. it is a healthy-vs-degrading detector, not an RUL predictor.
3. **The proposed CNN-BiLSTM-Attention model is not the best model and its two claimed strengths did not hold up**: its low false-alarm rate exists only through its classification head (2.9%); alarms derived from its RUL output give 38.3% false alarms. Attention is functionally almost irrelevant (below).
4. **Retraction:** my earlier statement that the proposed model's LOBO AUC is "0.86 ± 0.26 (unstable)" was mostly run-to-run nondeterminism. With deterministic ops the seed-42 LOBO AUC is 0.938 ± 0.069 (per fold 0.80–1.00).

## 1. Onset label validity (A2, `figA1_onset_audit_all_bearings.png`, `v2_audit_onset_table.csv`)
| Bearing | onset % life | phase (min) | failure (min) | verdict after visual inspection |
|---|---|---|---|---|
| 1_1 (L) | 51.8 | 225.0 | 467.0 | gradual, sustained (kurtosis corroborates, z=18) |
| 1_2 (L) | 94.6 | 7.7 | 145.0 | abrupt terminal; kurtosis does not corroborate (z=0.7) |
| **2_1 (L)** | **16.5** | 126.7 | 151.7 | **step then plateau: label is wrong for ~80% of the "degradation" phase** |
| **2_2 (L)** | **24.2** | 100.5 | 132.7 | **hump (rises then falls), not sustained degradation** (automatic flag missed this; visual only) |
| 3_1 (L) | 95.3 | 3.8 | 85.7 | abrupt terminal; no kurtosis shift (z=-1.4) |
| 3_2 (L) | 96.8 | 8.7 | 272.7 | abrupt terminal |
| 1_3 | 67.6 | 128.2 | 395.7 | gradual, sustained |
| 1_4 | 76.1 | 56.7 | 237.8 | sustained |
| 1_5 / 1_6 | 97.8 / 98.5 | 9.0 / 5.8 | 410.3 / 407.8 | abrupt terminal |
| **1_7** | **53.4** | 175.2 | 376.3 | **plateau at the threshold for 175 min, real rise only at the end: doubtful** |
| 2_3 / 2_4 / 2_5 / 2_6 / 2_7 | 99.5 / 98.8 / 99.5 / 98.0 / 96.5 | 1.3 / 1.3 / 1.7 / 2.2 / 1.2 | 325.7 / 125.0 / 385.0 / 116.7 / 38.2 | abrupt terminal (7–13 degradation windows only) |
| 3_3 | 71.7 | 20.3 | 72.2 | sustained |

* The rule is implemented consistently: a loop-based re-implementation equals the vectorised one for all 17 bearings; HI ≥ threshold from onset to failure and < threshold just before onset (asserted). No bearing was clamped.
* Two of six **training** bearings (2_1, 2_2) and one test bearing (1_7) carry label noise. Training bearings 2_1/2_2 are the only Condition-2 training histories, so this matters.

## 2. Leakage / hindsight (A3)
Tests that pass: (a) 30 inputs are vibration features only, no label/time/RUL/metadata columns; (b) **truncation test**: deleting all data after time *t* leaves the windows and the model's predictions up to *t* bit-identical (3 bearings × 3 cut points); (c) window *k* ends at snapshot *k+15* and carries that snapshot's label; (d) 27 feature values recomputed from a single raw CSV match (features use one snapshot only); (e) final scaler = mean of the 6 learning bearings, ≠ all-bearing mean.
**Offline vs online:** inputs at time *t* use only snapshots ≤ *t*. Future information enters **only** the onset *label* (centred median smoothing, quantile baseline, max failure level, final excursion). That is legitimate for building targets and for scoring, but it means the evaluation target for a test bearing is defined using that bearing's whole life.

## 3. LOBO (A4)
Asserted for all 6 folds: held-out bearing absent from training; scaler mean == mean of the 5 training bearings (≠ 6-bearing mean); **poison test**: replacing the held-out bearing's features/labels with garbage leaves training windows, labels, weights and scaler bit-identical. No thresholds, hyper-parameters, early stopping or model selection exist inside LOBO (all fixed in `HPARAMS`).
Per-fold proposed AUC (seed 42): 1_1 0.802, 1_2 0.942, 2_1 0.999, 2_2 0.964, 3_1 0.951, 3_2 0.971. Over 3 seeds: std across folds 0.061, std across seeds 0.014. Weakest fold: **Bearing1_1** (AUC 0.77–0.93, the long gradual bearing). CNN is more seed-sensitive (1_1 0.72–0.93, 3_1 0.76–1.00). Full tables in the notebook.

## 4. Baseline fairness (A5)
Same arrays, targets, sample weights and evaluation windows for every model (one `make_windows` call). Differences that remain: RF/Ridge see only the last snapshot (RF on the flattened window is *worse*: MAE 0.062, AUC 0.942, so RF's edge is not a last-snapshot artefact); RF is a 200-tree ensemble vs single networks; RF settings come from an earlier experiment on the old target that scored test bearings; **the proposed model's alarms use its own head while all others use y-hat thresholds**: under the same y-hat rule its false alarms are 38.3%.

## 5. Test-set integrity (A6, `v2_audit_test_integrity.csv`)
Test information **did** influence: (i) the onset-rule design (I inspected all 17 HI curves incl. test bearings and tried variants; the 10% value was fixed before v2 training); (ii) the decision to abandon the old target (poor test results); (iii) RF settings / fixed-epoch idea (earlier experiments). It did **not** influence: architecture, alarm thresholds, early stopping (none), model selection (none). Onset-fraction robustness (5/10/20%): detection AUC ordering stays constant (0.50) ≪ others; RF 0.96–0.97, CNN 0.92–0.96, proposed 0.886/0.963/0.970; RF MAE 0.057/0.054/0.027. Reported, not used to choose anything.

## 6. RUL metric (A7)
MAE = mean over all windows of |y − ŷ|, y = 1 before onset, y = (T_fail − t)/(T_fail − t0) after (fraction of the degradation phase remaining). Healthy windows dominate, so it mostly measures calibration at y = 1. It is **not** the PHM-2012 metric (minutes, one truncation time per bearing, asymmetric % error) and cannot be compared with challenge results. Not converted to minutes (phase length unknown at prediction time). Degradation-phase MAE per bearing ranges 0.10–0.60; skill vs constant in that phase: RF −3%, CNN +7%, proposed +12%.

## 7. Early warning (A8, symmetric y-hat alarm rule, 3-window persistence)
Detected phases (11 bearings × 3 seeds): RF 30/33, CNN 30/33, proposed 32/33 (the proposed gets there by alarming on healthy data: its false-alarm episodes reach 22–62 on several bearings). Useful warning exists for gradual bearings: 1_3 (128 min phase, RF 126 min lead), 1_4 (56 min), 1_7 (28–175 min depending on model), 3_3 (~16–17 min). Abrupt bearings: 1_5/1_6 (5.7–9 min phase, 3–8 min lead), 2_3–2_7 (phase 1–2 min, lead ≤ 2 min; 2_5 missed by RF and CNN). For these, degradation is too abrupt for maintenance action. Native-head proposed misses 1/11, 3/11, 2/11 bearings across seeds.

## 8. Explainability (A9)
* Attention weight on the 4 newest steps (uniform 0.25): post-onset minus pre-onset = **+0.032 on average, positive in only 6/11 bearings** (Bearing1_3 goes *down*, 0.275→0.176). The claim "attention focuses on recent vibration when degrading" is **not supported**.
* Replacing attention by uniform weights in the same trained network changes MAE by 0.002–0.007 and predictions correlate ≥ 0.99: attention has almost no functional role.
* Permutation importance (trained models fixed, test windows shuffled): proposed top = vert_rmsf (ΔAUC +0.133), horiz_spec_spread (+0.092); RF top = horiz_spec_energy (+0.063), horiz_spec_spread (+0.052). Only `horiz_spec_spread` overlaps, so the models rely on different cues.

## 9. Reproducibility (A10)
TF deterministic ops on, seeds fixed, hyper-parameters + library versions in `results/v2_run_config.json` (Python 3.12.6, TF 2.20.0, sklearn 1.5.2, numpy 2.2.6, pandas 2.2.2). The notebook executes top-to-bottom via `nbconvert` with no manual step. Not yet verified: a *second* full deterministic run reproducing this run bit-for-bit. The earlier non-deterministic run gave different proposed-model numbers (LOBO AUC 0.855 vs 0.938), so only this run should be cited.

## 10. Final research assessment
**A. Scientifically valid in v2:** no future information in model inputs (tested); LOBO isolation (tested); a healthy-vs-degrading detection signal with AUC 0.92–0.97 on 11 unseen bearings, robust to the onset fraction (5–20%); honest baselines; abrupt-vs-gradual failure characterisation.
**B. Questionable:** onset labels of 2_1, 2_2 (training) and 1_7 (test); only 7–13 degradation windows for five test bearings; evaluation target defined with hindsight on test bearings; RUL grading inside the degradation phase ≈ constant baseline; deep models' seed variance; unverified second-run reproducibility.
**C. Must fix before publication:** (1) replace the hindsight onset with a rule validated without test bearings (e.g. fix it on learning bearings, or use an established first-prediction-time method) and handle 2_1/2_2/1_7 explicitly; (2) evaluate RUL in the degradation phase with a metric that is meaningful there (and, if minutes are wanted, a deployable predictor of phase length); (3) calibrate the y-hat output for healthy windows (currently ≈0.90 not 1.0) or drop y-hat-based alarms; (4) more seeds, repeat-run reproducibility check; (5) report the PHM-2012 challenge protocol separately if a benchmark comparison is claimed.
**D. Headline experiment?** **Not yet.** It is a valid *degradation-onset detection* study; it is not an RUL-prediction result. It can be the headline only if framed exactly that way, with Random Forest / CNN as the strongest models.
**E. Claims you can safely make:** (i) a leakage-checked pipeline (inputs strictly causal, LOBO with held-out bearings excluded from all fitting); (ii) with a degradation-timeline label, the vibration features separate healthy from final-degradation windows on 11 unseen bearings (AUC 0.92–0.97; constant 0.50), robust to the onset threshold; (iii) simple models (RF, CNN) match or beat the CNN-BiLSTM-Attention model; (iv) about half the test bearings fail abruptly, with 1–2 minutes of warning for the Condition 2 ones; (v) results depend on the label definition, which was designed with access to all bearings' health-indicator curves.
**F. Claims you must NOT make:** that v2 predicts RUL accurately (in the degradation phase it does not beat a constant by more than 12%); that the proposed architecture is the best or improves accuracy; that its dual head gives low false alarms in general (only via the classifier head); that attention explains or focuses on recent/shock-related steps; that results are comparable to PHM-2012 leaderboard scores or in minutes; that the test bearings played no role in the target design; that the earlier "AUC 0.86 ± 0.26" or the earlier v2 tables are valid.
