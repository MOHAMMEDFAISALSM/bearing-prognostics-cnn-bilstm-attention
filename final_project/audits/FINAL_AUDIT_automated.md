# FINAL AUDIT (automated, independent re-implementations)

Notebook: `final_bearing_prognostics_review.ipynb`  |  audited at 2026-09-30 20:58

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | zero error outputs in the executed notebook | PASS | [] |
| 2 | all code cells executed, in order, from a clean kernel (execution_count = 1..N) | PASS | [1, 2, 3]...[47, 48, 49] |
| 3 | figures are embedded as outputs | PASS | 34 embedded images |
| 4 | required artifacts exist | PASS | [] |
| 5 | figures directory holds >= 30 non-empty PNG files | PASS | 35 files |
| 6 | selection.json written after LOBO and BEFORE the test predictions | PASS | lobo 1790781102 <= selection 1790781106 < test 1790781889 |
| 7 | LOBO predictions contain only learning bearings | PASS |  |
| 8 | test predictions contain exactly the 11 test bearings | PASS |  |
| 9 | each learning bearing is held out exactly once per model/seed | PASS |  |
| 10 | MAE, actionable MAE, mean AUC and official PHM score re-computed independently match the notebook for every model/seed | PASS | [] |
| 11 | official RUL table vs Full_Test_Set: only Bearing1_4 inconsistent (documented and corrected) | PASS | ['Bearing1_4'] |
| 12 | Bearing1_4 evaluation stops at the official end of life (last snapshot within one 10 s interval of it) | PASS | min RUL 9 s |
| 13 | Bearing1_4 RUL at the truncation point equals the official 339 s | PASS | 339.0 s |
| 14 | recorded configuration equals src/config.py | PASS |  |
| 15 | quick/debug mode is OFF in the recorded run | PASS |  |
| 16 | retraining the proposed model (seed 42) reproduces the saved test predictions | PASS | max |difference| = 1.42e-14 min |

**Overall: ALL AUTOMATED CHECKS PASSED**

## Strong-claim words in static markdown (for manual review)

* cell 40: ... size / max epochs / early stopping | 64 / 40 / patience 6, best weights restored | | Loss (RUL head) | Huber, δ = 0.1, on c...
* cell 43: ...histories, **the networks overfit almost immediately**: the best epoch is early, and the gap between training and validation...
* cell 78: ...e **median** of the 11 (so the demonstration is neither the best nor the worst case). This choice was made *after* the test ...
* cell 82: ...rms, RUL performance, detection, size). **A model is called best only for the criterion where the numbers say so.**...

## Strong-claim words in generated (data-driven) markdown output

* cell 42: ...handful of epochs while training loss keeps falling. Median best epoch of the proposed model: **1.0**. At the end of trainin...
* cell 47: ...Epochs used for the final training (median best LOBO epoch, at least 3): `{'cnn': 3, 'lstm': 3, 'cnn_lstm':...
* cell 48: ....8.  **Detection (mean AUC over horizons, constant 0.50):** best = **LSTM** (0.89); proposed = 0.85; best tabular = 0.72.  *...
* cell 48: ... constant 0.50):** best = **LSTM** (0.89); proposed = 0.85; best tabular = 0.72.  **Lowest false-alarm rate:** Gradient Boos...
* cell 76: ...+ Attention (dual head) (0.749). Models within 0.020 of the best are **statistically indistinguishable** here (seed noise): ...
* cell 86: ... 0.124 for a constant.  * **RQ4 - is the proposed model the best?** Highest mean detection AUC on test: **PROPOSED: CNN + Bi...
* cell 91: ...ction 13 shows the inflation. 2. *Is the proposed model the best?* Only where the tables say so (Section 24); simple models ...
