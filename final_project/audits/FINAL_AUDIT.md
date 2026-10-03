# FINAL AUDIT (manual, complements FINAL_AUDIT_automated.md)

Scope: final clean run of `final_bearing_prognostics_review.ipynb` (49 code cells, executed 1..49 in order, 0 error outputs, 34 embedded figures, 35 PNG files).

| Area | Finding |
|---|---|
| Leakage (splits/scaler/windows) | 20 automated checks in the notebook passed (bearing-disjoint splits, scaler fitted on training bearings only, poison test, truncation/causality, window boundaries, independent feature recomputation, target shuffle, frozen config). |
| Target leakage | Targets use only failure time (official durations; test: truncation point + official RUL). No vibration-derived label, no onset label, no life-fraction target used for the final models. Life-fraction (formulation B) is shown and rejected as non-deployable. |
| Test contamination | `selection.json` (LOBO only) written before `final_predictions_test.csv` (file timestamps checked). Test scored once. Test not used to choose model, epochs, target or thresholds. |
| Metric consistency | MAE, actionable MAE, mean AUC and official PHM score re-implemented independently for every model/seed: all match. Report figures are generated from the notebook outputs. |
| Reproducibility | Retraining the proposed model (seed 42) reproduces saved test predictions, max difference 1.4e-14 min. |
| Unsupported claims | Static text that contradicted the data was found and corrected during this audit (trend-indicator claim, stage-classification claim, lenient H2, "clearly better than chance" wording). Remaining strong words reviewed in FINAL_AUDIT_automated.md; "best" is only used tie-aware. |
| Dataset anomalies | Bearing1_4 Full_Test_Set runs past the official end of life (evaluation truncated at official EoL); Bearing1_1 clock glitches; 20 g criterion not observable in the snapshots of several bearings. |

## Disclosures that cannot be removed by code
1. **Prior exposure:** earlier experiments (v1/v2) looked at test-bearing curves, and the choice of formulation E was motivated by what those experiments showed. The final protocol never tunes on test, but the test set is therefore not perfectly "virgin".
2. RF and epoch settings evolved during development on learning bearings; final settings are frozen in `experiment_config.json`.
3. Only 6 learning bearings: LOBO rankings are unstable (rank correlation LOBO vs test: 0.47 detection, -0.52 actionable RUL error).
4. Attention is functionally weak (uniform-attention replacement changes predictions by 1.4 min) and not causal.
