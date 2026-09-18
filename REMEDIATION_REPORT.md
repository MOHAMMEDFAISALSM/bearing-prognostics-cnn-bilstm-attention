# REMEDIATION REPORT: Explainable Dual-Head Bearing Prognostics

**Project Title**: Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring  
**Executed Notebook**: `explainable_dual_head_bearing_prognostics.ipynb`  
**Execution State**: Clean execution completed (Notebook size: 6.37 MB with all cell outputs, tables, and 10 figures embedded)  
**Date**: 2026-09-17  

---

## 1. Executive Summary

In response to the audit findings documented in `AUDIT_REPORT.md`, all required remediation items have been successfully implemented without redesigning the core CNN–BiLSTM–Temporal-Attention architecture or modifying the 30-feature engineering pipeline. 

The primary breakthrough is the **complete elimination of test-set information leakage** during RUL de-normalization, the adoption of a physically grounded **piecewise-capped RUL formulation**, active model training execution with **captured convergence curves (Figure 4)**, and precise academic documentation for auxiliary RUL-stage classification and the Trajectory-Averaged PHM Prognostic Score.

The notebook was executed from a completely clean state (all stale caches and models purged prior to execution) and verified end-to-end.

---

## 2. Cell-by-Cell Remediation Inventory

| Cell Index & Type | Section Title | Specific Modifications Made |
| :--- | :--- | :--- |
| **Cell 0 [Markdown]** | Academic Header & Abstract | Updated abstract and innovations list: added piecewise-capped RUL formulation, zero test-leakage de-normalization, auxiliary RUL-stage classification, and trajectory-averaged PHM prognostic score. |
| **Cell 3 [Code]** | Section 1: Empirical Dataset Discovery | Added explicit computation of `GLOBAL_MAX_TRAIN_LIFETIME = 28,020.0 s` ($7.78\text{ hrs}$) strictly from training bearings (`Bearing1_1, 2_1, 3_1`). Added configurable `RUL_CAP_RATIO = 0.60` ($RUL_{cap} = 16,812.0\text{ s} = 280.2\text{ min}$). |
| **Cell 8 [Markdown]** | Section 4: Multi-Domain Features & Targets | Formulated physical piecewise capped RUL target $RUL_{capped}(t) = \min(RUL_{cap}, T_{EOL} - t)$ and normalized target $y_{rul} = RUL_{capped} / RUL_{cap} \in [0, 1]$. Documented that health stages (Normal, Warning, Critical) represent operational RUL horizons for multi-task regularization. |
| **Cell 9 [Code]** | Section 4: Parallel Feature Extraction | Replaced linear-from-zero RUL with piecewise capped formulation ($RUL_{capped}$). Updated auxiliary health stages based on capped RUL horizons. Cached to `results/extracted_bearing_features.csv`. |
| **Cell 12 [Markdown]** | Section 6: Ground-Truth Formulation | Documented leakage-free sequence generation and verified that sequence metadata stores only elapsed operating time and capped targets, with zero test lifetime storage. |
| **Cell 13 [Code]** | Section 6: Bearing-Level Splitting | Removed `total_lifetime_s` from sequence metadata. Sequences ($W=16$) built strictly within individual bearing trajectories. `StandardScaler` fitted strictly on training bearings. |
| **Cell 14 [Markdown]** | Section 7: Model Architecture | Clarified that `TemporalAttention` computes normalized attention weights $\alpha_t \in \mathbb{R}^{16}$ across temporal sequence steps (temporal explainability). |
| **Cell 16 [Markdown]** | Section 8: Compilation & Training | Updated description to reflect active training execution and convergence visualization. |
| **Cell 17 [Code]** | Section 8: Model Training & Convergence | **Removed bypass conditional** (`if os.path.exists`). Executed `model.fit()` cleanly, captured `history`, and generated **Figure 4 (`fig4_training_convergence.png`)** with 3 subplots: Multi-Task Loss, RUL MAE, and Risk Head Accuracy. |
| **Cell 18 [Markdown]** | Section 9: Academic Evaluation Metrics | Defined leak-free RUL de-normalization using fixed training constant $RUL_{cap}$. Explicitly documented the Trajectory-Averaged PHM Prognostic Score (T-Score) across continuous operational windows. |
| **Cell 19 [Code]** | Section 9: Evaluation on Unseen Test Bearings | **CRITICAL FIX**: Replaced `pred_norm_rul * total_lifetime_s` with `pred_norm_rul * RUL_CAP_SECONDS`. Calculated MAE, RMSE, Trajectory PHM score, classification metrics, and generated Figure 8 (Confusion Matrix). |
| **Cell 20 [Markdown]** | Section 10: Degradation Trajectory | Clarified that Figure 5 displays ground-truth capped RUL trajectory against model predictions. |
| **Cell 21 [Code]** | Section 10: Trajectories & Residuals | Updated Figure 5 to plot actual capped RUL vs predicted RUL; updated Figure 6 residuals; updated Figure 7 dynamic risk probabilities. |
| **Cell 22 [Markdown]** | Section 11: Temporal Attention | Emphasized that attention weights explain temporal sequence step importance across the 160s observation window. |
| **Cell 23 [Code]** | Section 11: Attention Extraction | Generated Figure 9 comparing attention distribution for early healthy phase vs late critical phase. |
| **Cell 24 [Markdown]** | Section 12: Industrial Dashboard | Documented real-time decision panel powered by leak-free RUL de-normalization. |
| **Cell 25 [Code]** | Section 12: Dashboard Generation | Generated Figure 10 (Industrial Decision Dashboard) with leak-free de-normalized RUL metrics. |
| **Cell 27 [Code]** | Section 13: Artifact Persistence | Audited all saved models, scalers, results, metrics, and all 10 figures. |

---

## 3. New Methodology Description

### 3.1 Strict Zero-Leakage Lifetime Normalization & De-Normalization
1. **Global Maximum Training Lifetime**:
   $$T_{max, train} = \max_{b \in \text{Train}} T_{EOL}(b) = 28,020.0\text{ s} \quad (7.78\text{ hrs, Bearing1\_1})$$
2. **Configurable RUL Cap**:
   $$RUL_{cap} = 0.60 \times T_{max, train} = 16,812.0\text{ s} \quad (280.2\text{ min}, 4.67\text{ hrs})$$
3. **Target Normalization**:
   $$RUL_{capped}(t) = \min(RUL_{cap}, \max(0, T_{EOL} - t)), \quad y_{rul} = \frac{RUL_{capped}(t)}{RUL_{cap}} \in [0, 1]$$
4. **Prediction De-Normalization (Train, Val, & Test)**:
   $$\hat{RUL}_{seconds} = \hat{y}_{rul} \times RUL_{cap}, \quad \hat{RUL}_{minutes} = \frac{\hat{RUL}_{seconds}}{60.0}$$
   **Zero test-bearing lifetime is accessed or utilized at any stage of inference.**

### 3.2 Auxiliary RUL-Stage Classification Head
To provide regularizing multi-task guidance to the regression backbone:
- **Normal (Stage 0)**: $y_{rul} > 0.40$ ($RUL > 112\text{ min}$)
- **Warning (Stage 1)**: $0.15 < y_{rul} \le 0.40$ ($42\text{ min} < RUL \le 112\text{ min}$)
- **Critical (Stage 2)**: $y_{rul} \le 0.15$ ($RUL \le 42\text{ min}$, failure imminent)

### 3.3 Trajectory-Averaged PHM Prognostic Score (T-Score)
Evaluates the asymmetric penalty function across all continuous operational windows $M = 4,719$:
$$T\text{-}Score = \frac{1}{M}\sum_{i=1}^M A_i, \quad A_i = \begin{cases} \exp\left(-\ln(0.5) \cdot \frac{\%Er_i}{5}\right) & \%Er_i \le 0 \\ \exp\left(\ln(0.5) \cdot \frac{\%Er_i}{20}\right) & \%Er_i > 0 \end{cases}$$

---

## 4. Empirical Evaluation Comparison (Unseen Test Bearings)

| Metric | Pre-Remediation (Methodologically Contaminated) | Post-Remediation (Strictly Leak-Free & Piecewise Capped) | Interpretation |
| :--- | :--- | :--- | :--- |
| **RUL MAE (minutes)** | $85.25\text{ min}$ | **$104.41\text{ min}$** | True leak-free error against capped ground truth without test lifetime leakage. |
| **RUL MAE (seconds)** | $5,115.20\text{ s}$ | **$6,264.38\text{ s}$** | Reflects realistic prognostic divergence across operating conditions. |
| **RUL RMSE (minutes)** | $108.94\text{ min}$ | **$122.10\text{ min}$** | Robust root mean square error across all 4,719 test sequences. |
| **RUL RMSE (seconds)** | $6,536.39\text{ s}$ | **$7,326.13\text{ s}$** | Unbiased evaluation without label injection. |
| **Normalized RUL MAE** | $0.2488$ | **$0.3726$** | True normalized error across $[0, 1]$ horizon. |
| **Trajectory PHM Score** | $0.2562$ | **$0.1244$** | Continuous trajectory penalty score (T-Score). |
| **Risk Head Accuracy** | $63.64\%$ | **$50.16\%$** | Auxiliary 3-stage classification accuracy across domain shift. |
| **Risk Weighted F1-Score** | $55.14\%$ | **$48.46\%$** | Balanced classification performance across operational stages. |
| **Risk Macro F1-Score** | $42.83\%$ | **$33.12\%$** | Unweighted macro F1 across Normal, Warning, and Critical. |

> [!NOTE]
> The pre-remediation metrics were artificially optimistic because multiplying normalized predictions by the test bearing's true lifetime gave the model an unfair scaling advantage on bearings with short lifespans. The post-remediation metrics represent the **true, unassisted generalization performance** of the model under real industrial conditions.

---

## 5. Verification of Zero Test-Set Contamination

A formal code and data-flow audit confirms:
1. **Preprocessing & Scaling**: `StandardScaler.fit()` is called **only** on `train_bearings` (`Bearing1_1, 2_1, 3_1`).
2. **Label Generation**: `RUL_CAP_SECONDS = 16,812.0 s` is calculated strictly from `GLOBAL_MAX_TRAIN_LIFETIME = 28,020.0 s`. No test bearing was included in this calculation.
3. **Sequence Construction**: The variable `total_lifetime_s` was completely removed from sequence metadata.
4. **Prediction De-Normalization**: In Cell 19, `pred_rul_s` is calculated as `pred_norm_rul * RUL_CAP_SECONDS`.
5. **Automated Search Audit**: A programmatic scan over all 28 notebook cells confirmed that `total_lifetime_s` is **0 times present** in the notebook.

---

## 6. Generated Publication Figures Inventory

All 10 figures are now sequentially numbered and present in `results/figures/`:
- `fig1_raw_vibration_healthy_vs_degraded.png` (853.3 KB)
- `fig2_frequency_spectrum_fft.png` (403.5 KB)
- `fig3_vibration_feature_trends.png` (586.9 KB)
- `fig4_training_convergence.png` (326.8 KB) — **NEW: Multi-Task Loss, RUL MAE, Risk Accuracy over Epochs**
- `fig5_rul_actual_vs_predicted.png` (549.0 KB)
- `fig6_rul_residuals_error.png` (341.4 KB)
- `fig7_failure_risk_probabilities.png` (654.7 KB)
- `fig8_confusion_matrix.png` (211.8 KB)
- `fig9_temporal_attention_weights.png` (238.1 KB)
- `fig10_industrial_decision_dashboard.png` (352.5 KB)

---
*Remediation successfully verified and documented.*
