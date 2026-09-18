# RESEARCH PAPER RESULTS PACKAGE

**Project Title**: Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring  
**Target Venue**: IEEE Transactions on Instrumentation and Measurement / Mechanical Systems and Signal Processing (MSSP)  
**Dataset**: PRONOSTIA / IEEE PHM 2012 Accelerated Bearing Run-to-Failure Dataset (FEMTO-ST Institute)  
**Trained Architecture**: Dual-Head 1D-CNN–BiLSTM–Temporal-Attention (113,924 trainable parameters)  
**Pre-Processing Protocol**: Strictly Zero-Leakage (Fitted exclusively on training bearings; $RUL_{cap} = 16,812.0\text{ s}$)  
**Freeze Status**: Model weights, feature extractors, scaling parameters, and empirical metrics are **strictly frozen**.  
**Date**: 2026-09-18  

---

## 1. Final Aggregate Metrics Table

The table below presents the verified macro-performance across both the baseline 3-bearing test experiment and the full 11-bearing test suite. Predictions are evaluated strictly out-of-sample with zero test-bearing lifetime or EOL leakage.

| Metric Category | Performance Indicator | Baseline 3-Bearing Experiment (`1_3, 2_3, 3_3`) | Full 11-Bearing Benchmark (`1_3`–`1_7`, `2_3`–`2_7`, `3_3`) |
| :--- | :--- | :---: | :---: |
| **Dataset Scale** | Number of Test Bearings | 3 bearings | **11 bearings** ($3.67\times$ coverage) |
| | Evaluated Vibration Snapshots | 4,764 snapshots | **17,355 snapshots** ($44.4\text{M}$ data points) |
| | Evaluated Temporal Windows ($W=16, S=1$) | 4,719 sequences | **17,190 sequences** |
| **RUL Prognosis** | **RUL Mean Absolute Error (MAE)** | **$104.41\text{ min}$** ($6,264.38\text{ s}$) | **$93.40\text{ min}$** ($5,603.87\text{ s}$) |
| | **RUL Root Mean Square Error (RMSE)** | **$122.10\text{ min}$** ($7,326.13\text{ s}$) | **$110.49\text{ min}$** ($6,629.43\text{ s}$) |
| | **Normalized RUL MAE** | **$0.3726$** | **$0.3333$** |
| | **Trajectory-Averaged PHM T-Score** | **$0.1244$** | **$0.1667$** |
| **Auxiliary Risk Classification** | **Classification Accuracy** | **$50.16\%$** | **$52.82\%$** |
| | **Weighted Precision** | **$47.44\%$** | **$50.73\%$** |
| | **Weighted Recall** | **$50.16\%$** | **$52.82\%$** |
| | **Weighted F1-Score** | **$48.46\%$** | **$51.61\%$** |
| | **Macro F1-Score** | **$33.12\%$** | **$40.22\%$** |

---

## 2. Itemized Per-Bearing Evaluation Table

Complete empirical breakdown across all 11 individual test bearings in `Full_Test_Set`, reflecting the true operating conditions ($1800\text{ rpm}/4000\text{ N}$, $1650\text{ rpm}/4200\text{ N}$, $1500\text{ rpm}/5000\text{ N}$):

| Bearing ID | Operating Condition | Total Snapshots | Valid Sequences ($W=16$) | Physical Lifetime (hrs) | RUL MAE (min) | RUL RMSE (min) | Normalized RUL MAE | PHM T-Score | Risk Accuracy (%) | Risk Precision (%) | Risk Recall (%) | Risk Weighted F1 (%) | Risk Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bearing1_3** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | 2,375 | 2,360 | 6.59 | 114.43 | 128.51 | 0.4084 | 0.1150 | 77.80% | 75.12% | 77.80% | 74.64% | 49.16% |
| **Bearing1_4** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | 1,428 | 1,413 | 3.96 | **56.23** | **67.08** | **0.2007** | **0.2668** | 70.28% | 49.43% | 70.28% | 58.03% | 55.59% |
| **Bearing1_5** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | 2,463 | 2,448 | 6.84 | 89.57 | 101.31 | 0.3197 | 0.2092 | 75.04% | 64.28% | 75.04% | 65.93% | 41.56% |
| **Bearing1_6** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | 2,448 | 2,433 | 6.80 | 129.52 | 146.49 | 0.4622 | 0.0951 | 23.35% | 79.17% | 23.35% | 19.73% | 31.57% |
| **Bearing1_7** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | 2,259 | 2,244 | 6.27 | 95.03 | 105.63 | 0.3392 | 0.1716 | 58.42% | 63.50% | 58.42% | 59.78% | 33.47% |
| **Bearing2_3** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | 1,955 | 1,940 | 5.43 | 101.50 | 123.41 | 0.3623 | 0.1615 | 25.52% | 29.50% | 25.52% | 27.27% | 14.90% |
| **Bearing2_4** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | 751 | 736 | 2.08 | **54.12** | **64.94** | **0.1931** | **0.1887** | 42.39% | 49.89% | 42.39% | 44.90% | 26.67% |
| **Bearing2_5** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | 2,311 | 2,296 | 6.42 | 82.85 | 100.21 | 0.2957 | 0.2306 | 69.95% | 58.63% | 69.95% | 62.18% | 39.93% |
| **Bearing2_6** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | 701 | 686 | 1.94 | 59.27 | 68.35 | 0.2115 | 0.1529 | 8.60% | 98.14% | 8.60% | 12.56% | 10.42% |
| **Bearing2_7** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | 230 | 215 | 0.64 | 69.78 | 80.96 | 0.2491 | 0.0000 | 12.09% | 100.00% | 12.09% | 21.58% | 7.19% |
| **Bearing3_3** | Cond 3 ($1500\text{ rpm}, 5000\text{ N}$) | 434 | 419 | 1.20 | 61.40 | 66.15 | 0.2191 | 0.0051 | 8.59% | 60.38% | 8.59% | 15.04% | 8.30% |

---

## 3. Official IEEE PHM 2012 Single-Inspection Score Table

### A. Context & Evaluation Protocol
In the IEEE PHM 2012 competition, model predictions were tested **at a single discrete inspection snapshot** (the terminal file of `Test_set`, $T_{trunc}$) before failure. The scoring metric $A_i$ applies an asymmetric exponential penalty ($2^{\%Er_i/5}$ for early predictions, $2^{-\%Er_i/20}$ for late predictions).

### B. Single-Inspection Results & Discrepancy Transparency
There is an established discrepancy in the literature regarding `Bearing1_4`:
- **Official PDF Table 3 Value**: The challenge document (`IEEEPHM2012-Challenge-Details.pdf`, Table 3) listed the remaining life for `Bearing1_4` as **$339\text{ s}$**.
- **Physical Run-to-Failure Ground Truth**: In the released `Full_Test_Set`, `Bearing1_4` ran for 1,428 files, whereas `Test_set` ended at file 1,139. The physical unreleased lifetime was $(1428 - 1139) \times 10\text{ s} = \mathbf{2,890.0\text{ s}}$ ($48.17\text{ min}$).

To uphold rigorous academic transparency, both scores are explicitly reported:

| Bearing ID | Condition | Truncation File | Operating Time ($T_{trunc}$) | Physical Actual RUL ($RUL_i$) | PDF Literal RUL (Table 3) | Predicted RUL ($\hat{RUL}_i$) | Absolute Error (Physical) | $\%Er_i$ (Physical) | Score $A_i$ (Physical EOL) | Score $A_i$ (PDF Table 3 Literal) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bearing1_3** | Cond 1 | `acc_01802.csv` | $5.00\text{ hrs}$ | $5,730.0\text{ s}$ | $5,730.0\text{ s}$ | $0.0\text{ s}$ | $95.50\text{ min}$ | $+100.00\%$ | **$0.0313$** | **$0.0313$** |
| **Bearing1_4** | Cond 1 | `acc_01139.csv` | $3.16\text{ hrs}$ | **$2,890.0\text{ s}$** | **$339.0\text{ s}$** | $2,365.7\text{ s}$ | **$8.74\text{ min}$** | **$+18.14\%$** | **$0.5333$** | **$0.0000$** |
| **Bearing1_5** | Cond 1 | `acc_02302.csv` | $6.39\text{ hrs}$ | $1,610.0\text{ s}$ | $1,610.0\text{ s}$ | $7,485.7\text{ s}$ | $97.93\text{ min}$ | $-364.95\%$ | **$0.0000$** | **$0.0000$** |
| **Bearing1_6** | Cond 1 | `acc_02302.csv` | $6.39\text{ hrs}$ | $1,460.0\text{ s}$ | $1,460.0\text{ s}$ | $3,612.2\text{ s}$ | $35.87\text{ min}$ | $-147.41\%$ | **$0.0000$** | **$0.0000$** |
| **Bearing1_7** | Cond 1 | `acc_01502.csv` | $4.17\text{ hrs}$ | $7,570.0\text{ s}$ | $7,570.0\text{ s}$ | $3,345.9\text{ s}$ | $70.40\text{ min}$ | $+55.80\%$ | **$0.1446$** | **$0.1446$** |
| **Bearing2_3** | Cond 2 | `acc_01202.csv` | $3.34\text{ hrs}$ | $7,530.0\text{ s}$ | $7,530.0\text{ s}$ | $8,894.7\text{ s}$ | $22.74\text{ min}$ | $-18.12\%$ | **$0.0811$** | **$0.0811$** |
| **Bearing2_4** | Cond 2 | `acc_00612.csv` | $1.70\text{ hrs}$ | $1,390.0\text{ s}$ | $1,390.0\text{ s}$ | $3,597.4\text{ s}$ | $36.79\text{ min}$ | $-158.81\%$ | **$0.0000$** | **$0.0000$** |
| **Bearing2_5** | Cond 2 | `acc_02002.csv` | $5.56\text{ hrs}$ | $3,090.0\text{ s}$ | $3,090.0\text{ s}$ | $6,111.7\text{ s}$ | $50.36\text{ min}$ | $-97.79\%$ | **$0.0000$** | **$0.0000$** |
| **Bearing2_6** | Cond 2 | `acc_00572.csv` | $1.59\text{ hrs}$ | $1,290.0\text{ s}$ | $1,290.0\text{ s}$ | $8,661.3\text{ s}$ | $122.86\text{ min}$ | $-571.42\%$ | **$0.0000$** | **$0.0000$** |
| **Bearing2_7** | Cond 2 | `acc_00172.csv` | $0.47\text{ hrs}$ | $580.0\text{ s}$ | $580.0\text{ s}$ | $9,723.7\text{ s}$ | $152.39\text{ min}$ | $-1576.49\%$ | **$0.0000$** | **$0.0000$** |
| **Bearing3_3** | Cond 3 | `acc_00352.csv` | $0.97\text{ hrs}$ | $820.0\text{ s}$ | $820.0\text{ s}$ | $4,481.5\text{ s}$ | $61.02\text{ min}$ | $-446.52\%$ | **$0.0000$** | **$0.0000$** |
| **AGGREGATE** | **Overall** | — | — | — | — | — | **$68.60\text{ min}$** | — | **$\mathbf{0.0718}$** | **$\mathbf{0.0234}$** |

### Summary of Challenge-Aligned Scores:
- **Official Challenge Score (PDF Table 3 Literal $339\text{ s}$ for Bearing1_4)**: **$\mathbf{0.0234}$**
- **Physical-EOL Sensitivity Analysis ($2,890\text{ s}$ from Full_Test_Set)**: **$\mathbf{0.0718}$**
- **3-Bearing Subset Challenge Score (`Bearing1_3, 2_3, 3_3`)**: **$\mathbf{0.0375}$**
- **Single-Inspection RUL Mean Absolute Error**: **$68.60\text{ min}$** ($4,116.0\text{ s}$)
- **Single-Inspection RUL Root Mean Square Error**: **$80.61\text{ min}$** ($4,836.6\text{ s}$)

---

## 4. Key Figures: Attention, Explainability & Training Convergence

### A. Temporal Attention Weights & Model Explainability (Figure 9)
The temporal attention mechanism generates sequence importance weights $\alpha_t \in [0, 1]$ across the 16 historical steps ($160\text{ s}$ of operational telemetry):

![Temporal Attention Weights Explainability](file:///C:/Users/moham/.gemini/antigravity-ide/brain/9012ac61-cc5a-45d6-b105-8a104f31fa45/fig9_temporal_attention_weights.png)

- **Early Stationary Phase (Healthy)**: Attention is broadly and uniformly distributed ($\alpha_t \approx 0.06$) across the entire sequence window, indicating that no localized shock events dominate.
- **Catastrophic Failure Phase (Critical)**: Attention concentrates sharply on the most recent 1–3 snapshots ($\alpha_{16} \to 0.35\text{--}0.55$), validating that the neural network autonomously detects and isolates high-energy impact bursts during rapid wear.

### B. Multi-Task Training Convergence & Loss Dynamics (Figure 4)
Training history showing total loss, RUL MAE, and auxiliary risk classification accuracy over training epochs with `restore_best_weights=True`:

![Training Convergence Curves](file:///C:/Users/moham/.gemini/antigravity-ide/brain/9012ac61-cc5a-45d6-b105-8a104f31fa45/fig4_training_convergence.png)

- **Convergence Path**: Total loss dropped monotonically from $0.2339 \to 0.0202$ on training bearings.
- **Best Validation Checkpoint**: Epoch 1 achieved best validation loss ($0.7702$); Epoch 2 achieved best validation RUL MAE ($0.2002$). Training halted at Epoch 8 via early stopping (patience = 6).
- **Physical Domain Gap**: Identifies the inherent fatigue variance in PRONOSTIA (`Bearing1_1` lived $7.78\text{ hrs}$ vs `Bearing1_2` $2.42\text{ hrs}$ under identical conditions).

### C. Industrial Health Decision Dashboard (Figure 10)
Real-time operational telemetry dashboard for plant operators showing dual-head outputs: continuous remaining hours, active risk alert banner, and multi-class probability gauges:

![Industrial Health Decision Dashboard](file:///C:/Users/moham/.gemini/antigravity-ide/brain/9012ac61-cc5a-45d6-b105-8a104f31fa45/fig10_industrial_decision_dashboard.png)

---

## 5. Proposed Research-Paper Structure

Below is the recommended outline and section-by-section drafting plan for submission to an IEEE Transactions or Elsevier instrumentation journal:

```
TITLE: Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring Under Physical Lifespan Heterogeneity

ABSTRACT (250 words)
- Problem: Accelerated bearing degradation exhibits non-linear progression, physical lifespan heterogeneity, and black-box opacity.
- Proposed Architecture: Multi-task 1D-CNN (local feature extraction) + BiLSTM (temporal degradation modeling) + Temporal Attention (time-step explainability) + Dual Heads (Piecewise RUL regression + 3-stage auxiliary operational risk classification).
- Dataset & Integrity: PRONOSTIA / IEEE PHM 2012 benchmark across 17 bearings (72.1 hours, 24,889 snapshots). Evaluated under strictly zero-leakage conditions (training-derived RUL cap = 16,812 s; scaler fitted only on training bearings).
- Key Empirical Results: Full 11-bearing benchmark achieves RUL MAE of 93.40 min (0.3333 normalized MAE), Trajectory T-Score of 0.1667, and Risk F1 of 51.61%. Official single-inspection challenge score achieves 0.0234 (PDF Table 3 literal) and 0.0718 (physical EOL sensitivity).
- Interpretability: Attention weights demonstrate physical transition from uniform baseline context to localized shock focus.

I. INTRODUCTION
  A. Context & Industrial Motivation (Predictive Maintenance in Industry 4.0)
  B. Methodological Limitations of Existing Deep Learning Prognostics (Data Leakage, Linear RUL Penalties, Black-Box Opacity)
  C. Core Contributions of this Paper:
     1. Strict Zero-Leakage Preprocessing & Normalization Framework.
     2. Dual-Head Multi-Task Formulation with Piecewise Capped RUL.
     3. Self-Attention Temporal Explainability.
     4. Dual Benchmarking: Full Lifecycle Trajectory vs. Official IEEE PHM 2012 Single-Inspection Score.

II. RELATED WORK & METHODOLOGICAL GAPS
  A. Physics-Informed vs. Data-Driven Prognostics.
  B. Hybrid Deep Architectures (CNN, LSTM, Transformers).
  C. The Data Leakage Trap in Bearing Prognostics (Critical Review of Test EOL Contamination in Literature).
  D. Single-Inspection Challenge Scoring vs. Trajectory Monitoring.

III. PHYSICAL DATASET & MULTI-DOMAIN FEATURE ENGINEERING
  A. The PRONOSTIA / IEEE PHM 2012 Accelerated Testbed (FEMTO-ST).
  B. Operating Regimes: Loads (4000 N to 5000 N) and Speeds (1500 rpm to 1800 rpm).
  C. 30-Dimensional Multi-Domain Feature Vector (Time-Domain Statistical & Spectral FFT Indicators).
  D. Degradation Monotonicity: Kurtosis as Early-Warning Shock Trigger vs. RMS as Catastrophic Wear Regressor.

IV. PROPOSED EXPLAINABLE DUAL-HEAD ARCHITECTURE
  A. Mathematical Problem Formulation.
  B. 1D-CNN Local Spatio-Temporal Representation Extraction.
  C. Bidirectional LSTM Sequence Progression Modeling.
  D. Self-Attention Mechanism for Temporal Time-Step Interpretability.
  E. Multi-Task Objective Function (Joint Smooth L1 / Huber RUL Loss + Sparse Categorical Cross-Entropy Risk Loss).
  F. Piecewise-Capped Target Formulation: Physical Justification of 60% Lifetime Cap.

V. EXPERIMENTAL SETUP & ZERO-LEAKAGE EVALUATION PROTOCOL
  A. Mutually Exclusive Bearing Splits (Train, Validation, Test).
  B. Derivation of Fixed Constants (GLOBAL_MAX_TRAIN_LIFETIME = 28,020 s; RUL_CAP_SECONDS = 16,812 s).
  C. Sliding Window Formulation (Window Length W = 16, Step S = 1).
  D. Evaluation Metrics: MAE, RMSE, Normalized MAE, Macro/Weighted F1, Trajectory T-Score, and Official Challenge Score.

VI. EXPERIMENTAL RESULTS & COMPARATIVE BENCHMARKING
  A. Training Dynamics & Convergence (Analysis of Early Stopping Checkpoint).
  B. 3-Bearing Validation Baseline vs. Full 11-Bearing Out-of-Sample Benchmark.
  C. Analysis of Operating Condition Robustness (Cond 1 vs Cond 2 vs Cond 3).
  D. Official IEEE PHM 2012 Challenge Single-Inspection Evaluation:
     - Discrepancy Analysis on Bearing1_4 (Table 3 Literal 339 s vs Physical 2,890 s).
     - Standout Single-Bearing Prediction (Bearing1_4 Error: 8.74 min, Score: 0.5333).
     - Global Scores: 0.0234 (Literal) vs. 0.0718 (Physical EOL).
  E. Comparison: Discrete Single-Inspection Score vs. Continuous Lifecycle Trajectory Monitoring.

VII. MODEL EXPLAINABILITY & OPERATIONAL DECISION DASHBOARD
  A. Qualitative & Quantitative Attention Weight Analysis Across Lifecycles.
  B. Dynamic Failure Risk Probability Progression Over Time.
  C. Plant-Floor Telemetry Panel for Predictive Maintenance Scheduling.

VIII. THREATS TO VALIDITY & LIMITATIONS
  A. Lifespan Heterogeneity & Cross-Bearing Domain Shift in PRONOSTIA.
  B. Limited Representation of Condition 3 (1 Test Bearing).
  C. Computational Overhead of Multi-Domain Spectral Transforms in Embedded Edge Devices.

IX. CONCLUSION & FUTURE WORK
  A. Summary of Scientific & Empirical Findings.
  B. Future Directions: Cross-Asset Domain Adaptation, Contrastive Pre-Training, and Edge-Deployment on Micro-controllers.
```

---

## 6. Persisted Artifacts & Integrity Verification

All outputs, tables, and scripts are stored in the project workspace:
- **Paper Package Summary**: [`PAPER_RESULTS_PACKAGE.md`](file:///d:/faisal-VS/faisal%20project/MA_project/PAPER_RESULTS_PACKAGE.md)
- **11-Bearing Complete Report**: [`ALL_11_TEST_BEARINGS_REPORT.md`](file:///d:/faisal-VS/faisal%20project/MA_project/ALL_11_TEST_BEARINGS_REPORT.md)
- **Single-Inspection Challenge Report**: [`CHALLENGE_SINGLE_INSPECTION_SCORE.md`](file:///d:/faisal-VS/faisal%20project/MA_project/CHALLENGE_SINGLE_INSPECTION_SCORE.md)
- **Post-Remediation Verification**: [`POST_REMEDIATION_VERIFICATION.md`](file:///d:/faisal-VS/faisal%20project/MA_project/POST_REMEDIATION_VERIFICATION.md)
- **Official Challenge Score Script**: [`evaluate_challenge_single_inspection.py`](file:///d:/faisal-VS/faisal%20project/MA_project/evaluate_challenge_single_inspection.py)
- **11-Bearing Benchmark Script**: [`evaluate_all_11_test_bearings.py`](file:///d:/faisal-VS/faisal%20project/MA_project/evaluate_all_11_test_bearings.py)
- **Detailed Challenge Inspection CSV**: [`results/challenge_single_inspection_table.csv`](file:///d:/faisal-VS/faisal%20project/MA_project/results/challenge_single_inspection_table.csv)
- **Challenge Metrics JSON**: [`results/challenge_single_inspection_metrics.json`](file:///d:/faisal-VS/faisal%20project/MA_project/results/challenge_single_inspection_metrics.json)
- **11-Bearing Per-Bearing CSV Table**: [`results/per_bearing_metrics_all_11_bearings.csv`](file:///d:/faisal-VS/faisal%20project/MA_project/results/per_bearing_metrics_all_11_bearings.csv)
- **11-Bearing Predictions CSV**: [`results/test_predictions_all_11_bearings.csv`](file:///d:/faisal-VS/faisal%20project/MA_project/results/test_predictions_all_11_bearings.csv)
- **Publication Figures (Figures 1–10)**: `results/figures/fig*.png`
