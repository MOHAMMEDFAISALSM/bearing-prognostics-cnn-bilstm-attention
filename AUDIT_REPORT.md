# ACADEMIC AUDIT REPORT: Explainable Dual-Head CNN–BiLSTM–Attention Bearing Prognostics

**Project Title**: Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring  
**Target File**: `explainable_dual_head_bearing_prognostics.ipynb`  
**Dataset**: PRONOSTIA / IEEE PHM 2012 Accelerated Bearing Dataset (FEMTO-ST Institute)  
**Audit Date**: 2026-09-17  
**Audit Objective**: Rigorous academic audit of dataset fidelity, labeling, feature engineering, sequence generation, leakage prevention, architecture, multi-task targets, training convergence, evaluation metrics, and explainability.

---

## 1. Executive Summary & Verdict

The executed notebook demonstrates a well-structured, functional end-to-end deep learning pipeline that successfully processes real PRONOSTIA vibration data, extracts multi-domain features, trains a dual-head CNN–BiLSTM–Attention model, and generates publication-grade visualizations. 

However, this audit identified **one critical methodological leakage/contamination issue** (test-set ground truth lifetime used during test RUL de-normalization), **three questionable target/metric formulations**, and **several academic opportunities for enhancement**.

### Audit Status Breakdown
- **Correct (No issues)**: 8 components (Dataset alignment, Raw signals, Feature extraction, Bearing-level split, Feature scaling, Sequence generation, Core architecture, Attention explainability).
- **Questionable (Needs clarification/justification)**: 3 components (Health stage target formulation, Trajectory-averaged PHM score, Generalization gap).
- **Needs Correction (Methodological flaw / Incomplete)**: 3 components (Test-set lifetime de-normalization leakage, Linear RUL without degradation onset modeling, Bypassed training history plot).

---

## 2. Comprehensive Audit Matrix

| ID | Notebook Section / Cell | Component Audited | Audit Status | Primary Finding & Academic Impact |
| :--- | :--- | :--- | :--- | :--- |
| **01** | Section 1 (Cell 2–3) | **Dataset Loading & Inventory** | **Correct (Minor Limitation)** | Scanned `Learning_set` and `Full_Test_Set` accurately (17 bearings, 12,298 snapshots). `Test_set` (truncated challenge files) was omitted from audit table. |
| **02** | Section 1 & 4 (Cell 3, 9) | **Bearing Selection & Test Set Coverage** | **Questionable** | Feature extraction and evaluation processed 9 bearings (6 learning, 3 test). 8 of the 11 full test bearings were excluded from inference. |
| **03** | Section 2 (Cell 4–5) | **Raw Vibration & EOL 20g Threshold** | **Correct** | Accurate dual-axis loading ($25.6\text{ kHz}$, 2560 samples). $20g$ physical threshold is accurately referenced. |
| **04** | Section 3 (Cell 6–7) | **Spectral FFT Analysis** | **Correct** | Proper single-sided FFT amplitude spectrum ($0$ to $12.8\text{ kHz}$) with DC offset removal and accurate Nyquist limits. |
| **05** | Section 4 (Cell 8–9) | **Feature Engineering** | **Correct** | 30 robust multi-domain features (15 per channel $\times$ 2 channels) with $\epsilon$-stabilized formulas. Multithreaded extraction and disk caching verified. |
| **06** | Section 5 (Cell 10–11) | **Feature Degradation Progression** | **Correct** | Accurately identifies Kurtosis as an early transient impulse indicator and RMS as a late wear monotonic indicator. |
| **07** | Section 6 (Cell 12–13) | **RUL / EOL Label Formulation** | **Needs Correction** | Assumes linear RUL starting from $t=0$, which contradicts bearing physics (bearings operate with zero degradation for 60–80% of life). |
| **08** | Section 6 (Cell 12–13) | **Bearing-Level Partitioning** | **Correct (Zero Leakage)** | Strict bearing-level isolation across Train (`1_1, 2_1, 3_1`), Val (`1_2, 2_2, 3_2`), and Test (`1_3, 2_3, 3_3`). No cross-bearing contamination. |
| **09** | Section 6 (Cell 12–13) | **Feature Scaling** | **Correct (Zero Leakage)** | `StandardScaler` fitted **strictly on training bearings**. Validation and test bearings transformed without leakage. |
| **10** | Section 6 (Cell 12–13) | **Temporal Sequence Generation** | **Correct (Zero Leakage)** | Sliding windows ($W=16$ snapshots = $160\text{ s}$) constructed strictly within individual bearing trajectories. |
| **11** | Section 7 (Cell 14–15) | **CNN–BiLSTM–Attention Architecture** | **Correct** | Valid dual-head architecture: Conv1D (local features), BiLSTM (temporal dynamics), custom serializable `TemporalAttention`, dual heads. |
| **12** | Section 7 & 9 (Cell 9, 13, 19) | **Failure-Risk Classification Target** | **Questionable** | Health stages (`Normal`, `Warning`, `Critical`) are derived solely by thresholding normalized RUL rather than empirical vibration severity. |
| **13** | Section 8 (Cell 16–17) | **Training Procedure & Convergence** | **Needs Correction** | Cell 17 bypasses `model.fit()` if checkpoint exists. `history` is not captured, and Figure 4 (training convergence) was omitted from the notebook. |
| **14** | Section 9 (Cell 18–19) | **Test RUL De-Normalization** | **CRITICAL: Needs Correction** | Test normalized RUL is de-normalized using `sub['operating_time_s'].iloc[-1]` (the true failure time of the test bearing), which is unknown in deployment. |
| **15** | Section 9 (Cell 18–19) | **PHM Prognostic Score Calculation** | **Questionable** | Formula mathematically matches challenge equations, but evaluates a continuous trajectory average (4,719 windows) rather than the challenge's single inspection point. |
| **16** | Section 11 (Cell 22–23) | **Explainability Implementation** | **Correct (Clarification Needed)** | Successfully extracts temporal attention weights $\alpha_t \in \mathbb{R}^{16}$. Explains *time-step importance*, not *feature importance*. |
| **17** | Section 12 (Cell 24–25) | **Industrial Decision Dashboard** | **Correct** | Functional operational interface integrating telemetry, RUL, remaining time, risk probabilities, and alert states. |

---

## 3. Detailed Technical Findings

### Finding 1: Test-Set Information Leakage in RUL De-normalization (CRITICAL)
- **Location**: Notebook Cell 13 and Cell 19 (`df_meta_test['pred_rul_s'] = pred_norm_rul * df_meta_test['total_lifetime_s']`).
- **Nature**: Methodological Contamination / Information Leakage.
- **Analysis**:
  In Cell 13, `total_lifetime_s` is recorded as `sub['operating_time_s'].iloc[-1]` for test bearings. In Cell 19, the model's normalized prediction $\hat{y}_{norm} \in [0, 1]$ is multiplied by `total_lifetime_s` of that specific test bearing to convert it into physical seconds.
  In a real industrial setting or blind benchmark, **the total lifetime of an unseen test bearing is unknown** (it is the exact quantity the system is deployed to predict). If Bearing A fails at $23,740\text{ s}$ and Bearing B fails at $4,330\text{ s}$, multiplying a model output of $0.5$ by their true total lifetimes injects ground-truth label information into the evaluation.
- **Remediation**:
  1. Define a global maximum lifetime derived **strictly from the training set**:
     $$T_{max} = \max_{b \in \text{Train}} T_{EOL}(b) = 28,030\text{ s} \quad (\text{Bearing1\_1})$$
  2. Normalize training targets by $T_{max}$: $y_{norm} = RUL_{seconds} / T_{max}$.
  3. At test time, de-normalize all predictions using the fixed training constant:
     $$\hat{RUL}_{seconds} = \hat{y}_{norm} \times T_{max}$$
  This completely eliminates test-set lifetime dependency.

---

### Finding 2: Linear RUL vs Piecewise-Linear Degradation Physics
- **Location**: Notebook Cell 8–9 (`rul_s = max(0.0, (total_snapshots - 1 - snapshot_idx) * 10.0)`).
- **Nature**: Model Formulation / Physical Realism.
- **Analysis**:
  The notebook models RUL as a strictly decreasing linear line starting from the very first snapshot ($t=0$).
  However, in bearing degradation physics (e.g., Heimes 2008, Babu et al. 2016, IEEE PHM literature), brand-new bearings exhibit baseline stationary vibration with no physical fault for the first $60\%\text{--}80\%$ of their life. Forcing a neural network to predict that $RUL$ has dropped from $1.0$ to $0.7$ when the vibration signal is identical Gaussian noise forces the model to memorize arbitrary noise patterns.
- **Remediation**:
  Adopt the standard **piecewise-linear RUL model** with an inflection point $RUL_{max}$:
  $$RUL(t) = \min(RUL_{max}, T_{EOL} - t)$$
  where $RUL_{max} \approx 0.6 \times T_{max}$ or determined via the first Kurtosis/RMS divergence point.

---

### Finding 3: Health Stage Target Grounding (RUL vs Vibration Severity)
- **Location**: Notebook Cell 9 (`if norm_rul > 0.40: 0 elif norm_rul > 0.15: 1 else: 2`).
- **Nature**: Target Design / Semantic Consistency.
- **Analysis**:
  The second head is described as predicting "Bearing Failure-Risk Stage (Normal, Warning, Critical)". However, its ground truth is defined purely by binning the normalized RUL target (`> 0.40`, `0.15--0.40`, `\le 0.15`).
  Because degradation onset varies wildly across bearings (e.g., `Bearing1_1` maintains low vibration until the last $10\%$ of life, whereas `Bearing2_1` begins rising earlier), an RUL-based threshold does not correspond to a constant vibration severity level.
- **Remediation**:
  Either:
  - **Option A (Clarification)**: Explicitly document in the Markdown text that Head 2 performs **RUL-Stage Discretization** to provide multi-task auxiliary regularization for the regression head.
  - **Option B (Physical Severity Grounding)**: Define health stages using empirical vibration degradation criteria (e.g., Stage 0: $RMS < 2g$, Stage 1: $2g \le RMS < 10g$, Stage 2: $RMS \ge 10g$ or Peak $\ge 20g$).

---

### Finding 4: Bypassed Training History & Missing Convergence Plot
- **Location**: Notebook Cell 16–17.
- **Nature**: Pipeline Completeness / Artifact Omission.
- **Analysis**:
  Cell 17 checks `if os.path.exists(MODEL_CHECKPOINT_PATH): load_model(...) else: model.fit(...)`. Because the pre-trained weights were already saved on disk, `nbconvert` executed the `load_model` branch.
  Consequently, `history` was never generated during that execution, and the training vs validation loss/accuracy curves were not plotted in the notebook. Notice that in `results/figures/`, `fig4_training_convergence.png` is missing.
- **Remediation**:
  Remove the bypass conditional or explicitly log/plot the training history curves so that the learning dynamics, loss curves, and validation metrics are visibly embedded in the notebook.

---

### Finding 5: Generalization Gap & Operating Condition Shift
- **Location**: Training Execution Logs & Validation Metrics.
- **Nature**: Empirical Performance / Academic Analysis.
- **Analysis**:
  Training logs show:
  - Training Loss: $0.0280$ | Risk Accuracy: $98.30\%$ | RUL MAE: $0.0701$
  - Validation Loss: $1.5559$ | Risk Accuracy: $44.17\%$ | RUL MAE: $0.2226$
  There is a pronounced generalization gap between the training bearings (`Bearing1_1, 2_1, 3_1`) and validation bearings (`Bearing1_2, 2_2, 3_2`).
  In the PRONOSTIA dataset, `Bearing1_1` lasted $7.79\text{ hrs}$, whereas `Bearing1_2` under the exact same condition lasted only $2.42\text{ hrs}$ ($3.2\times$ difference). The model tends to predict a slower degradation slope based on `Bearing1_1`. This domain divergence is a recognized hallmark of PRONOSTIA and should be analyzed in the academic discussion.

---

### Finding 6: IEEE PHM 2012 Score Interpretation
- **Location**: Notebook Cell 18–19.
- **Nature**: Metric Definition Alignment.
- **Analysis**:
  The mathematical formulation of the asymmetric penalty function:
  $$A_i = \begin{cases} \exp(-\ln(0.5) \cdot (\%Er_i / 5)) & \%Er_i \le 0 \\ \exp(\ln(0.5) \cdot (\%Er_i / 20)) & \%Er_i > 0 \end{cases}$$
  is implemented correctly in Python.
  However, the official IEEE PHM 2012 competition computed $A_i$ **once per test bearing** at its designated truncated inspection snapshot (11 scores averaged). The notebook evaluates $A_i$ across **all 4,719 sliding windows** along the entire trajectory.
- **Remediation**:
  Document in the Markdown cell that this metric represents a **Trajectory-Averaged Prognostic Score (T-Score)** across continuous operational inspection windows, rather than the competition's single truncation snapshot score.

---

### Finding 7: Test Set Bearing Coverage
- **Location**: Notebook Cell 9 (`b in ['Bearing1_3', 'Bearing2_3', 'Bearing3_3']`).
- **Nature**: Dataset Utilization.
- **Analysis**:
  `Full_Test_Set` contains 11 bearings (`Bearing1_3` to `1_7`, `Bearing2_3` to `2_7`, `Bearing3_3`). The notebook processes only 3 test bearings (`1_3, 2_3, 3_3`), representing one bearing per operating condition.
  While this covers all three load regimes, the remaining 8 bearings are omitted from evaluation.
- **Remediation**:
  Either state in the methodology that three representative run-to-failure bearings (one per condition) were selected for benchmark prototyping, or extend the evaluation loop to all 11 test bearings.

---

### Finding 8: Temporal vs Feature Explainability Scope
- **Location**: Notebook Cell 22–23.
- **Nature**: Academic Claim Precision.
- **Analysis**:
  The custom `TemporalAttention` layer accurately calculates time-step weights $\alpha_t \in \mathbb{R}^{16}$.
  However, this explains **temporal relevance** (which past 10-second intervals triggered the alarm), not **feature relevance** (which of the 30 vibration features contributed most).
- **Remediation**:
  Ensure the paper and notebook text explicitly specify "Temporal Attention Explainability" rather than generalized "Feature Explainability" to maintain academic precision.

---

## 4. Concrete Remediation Plan

To elevate the notebook from a working prototype to a publication-ready academic submission:

1. **Eliminate De-normalization Leakage**:
   Replace `pred_norm_rul * total_lifetime_s` with `pred_norm_rul * GLOBAL_MAX_TRAIN_LIFETIME` across Cell 13, Cell 19, and Cell 25.
2. **Implement Piecewise-Linear RUL**:
   Add a parameter `RUL_CAP = 1.0` or `inflection_threshold` to prevent early-life linear penalty.
3. **Capture and Plot Training Convergence**:
   Refactor Cell 17 to ensure `history` is retained and plot `fig4_training_convergence.png` showing Loss, MAE, and Accuracy curves.
4. **Clarify Metric Definitions**:
   Add markdown notes clarifying the Trajectory-Averaged PHM Score and the distinction between RUL-stage classification and vibration severity zones.
5. **Expand Test Set Evaluation**:
   Add an optional evaluation switch to loop over all 11 test bearings in `Full_Test_Set`.

---
*Report compiled autonomously following comprehensive codebase and artifact audit.*
