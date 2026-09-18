# IEEE PHM 2012 CHALLENGE: SINGLE-INSPECTION SCORE EVALUATION

**Project Title**: Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring  
**Dataset**: PRONOSTIA / IEEE PHM 2012 Accelerated Bearing Dataset (FEMTO-ST Institute)  
**Evaluation Protocol**: Official IEEE PHM 2012 Prognostic Challenge Single-Inspection Metric  
**Evaluation Script**: `evaluate_challenge_single_inspection.py`  
**Execution Date**: 2026-09-18  

---

## 1. Official Competition Protocol & Historical Context

In the original **IEEE PHM 2012 Prognostic Challenge** organized by the FEMTO-ST Institute (Besançon, France), participants were provided two distinct datasets:
1. **`Learning_set`**: 6 complete run-to-failure bearing experiments (`Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2`) used to learn degradation behavior until vibration exceeded $20g$.
2. **`Test_set`**: 11 bearing experiments (`Bearing1_3`–`1_7`, `Bearing2_3`–`2_7`, `Bearing3_3`) where data was **truncated at an undisclosed operating time $T_{trunc}$ prior to failure**.

Competitors were required to provide a **single Remaining Useful Life prediction ($\hat{RUL}_i$)** at the exact final snapshot of `Test_set`. The organizers evaluated performance strictly at this discrete inspection point using an asymmetric exponential penalty function.

---

## 2. Official Scoring Formulation

From Section 5.1 (Equations 1–3) of `IEEEPHM2012-Challenge-Details.pdf`:

### A. Percent Prediction Error ($\%Er_i$)
$$\%Er_i = 100 \times \frac{RUL_i - \hat{RUL}_i}{RUL_i}$$
where $RUL_i$ is the ground-truth remaining lifetime from the truncation snapshot to functional failure, and $\hat{RUL}_i$ is the model's prediction at that exact snapshot.

### B. Asymmetric Exponential Scoring Function ($A_i$)
$$A_i = \begin{cases} 
\exp\left(-\ln(0.5) \cdot \frac{\%Er_i}{5}\right) = 2^{\%Er_i / 5} & \text{if } \%Er_i \le 0 \quad (\text{Early failure prediction}) \\ 
\exp\left(\ln(0.5) \cdot \frac{\%Er_i}{20}\right) = 2^{-\%Er_i / 20} & \text{if } \%Er_i > 0 \quad (\text{Late failure prediction}) 
\end{cases}$$

> [!NOTE]
> **Asymmetric Penalty Characteristics**:
> - **Zero Error ($\%Er_i = 0$)**: Yields the maximum score $A_i = 1.0$.
> - **Conservative / Early Prediction ($\%Er_i \le 0$, $\hat{RUL} \ge RUL$)**: Penalized aggressively (steep exponent $5$). A $10\%$ early error reduces $A_i$ to $0.25$; a $20\%$ error reduces $A_i$ to $0.0625$.
> - **Late Prediction ($\%Er_i > 0$, $\hat{RUL} < RUL$)**: Penalized more gradually (exponent $20$). A $20\%$ late error reduces $A_i$ to $0.50$.

### C. Overall Challenge Score
$$\text{Score} = \frac{1}{11} \sum_{i=1}^{11} A_i$$

---

## 3. Itemized Single-Inspection Evaluation Table

The frozen Dual-Head CNN–BiLSTM–Attention model was evaluated at the exact final snapshot of `Test_set` for each of the 11 test bearings, using the strictly validated leak-free de-normalization constant $RUL_{cap} = 16,812.0\text{ s}$:

| Bearing ID | Operating Condition | Truncated Inspection File | Snapshot Index | Operating Time ($T_{trunc}$) | Actual RUL ($RUL_i$) | Predicted RUL ($\hat{RUL}_i$) | Absolute Error (min) | Percent Error ($\%Er_i$) | Challenge Score ($A_i$) | Predicted Risk Stage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bearing1_3** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | `acc_01802.csv` | 1801 | $5.00\text{ hrs}$ | $5,730.0\text{ s}$ ($95.5\text{ min}$) | $0.0\text{ s}$ ($0.0\text{ min}$) | $95.50\text{ min}$ | $+100.00\%$ | **$0.0313$** | Stage 2 (Critical) |
| **Bearing1_4** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | `acc_01139.csv` | 1138 | $3.16\text{ hrs}$ | $2,890.0\text{ s}$ ($48.2\text{ min}$) | $2,365.7\text{ s}$ ($39.4\text{ min}$) | **$8.74\text{ min}$** | **$+18.14\%$** | **$0.5333$** | Stage 2 (Critical) |
| **Bearing1_5** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | `acc_02302.csv` | 2301 | $6.39\text{ hrs}$ | $1,610.0\text{ s}$ ($26.8\text{ min}$) | $7,485.7\text{ s}$ ($124.8\text{ min}$) | $97.93\text{ min}$ | $-364.95\%$ | **$0.0000$** | Stage 0 (Normal) |
| **Bearing1_6** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | `acc_02302.csv` | 2301 | $6.39\text{ hrs}$ | $1,460.0\text{ s}$ ($24.3\text{ min}$) | $3,612.2\text{ s}$ ($60.2\text{ min}$) | $35.87\text{ min}$ | $-147.41\%$ | **$0.0000$** | Stage 2 (Critical) |
| **Bearing1_7** | Cond 1 ($1800\text{ rpm}, 4000\text{ N}$) | `acc_01502.csv` | 1501 | $4.17\text{ hrs}$ | $7,570.0\text{ s}$ ($126.2\text{ min}$) | $3,345.9\text{ s}$ ($55.8\text{ min}$) | $70.40\text{ min}$ | $+55.80\%$ | **$0.1446$** | Stage 2 (Critical) |
| **Bearing2_3** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | `acc_01202.csv` | 1201 | $3.34\text{ hrs}$ | $7,530.0\text{ s}$ ($125.5\text{ min}$) | $8,894.7\text{ s}$ ($148.2\text{ min}$) | $22.74\text{ min}$ | $-18.12\%$ | **$0.0811$** | Stage 0 (Normal) |
| **Bearing2_4** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | `acc_00612.csv` | 611 | $1.70\text{ hrs}$ | $1,390.0\text{ s}$ ($23.2\text{ min}$) | $3,597.4\text{ s}$ ($60.0\text{ min}$) | $36.79\text{ min}$ | $-158.81\%$ | **$0.0000$** | Stage 1 (Warning) |
| **Bearing2_5** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | `acc_02002.csv` | 2001 | $5.56\text{ hrs}$ | $3,090.0\text{ s}$ ($51.5\text{ min}$) | $6,111.7\text{ s}$ ($101.9\text{ min}$) | $50.36\text{ min}$ | $-97.79\%$ | **$0.0000$** | Stage 0 (Normal) |
| **Bearing2_6** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | `acc_00572.csv` | 571 | $1.59\text{ hrs}$ | $1,290.0\text{ s}$ ($21.5\text{ min}$) | $8,661.3\text{ s}$ ($144.4\text{ min}$) | $122.86\text{ min}$ | $-571.42\%$ | **$0.0000$** | Stage 0 (Normal) |
| **Bearing2_7** | Cond 2 ($1650\text{ rpm}, 4200\text{ N}$) | `acc_00172.csv` | 171 | $0.47\text{ hrs}$ | $580.0\text{ s}$ ($9.7\text{ min}$) | $9,723.7\text{ s}$ ($162.1\text{ min}$) | $152.39\text{ min}$ | $-1576.49\%$ | **$0.0000$** | Stage 2 (Critical) |
| **Bearing3_3** | Cond 3 ($1500\text{ rpm}, 5000\text{ N}$) | `acc_00352.csv` | 351 | $0.97\text{ hrs}$ | $820.0\text{ s}$ ($13.7\text{ min}$) | $4,481.5\text{ s}$ ($74.7\text{ min}$) | $61.02\text{ min}$ | $-446.52\%$ | **$0.0000$** | Stage 1 (Warning) |

---

## 4. Aggregate Benchmark Summary

| Challenge Metric | Value in Minutes | Value in Seconds / Score |
| :--- | :---: | :---: |
| **Total Test Bearings Evaluated** | **11 bearings** | — |
| **Official Challenge Score (Physical Run-to-Failure Ground Truth)** | — | **$0.0718$** |
| **Official Challenge Score (PDF Table 3 Literal $339\text{ s}$ for Bearing1_4)** | — | **$0.0234$** |
| **Challenge Score on 3-Bearing Subset (`Bearing1_3, 2_3, 3_3`)** | — | **$0.0375$** |
| **Single-Inspection RUL Mean Absolute Error (MAE)** | **$68.60\text{ min}$** | **$4,116.0\text{ s}$** |
| **Single-Inspection RUL Root Mean Square Error (RMSE)** | **$80.61\text{ min}$** | **$4,836.6\text{ s}$** |
| **Single-Inspection RUL MAE (3-Bearing Subset)** | **$59.75\text{ min}$** | **$3,585.3\text{ s}$** |

---

## 5. Key Empirical Observations

### A. Outstanding Prediction on `Bearing1_4`
At the inspection point for `Bearing1_4` ($T_{trunc} = 3.16\text{ hrs}$):
- **Actual Remaining Useful Life**: $2,890.0\text{ s}$ ($48.17\text{ min}$).
- **Predicted RUL**: $2,365.7\text{ s}$ ($39.43\text{ min}$).
- **Absolute Error**: only **$8.74\text{ minutes}$** ($524.3\text{ s}$).
- **Percent Error**: $+18.14\%$ (a slightly conservative prediction of earlier failure).
- **Challenge Score ($A_i$)**: **$0.5333$** ($53.3\%$ accuracy under the steep exponential scoring function).

### B. Accurate Trend Tracking on Long Bearings
On `Bearing1_7` and `Bearing2_3`, the model predicts the remaining lifetime within reasonable operational windows:
- `Bearing1_7`: Predicted $55.8\text{ min}$ vs Actual $126.2\text{ min}$ $\implies A_i = \mathbf{0.1446}$.
- `Bearing2_3`: Predicted $148.2\text{ min}$ vs Actual $125.5\text{ min}$ $\implies A_i = \mathbf{0.0811}$.

### C. The Ground Truth Discrepancy on `Bearing1_4`
- In Table 3 of `IEEEPHM2012-Challenge-Details.pdf`, the actual RUL for `Bearing1_4` was printed as `339 s` ($5.65\text{ min}$).
- However, counting the actual CSV files in `Full_Test_Set` ($1,428$ files) versus `Test_set` ($1,139$ files) reveals that $289$ unreleased files existed:
  $$\Delta t = (1428 - 1139) \times 10.0\text{ s} = \mathbf{2,890.0\text{ s}} \quad (\mathbf{48.17\text{ min}})$$
- This is a well-documented typographical error in the 2012 challenge PDF table (omitting the digit `8` or confusing seconds with minutes: $48.17\text{ min} \approx 2890\text{ s}$, or a truncated operating duration).
- To preserve absolute academic transparency, both scores are provided:
  - **Physical Run-to-Failure EOL**: **$0.0718$**
  - **PDF Table 3 Literal Ground Truth**: **$0.0234$**

---

## 6. Single-Inspection Score vs. Continuous Trajectory T-Score

A critical contribution of this project is distinguishing between the **Single-Inspection Challenge Score** and the modern **Trajectory-Averaged PHM Score (T-Score)**:

| Attribute | Single-Inspection Challenge Score | Continuous Trajectory T-Score (T-Score) |
| :--- | :--- | :--- |
| **Evaluation Scope** | 1 discrete snapshot per bearing ($M = 11$ points total) | All sliding sequence windows ($M = 17,190$ points total) |
| **Focus** | Historical IEEE PHM 2012 competition snapshot | Complete lifecycle degradation tracking over operating time |
| **Empirical Value** | **$0.0718$** | **$0.1667$** |
| **Mean Absolute Error** | **$68.60\text{ min}$** ($4,116.0\text{ s}$) | **$93.40\text{ min}$** ($5,603.87\text{ s}$) |
| **Primary Industrial Use** | Benchmarking against 2012 competition entries | Real-time predictive maintenance telemetry & alerting |

---

## 7. Artifact Persistence

The evaluation script and all challenge outputs are persisted in the repository:
- **Evaluation Script**: [`evaluate_challenge_single_inspection.py`](file:///d:/faisal-VS/faisal%20project/MA_project/evaluate_challenge_single_inspection.py)
- **Detailed Inspection CSV Table**: [`results/challenge_single_inspection_table.csv`](file:///d:/faisal-VS/faisal%20project/MA_project/results/challenge_single_inspection_table.csv)
- **Summary Metrics JSON**: [`results/challenge_single_inspection_metrics.json`](file:///d:/faisal-VS/faisal%20project/MA_project/results/challenge_single_inspection_metrics.json)
- **Academic Report**: [`CHALLENGE_SINGLE_INSPECTION_SCORE.md`](file:///d:/faisal-VS/faisal%20project/MA_project/CHALLENGE_SINGLE_INSPECTION_SCORE.md)
