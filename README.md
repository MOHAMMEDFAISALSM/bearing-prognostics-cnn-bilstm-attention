# Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Benchmark-IEEE%20PHM%202012%20%2F%20PRONOSTIA-green.svg)](https://github.com/MOHAMMEDFAISALSM/bearing-prognostics-cnn-bilstm-attention)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official research code, evaluation suites, reproduction artifacts, publication-quality figures, and documentation for the paper:
**"Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring"**.

---

## 📌 Abstract & Overview

Remaining Useful Life (RUL) estimation for critical rotating machinery is often hampered by non-stationary degradation dynamics and opaque black-box deep learning models. This repository implements an end-to-end explainable multi-task prognostics framework combining:
1. **1D-CNN Multi-Scale Feature Extractor**: Captures local transient spatio-temporal vibration patterns from horizontal and vertical acceleration signals.
2. **Bidirectional LSTM (BiLSTM)**: Models long-range sequential degradation progression across operating hours.
3. **Temporal Attention Mechanism**: Dynamically assigns interpretability weights to identify critical failure-inducing degradation epochs.
4. **Dual-Head Multi-Objective Output**: Simultaneously predicts continuous **RUL** (Huber/Smooth L1 loss) and discrete **Degradation Risk State** (Normal, Degrading, Critical).

---

## 🏆 Key Benchmark Results (IEEE PHM 2012 / PRONOSTIA)

Evaluated under a **strict zero-leakage protocol** (pre-processing scalers and $RUL_{cap} = 16,812\text{ s}$ derived exclusively from training bearings):

| Benchmark Suite | Evaluated Bearings | Vibration Snapshots | RUL MAE | RUL RMSE | Normalized MAE | Risk Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 3-Bearing Set** | `1_3`, `2_3`, `3_3` | 4,764 | 104.41 min | 122.10 min | 0.3726 | 50.16% |
| **Full 11-Bearing Suite** | `1_3`–`1_7`, `2_3`–`2_7`, `3_3` | 17,355 | **93.40 min** | **110.49 min** | **0.3333** | **52.82%** |

- **Official Challenge Single-Inspection Evaluation**: Evaluated at the competition terminal test snapshot $T_{trunc}$ with comprehensive discrepancy handling on `Bearing1_4` (Physical EOL score: **0.0718**).

For full details, see the [`PAPER_RESULTS_PACKAGE.md`](PAPER_RESULTS_PACKAGE.md) and [`ALL_11_TEST_BEARINGS_REPORT.md`](ALL_11_TEST_BEARINGS_REPORT.md).

---

## 📁 Repository Structure

```
├── .gitignore                                     # Comprehensive exclusion of raw dataset (~3GB), caches, envs
├── README.md                                      # Project overview and reproduction guide
├── ALL_11_TEST_BEARINGS_REPORT.md                 # Complete audit across all 11 test bearings
├── AUDIT_REPORT.md                                # Methodological and zero-leakage protocol audit
├── CHALLENGE_SINGLE_INSPECTION_SCORE.md           # IEEE PHM 2012 challenge single-snapshot scoring report
├── PAPER_RESULTS_PACKAGE.md                       # Comprehensive paper results, tables, and claim-evidence map
├── POST_REMEDIATION_VERIFICATION.md               # Post-audit verification of metric consistency
├── REMEDIATION_REPORT.md                          # Action report resolving experimental discrepancies
├── build_notebook.py                              # Script generating the reproducible Jupyter research notebook
├── evaluate_all_11_test_bearings.py               # Out-of-sample evaluation on full 11 test bearing dataset
├── evaluate_challenge_single_inspection.py        # Official IEEE PHM 2012 single-inspection scoring script
├── explainable_dual_head_bearing_prognostics.ipynb# Self-contained research notebook with full pipeline
├── models/                                        # Frozen trained weights & preprocessing artifacts
│   ├── dual_head_cnn_bilstm_attention.keras       # Trained dual-head Keras model
│   ├── feature_names.json                         # Feature vector ordering
│   └── scaler.pkl                                 # Zero-leakage fitted StandardScaler
├── paper/                                         # Manuscript materials, LaTeX equations & assets
│   ├── Explainable_DualHead_CNN_BiLSTM_Attention_Bearing_Prognostics.pdf
│   ├── Explainable_DualHead_CNN_BiLSTM_Attention_Bearing_Prognostics.docx
│   ├── CLAIM_EVIDENCE_MAP.md
│   ├── make_figures.py                            # Script regenerating paper figures
│   ├── make_equations.py                          # Script generating equation JSONs
│   ├── build_paper.js                             # Document assembler
│   ├── attn_b13.npy                               # Attention weight data artifact
│   ├── equations.omml.json
│   ├── equations.tex.json
│   └── figures/                                   # High-resolution paper figures (PNG + SVG)
└── results/                                       # Empirical metrics, predictions, and figures
    ├── challenge_single_inspection_metrics.json
    ├── challenge_single_inspection_table.csv
    ├── evaluation_metrics.json
    ├── evaluation_metrics_all_11_bearings.json
    ├── extracted_bearing_features.csv
    ├── extracted_features_all_11_test_bearings.csv
    ├── per_bearing_metrics_all_11_bearings.csv
    ├── test_predictions.csv
    ├── test_predictions_all_11_bearings.csv
    └── figures/                                   # 10 core publication figures (Fig 1–10)
```

---

## ⚡ Quick Start & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/MOHAMMEDFAISALSM/bearing-prognostics-cnn-bilstm-attention.git
cd bearing-prognostics-cnn-bilstm-attention

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install numpy scipy pandas scikit-learn tensorflow matplotlib seaborn
```

### 2. Dataset Acquisition
Download the IEEE PHM 2012 / PRONOSTIA Bearing Dataset from the official FEMTO-ST repository. Place the uncompressed dataset inside the repository root (e.g. `ieee-phm-2012-data-challenge-dataset-master/`), which is automatically ignored by `.gitignore`.

### 3. Running Independent Out-of-Sample Evaluations
Evaluate all 11 test bearings using the frozen trained model:
```bash
python evaluate_all_11_test_bearings.py
```

Evaluate the official IEEE PHM 2012 single-inspection metric:
```bash
python evaluate_challenge_single_inspection.py
```

Generate the complete Jupyter research notebook:
```bash
python build_notebook.py
```

---

## 📊 Publication Figures

The repository includes high-resolution, vector-quality figures located in [`results/figures/`](results/figures/) and [`paper/figures/`](paper/figures/):
- **Figure 1**: Raw horizontal/vertical vibration profiles in healthy vs. degraded regimes.
- **Figure 2**: FFT frequency spectra highlighting harmonic emergence.
- **Figure 3**: Feature trends (RMS, Kurtosis, Peak-to-Peak, Crest Factor) across operating lifetime.
- **Figure 4**: Multi-objective training convergence (Huber RUL loss & Categorical Cross-Entropy).
- **Figure 5**: Continuous ground truth vs. predicted RUL trajectories.
- **Figure 6**: Error residual distribution and $95\%$ confidence bounds.
- **Figure 7**: Dynamic failure risk probability evolution over time.
- **Figure 8**: Confusion matrix for auxiliary 3-state degradation classification.
- **Figure 9**: Temporal attention weight heatmaps demonstrating localized degradation awareness.
- **Figure 10**: Industrial plant-floor predictive maintenance decision dashboard.

---

## 📜 Citation

If you use this codebase, model architecture, or benchmark results in your research, please cite:

```bibtex
@article{faisal2026explainable,
  title={Explainable Dual-Head CNN--BiLSTM--Attention Prognostics for Industrial Bearing Health Monitoring},
  author={Mohammed Faisal, S. M.},
  journal={Preprint / Working Paper},
  year={2026}
}
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
