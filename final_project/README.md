# Final project: PRONOSTIA / IEEE PHM 2012 bearing prognostics

**Primary deliverable:** `final_bearing_prognostics_review.ipynb` - fully executed from a clean kernel (zero cell errors), with explanations before/after every major section, 30+ generated figures, and a "How to Explain This Project in a Review" section.

## Project Structure / Where Everything Is
```
final_project/
├── final_bearing_prognostics_review.ipynb   MAIN REVIEW NOTEBOOK (must stay next to src/)
├── README.md, PROJECT_INVENTORY.md          this guide, full file inventory
├── src/        source code (config, features, data, models, metrics, experiment, leakage, analysis, report)
├── nb/         builder of the notebook (part1-3.py, build.py); not needed to run the notebook
├── artifacts/  MODEL + RESULTS: final_proposed_model.keras (trained model), final_scaler.pkl (scaler),
│               feature_names.json / final_feature_config.json (feature list), selection.json,
│               experiment_config.json (configuration), *_predictions*.csv, *metrics*.csv/json, ablation_*.csv, perm_importance_*.csv
├── figures/    34 final figures (PNG, 300 dpi)
├── reports/    FINAL_RESEARCH_REPORT.md, DATASET_AUDIT_REPORT.md, REFERENCES.md
├── audits/     FINAL_AUDIT.md (manual), FINAL_AUDIT_automated.md
├── scripts/    audit_final.py (independent audit), make_report.py (regenerates the research report)
└── archive/obsolete/   fig_sample_prediction_Bearing1_7.png (older demo figure, superseded by the _t346min / _t94min files)
```
Note: re-running the notebook re-creates `DATASET_AUDIT_REPORT.md` in the project root (the notebook writes it there); the curated copy is `reports/DATASET_AUDIT_REPORT.md`.

## Run
```bash
cd final_project
jupyter nbconvert --to notebook --execute --inplace final_bearing_prognostics_review.ipynb --ExecutePreprocessor.timeout=14400
python scripts/audit_final.py      # independent audit (writes audits/FINAL_AUDIT_automated.md)
python scripts/make_report.py      # regenerates reports/FINAL_RESEARCH_REPORT.md from the executed notebook
```
Runtime is about one hour on a CPU (deterministic TensorFlow). `BP_QUICK=1` is a debugging switch (3 epochs, 1 seed) and must never be used for reported results.

To demonstrate a single prediction, edit `DEMO_BEARING` / `DEMO_TIME_MIN` in notebook Section 23.
