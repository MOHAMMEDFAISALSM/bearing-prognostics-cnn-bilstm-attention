# PROJECT INVENTORY (final_project/)

Categories: A main notebook, B source, C model artifacts, D results, E figures, F reports, G audit/reproducibility. H development leftovers, I duplicates: **none remain** (deleted in the previous cleanup step). J unknown/unreferenced: only the obsolete figure noted below.

| path | ext | category | referenced/used by final notebook | needed for reproducibility | final artifact |
|---|---|---|---|---|---|
| artifacts/ablation_lobo.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/ablation_test.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/dataset_audit_table.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/experiment_config.json | .json | G Config / selection | written by it | yes | yes |
| artifacts/feature_documentation.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/feature_names.json | .json | C Model artifact | written by it | yes | yes |
| artifacts/features_raw.csv | .csv | D Intermediate (regenerated each run) | written by it | no (recomputed) | yes |
| artifacts/final_feature_config.json | .json | C Model artifact | written by it | yes | yes |
| artifacts/final_metrics.json | .json | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/final_metrics_per_model_seed.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/final_model_comparison_test.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/final_multicriteria_comparison.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/final_per_bearing_proposed.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/final_predictions_test.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/final_proposed_model.keras | .keras | C Model artifact | written by it | yes | yes |
| artifacts/final_scaler.pkl | .pkl | C Model artifact | written by it | yes | yes |
| artifacts/lobo_metrics_per_fold_seed.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/lobo_model_comparison.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/lobo_predictions.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/perm_importance_proposed_lobo.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/perm_importance_proposed_test.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/perm_importance_rf_lobo.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| artifacts/selection.json | .json | G Config / selection | written by it | yes | yes |
| artifacts/test_phase_lengths_descriptive.csv | .csv | D Result/metric/prediction | written by it | audit reads it | yes |
| scripts/audit_final.py | .py | G Audit | no | yes | yes |
| reports/DATASET_AUDIT_REPORT.md | .md | F Report/documentation | no | no | yes |
| figures/fig01_dataset_overview.png | .png | E Figure | written by it | no | yes |
| figures/fig02_sample_raw_signal.png | .png | E Figure | written by it | no | yes |
| figures/fig03_healthy_vs_degraded.png | .png | E Figure | written by it | no | yes |
| figures/fig04_health_indicator_learning.png | .png | E Figure | written by it | no | yes |
| figures/fig05_feature_vs_life_learning.png | .png | E Figure | written by it | no | yes |
| figures/fig06_feature_distributions.png | .png | E Figure | written by it | no | yes |
| figures/fig07_feature_correlation.png | .png | E Figure | written by it | no | yes |
| figures/fig08_horizon_detectability.png | .png | E Figure | written by it | no | yes |
| figures/fig09_split_protocol.png | .png | E Figure | written by it | no | yes |
| figures/fig_ablation_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_ablation_test.png | .png | E Figure | written by it | no | yes |
| figures/fig_actual_vs_predicted_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_actual_vs_predicted_test.png | .png | E Figure | written by it | no | yes |
| figures/fig_architecture.png | .png | E Figure | written by it | no | yes |
| figures/fig_attention_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_confusion_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_confusion_test.png | .png | E Figure | written by it | no | yes |
| figures/fig_early_warning_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_early_warning_test.png | .png | E Figure | written by it | no | yes |
| figures/fig_error_distribution_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_error_distribution_test.png | .png | E Figure | written by it | no | yes |
| figures/fig_explainability_test.png | .png | E Figure | written by it | no | yes |
| figures/fig_explanation_example_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_feature_importance_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_health_indicator_all17.png | .png | E Figure | written by it | no | yes |
| figures/fig_model_comparison_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_per_bearing_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_per_bearing_test.png | .png | E Figure | written by it | no | yes |
| figures/fig_roc_lobo.png | .png | E Figure | written by it | no | yes |
| figures/fig_roc_test.png | .png | E Figure | written by it | no | yes |
| archive/obsolete/fig_sample_prediction_Bearing1_7.png | .png | E Figure | stale name listed in an output | no | OBSOLETE (see note) |
| figures/fig_sample_prediction_Bearing1_7_t346min.png | .png | E Figure | written by it | no | yes |
| figures/fig_sample_prediction_Bearing1_7_t94min.png | .png | E Figure | written by it | no | yes |
| figures/fig_training_curves.png | .png | E Figure | written by it | no | yes |
| figures/fig_training_metrics.png | .png | E Figure | written by it | no | yes |
| audits/FINAL_AUDIT.md | .md | G Audit | no | yes | yes |
| audits/FINAL_AUDIT_automated.md | .md | G Audit | no | yes | yes |
| final_bearing_prognostics_review.ipynb | .ipynb | A Main review notebook | - | yes | yes |
| reports/FINAL_RESEARCH_REPORT.md | .md | F Report/documentation | no | no | yes |
| scripts/make_report.py | .py | G Report generator | no | no | yes |
| nb/build.py | .py | B Source code (notebook builder) | no (only builds it) | yes (regenerates notebook) | no |
| nb/nbdefs.py | .py | B Source code (notebook builder) | no (only builds it) | yes (regenerates notebook) | no |
| nb/part1.py | .py | B Source code (notebook builder) | no (only builds it) | yes (regenerates notebook) | no |
| nb/part2.py | .py | B Source code (notebook builder) | no (only builds it) | yes (regenerates notebook) | no |
| nb/part3.py | .py | B Source code (notebook builder) | no (only builds it) | yes (regenerates notebook) | no |
| README.md | .md | F Report/documentation | no | no | yes |
| reports/REFERENCES.md | .md | F Report/documentation | no | no | yes |
| src/__init__.py | .py | B Source code | yes (imported) | yes | no |
| src/analysis.py | .py | B Source code | yes (imported) | yes | no |
| src/config.py | .py | B Source code | yes (imported) | yes | no |
| src/data.py | .py | B Source code | yes (imported) | yes | no |
| src/experiment.py | .py | B Source code | yes (imported) | yes | no |
| src/features.py | .py | B Source code | yes (imported) | yes | no |
| src/leakage.py | .py | B Source code | yes (imported) | yes | no |
| src/metrics.py | .py | B Source code | yes (imported) | yes | no |
| src/models.py | .py | B Source code | yes (imported) | yes | no |
| src/report.py | .py | B Source code | yes (imported) | yes | no |

Note: `figures/fig_sample_prediction_Bearing1_7.png` is obsolete (older demo naming) but is listed in a saved notebook output; moved to `archive/obsolete/`. The notebook's saved output still lists it in a directory listing (executed output left untouched). Re-running the notebook re-creates `DATASET_AUDIT_REPORT.md` at the project root.