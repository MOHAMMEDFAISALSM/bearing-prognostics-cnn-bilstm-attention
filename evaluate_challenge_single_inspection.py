# ==============================================================================
# IEEE PHM 2012 CHALLENGE-ALIGNED SINGLE-INSPECTION SCORE EVALUATION
# PRONOSTIA / FEMTO-ST Bearing Prognostic Challenge Protocol
# ==============================================================================
import os
import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKSPACE_DIR, 'results')
PREDICTIONS_CSV = os.path.join(RESULTS_DIR, 'test_predictions_all_11_bearings.csv')

# ------------------------------------------------------------------------------
# 1. Truncated Inspection Points Matching Test_set
# ------------------------------------------------------------------------------
# In the official challenge, participants were provided Test_set where vibration
# was truncated at an unknown operational snapshot prior to failure.
# The single prediction was evaluated at the final snapshot of Test_set.
INSPECTION_METADATA = [
    {
        'bearing': 'Bearing1_3',
        'condition': 'Cond 1',
        'test_set_files': 1802,
        'full_test_files': 2375,
        'last_file': 'acc_01802.csv',
        'snapshot_idx': 1801,
        'actual_rul_s': 5730.0,
        'actual_rul_pdf_literal_s': 5730.0
    },
    {
        'bearing': 'Bearing1_4',
        'condition': 'Cond 1',
        'test_set_files': 1139,
        'full_test_files': 1428,
        'last_file': 'acc_01139.csv',
        'snapshot_idx': 1138,
        'actual_rul_s': 2890.0,             # Physical files (1428 - 1139) * 10 s
        'actual_rul_pdf_literal_s': 339.0   # Printed in PDF Table 3 (typographical error in challenge doc)
    },
    {
        'bearing': 'Bearing1_5',
        'condition': 'Cond 1',
        'test_set_files': 2302,
        'full_test_files': 2463,
        'last_file': 'acc_02302.csv',
        'snapshot_idx': 2301,
        'actual_rul_s': 1610.0,
        'actual_rul_pdf_literal_s': 1610.0
    },
    {
        'bearing': 'Bearing1_6',
        'condition': 'Cond 1',
        'test_set_files': 2302,
        'full_test_files': 2448,
        'last_file': 'acc_02302.csv',
        'snapshot_idx': 2301,
        'actual_rul_s': 1460.0,
        'actual_rul_pdf_literal_s': 1460.0
    },
    {
        'bearing': 'Bearing1_7',
        'condition': 'Cond 1',
        'test_set_files': 1502,
        'full_test_files': 2259,
        'last_file': 'acc_01502.csv',
        'snapshot_idx': 1501,
        'actual_rul_s': 7570.0,
        'actual_rul_pdf_literal_s': 7570.0
    },
    {
        'bearing': 'Bearing2_3',
        'condition': 'Cond 2',
        'test_set_files': 1202,
        'full_test_files': 1955,
        'last_file': 'acc_01202.csv',
        'snapshot_idx': 1201,
        'actual_rul_s': 7530.0,
        'actual_rul_pdf_literal_s': 7530.0
    },
    {
        'bearing': 'Bearing2_4',
        'condition': 'Cond 2',
        'test_set_files': 612,
        'full_test_files': 751,
        'last_file': 'acc_00612.csv',
        'snapshot_idx': 611,
        'actual_rul_s': 1390.0,
        'actual_rul_pdf_literal_s': 1390.0
    },
    {
        'bearing': 'Bearing2_5',
        'condition': 'Cond 2',
        'test_set_files': 2002,
        'full_test_files': 2311,
        'last_file': 'acc_02002.csv',
        'snapshot_idx': 2001,
        'actual_rul_s': 3090.0,
        'actual_rul_pdf_literal_s': 3090.0
    },
    {
        'bearing': 'Bearing2_6',
        'condition': 'Cond 2',
        'test_set_files': 572,
        'full_test_files': 701,
        'last_file': 'acc_00572.csv',
        'snapshot_idx': 571,
        'actual_rul_s': 1290.0,
        'actual_rul_pdf_literal_s': 1290.0
    },
    {
        'bearing': 'Bearing2_7',
        'condition': 'Cond 2',
        'test_set_files': 172,
        'full_test_files': 230,
        'last_file': 'acc_00172.csv',
        'snapshot_idx': 171,
        'actual_rul_s': 580.0,
        'actual_rul_pdf_literal_s': 580.0
    },
    {
        'bearing': 'Bearing3_3',
        'condition': 'Cond 3',
        'test_set_files': 352,
        'full_test_files': 434,
        'last_file': 'acc_00352.csv',
        'snapshot_idx': 351,
        'actual_rul_s': 820.0,
        'actual_rul_pdf_literal_s': 820.0
    }
]

# ------------------------------------------------------------------------------
# 2. Official Challenge Scoring Function (Eq. 1-3 from Challenge PDF)
# ------------------------------------------------------------------------------
def compute_phm_score(actual_rul, pred_rul):
    """
    Computes percent error and official IEEE PHM 2012 challenge score A_i:
      Er_i = 100 * (actual_rul - pred_rul) / actual_rul
      A_i = exp(-ln(0.5) * (Er_i / 5))   if Er_i <= 0  (early prediction)
            exp(+ln(0.5) * (Er_i / 20))  if Er_i > 0   (late prediction)
    """
    err_pct = 100.0 * (actual_rul - pred_rul) / (actual_rul + 1e-6)
    if err_pct <= 0:
        a_i = np.exp(-np.log(0.5) * (err_pct / 5.0))
    else:
        a_i = np.exp(np.log(0.5) * (err_pct / 20.0))
    return float(err_pct), float(a_i)

# ------------------------------------------------------------------------------
# 3. Load Predictions & Extract Single-Inspection Points
# ------------------------------------------------------------------------------
if not os.path.exists(PREDICTIONS_CSV):
    raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_CSV}. Please run evaluate_all_11_test_bearings.py first.")

df_preds = pd.read_csv(PREDICTIONS_CSV)
records = []

for meta in INSPECTION_METADATA:
    b = meta['bearing']
    s_idx = meta['snapshot_idx']
    
    match = df_preds[(df_preds['bearing'] == b) & (df_preds['snapshot_idx'] == s_idx)]
    if len(match) == 0:
        raise ValueError(f"No prediction found for {b} at snapshot {s_idx}!")
    
    pred_row = match.iloc[0]
    pred_rul_s = float(pred_row['pred_rul_s'])
    pred_rul_m = float(pred_row['pred_rul_m'])
    pred_risk = int(pred_row['pred_risk_class'])
    prob_norm = float(pred_row['prob_normal'])
    prob_warn = float(pred_row['prob_warning'])
    prob_crit = float(pred_row['prob_critical'])
    
    act_phys_s = meta['actual_rul_s']
    act_phys_m = act_phys_s / 60.0
    err_phys_pct, a_phys = compute_phm_score(act_phys_s, pred_rul_s)
    abs_err_phys_m = abs(act_phys_m - pred_rul_m)
    
    act_lit_s = meta['actual_rul_pdf_literal_s']
    act_lit_m = act_lit_s / 60.0
    err_lit_pct, a_lit = compute_phm_score(act_lit_s, pred_rul_s)
    
    records.append({
        'Bearing': b,
        'Condition': meta['condition'],
        'Inspection_Snapshot': s_idx,
        'Inspection_File': meta['last_file'],
        'Operating_Hours': round((s_idx * 10.0) / 3600.0, 2),
        'Actual_RUL_Physical_s': round(act_phys_s, 1),
        'Actual_RUL_Physical_m': round(act_phys_m, 2),
        'Predicted_RUL_s': round(pred_rul_s, 1),
        'Predicted_RUL_m': round(pred_rul_m, 2),
        'Absolute_Error_m': round(abs_err_phys_m, 2),
        'Percent_Error_Physical': round(err_phys_pct, 2),
        'Score_A_Physical': round(a_phys, 4),
        'Actual_RUL_PDF_Literal_s': round(act_lit_s, 1),
        'Percent_Error_Literal': round(err_lit_pct, 2),
        'Score_A_Literal': round(a_lit, 4),
        'Predicted_Risk_Class': pred_risk,
        'Prob_Normal': round(prob_norm, 3),
        'Prob_Warning': round(prob_warn, 3),
        'Prob_Critical': round(prob_crit, 3)
    })

df_challenge = pd.DataFrame(records)

# ------------------------------------------------------------------------------
# 4. Aggregate Official Challenge Scores
# ------------------------------------------------------------------------------
# 1) Across All 11 Test Bearings (Physical Ground Truth EOL)
challenge_score_11_phys = float(df_challenge['Score_A_Physical'].mean())
mae_minutes_11 = float(df_challenge['Absolute_Error_m'].mean())
rmse_minutes_11 = float(np.sqrt(np.mean(df_challenge['Absolute_Error_m']**2)))

# 2) Across All 11 Test Bearings (PDF Literal Table 3)
challenge_score_11_lit = float(df_challenge['Score_A_Literal'].mean())

# 3) 3-Bearing Subset (Bearing1_3, 2_3, 3_3)
sub_3 = df_challenge[df_challenge['Bearing'].isin(['Bearing1_3', 'Bearing2_3', 'Bearing3_3'])]
challenge_score_3_phys = float(sub_3['Score_A_Physical'].mean())
mae_minutes_3 = float(sub_3['Absolute_Error_m'].mean())

summary_dict = {
    'Protocol': 'IEEE PHM 2012 Prognostic Challenge (Single-Inspection Point)',
    'Total_Bearings_Evaluated': len(df_challenge),
    'Official_Challenge_Score_All_11_Physical': challenge_score_11_phys,
    'Official_Challenge_Score_All_11_Literal': challenge_score_11_lit,
    'Official_Challenge_Score_3_Bearings': challenge_score_3_phys,
    'Single_Inspection_RUL_MAE_minutes_11': mae_minutes_11,
    'Single_Inspection_RUL_RMSE_minutes_11': rmse_minutes_11,
    'Single_Inspection_RUL_MAE_minutes_3': mae_minutes_3,
    'Scoring_Equation': 'Score = (1/N) * sum(A_i), A_i = exp(-ln(0.5)*(Er_i/5)) if Er_i<=0 else exp(ln(0.5)*(Er_i/20))'
}

# ------------------------------------------------------------------------------
# 5. Persist Results
# ------------------------------------------------------------------------------
output_csv = os.path.join(RESULTS_DIR, 'challenge_single_inspection_table.csv')
output_json = os.path.join(RESULTS_DIR, 'challenge_single_inspection_metrics.json')

df_challenge.to_csv(output_csv, index=False)
with open(output_json, 'w') as f:
    json.dump(summary_dict, f, indent=2)

print("=" * 115)
print("IEEE PHM 2012 PROGNOSTIC CHALLENGE — SINGLE-INSPECTION POINT EVALUATION")
print("=" * 115)
display_cols = [
    'Bearing', 'Condition', 'Inspection_File', 'Operating_Hours',
    'Actual_RUL_Physical_s', 'Predicted_RUL_s', 'Absolute_Error_m',
    'Percent_Error_Physical', 'Score_A_Physical', 'Predicted_Risk_Class'
]
try:
    print(df_challenge[display_cols].to_markdown(index=False))
except Exception:
    print(df_challenge[display_cols].to_string(index=False))

print("\n" + "=" * 80)
print("CHALLENGE SCORE SUMMARY")
print("=" * 80)
print(f"  Official Challenge Score (All 11 Bearings - Physical Run-to-Failure): {challenge_score_11_phys:.4f}")
print(f"  Official Challenge Score (All 11 Bearings - PDF Literal Table 3)    : {challenge_score_11_lit:.4f}")
print(f"  Challenge Score (3-Bearing Subset: Bearing1_3, 2_3, 3_3)             : {challenge_score_3_phys:.4f}")
print(f"  Single-Inspection RUL MAE (All 11 Bearings)                         : {mae_minutes_11:.2f} min ({mae_minutes_11*60.0:.1f} s)")
print(f"  Single-Inspection RUL RMSE (All 11 Bearings)                        : {rmse_minutes_11:.2f} min ({rmse_minutes_11*60.0:.1f} s)")
print("=" * 80)
print(f"\nDetailed CSV table saved to : {output_csv}")
print(f"Summary JSON metrics saved to: {output_json}")
