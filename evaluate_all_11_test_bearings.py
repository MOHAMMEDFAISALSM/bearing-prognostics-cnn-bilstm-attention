# ==============================================================================
# EVALUATION SCRIPT: ALL 11 TEST BEARINGS IN FULL_TEST_SET
# PRONOSTIA / IEEE PHM 2012 Accelerated Bearing Run-to-Failure Dataset
# ==============================================================================
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers

sys.stdout.reconfigure(encoding='utf-8')

# ------------------------------------------------------------------------------
# 1. Configuration & Paths
# ------------------------------------------------------------------------------
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE_DIR, 'ieee-phm-2012-data-challenge-dataset-master')
MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')
RESULTS_DIR = os.path.join(WORKSPACE_DIR, 'results')

FULL_TEST_BEARINGS = [
    'Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
    'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
    'Bearing3_3'
]

# Fixed physical constants derived strictly from training bearings in the validated experiment
GLOBAL_MAX_TRAIN_LIFETIME = 28020.0  # s (derived strictly from Bearing1_1)
RUL_CAP_RATIO = 0.60
RUL_CAP_SECONDS = 16812.0  # s (0.60 * 28020.0 s = 280.2 min)
WINDOW_LEN = 16

operating_conditions_map = {
    'Bearing1_1': (1800, 4000, 1), 'Bearing1_2': (1800, 4000, 1),
    'Bearing1_3': (1800, 4000, 1), 'Bearing1_4': (1800, 4000, 1),
    'Bearing1_5': (1800, 4000, 1), 'Bearing1_6': (1800, 4000, 1), 'Bearing1_7': (1800, 4000, 1),
    'Bearing2_1': (1650, 4200, 2), 'Bearing2_2': (1650, 4200, 2),
    'Bearing2_3': (1650, 4200, 2), 'Bearing2_4': (1650, 4200, 2),
    'Bearing2_5': (1650, 4200, 2), 'Bearing2_6': (1650, 4200, 2), 'Bearing2_7': (1650, 4200, 2),
    'Bearing3_1': (1500, 5000, 3), 'Bearing3_2': (1500, 5000, 3),
    'Bearing3_3': (1500, 5000, 3)
}

# ------------------------------------------------------------------------------
# 2. Register Custom Layer & Load Model & Scaler
# ------------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(layers.Layer):
    """Custom Attention Layer calculating temporal importance weights across sequence time steps."""
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name='attention_weight', shape=(d, d),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(d,),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(d, 1),
                                 initializer='glorot_uniform', trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        weights = tf.nn.softmax(ait, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context, weights

    def get_config(self):
        return super(TemporalAttention, self).get_config()

print("Loading pre-trained model and pre-fitted scaler...")
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
scaler = joblib.load(scaler_path)
print(f"  Scaler loaded: {scaler_path} (Fitted features: {len(scaler.mean_)})")

feat_path = os.path.join(MODELS_DIR, 'feature_names.json')
with open(feat_path, 'r') as f:
    feature_columns = json.load(f)
print(f"  Feature names loaded: {len(feature_columns)} columns")

model_path = os.path.join(MODELS_DIR, 'dual_head_cnn_bilstm_attention.keras')
model = tf.keras.models.load_model(model_path, compile=False)
print(f"  Model loaded: {model_path} ({model.count_params():,} parameters)")

# ------------------------------------------------------------------------------
# 3. Feature Extractor for Snapshots
# ------------------------------------------------------------------------------
def extract_snapshot_features(filepath, bearing_name, snapshot_idx, total_snapshots, fs=25600.0):
    """Extracts 30 vibration features and generates piecewise capped RUL targets."""
    # Handle delimiter variance: Bearing1_4 uses ';' delimiter, other bearings use ','
    sep = ';' if bearing_name == 'Bearing1_4' else ','
    try:
        data = pd.read_csv(filepath, sep=sep, header=None, usecols=[4, 5], dtype=np.float32).values
    except Exception:
        # Robust fallback with auto-detection
        data = pd.read_csv(filepath, sep=None, engine='python', header=None, usecols=[4, 5], dtype=np.float32).values
    h, v = data[:, 0], data[:, 1]
    
    speed, load, cond = operating_conditions_map.get(bearing_name, (1800, 4000, 1))
    
    operating_time_s = snapshot_idx * 10.0
    actual_rul_s = max(0.0, (total_snapshots - 1 - snapshot_idx) * 10.0)
    capped_rul_s = min(RUL_CAP_SECONDS, actual_rul_s)
    norm_rul = capped_rul_s / RUL_CAP_SECONDS
    
    if norm_rul > 0.40:
        health_stage = 0
    elif norm_rul > 0.15:
        health_stage = 1
    else:
        health_stage = 2
        
    feat = {
        'bearing': bearing_name,
        'snapshot_idx': snapshot_idx,
        'operating_time_s': operating_time_s,
        'operating_time_m': operating_time_s / 60.0,
        'actual_rul_s': actual_rul_s,
        'actual_rul_m': actual_rul_s / 60.0,
        'capped_rul_s': capped_rul_s,
        'capped_rul_m': capped_rul_s / 60.0,
        'norm_rul': norm_rul,
        'health_stage': health_stage,
        'speed_rpm': speed,
        'load_n': load,
        'condition_id': cond
    }
    
    for prefix, s in [('horiz', h), ('vert', v)]:
        mean_val = np.mean(s)
        s_centered = s - mean_val
        std_val = np.std(s) + 1e-8
        rms = np.sqrt(np.mean(s**2))
        peak = np.max(np.abs(s))
        p2p = np.ptp(s)
        mean_abs = np.mean(np.abs(s)) + 1e-8
        variance = np.mean(s_centered**2) + 1e-8
        
        # Time-domain statistical indicators
        kurt = np.mean(s_centered**4) / (variance**2) - 3.0
        skew = np.mean(s_centered**3) / (variance**1.5)
        crest_factor = peak / (rms + 1e-8)
        shape_factor = rms / mean_abs
        margin_factor = peak / ((np.mean(np.sqrt(np.abs(s)))**2) + 1e-8)
        impulse_factor = peak / mean_abs
        
        # Frequency-domain spectral indicators
        fft_vals = np.abs(np.fft.rfft(s_centered))
        f_axis = np.fft.rfftfreq(len(s), d=1.0/fs)
        spec_energy = np.sum(fft_vals**2)
        fft_sum = np.sum(fft_vals) + 1e-8
        spec_centroid = np.sum(f_axis * fft_vals) / fft_sum
        spec_spread = np.sqrt(np.sum(((f_axis - spec_centroid)**2) * fft_vals) / fft_sum)
        spec_peak_freq = f_axis[np.argmax(fft_vals)]
        rmsf = np.sqrt(np.sum((f_axis**2) * fft_vals) / fft_sum)
        
        feat[f'{prefix}_rms'] = rms
        feat[f'{prefix}_peak'] = peak
        feat[f'{prefix}_p2p'] = p2p
        feat[f'{prefix}_std'] = std_val
        feat[f'{prefix}_kurtosis'] = kurt
        feat[f'{prefix}_skewness'] = skew
        feat[f'{prefix}_crest_factor'] = crest_factor
        feat[f'{prefix}_shape_factor'] = shape_factor
        feat[f'{prefix}_margin_factor'] = margin_factor
        feat[f'{prefix}_impulse_factor'] = impulse_factor
        feat[f'{prefix}_spec_energy'] = spec_energy
        feat[f'{prefix}_spec_centroid'] = spec_centroid
        feat[f'{prefix}_spec_spread'] = spec_spread
        feat[f'{prefix}_spec_peak_freq'] = spec_peak_freq
        feat[f'{prefix}_rmsf'] = rmsf
        
    return feat

def process_bearing_folder(subset_name, bearing_name):
    folder = os.path.join(DATASET_DIR, subset_name, bearing_name)
    csv_files = sorted([f for f in os.listdir(folder) if f.startswith('acc_') and f.endswith('.csv')])
    total_files = len(csv_files)
    
    tasks = [(os.path.join(folder, f), bearing_name, idx, total_files) for idx, f in enumerate(csv_files)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda args: extract_snapshot_features(*args), tasks))
    return results

# ------------------------------------------------------------------------------
# 4. Load or Extract Features for All 11 Test Bearings
# ------------------------------------------------------------------------------
CACHE_EXISTING = os.path.join(RESULTS_DIR, 'extracted_bearing_features.csv')
CACHE_11_TEST = os.path.join(RESULTS_DIR, 'extracted_features_all_11_test_bearings.csv')

print("\nAssembling feature dataset for all 11 test bearings...")
all_test_features = []

if os.path.exists(CACHE_11_TEST):
    print(f"  Loading all 11 test bearings from precomputed cache: {CACHE_11_TEST}")
    df_test_features = pd.read_csv(CACHE_11_TEST)
    print(f"  Loaded {len(df_test_features):,} snapshots across {df_test_features['bearing'].nunique()} bearings.")
else:
    # Check existing extracted features cache
    existing_df = None
    if os.path.exists(CACHE_EXISTING):
        existing_df = pd.read_csv(CACHE_EXISTING)
        print(f"  Existing feature cache loaded: {len(existing_df):,} rows")

    for b in FULL_TEST_BEARINGS:
        if existing_df is not None and b in existing_df['bearing'].values:
            sub = existing_df[existing_df['bearing'] == b].to_dict('records')
            all_test_features.extend(sub)
            print(f"  Loaded {b} from existing cache ({len(sub):,} snapshots)")
        else:
            t0 = time.time()
            res = process_bearing_folder('Full_Test_Set', b)
            all_test_features.extend(res)
            print(f"  Extracted {b} ({len(res):,} files) in {time.time()-t0:.2f}s")

    df_test_features = pd.DataFrame(all_test_features)
    df_test_features.to_csv(CACHE_11_TEST, index=False)
    print(f"  All 11 test bearings features saved to: {CACHE_11_TEST} ({len(df_test_features):,} snapshots)")

# ------------------------------------------------------------------------------
# 5. Form Sequences with Zero Leakage
# ------------------------------------------------------------------------------
print("\nForming sliding sequences (W=16, S=1) and standardizing features with pre-fitted scaler...")

def create_bearing_test_sequences(df, bearing_name, scaler_obj, feat_cols, window_len=16):
    b_df = df[df['bearing'] == bearing_name].sort_values('snapshot_idx').reset_index(drop=True)
    N = len(b_df)
    if N < window_len:
        return None, None
    
    # Scale strictly using training-fitted scaler
    X_raw = b_df[feat_cols].values
    X_scaled = scaler_obj.transform(X_raw)
    
    X_seqs = []
    meta_list = []
    for i in range(window_len - 1, N):
        X_seqs.append(X_scaled[i - window_len + 1 : i + 1])
        meta_list.append({
            'bearing': bearing_name,
            'snapshot_idx': b_df.loc[i, 'snapshot_idx'],
            'operating_time_s': b_df.loc[i, 'operating_time_s'],
            'operating_time_m': b_df.loc[i, 'operating_time_m'],
            'actual_rul_s': b_df.loc[i, 'actual_rul_s'],
            'actual_rul_m': b_df.loc[i, 'actual_rul_m'],
            'capped_rul_s': b_df.loc[i, 'capped_rul_s'],
            'capped_rul_m': b_df.loc[i, 'capped_rul_m'],
            'norm_rul_target': b_df.loc[i, 'norm_rul'],
            'health_stage': int(b_df.loc[i, 'health_stage']),
            'condition_id': int(b_df.loc[i, 'condition_id'])
        })
        
    return np.array(X_seqs, dtype=np.float32), pd.DataFrame(meta_list)

# ------------------------------------------------------------------------------
# 6. Evaluation Function (IEEE PHM T-Score, MAE, RMSE, Classification Metrics)
# ------------------------------------------------------------------------------
def compute_phm_t_score(true_rul, pred_rul):
    """Computes NASA / IEEE PHM 2012 Prognostic Score (Trajectory T-Score)."""
    err_pct = 100.0 * (true_rul - pred_rul) / (true_rul + 1e-6)
    penalties = np.where(err_pct <= 0,
                         np.exp(-np.log(0.5) * (err_pct / 5.0)),
                         np.exp(np.log(0.5) * (err_pct / 20.0)))
    return float(np.mean(penalties))

all_meta_predictions = []
per_bearing_records = []

for b in FULL_TEST_BEARINGS:
    X_b, df_meta_b = create_bearing_test_sequences(df_test_features, b, scaler, feature_columns, WINDOW_LEN)
    if X_b is None:
        print(f"Warning: {b} has fewer than {WINDOW_LEN} snapshots. Skipped.")
        continue
        
    preds = model.predict(X_b, batch_size=128, verbose=0)
    if isinstance(preds, dict):
        pred_risk_probs = preds['risk_output']
        pred_norm_rul = np.clip(preds['rul_output'].flatten(), 0.0, 1.0)
    else:
        pred_risk_probs = preds[0]
        pred_norm_rul = np.clip(preds[1].flatten(), 0.0, 1.0)
    pred_risk_class = np.argmax(pred_risk_probs, axis=-1)
    
    # Strictly training-derived de-normalization: RUL_CAP_SECONDS = 16812.0 s
    df_meta_b['pred_norm_rul'] = pred_norm_rul
    df_meta_b['pred_rul_s'] = pred_norm_rul * RUL_CAP_SECONDS
    df_meta_b['pred_rul_m'] = df_meta_b['pred_rul_s'] / 60.0
    df_meta_b['pred_risk_class'] = pred_risk_class
    df_meta_b['prob_normal'] = pred_risk_probs[:, 0]
    df_meta_b['prob_warning'] = pred_risk_probs[:, 1]
    df_meta_b['prob_critical'] = pred_risk_probs[:, 2]
    
    # Bearing-level metrics
    b_mae_s = mean_absolute_error(df_meta_b['capped_rul_s'], df_meta_b['pred_rul_s'])
    b_mae_m = b_mae_s / 60.0
    b_rmse_s = np.sqrt(mean_squared_error(df_meta_b['capped_rul_s'], df_meta_b['pred_rul_s']))
    b_rmse_m = b_rmse_s / 60.0
    b_norm_mae = mean_absolute_error(df_meta_b['norm_rul_target'], df_meta_b['pred_norm_rul'])
    b_t_score = compute_phm_t_score(df_meta_b['capped_rul_s'].values, df_meta_b['pred_rul_s'].values)
    
    y_true_risk = df_meta_b['health_stage'].values
    y_pred_risk = df_meta_b['pred_risk_class'].values
    b_acc = accuracy_score(y_true_risk, y_pred_risk)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_risk, y_pred_risk, average='weighted', zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true_risk, y_pred_risk, average='macro', zero_division=0)
    
    cond_id = df_meta_b['condition_id'].iloc[0]
    total_life_s = df_meta_b['operating_time_s'].iloc[-1]
    
    per_bearing_records.append({
        'Bearing': b,
        'Condition': f"Cond {cond_id}",
        'Snapshots': len(df_meta_b) + WINDOW_LEN - 1,
        'Sequences': len(df_meta_b),
        'Lifetime_hrs': round(total_life_s / 3600.0, 2),
        'RUL_MAE_min': round(b_mae_m, 2),
        'RUL_RMSE_min': round(b_rmse_m, 2),
        'Norm_RUL_MAE': round(b_norm_mae, 4),
        'PHM_T_Score': round(b_t_score, 4),
        'Risk_Acc_pct': round(b_acc * 100.0, 2),
        'Risk_Prec_pct': round(prec * 100.0, 2),
        'Risk_Recall_pct': round(rec * 100.0, 2),
        'Risk_W_F1_pct': round(f1 * 100.0, 2),
        'Risk_Macro_F1_pct': round(macro_f1 * 100.0, 2)
    })
    
    all_meta_predictions.append(df_meta_b)
    print(f"  {b:12s} ({len(df_meta_b):5d} seqs) | RUL MAE: {b_mae_m:6.2f} min | RMSE: {b_rmse_m:6.2f} min | Norm MAE: {b_norm_mae:.4f} | Risk Acc: {b_acc*100.0:5.2f}% | T-Score: {b_t_score:.4f}")

# Combine all predictions
df_all_preds = pd.concat(all_meta_predictions, ignore_index=True)
output_preds_csv = os.path.join(RESULTS_DIR, 'test_predictions_all_11_bearings.csv')
df_all_preds.to_csv(output_preds_csv, index=False)
print(f"\nAll 11 bearings predictions saved to: {output_preds_csv} (Total sequences: {len(df_all_preds):,})")

# ------------------------------------------------------------------------------
# 7. Aggregate Metrics across All 11 Bearings
# ------------------------------------------------------------------------------
agg_mae_s = mean_absolute_error(df_all_preds['capped_rul_s'], df_all_preds['pred_rul_s'])
agg_mae_m = agg_mae_s / 60.0
agg_rmse_s = np.sqrt(mean_squared_error(df_all_preds['capped_rul_s'], df_all_preds['pred_rul_s']))
agg_rmse_m = agg_rmse_s / 60.0
agg_norm_mae = mean_absolute_error(df_all_preds['norm_rul_target'], df_all_preds['pred_norm_rul'])
agg_t_score = compute_phm_t_score(df_all_preds['capped_rul_s'].values, df_all_preds['pred_rul_s'].values)

y_true_all = df_all_preds['health_stage'].values
y_pred_all = df_all_preds['pred_risk_class'].values

agg_acc = accuracy_score(y_true_all, y_pred_all)
agg_w_prec, agg_w_rec, agg_w_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='weighted', zero_division=0)
agg_m_prec, agg_m_rec, agg_m_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)

agg_results = {
    'Total_Test_Bearings': len(FULL_TEST_BEARINGS),
    'Total_Test_Snapshots': int(df_all_preds['bearing'].nunique() * (WINDOW_LEN - 1) + len(df_all_preds)),
    'Total_Test_Sequences': len(df_all_preds),
    'RUL_MAE_seconds': float(agg_mae_s),
    'RUL_MAE_minutes': float(agg_mae_m),
    'RUL_RMSE_seconds': float(agg_rmse_s),
    'RUL_RMSE_minutes': float(agg_rmse_m),
    'RUL_MAE_normalized': float(agg_norm_mae),
    'Trajectory_Averaged_PHM_Score': float(agg_t_score),
    'Risk_Accuracy': float(agg_acc),
    'Risk_Weighted_Precision': float(agg_w_prec),
    'Risk_Weighted_Recall': float(agg_w_rec),
    'Risk_Weighted_F1': float(agg_w_f1),
    'Risk_Macro_Precision': float(agg_m_prec),
    'Risk_Macro_Recall': float(agg_m_rec),
    'Risk_Macro_F1': float(agg_m_f1)
}

output_agg_json = os.path.join(RESULTS_DIR, 'evaluation_metrics_all_11_bearings.json')
with open(output_agg_json, 'w') as f:
    json.dump(agg_results, f, indent=2)
print(f"Aggregate metrics saved to: {output_agg_json}")

df_per_bearing = pd.DataFrame(per_bearing_records)
output_per_bearing_csv = os.path.join(RESULTS_DIR, 'per_bearing_metrics_all_11_bearings.csv')
df_per_bearing.to_csv(output_per_bearing_csv, index=False)
print(f"Per-bearing metrics saved to: {output_per_bearing_csv}")

print("\n" + "=" * 100)
print("PER-BEARING EVALUATION TABLE (ALL 11 BEARINGS IN FULL_TEST_SET)")
print("=" * 100)
try:
    print(df_per_bearing.to_markdown(index=False))
except Exception:
    print(df_per_bearing.to_string(index=False))

print("\n" + "=" * 65)
print("AGGREGATE BENCHMARK SUMMARY (ALL 11 TEST BEARINGS - ZERO LEAKAGE)")
print("=" * 65)
for k, v in agg_results.items():
    if isinstance(v, float):
        print(f"  {k:30s} : {v:.4f}")
    else:
        print(f"  {k:30s} : {v}")
print("=" * 65)
