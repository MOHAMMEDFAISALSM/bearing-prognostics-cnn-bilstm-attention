"""Per-snapshot feature extraction. Each snapshot is processed ALONE: features at time t can only depend on the
2560 samples recorded at time t (causality is verified by tests in the notebook)."""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from . import config as C

BANDS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12800)]   # Hz, spectral band-energy fractions
BASE = ["rms", "peak", "p2p", "kurtosis", "skewness", "crest_factor", "shape_factor", "impulse_factor", "margin_factor",
        "spec_energy", "spec_centroid", "spec_spread", "spec_peak_freq", "band_0_1k", "band_1_3k", "band_3_6k", "band_6_12k"]
FEATURES = [f"{ch}_{b}" for ch in ("horiz", "vert") for b in BASE]
_LOG_BASE = ("rms", "peak", "p2p", "spec_energy", "crest_factor", "shape_factor", "impulse_factor", "margin_factor")   # positive, heavy-tailed
_FRAC_BASE = ("band_0_1k", "band_1_3k", "band_3_6k", "band_6_12k")                                                        # fractions in [0,1]
_SLOG_BASE = ("kurtosis", "skewness")                                                                                     # can be negative
LOG_FEATURES = [f for f in FEATURES if f.split("_", 1)[1] in _LOG_BASE]
FRAC_FEATURES = [f for f in FEATURES if f.split("_", 1)[1] in _FRAC_BASE]
SLOG_FEATURES = [f for f in FEATURES if f.split("_", 1)[1] in _SLOG_BASE]
EPS = 1e-12

FEATURE_DOC = {  # name -> (formula, physical meaning / why it may indicate degradation)
    "rms": ("sqrt(mean(x^2))", "Overall vibration energy; grows as defects enlarge and friction/impacts increase."),
    "peak": ("max|x|", "Largest instantaneous shock; sensitive to single strong impacts (also to noise spikes)."),
    "p2p": ("max(x)-min(x)", "Total excursion; similar to peak but based on the signed range."),
    "kurtosis": ("mean((x-mu)^4)/sigma^4 - 3", "Impulsiveness. About 0 for smooth Gaussian vibration, rises when sharp impacts appear (spalls, cracks)."),
    "skewness": ("mean((x-mu)^3)/sigma^3", "Asymmetry of the amplitude distribution; one-sided impacts change it."),
    "crest_factor": ("peak/rms", "Peak-to-average ratio; rises early with isolated impacts, falls when damage becomes continuous."),
    "shape_factor": ("rms/mean|x|", "Waveform shape; changes when the amplitude distribution departs from Gaussian."),
    "impulse_factor": ("peak/mean|x|", "Sharpness of impulses relative to the mean level."),
    "margin_factor": ("peak/(mean(sqrt|x|))^2", "Peak relative to a low-order mean; emphasises rare large peaks (classical incipient-fault indicator)."),
    "spec_energy": ("sum(P_k), P=|FFT(x-mu)|^2/N^2", "Total spectral power (Parseval: equals variance); tracks overall energy growth."),
    "spec_centroid": ("sum(f_k P_k)/sum(P_k)", "Centre of gravity of the power spectrum; moves when high-frequency resonances are excited."),
    "spec_spread": ("sqrt(sum((f_k-fc)^2 P_k)/sum(P_k))", "Bandwidth of the spectrum; broadens with damage."),
    "spec_peak_freq": ("argmax_f P(f)", "Dominant frequency (shaft harmonics vs structural resonance)."),
    "band_0_1k": ("sum(P_k, 0-1 kHz)/sum(P_k)", "Fraction of power at low frequency (shaft rotation harmonics; shaft speed is 25-30 Hz)."),
    "band_1_3k": ("sum(P_k, 1-3 kHz)/sum(P_k)", "Mid band."),
    "band_3_6k": ("sum(P_k, 3-6 kHz)/sum(P_k)", "Mid-high band."),
    "band_6_12k": ("sum(P_k, 6-12.8 kHz)/sum(P_k)", "High band, where impact-excited resonances appear."),
}
_F_AXIS = np.fft.rfftfreq(C.N_SAMPLES, d=1.0 / C.FS)


def read_snapshot(path):
    """Robust reader. Returns (array n x 6, delimiter). The delimiter is detected from the first line."""
    with open(path, "r") as fh:
        first = fh.readline()
    sep = ";" if first.count(";") >= 5 else ","
    arr = pd.read_csv(path, sep=sep, header=None, dtype=np.float64).values
    return arr, sep


def channel_features(x, prefix):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(); c = x - mu
    rms = np.sqrt(np.mean(x ** 2)); peak = np.max(np.abs(x)); mean_abs = np.mean(np.abs(x)) + EPS
    var = np.mean(c ** 2) + EPS
    P = (np.abs(np.fft.rfft(c)) / len(x)) ** 2
    Ptot = P.sum() + EPS
    fc = np.sum(_F_AXIS * P) / Ptot
    out = {
        "rms": rms, "peak": peak, "p2p": np.ptp(x),
        "kurtosis": np.mean(c ** 4) / var ** 2 - 3.0, "skewness": np.mean(c ** 3) / var ** 1.5,
        "crest_factor": peak / (rms + EPS), "shape_factor": rms / mean_abs, "impulse_factor": peak / mean_abs,
        "margin_factor": peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + EPS),
        "spec_energy": Ptot, "spec_centroid": fc, "spec_spread": np.sqrt(np.sum((_F_AXIS - fc) ** 2 * P) / Ptot),
        "spec_peak_freq": _F_AXIS[np.argmax(P)],
    }
    for name, (lo, hi) in zip(("band_0_1k", "band_1_3k", "band_3_6k", "band_6_12k"), BANDS):
        out[name] = P[(_F_AXIS >= lo) & (_F_AXIS < hi)].sum() / Ptot
    return {f"{prefix}_{k}": float(v) for k, v in out.items()}


def snapshot_record(path):
    """Audit fields + features for ONE snapshot file."""
    arr, sep = read_snapshot(path)
    n_rows, n_cols = arr.shape
    h, v = arr[:, 4], arr[:, 5]
    t0 = arr[0, 0] * 3600 + arr[0, 1] * 60 + arr[0, 2] + arr[0, 3] * 1e-6
    rec = {"n_rows": n_rows, "n_cols": n_cols, "sep": sep, "n_nan": int(np.isnan(arr).sum()), "n_inf": int(np.isinf(arr).sum()),
           "t0_clock_s": t0, "max_abs_h": float(np.nanmax(np.abs(h))), "max_abs_v": float(np.nanmax(np.abs(v)))}
    if np.isnan(h).any() or np.isnan(v).any():          # never silently impute: flag it (n_nan) and fill for feature computation only
        h = np.nan_to_num(h); v = np.nan_to_num(v)
    rec.update(channel_features(h, "horiz")); rec.update(channel_features(v, "vert"))
    return rec


def extract_bearing(bearing, workers=8):
    folder = C.DATASET / C.subset_of(bearing) / bearing
    files = sorted(f for f in folder.iterdir() if f.name.startswith("acc_") and f.name.endswith(".csv"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(snapshot_record, files))
    df = pd.DataFrame(recs)
    df.insert(0, "snapshot_idx", np.arange(len(df)))
    df.insert(0, "bearing", bearing)
    df["file"] = [f.name for f in files]
    return df


def extract_all(cache=None, force=False):
    """Extract features + audit fields for all 17 bearings (cached to CSV, regenerated if the cache is missing)."""
    if cache is not None and cache.exists() and not force:
        return pd.read_csv(cache)
    df = pd.concat([extract_bearing(b) for b in C.LEARNING + C.TEST], ignore_index=True)
    if cache is not None:
        df.to_csv(cache, index=False)
    return df


def transform(df):
    """Fixed (non-fitted) transform chosen from the DISTRIBUTION diagnostics of Section 10 (heavy tails), not from any model result:
    log10 for positive amplitude/ratio features, log10(x+1e-6) for band fractions, signed log1p for kurtosis/skewness.
    No statistics are learned, so it cannot leak."""
    out = df[FEATURES].copy()
    out[LOG_FEATURES] = np.log10(out[LOG_FEATURES].clip(lower=1e-9))                    # positive amplitude / ratio features
    out[FRAC_FEATURES] = np.log10(out[FRAC_FEATURES].clip(lower=0) + 1e-6)              # band fractions: very small values dominate the tail
    out[SLOG_FEATURES] = np.sign(out[SLOG_FEATURES]) * np.log1p(np.abs(out[SLOG_FEATURES]))   # signed log for kurtosis / skewness
    return out
