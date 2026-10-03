from nbdefs import md, code

# =====================================================================================================================
md(r"""
# Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring
### A leakage-controlled, fully audited study on the PRONOSTIA / IEEE PHM 2012 bearing dataset

**Abstract.** ABSTRACT_PLACEHOLDER
""")

md(r"""
## 1. Research Questions

* **RQ1 – Data.** What does the PRONOSTIA / IEEE PHM 2012 dataset really contain, and is every fact we rely on (lifetimes, end of life, file formats) consistent?
* **RQ2 – Formulation.** Which prognostic formulation (raw RUL, capped RUL, normalised RUL, stage classification, degradation detection, multi-task) is scientifically defensible for this data?
* **RQ3 – Generalisation.** How well does vibration data predict remaining life and health stage of a bearing that the model has **never seen** (bearing-level validation, no leakage)?
* **RQ4 – Architecture.** Does each component of the proposed CNN–BiLSTM–Attention dual-head network (CNN, BiLSTM, attention, second head) measurably contribute, compared with simple baselines?
* **RQ5 – Explainability.** What can the attention weights and feature-importance analyses truthfully tell us?
* **RQ6 – Limits.** For which bearings is early warning physically possible, and for which is failure too abrupt?

**Hypotheses (stated before looking at any test result, and tested — not assumed).**
H1: sequence models detect impending failure better than tabular baselines. H2: the attention layer and the dual head improve accuracy. H3: a warning several tens of minutes before failure is possible for every bearing.
""")

# =====================================================================================================================
md(r"""
## 2. Dataset Research & Literature Review

*This section was written from primary sources that were actually opened while preparing the project: the official challenge document shipped with the dataset [2], the platform paper [1], and the peer-reviewed / arXiv papers listed in Section 28. Statements marked **[CITATION NEEDED]** could not be verified from a source I could open and should be confirmed before publication.*

### 2.1 Facts about the dataset

| # | Topic | Fact | Source |
|---|---|---|---|
| 1 | Origin | Collected on the **PRONOSTIA** platform of the AS2M department, **FEMTO-ST Institute**, Besançon, France | [1], [2] |
| 2 | Purpose | Data for the **IEEE PHM 2012 Prognostic Challenge**: estimate the remaining useful life (RUL) of bearings from monitoring data; "accelerated" degradation so that failure happens within hours | [1], [2] |
| 3 | Number of bearings | **17** run-to-failure experiments | [2] Table 1, [3] |
| 4 | Learning set | **6** bearings: Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2 (complete run-to-failure) | [2] Table 1 |
| 5 | Test set | **11** bearings: 1_3–1_7, 2_3–2_7, 3_3. In the *challenge* these were **truncated** and participants had to predict RUL | [2] §1.2, Table 1 |
| 6 | Operating conditions | C1: 1800 rpm, 4000 N; C2: 1650 rpm, 4200 N; C3: 1500 rpm, 5000 N (radial load applied by a pneumatic jack; the dynamic load rating of the bearing is 4000 N) | [2] §3.2, App. A.1 |
| 7 | Sampling frequency | Vibration: **25.6 kHz** (temperature: 10 Hz) | [2] §4.1 |
| 8 | Samples per snapshot | **2560** samples (= 1/10 s) | [2] §4.1 |
| 9 | Snapshot interval | one snapshot every **10 s** (temperature: 600 samples per minute) | [2] §4.1 |
| 10 | Channels | **Two** accelerometers (DYTRAN 3035B, 50 g range) mounted radially, **90° apart**, on the outer race: **horizontal** and **vertical** | [1] §III-C, [2] §2.4, App. A.2 |
| 11 | End of life | Tests were stopped when the vibration amplitude exceeded **20 g**; "RUL was defined as time to accelerometer exceeding 20 g" | [2] §3.1, Note 1 |
| 12 | File structure | `acc_xxxxx.csv` (vibration) and `temp_xxxxx.csv` (temperature); 6 columns: hour, minute, second, µ-second, horizontal acceleration, vertical acceleration | [2] §4.2, Table 2 |
| 13 | Repository | The GitHub copy used here contains `Learning_set/`, `Test_set/` (truncated, as in the challenge) and **`Full_Test_Set/`** (test bearings run to failure). `Full_Test_Set` is **not described** in the challenge document [2] or the README [3]; its consistency with [2] is checked in Section 5 | [3] + audit |
| 14 | Known characteristics | Very different durations (**1 h to 7 h**), a **small learning set**, and **very different degradation behaviours**; some degradations are **sudden** and non-monotonic; classical fault-frequency models (L10, BPFI, BPFO) **do not match** the observations; noise level is not controlled | [1] §IV-C–E, [2] §1.2, Notes 2–3 |
| 15 | Reported difficulties | Learning from a *small number of run-to-failure bearings* is the central difficulty | [2], [4] |

### 2.2 How researchers formulate the problem

* **Official challenge task.** Predict the RUL of each of the 11 test bearings at its truncation point; the error is a *percentage error* and the score is **asymmetric** (late predictions are punished harder than early ones) [2] §5.1. We reproduce this protocol exactly in Section 22.
* **Prognostic pipeline.** A widely cited review divides machinery prognostics into data acquisition, **health-indicator (HI) construction**, **health-stage division**, and RUL prediction [6].
* **Two-stage / capped targets.** In other prognostic benchmarks a *piece-wise linear* RUL target (constant while healthy, then linear decrease to 0) is common, because a healthy system gives no evidence about how much life remains [8] (turbofan data; the same reasoning applies to bearings). Using the *fraction of life* as target is also used with this dataset [4] (past-useful-life ratio) but it needs the total lifetime, which is unknown for a running machine.
* **Common metrics.** The official percentage-error score [2]; MAE / RMSE are widely used in follow-up work **[CITATION NEEDED: a specific PHM-2012 paper reporting MAE/RMSE]**.
* **Deep models on this data.** Deep architectures (CNN, LSTM, attention) have been benchmarked on this dataset, e.g. [5]. **[CITATION NEEDED]** for a peer-reviewed CNN–LSTM–attention PHM-2012 paper; I could not open one.

### 2.3 Leakage risks specific to this data and how experts view splitting

1. **Random splitting of sliding windows** puts almost identical overlapping windows into train and test. A 2026 preprint on multi-task fault-diagnosis/RUL models reports that naive window splitting can inflate accuracy from a genuine 20–60 % to 99.9 % [10] (preprint, not peer-reviewed; the effect is shown independently in Section 13).
2. **Using the test bearings' full runs** (`Full_Test_Set`) for anything except final scoring — scaling statistics, thresholds, target design, model selection — would leak information the challenge deliberately hid.
3. **Lifetime-normalised targets** (fraction of life) silently use the failure time of the evaluated bearing.
4. **The bearing is the independent unit.** The challenge itself separates whole bearings into learning and test sets [2]; that is the appropriate protocol. With only 6 learning bearings, *leave-one-bearing-out* validation is the natural way to use them.
""")

md(r"""
## 3. Literature-Based Methodology and Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| Unit of independence | the **bearing** | [2], [10]; windows of one bearing are highly dependent |
| Method-development data | the **6 learning bearings only** (leave-one-bearing-out) | test bearings must not influence any design choice |
| Final test | the **11 test bearings**, scored once after the design is frozen | official protocol [2] |
| End of life | official recording duration for learning bearings; *truncation point + official RUL* for test bearings | consistent with [2]; fixes an inconsistency found in the audit (Bearing1_4) |
| Input | last **16 snapshots (160 s)** of 34 vibration features | causal window; documented in Section 9 |
| Target | **capped RUL (minutes, cap 120)** + **3-class stage** (Normal / Warning / Critical) derived **only from the failure time** | Section 11 explains why this is the defensible formulation and why no vibration-based "onset" label is used for training |
| Baselines | Constant, Ridge, Random Forest, Gradient Boosting, LSTM, CNN, CNN+LSTM, CNN+BiLSTM | Section 14 |
| Selection | decided on **LOBO validation only**, written to disk **before** the test bearings are scored | Section 17 / 22 |
| Reporting | negative results are reported | Section 25 |

**Remaining hindsight / contamination that must be admitted.** (i) Before this notebook existed I inspected health-indicator curves of *all* 17 bearings, including test bearings, in earlier experiments (v1/v2). The target used here does **not** need any vibration-based label, which removes that dependence from training, but the *choice* of a capped RUL formulation was motivated by earlier poor results. (ii) Descriptive plots of test bearings are shown only *after* the design freeze (Section 22) and are never used for a decision.
""")

# =====================================================================================================================
md(r"""
## 4. Setup, Reproducibility and Dataset Loading

**WHAT** – import the project modules (`src/`), fix all random seeds, enable deterministic TensorFlow operations and print library versions. Then read **every raw vibration file once** (24,889 files) and compute both audit fields and features.
**WHY** – a result is only meaningful if it can be reproduced and if we know exactly which data produced it.
**EXPECT** – about 2 minutes (the result is written to `artifacts/features_raw.csv` as an output; the notebook never reads a stale cache).
**HOW IT HELPS** – all later sections work from this single, audited table.
""")

code(r'''
import os, sys, json, time, random, platform, warnings, filecmp, hashlib
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(Path.cwd()))                   # the notebook lives in final_project/
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sklearn, scipy
from IPython.display import display, Markdown
import tensorflow as tf
from src import config as C, features as F, data as D, models as M, metrics as Mx, experiment as E, analysis as A, leakage as L, report as R

SEED = 42
random.seed(SEED); np.random.seed(SEED); M.set_determinism(SEED)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 90, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9})
COND_COLOR = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
C.ARTIFACTS.mkdir(exist_ok=True); C.FIGURES.mkdir(exist_ok=True)
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                scipy=scipy.__version__, tensorflow=tf.__version__, platform=platform.platform())
print(json.dumps(VERSIONS, indent=1))
print("dataset folder:", C.DATASET, "| exists:", C.DATASET.exists())
t0 = time.time()
raw = F.extract_all(C.ARTIFACTS / "features_raw.csv", force=True)      # ALWAYS re-reads all raw files (no dependence on any cache); the CSV is only an output
print(f"raw feature table: {raw.shape[0]:,} snapshots x {raw.shape[1]} columns  ({time.time()-t0:.0f}s)")
''')

# =====================================================================================================================
md(r"""
## 5. Complete Dataset Audit

**WHAT** – inspect the *whole* dataset programmatically: bearing IDs, files per bearing, lifetimes, operating conditions, delimiters, rows/columns per file, missing or infinite values, clock stamps, where the 20 g threshold is actually crossed, and whether the test data is consistent with the official challenge document.
**WHY** – the earlier notebook silently assumed these facts. If any is wrong (for example the end of life of one bearing) every RUL label built on it is wrong.
**EXPECT** – 17 bearings, 2560 × 6 files, and lifetimes that match the official table. Anything else is an anomaly we must document and handle.
**HOW IT HELPS** – it decides how the end of life (and therefore every target) is defined.
""")

code(r'''
rows = []
for b in C.LEARNING + C.TEST:
    g = raw[raw.bearing == b].sort_values("snapshot_idx"); n = len(g)
    d = np.diff(g.t0_clock_s.values); d = np.where(d < -3600, d + 86400, d)          # clock stamps of the first sample of each file
    mx = np.maximum(g.max_abs_h.values, g.max_abs_v.values)
    trunc = D.truncated_length(b); rpm, load = C.CONDITION[C.condition_of(b)]
    folder = C.DATASET / C.subset_of(b) / b
    rows.append({"bearing": b, "set": "learning" if b in C.LEARNING else "test", "condition": C.condition_of(b), "rpm": rpm, "load_N": load,
                 "acc_files": n, "temp_files": len(list(folder.glob("temp_*.csv"))), "lifetime_min": round((n - 1) * 10 / 60, 1),
                 "rows/file": f"{g.n_rows.min()}-{g.n_rows.max()}", "cols": int(g.n_cols.max()), "delimiter": "".join(sorted(g.sep.unique())),
                 "NaN": int(g.n_nan.sum()), "Inf": int(g.n_inf.sum()), "peak_g": round(float(mx.max()), 1),
                 "snapshots>20g": int((mx > C.FAILURE_G).sum()), "first>20g_at_%life": round(100 * np.argmax(mx > C.FAILURE_G) / n, 1) if (mx > C.FAILURE_G).any() else np.nan,
                 "irregular_clock_steps": int((np.abs(d - 10) > 1).sum()),
                 "trunc_files": trunc, "official_RUL_s": C.OFFICIAL_RUL_S.get(b, np.nan), "implied_RUL_s": (n - trunc) * 10 if b in C.TEST else np.nan})
audit = pd.DataFrame(rows).set_index("bearing")
audit["RUL_consistent"] = np.where(audit.set == "test", audit.official_RUL_s == audit.implied_RUL_s, np.nan)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
display(audit)

# ---- hard assertions on the parts that must hold for the pipeline ----
assert len(audit) == 17 and (audit.set == "learning").sum() == 6 and (audit.set == "test").sum() == 11
assert (raw.n_rows == C.N_SAMPLES).all() and (raw.n_cols == 6).all(), "unexpected file shape"
assert raw.n_nan.sum() == 0 and raw.n_inf.sum() == 0, "missing / infinite raw values"
print(f"\nTotal snapshots: {audit.acc_files.sum():,} | raw vibration samples: {audit.acc_files.sum()*C.N_SAMPLES*2:,} (2 channels) | "
      f"delimiters: {raw.groupby('sep').size().to_dict()}")

# ---- the truncated Test_set must be the first part of Full_Test_Set (byte-identical files) ----
same_bytes, same_num, owners = [], [], []
for b in C.TEST:
    k = int(audit.loc[b, "trunc_files"])
    for i in (1, k // 2, k):
        f1 = C.DATASET / "Test_set" / b / f"acc_{i:05d}.csv"; f2 = C.DATASET / "Full_Test_Set" / b / f"acc_{i:05d}.csv"
        same_bytes.append(filecmp.cmp(f1, f2, shallow=False)); owners.append(b)
        a1, _ = F.read_snapshot(f1); a2, _ = F.read_snapshot(f2); same_num.append(a1.shape == a2.shape and bool(np.array_equal(a1, a2)))
print(f"Test_set vs Full_Test_Set (3 files per test bearing): byte-identical {sum(same_bytes)}/{len(same_bytes)}, numerically identical {sum(same_num)}/{len(same_num)}")
print("bearings whose files differ only in bytes:", sorted({b for b, sb, sn in zip(owners, same_bytes, same_num) if not sb and sn}), "(delimiter: Test_set uses ',', Full_Test_Set uses ';')")
assert all(same_num), "Full_Test_Set is not numerically identical to the truncated Test_set"

# ---- anomalies, derived from the table (not typed by hand) ----
anom = []
for b, r in audit[audit.set == "test"].iterrows():
    if not r.RUL_consistent:
        anom.append(f"**{b}**: Full_Test_Set has {int(r.acc_files - r.trunc_files)} snapshots after the truncation point (= {r.implied_RUL_s:.0f} s) but the official actual RUL is {r.official_RUL_s:.0f} s -> the extra {(r.implied_RUL_s - r.official_RUL_s):.0f} s of data lie **after the official end of life**")
for b, r in audit.iterrows():
    if r.irregular_clock_steps:
        anom.append(f"**{b}**: {r.irregular_clock_steps} irregular clock steps (logging glitch in the time stamps only; file order and the 10 s interval from [2] are used instead)")
    if r["snapshots>20g"] == 0:
        anom.append(f"**{b}**: no recorded 0.1 s snapshot ever exceeds 20 g (peak {r.peak_g} g) -> the 20 g end-of-life criterion is **not observable in the snapshots** for this bearing")
    elif r["first>20g_at_%life"] < 90:
        anom.append(f"**{b}**: first snapshot above 20 g already at {r['first>20g_at_%life']}% of the recorded life -> a 20 g threshold applied to the snapshots would give a very different end of life than the official duration")
R.show("### Anomalies found by the audit\n" + "\n".join(f"* {a}" for a in anom))
audit.to_csv(C.ARTIFACTS / "dataset_audit_table.csv")
''')

code(r'''
# Figure 1 - dataset overview: snapshots, lifetimes, operating conditions
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
order = C.LEARNING + C.TEST
cols = [COND_COLOR[C.condition_of(b)] for b in order]
bars = ax[0].bar(range(17), audit.loc[order, "acc_files"], color=cols, edgecolor="black")
for p_, b_ in zip(bars, order):
    if b_ in C.TEST: p_.set_hatch("//")
ax[0].set_xticks(range(17)); ax[0].set_xticklabels([b.replace("Bearing", "B") for b in order], rotation=60)
ax[0].set_ylabel("Number of vibration snapshots (10 s apart)"); ax[0].set_title("(a) Snapshots per bearing (hatched = test)")
ax[0].axvline(5.5, color="k", ls=":")
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(fc=COND_COLOR[c], ec="k", label=f"Condition {c}: {C.CONDITION[c][0]} rpm, {C.CONDITION[c][1]} N") for c in (1, 2, 3)], fontsize=8, loc="upper right")
srt = audit.sort_values("lifetime_min")
bars_b = ax[1].barh(srt.index, srt.lifetime_min, color=[COND_COLOR[c] for c in srt.condition], edgecolor="black")
for p_, b_ in zip(bars_b, srt.index):
    if b_ in C.TEST: p_.set_hatch("//")
ax[1].axvline(audit.lifetime_min.median(), color="k", ls="--", label=f"median {audit.lifetime_min.median():.0f} min")
ax[1].set_xlabel("Lifetime to end of life (minutes)"); ax[1].set_title("(b) Lifetime of every bearing (colour = condition, hatched = test)"); ax[1].legend(loc="lower right")
cnt = audit.groupby(["condition", "set"]).size().unstack(fill_value=0)
cnt.plot(kind="bar", stacked=True, ax=ax[2], color=["#cccccc", "#777777"], edgecolor="black")
ax[2].set_xticklabels([f"C{c}\n{C.CONDITION[c][0]} rpm, {C.CONDITION[c][1]} N" for c in cnt.index], rotation=0)
ax[2].set_ylabel("Number of bearings"); ax[2].set_title("(c) Learning (light grey) vs test (dark grey) bearings per condition"); ax[2].set_xlabel("")
for i, c in enumerate(cnt.index):
    ax[2].text(i, cnt.loc[c].sum() + 0.1, f"mean life {audit[audit.condition==c].lifetime_min.mean():.0f} min", ha="center", fontsize=9)
plt.tight_layout(); R.savefig(fig, "fig01_dataset_overview.png"); plt.show()
lt = audit.groupby("condition").lifetime_min.agg(["min", "median", "max"]).round(1)
R.show(f"**Figure 1.** (a) file counts, (b) lifetimes and (c) conditions. Lifetimes range from **{audit.lifetime_min.min():.0f} to {audit.lifetime_min.max():.0f} minutes** "
       f"(factor {audit.lifetime_min.max()/audit.lifetime_min.min():.0f}); per condition (min/median/max): " + "; ".join(f"C{c}: {r['min']}/{r['median']}/{r['max']}" for c, r in lt.iterrows()) +
       f". Condition 3 has only {int((audit.condition==3).sum())} bearings, so its learning history is a single-digit sample.")
''')

md(r"""
### 5.1 What did we find?
The dataset is clean in format: every one of the 24,889 files is a 2560 × 6 numeric table, no value is missing, and only Bearing1_4 uses `;` instead of `,`. The audit, however, exposed three facts that shape the whole project:

1. **Lifetimes differ by a factor of about 12** and there are only 6 learning bearings — the central difficulty named by the organisers [2].
2. **The 20 g criterion is not visible in every snapshot** because a snapshot covers only 0.1 s out of every 10 s. Some bearings never show > 20 g, others show it long before the end. Therefore the **end of life must come from the official recording duration and the official RUL table [2], not from thresholding the snapshots**.
3. **Bearing1_4** in `Full_Test_Set` contains data *after* its official end of life. We truncate it at the official end of life (Section 11), which is the only choice consistent with [2]. The truncated `Test_set` and `Full_Test_Set` are numerically identical where they overlap (Bearing1_4 differs only in the delimiter), so the official truncation points can be used in Section 22.

**Does this support our hypotheses?** It supports the need for bearing-level validation (H-related design), and it already warns that H3 (long warning for every bearing) is doubtful, because lifetimes and degradation dynamics vary so much.
""")

# =====================================================================================================================
md(r"""
## 6. Sample Raw Data

**WHAT** – open one raw file exactly as delivered, label its six columns using the official description [2, Table 2], and plot it.
**WHY** – to be sure we interpret the measurements correctly (units, channels, time base) before extracting features.
**EXPECT** – 2560 rows covering 0.1 s; two acceleration columns in units of *g* around zero.
""")

code(r'''
sample_path = C.DATASET / "Learning_set" / "Bearing1_1" / "acc_00010.csv"
arr, sep = F.read_snapshot(sample_path)
cols6 = ["hour", "minute", "second", "microsecond", "horizontal_acc_g", "vertical_acc_g"]
sample = pd.DataFrame(arr, columns=cols6)
print(f"file: {sample_path.name} | rows: {len(sample)} | delimiter: '{sep}' | duration: {len(sample)/C.FS*1000:.0f} ms at {C.FS/1000:.1f} kHz")
display(sample.head(5))
R.show("""| Column | Meaning (official description [2] Table 2) |
|---|---|
| 1 hour, 2 minute, 3 second | wall-clock time stamp of the sample |
| 4 microsecond | µs part of the time stamp (so consecutive rows are 1/25600 s = 39.06 µs apart) |
| 5 horizontal acceleration | accelerometer on the horizontal axis, in *g* |
| 6 vertical acceleration | accelerometer on the vertical axis, in *g* |
""")
assert np.allclose(np.diff(arr[:, 3]).mean(), 1e6 / C.FS, rtol=0.02), "time base of the file is not 25.6 kHz"
t_ms = np.arange(len(sample)) / C.FS * 1000
fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].plot(t_ms, sample.horizontal_acc_g, lw=0.7, color="#1f77b4"); ax[0].set_ylabel("Horizontal acceleration (g)")
ax[1].plot(t_ms, sample.vertical_acc_g, lw=0.7, color="#2ca02c"); ax[1].set_ylabel("Vertical acceleration (g)"); ax[1].set_xlabel("Time within the 0.1 s snapshot (ms)")
ax[0].set_title("Bearing1_1, snapshot 10 (healthy phase): one raw file")
plt.tight_layout(); R.savefig(fig, "fig02_sample_raw_signal.png"); plt.show()
R.show(f"**Figure 2.** One raw snapshot. The time base check passed (39 µs between samples). Horizontal peak {sample.horizontal_acc_g.abs().max():.2f} g, vertical peak {sample.vertical_acc_g.abs().max():.2f} g: a healthy bearing is a low-level, noise-like signal.")
''')

code(r'''
# Figure 3 - healthy vs degraded snapshot of the same bearing (time signal + spectrum)
b = "Bearing1_1"; n_b = int(audit.loc[b, "acc_files"])
early, late = 10, n_b - 1
fig, ax = plt.subplots(2, 2, figsize=(15, 7))
for j, (idx, lab) in enumerate([(early, "healthy (snapshot 10)"), (late, f"end of life (snapshot {late})")]):
    a, _ = F.read_snapshot(C.DATASET / "Learning_set" / b / f"acc_{idx+1:05d}.csv")
    ax[0, j].plot(t_ms, a[:, 4], lw=0.6, color="#d62728" if j else "#1f77b4"); ax[0, j].set_title(f"{b} horizontal - {lab}")
    ax[0, j].set_ylabel("Acceleration (g)"); ax[0, j].set_xlabel("ms")
    if j: ax[0, j].axhline(20, color="k", ls="--", lw=0.8, label="20 g"); ax[0, j].axhline(-20, color="k", ls="--", lw=0.8); ax[0, j].legend()
    x = a[:, 4] - a[:, 4].mean(); P = np.abs(np.fft.rfft(x)) / len(x)
    ax[1, j].plot(np.fft.rfftfreq(len(x), 1 / C.FS), P, lw=0.7, color="#d62728" if j else "#1f77b4")
    ax[1, j].set_xlabel("Frequency (Hz)"); ax[1, j].set_ylabel("Amplitude spectrum (g)"); ax[1, j].set_title("Spectrum (0 - 12.8 kHz)")
plt.tight_layout(); R.savefig(fig, "fig03_healthy_vs_degraded.png"); plt.show()
R.show("**Figure 3.** The same bearing early and at the end. The end-of-life snapshot shows impacts far beyond the healthy level and energy spread over a broad frequency range. This motivates amplitude features (RMS, peak, kurtosis) and spectral-shape features (centroid, spread, band energies).")
''')

# =====================================================================================================================
md(r"""
## 7. Exploratory Data Analysis (learning bearings only)

**WHAT** – look at a simple **health indicator** (the log of the combined RMS of both accelerometers) over the life of the 6 learning bearings.
**WHY** – to see *how* real bearings degrade: slowly, suddenly, or with temporary bumps. This decides what kind of prediction is even possible.
**EXPECT** – some bearings show a long gradual rise; others stay flat until the last minutes.
**RULE.** Only *learning* bearings are shown here. Test bearings are shown for description only after the design is frozen (Section 22).
""")

code(r'''
# Data table used from here on: keeps snapshots up to the OFFICIAL end of life and attaches RUL targets (from failure time only)
table = D.build_table(raw)
feats = F.transform(table)                 # fixed transform, no fitted statistics
assert set(table.bearing.unique()) == set(C.LEARNING + C.TEST)
print("snapshots kept:", len(table), "of", len(raw), "| Bearing1_4 kept", int((table.bearing == 'Bearing1_4').sum()), "of", int((raw.bearing == 'Bearing1_4').sum()), "(cut at the official end of life)")

lrn = table[table.bearing.isin(C.LEARNING)].copy()
lrn["hi"] = np.nan
for b in C.LEARNING:
    m = lrn.bearing == b; lrn.loc[m, "hi"] = A.combined_rms_hi(lrn[m])
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
for b in C.LEARNING:
    g = lrn[lrn.bearing == b]; c = COND_COLOR[C.condition_of(b)]
    ax[0].plot(g.time_min, g.hi, lw=1, color=c, alpha=0.9, label=b)
    ax[1].plot(g.time_min / g.time_min.max(), g.hi, lw=1, color=c, alpha=0.9)
    ax[2].plot(np.maximum(g.rul_min, 0.05), g.hi, lw=1, color=c, alpha=0.9)
ax[0].set_xlabel("Operating time (min)"); ax[0].set_title("(a) health indicator vs time"); ax[0].legend(ncol=2)
ax[1].set_xlabel("Fraction of life elapsed"); ax[1].set_title("(b) vs normalised life (needs the lifetime: descriptive only)")
ax[2].set_xscale("log"); ax[2].invert_xaxis(); ax[2].set_xlabel("Remaining useful life (min, log scale, failure at right)"); ax[2].set_title("(c) aligned at failure")
for a_ in ax: a_.set_ylabel("log combined RMS (g)")
plt.tight_layout(); R.savefig(fig, "fig04_health_indicator_learning.png"); plt.show()
ph = A.phase_table(table, C.LEARNING); display(ph)
R.show("**Figure 4.** Health indicator (log of combined RMS) of the six learning bearings. Panel (c) aligns all bearings at failure on a log axis: the indicator becomes clearly elevated only in the last minutes for several bearings. The table lists the length of the final degradation phase (descriptive, hindsight-based, learning bearings only) for three sensitivity values of the threshold.")
''')

md(r"""
### 7.1 What did we find?
Learning bearings degrade in **different ways**: Bearing1_1 shows a long, slowly rising indicator; others (e.g. Bearing3_1, 3_2, 1_2) stay flat and rise only in the last few minutes; Bearing2_1 and 2_2 show level changes early in life that are not clearly damage. The table quantifies the *length of the final degradation phase* at three thresholds: for half of the learning bearings it is of the order of **4–10 minutes**.

**What this means.** A model cannot know that a bearing will fail in 60 minutes if the vibration is indistinguishable from a healthy bearing until 5 minutes before failure. This is a property of the data, not of the model, and it is why we later measure prediction quality **as a function of the horizon** (Section 11) and report abrupt bearings separately.
""")

# =====================================================================================================================
md(r"""
## 8. Health Indicator Analysis

**WHAT** – screen all candidate features on the **learning bearings only** using three descriptive measures: *trend* (Spearman correlation with operating time), *consistency* (how many learning bearings show a strong trend with the same sign), and *late shift* (robust z-score of the value in the last 60 minutes relative to the first 30 % of life).
**WHY** – the task asks us not to use features blindly. A useful health indicator should change with degradation, do so in the same direction for different bearings, and change *before* the very end.
**EXPECT** – amplitude features (RMS, peak) change strongly; some shape features change only for some bearings; band fractions may be inconsistent.
**HOW IT HELPS** – we keep the full, documented feature set (each feature is physically motivated; deep models can learn combinations) but use the screening to explain which features carry the degradation signal and to choose the indicator used for illustration and demo. The screening does **not** change the model inputs, so it cannot leak.
""")

code(r'''
screen = A.feature_screening(table, feats, C.LEARNING).sort_values("consistency", ascending=False)
screen["abs_trend"] = screen.trend_spearman.abs()
display(screen.sort_values(["consistency", "abs_trend"], ascending=False).round(3))
top_c = screen.sort_values("consistency", ascending=False).head(2)
R.show(f"**Reading the screening (learning bearings only).** Most consistent whole-life trend: " + ", ".join(f"`{i}` (consistency {r.consistency:.2f}, trend {r.trend_spearman:+.2f})" for i, r in top_c.iterrows()) +
       f". For the amplitude features: `horiz_rms` trend {screen.loc['horiz_rms','trend_spearman']:+.2f}, consistency {screen.loc['horiz_rms','consistency']:.2f}; `vert_rms` trend {screen.loc['vert_rms','trend_spearman']:+.2f}, consistency {screen.loc['vert_rms','consistency']:.2f}. "
       "The combined-RMS indicator of Section 7 is kept only as the standard *descriptive* indicator for illustration and the demo; it is not a model-input choice.")
top6 = ["horiz_rms", "horiz_kurtosis", "horiz_crest_factor", "horiz_spec_centroid", "horiz_band_6_12k", "horiz_spec_spread"]
fig, axes = plt.subplots(2, 3, figsize=(19, 8)); axes = axes.ravel()
for a_, f in zip(axes, top6):
    for b in C.LEARNING:
        m = (table.bearing == b).values
        a_.plot(np.maximum(table.loc[m, "rul_min"], 0.05), feats.loc[m, f].rolling(9, min_periods=1, center=True).median(), lw=0.9, color=COND_COLOR[C.condition_of(b)], alpha=0.85, label=b)
    a_.set_xscale("log"); a_.invert_xaxis(); a_.set_title(f); a_.set_xlabel("RUL (min, log)"); a_.set_ylabel("feature value (transformed)")
axes[0].legend(ncol=2, fontsize=7)
plt.tight_layout(); R.savefig(fig, "fig05_feature_vs_life_learning.png"); plt.show()
R.show("**Figure 5.** Selected features against remaining life on the six learning bearings (median-smoothed for readability only; models use unsmoothed values). Features move mostly within the last tens of minutes, and their absolute level differs between bearings.")
''')

md(r"""
### 8.1 What did we find?
""")

code(r'''
n_cons = int((screen.consistency > 0.5).sum()); n_shift = int((screen.late_shift_z.abs() > 1).sum())
late_top = screen[screen.index != "horiz_spec_peak_freq"].late_shift_z.abs().sort_values(ascending=False).head(5)
R.show(f"* **Over the whole life, hardly any feature is a consistent monotone indicator across bearings**: only {n_cons} of 34 features have consistency > 0.5, and the most consistent ones (3-6 kHz band fractions) *decrease* as damage grows. A typical reason is visible in Figures 4-5: vibration often *falls* during run-in early in life and rises only near the end, so a whole-life trend statistic is a poor criterion.\n\n"
       f"* **Features do move late**: {n_shift} of 34 features shift by more than one robust standard deviation in the last 60 minutes relative to the early baseline; strongest: " + ", ".join(f"`{i}` ({v:.1f})" for i, v in late_top.items()) + ".\n\n"
       "* **Consequence.** No single feature or fixed threshold is a robust health indicator across these bearings. This supports giving a learned model a *window of many features*, and it warns that hand-set thresholds (for example 'RMS above x') would not transfer between bearings. Our screening is descriptive: it does not choose the model inputs.")
''')

# =====================================================================================================================
md(r"""
## 9. Feature Engineering

**WHAT** – one fixed feature pipeline: from each 0.1 s snapshot and each of the two channels we compute 17 features (9 time-domain, 4 spectral-shape, 4 band-energy fractions) = **34 features**. Heavy-tailed features are compressed with a **fixed, non-fitted** transform chosen from the distribution diagnostics of Section 10 (not from any model result): log10 for positive amplitude/ratio features (RMS, peak, peak-to-peak, spectral energy, crest/shape/impulse/margin factors), log10(x + 1e-6) for the four band fractions, and signed log1p for kurtosis and skewness.
**WHY** – the network receives compact, physically meaningful numbers instead of 2560 raw samples, which is important with only 6 learning bearings. The standard deviation was intentionally **not** included: with a near-zero mean it duplicates RMS.
**CAUSALITY** – every feature is computed from **one snapshot only**. A feature at time *t* therefore cannot depend on the future; Section 12 verifies this with automated tests.
""")

code(r'''
doc = pd.DataFrame([{"feature": k, "formula": v[0], "physical meaning / why it may indicate degradation": v[1],
                     "transform": ("log10" if k in F._LOG_BASE else ("log10(x+1e-6)" if k in F._FRAC_BASE else ("signed log1p" if k in F._SLOG_BASE else "none")))} for k, v in F.FEATURE_DOC.items()]).set_index("feature")
pd.set_option("display.max_colwidth", 140); display(doc)
print(f"{len(F.BASE)} features per channel x 2 channels = {len(F.FEATURES)} features; transforms: log10 on {len(F.LOG_FEATURES)}, log10(x+1e-6) on {len(F.FRAC_FEATURES)}, signed log1p on {len(F.SLOG_FEATURES)}")
doc.to_csv(C.ARTIFACTS / "feature_documentation.csv")
json.dump(F.FEATURES, open(C.ARTIFACTS / "feature_names.json", "w"), indent=1)
''')

md(r"""
## 10. Feature Analysis: quality checks, distributions, correlation, scaling

**WHAT** – check for NaN/Inf, look at distributions and outliers, and compute the correlation between features. **Learning bearings only** are used for these statistics.
**WHY** – redundant features (for example RMS and spectral energy) waste capacity and can make importance analyses misleading; heavy tails would dominate a neural network unless scaled.
**EXPECT** – strong correlations inside the amplitude group; long tails before the log transform.
""")

code(r'''
assert not feats.isna().any().any() and not np.isinf(feats.values).any(), "NaN/Inf in features"
lm = table.bearing.isin(C.LEARNING).values
print("NaN/Inf in the 34 features (all 17 bearings):", int(feats.isna().sum().sum()), int(np.isinf(feats.values).sum()))
raw_l = table.loc[lm, F.FEATURES]; tr_l = feats.loc[lm]
out_frac = (np.abs(tr_l - tr_l.median()) / (1.4826 * (tr_l - tr_l.median()).abs().median() + 1e-9) > 5).mean().sort_values(ascending=False)
sk_tbl = pd.DataFrame({"skew_raw": raw_l.skew(), "skew_after_transform": tr_l.skew(), "frac_beyond_5MAD": out_frac}); display(sk_tbl.sort_values("skew_raw", key=abs, ascending=False).head(10).round(2))
print(f"max |skewness|: raw {sk_tbl.skew_raw.abs().max():.1f} -> after transform {sk_tbl.skew_after_transform.abs().max():.1f}")

fig, ax = plt.subplots(2, 4, figsize=(20, 7)); ax = ax.ravel()
for a_, f in zip(ax, ["horiz_rms", "horiz_kurtosis", "horiz_crest_factor", "horiz_spec_centroid", "horiz_spec_energy", "horiz_band_6_12k", "vert_rms", "vert_kurtosis"]):
    for b in C.LEARNING:
        sns.kdeplot(feats.loc[(table.bearing == b).values, f], ax=a_, lw=1, color=COND_COLOR[C.condition_of(b)], warn_singular=False)
    a_.set_title(f"{f}"); a_.set_xlabel("value (after transform)"); a_.set_ylabel("density")
plt.tight_layout(); R.savefig(fig, "fig06_feature_distributions.png"); plt.show()
R.show("**Figure 6.** Feature distributions per learning bearing (one curve each, colour = operating condition). The curves differ between bearings even in the healthy phase, confirming that a global scaling fitted on the training bearings is required (Section 13) and that per-bearing offsets are a real source of error.")

corr = feats.loc[lm].corr(method="spearman")
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, xticklabels=True, yticklabels=True, cbar_kws={"label": "Spearman correlation"})
ax.set_title("Feature correlation matrix (learning bearings, Spearman)"); plt.tight_layout(); R.savefig(fig, "fig07_feature_correlation.png"); plt.show()
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack(); hi_pairs = pairs[pairs.abs() > 0.95].sort_values(ascending=False)
R.show(f"**Figure 7.** Spearman correlation of the 34 features. **{len(hi_pairs)} feature pairs have |rho| > 0.95** (for example: " + ", ".join(f"`{a}`~`{b}`" for (a, b) in hi_pairs.index[:4]) +
       "). Amplitude features form one highly redundant block; shape and band features are less correlated. Redundant inputs are kept (they are cheap and each is documented), and the importance analysis of Section 20 is therefore reported per feature *and* interpreted with this redundancy in mind.")
''')

md(r"""
### 10.1 What did we find?
No NaN or infinite values exist. The fixed log / signed-log transforms remove most of the heavy tails (see the skewness table: raw skewness up to about 16 falls to about 6 at most). Correlation analysis shows that amplitude features are largely redundant, and per-bearing distributions differ noticeably. **Consequence:** a `StandardScaler` **fitted only on training bearings** is used inside every experiment, and importance results must not be over-interpreted for individual members of a correlated block.
""")

# =====================================================================================================================
md(r"""
## 11. Target Definition — choosing the formulation

### 11.1 Candidate formulations

| | Formulation | Advantage | Limitation for this dataset |
|---|---|---|---|
| A | **Direct RUL (minutes)** | the official challenge task [2] | a healthy-looking bearing may have 400 or 20 minutes left: uncapped targets contain unpredictable variance |
| A′ | **Capped RUL (minutes)** | removes the unpredictable early part; standard in prognostics [8] | needs a cap (design choice → sensitivity analysis) |
| B | **Normalised RUL (fraction of life)** | comparable across bearings | needs the total lifetime = future information; **not deployable** |
| C | **Stage classification** (Normal / Warning / Critical by remaining time) | operationally meaningful, robust | coarse; boundaries are a design choice |
| D | **Degradation vs healthy detection** | physically motivated | the label needs a vibration-based *onset*, which is found with **hindsight** and depends on a threshold; not usable as a real-time claim |
| E | **Multi-task (A′ + C)** | one model gives a number and an alarm level; the stage task can regularise the regression | only helps if the tasks are consistent |

**Decision principle (fixed before running the study below).** Prefer targets that (i) use *only the failure time* (no hindsight on the vibration), (ii) are deployable for a running machine, (iii) match the official protocol. This selects **E = capped RUL + stage**, with cap = 120 min (a two-hour planning horizon) and stage boundaries at 60 min (Warning) and 20 min (Critical). B is excluded because it is not deployable; D is used only for description (Section 7 / 22) and never as a training label.

### 11.2 Empirical check on the learning bearings (leave-one-bearing-out, fast models only)
**WHAT** – measure, on held-out learning bearings, how well vibration predicts each formulation compared with predicting a constant.
**WHY** – to test the choice above against the data rather than only against arguments, and to see how prediction quality changes with the horizon.
""")

code(r'''
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, f1_score

folds_dev = D.lobo_folds()
L.check_splits(folds_dev)
def tab_windows(fold):
    sc = D.fit_scaler(feats, table, fold["train"] + [fold["val"]])          # here validation bearing is simply another training bearing
    Xtr, mtr = D.make_windows(table, feats, fold["train"] + [fold["val"]], sc); Xte, mte = D.make_windows(table, feats, [fold["held_out"]], sc)
    return D.window_summary(Xtr), mtr, D.window_summary(Xte), mte

rows = []
for fi, f in enumerate(folds_dev):
    Ttr, mtr, Tte, mte = tab_windows(f)
    life_tr = table[table.bearing.isin(f["train"] + [f["val"]])].groupby("bearing").eol_s.first() / 60
    def frac(meta): return (meta.time_min / (meta.time_min + meta.rul_min)).values                       # B: fraction of life elapsed
    rf = RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=SEED)
    # A: uncapped RUL minutes
    rf.fit(Ttr, mtr.rul_min); pa = rf.predict(Tte); rows.append(dict(fold=fi, form="A  uncapped RUL (min)", model_mae=np.abs(pa - mte.rul_min).mean(), const_mae=np.abs(mtr.rul_min.mean() - mte.rul_min).mean()))
    # A': capped RUL minutes (cap 120)
    rf.fit(Ttr, mtr.rul_cap_min); pc = rf.predict(Tte); rows.append(dict(fold=fi, form="A' capped RUL, cap 120 (min)", model_mae=np.abs(pc - mte.rul_cap_min).mean(), const_mae=np.abs(mtr.rul_cap_min.mean() - mte.rul_cap_min).mean()))
    # B: fraction of life (NOT deployable) - shown for comparison only
    rf.fit(Ttr, frac(mtr)); pb = rf.predict(Tte); rows.append(dict(fold=fi, form="B  life fraction (not deployable)", model_mae=np.abs(pb - frac(mte)).mean(), const_mae=np.abs(frac(mtr).mean() - frac(mte)).mean()))
    # C/E: stage classification macro-F1
    clf = RandomForestClassifier(150, min_samples_leaf=5, n_jobs=-1, random_state=SEED, class_weight="balanced").fit(Ttr, mtr.stage)
    rows.append(dict(fold=fi, form="C  3-stage classification (macro-F1)", model_mae=f1_score(mte.stage, clf.predict(Tte), average="macro", labels=[0, 1, 2], zero_division=0),
                     const_mae=f1_score(mte.stage, np.full(len(mte), np.bincount(mtr.stage).argmax()), average="macro", labels=[0, 1, 2], zero_division=0)))
form = pd.DataFrame(rows)
summ = form.groupby("form", sort=False).agg(model=("model_mae", "mean"), constant=("const_mae", "mean"))
summ["model_better_in_folds"] = form.assign(win=lambda d: np.where(d.form.str.contains("F1"), d.model_mae > d.const_mae, d.model_mae < d.const_mae)).groupby("form", sort=False).win.sum().astype(int).astype(str) + "/6"
display(summ.round(3))
R.show("**Reading the table.** For A, A' and B lower is better (minutes or fraction); for C higher is better. *constant* = predict the training mean (or majority class). Random Forest on tabular window summaries is used here only as a fast probe.")
''')

code(r'''
# Horizon analysis: how well can vibration detect "failure within H minutes"?  (Random Forest probe, LOBO, learning bearings only)
H = [3, 5, 10, 20, 30, 60, 90]
hz = []
for fi, f in enumerate(folds_dev):
    Ttr, mtr, Tte, mte = tab_windows(f)
    rf = RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Ttr, mtr.rul_cap_min)
    score = -rf.predict(Tte)
    for h in H:
        y = (mte.rul_min <= h).astype(int)
        hz.append(dict(fold=fi, held_out=f["held_out"], H=h, auc=roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan, positives=float(y.mean())))
hz = pd.DataFrame(hz)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for b, g in hz.groupby("held_out"):
    ax[0].plot(g.H, g.auc, marker="o", lw=1.2, label=b, color=COND_COLOR[C.condition_of(b)], alpha=0.8)
ax[0].plot(hz.groupby("H").auc.mean(), "k-", lw=3, label="mean"); ax[0].axhline(0.5, color="gray", ls=":")
ax[0].set_xscale("log"); ax[0].set_xlabel("Detection horizon H (minutes before failure)"); ax[0].set_ylabel("ROC-AUC of 'failure within H min'"); ax[0].set_title("(a) Detectability vs horizon, each held-out learning bearing"); ax[0].legend(fontsize=7, ncol=2)
lifemin = table.groupby("bearing").eol_s.first() / 60
ph10 = A.phase_table(table, C.LEARNING)["phase_min@10%"]
ax[1].bar(range(6), ph10.values, color=[COND_COLOR[C.condition_of(b)] for b in ph10.index], edgecolor="black")
ax[1].set_xticks(range(6)); ax[1].set_xticklabels([b.replace("Bearing", "B") for b in ph10.index]); ax[1].set_yscale("log")
ax[1].axhline(20, color="r", ls="--", label="Critical horizon (20 min)"); ax[1].axhline(60, color="orange", ls="--", label="Warning horizon (60 min)")
ax[1].set_ylabel("Length of final degradation phase (min, log)"); ax[1].set_title("(b) Visible degradation phase (descriptive, hindsight)"); ax[1].legend()
plt.tight_layout(); R.savefig(fig, "fig08_horizon_detectability.png"); plt.show()
m_auc = hz.groupby("H").auc.mean()
R.show(f"**Figure 8.** (a) Mean AUC of detecting 'failure within H minutes' falls from **{m_auc[5]:.2f} (H=5 min)** to **{m_auc[60]:.2f} (H=60 min)**, and for some bearings it is close to chance at long horizons. (b) The visible degradation phase of several learning bearings is shorter than the Warning horizon.")
''')

md(r"""
### 11.3 What did we find, and what is the frozen target?
""")

code(r'''
r = summ.copy(); a_, ac_, b_, c_ = r.iloc[0], r.iloc[1], r.iloc[2], r.iloc[3]
R.show("\n\n".join([
    f"* **Direct / capped RUL in minutes (A, A'):** the Random-Forest probe does **not** clearly beat the constant prediction (A: {a_.model:.1f} vs {a_.constant:.1f} min, better in {r.model_better_in_folds.iloc[0]} folds; A': {ac_.model:.1f} vs {ac_.constant:.1f} min, {r.model_better_in_folds.iloc[1]} folds). Minute-level RUL across bearings is therefore **not supported** by this data.",
    f"* **Life fraction (B)** gives lower error ({b_.model:.2f} vs {b_.constant:.2f}, {r.model_better_in_folds.iloc[2]} folds) because the target itself is normalised by the bearing's total lifetime - information that is unknown for a running machine. **Not deployable; not used.**",
    f"* **Stage classification (C)** beats the majority-class predictor in {r.model_better_in_folds.iloc[3]} folds (macro-F1 {c_.model:.2f} vs {c_.constant:.2f}), i.e. there is real but moderate signal about *how close to failure* a bearing is.",
    f"* **Detectability depends strongly on the horizon** (Figure 8): the mean AUC of 'failure within H min' is {m_auc[5]:.2f} at 5 min, {m_auc[20]:.2f} at 20 min and {m_auc[60]:.2f} at 60 min, and it is close to chance for the abrupt bearings at long horizons.",
    "**Frozen decision.** Keep formulation **E** (capped RUL + stage) with the *a priori* horizons (cap 120 min, Warning <= 60 min, Critical <= 20 min); they are **not tuned** to obtain better scores. Because horizon matters so much, every model is also evaluated at several horizons (AUC at 5, 10, 20, 30, 60 min). We record explicitly that **the dataset does not support reliable early RUL prediction for bearings that fail abruptly**. No vibration-based onset label is used for training; the onset-style analysis (D) is descriptive only.",
    "**Remaining hindsight assumptions.** (1) Training uses the failure time of the learning bearings (standard supervised learning). (2) The cap and stage boundaries are design choices fixed a priori; a sensitivity analysis follows. (3) The descriptive onset/phase-length tables use the whole trajectory (hindsight) and are never used as labels."]))
''')

code(r'''
# Sensitivity of the conclusion to the cap (learning bearings, LOBO, Random Forest probe + Ridge): "skill" = 1 - MAE_model/MAE_constant
sens = []
for cap in [60, 120, 240]:
    for fi, f in enumerate(folds_dev):
        Ttr, mtr, Tte, mte = tab_windows(f)
        ytr = np.minimum(mtr.rul_min, cap); yte = np.minimum(mte.rul_min, cap)
        for name, mdl in [("Random Forest", RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=SEED)), ("Ridge", Ridge(1.0))]:
            p = mdl.fit(Ttr, ytr).predict(Tte)
            sens.append(dict(cap=cap, model=name, fold=fi, skill=1 - np.abs(p - yte).mean() / np.abs(ytr.mean() - yte).mean(),
                             auc60=roc_auc_score(mte.rul_min <= 60, -p) if 0 < (mte.rul_min <= 60).sum() < len(mte) else np.nan))
sens = pd.DataFrame(sens)
display(sens.groupby(["cap", "model"]).agg(mean_skill=("skill", "mean"), folds_with_skill_gt0=("skill", lambda s: f"{(s>0).sum()}/6"), mean_AUC_60min=("auc60", "mean")).round(3))
R.show("**Cap sensitivity (learning bearings only).** Skill > 0 means better than the constant. The picture (little or negative skill in minutes, moderate detection AUC) does not change with the cap, so the conclusion is not an artefact of the chosen cap. Nothing was tuned on this table.")
''')

# =====================================================================================================================
md(r"""
## 12. Leakage Prevention — automated checks

**WHAT** – executable tests. If any of them fails, the notebook stops with an `AssertionError`.
**WHY** – claims of "no leakage" are worthless unless they are tested. Each test corresponds to a leakage route relevant to this dataset.

| Test | Leakage route it closes |
|---|---|
| splits are disjoint, use only learning bearings, every learning bearing is held out once | train/test contamination |
| scaler statistics equal the training-bearing statistics and differ from all-data statistics | normalisation leakage |
| **poison test**: destroying the held-out bearing's data changes nothing used for training | any hidden use of held-out data |
| **truncation test**: deleting all snapshots after time *t* leaves windows up to *t* unchanged | future information in inputs |
| windows are contiguous inside one bearing | windows across bearing boundaries |
| features recomputed independently from a single raw file match | cross-snapshot information in features |
| **targets are unchanged when the vibration features are shuffled** | vibration-based hindsight in targets |
| configuration snapshot is stored and re-checked before the test | test-driven design changes |
""")

code(r'''
checks = {}
folds = D.lobo_folds()
checks["splits disjoint, learning-only, each bearing held out once"] = L.check_splits(folds)
for fi, f in enumerate(folds):
    sc_f = D.fit_scaler(feats, table, f["train"])
    checks[f"fold {fi}: scaler fitted on training bearings only"] = L.check_scaler_train_only(sc_f, feats, table, f["train"])
    checks[f"fold {fi}: poison test (held-out data cannot reach training)"] = L.check_poison(table, feats, f)
sc_all_chk = D.fit_scaler(feats, table, C.LEARNING)
for b in ("Bearing1_1", "Bearing2_4", "Bearing1_3"):
    n_b = int((table.bearing == b).sum())
    checks[f"{b}: truncation / causality test"] = L.check_window_causality(table, feats, sc_all_chk, b, [40, n_b // 2, n_b - 30])
Xchk, mchk = D.make_windows(table, feats, C.LEARNING, sc_all_chk)
checks["windows contiguous inside each bearing"] = L.check_window_boundaries(Xchk, mchk)
checks["features recomputed independently from single raw files (all 17 bearings)"] = L.check_features_single_snapshot(raw)
checks["targets depend only on failure time (feature-shuffle test)"] = L.check_targets_only_from_failure_time(raw, table)
assert not (set(C.TEST) & set(D.lobo_folds()[0]["train"])), "test bearing in training"
CFG_FROZEN = L.config_snapshot()
checks["configuration snapshot stored (re-checked before the test)"] = True
display(pd.DataFrame({"check": list(checks), "passed": list(checks.values())}))
assert all(checks.values())
R.show(f"**All {len(checks)} leakage checks passed.** Frozen configuration: `{CFG_FROZEN}`")
''')

md(r"""
### 12.1 What did we find?
All automated checks passed. In particular the **poison test** shows that everything derived for training (windows, scaler, targets) is bit-identical even if the held-out bearing's data is replaced by noise, and the **truncation test** shows that model inputs at time *t* do not depend on any later snapshot. **This supports the claim that the validation estimates below are free of leakage from the held-out bearings.** It does not remove the *selection* effects discussed in Section 3 (which are disclosed, not hidden).
""")

# =====================================================================================================================
md(r"""
## 13. Train / Validation / Test Split

**WHAT** – define the data protocol and demonstrate why random window splitting would be wrong.
**PROTOCOL.**
* **Method development and model selection:** the 6 learning bearings with **leave-one-bearing-out (LOBO)**. In each of the 6 folds: one bearing is *held out* (scored), the next learning bearing in cyclic order is the *validation* bearing (used only for early stopping), the other 4 train the model.
* **Final test:** the 11 test bearings, scored once after the design is frozen (Section 22). The final models train on all 6 learning bearings for a number of epochs fixed from the LOBO early-stopping results.

**WHY random splitting is wrong.** Windows overlap by 15 of 16 snapshots; a random split puts near-copies of test windows into training. The cell below measures the effect with the same Random Forest on the same learning bearings.
""")

code(r'''
fold_tbl = pd.DataFrame([{"fold": i, "held-out (scored)": f["held_out"], "validation (early stopping)": f["val"], "training": ", ".join(f["train"])} for i, f in enumerate(D.lobo_folds())])
display(fold_tbl)
fig, ax = plt.subplots(figsize=(11, 3.4))
colmap = {"train": "#4c72b0", "val": "#dd8452", "held": "#c44e52", "-": "#eeeeee", "test": "#55a868"}
grid = np.full((6, 17), "-", dtype=object)
for i, f in enumerate(D.lobo_folds()):
    for b in f["train"]: grid[i, (C.LEARNING + C.TEST).index(b)] = "train"
    grid[i, (C.LEARNING + C.TEST).index(f["val"])] = "val"; grid[i, (C.LEARNING + C.TEST).index(f["held_out"])] = "held"
    for b in C.TEST: grid[i, (C.LEARNING + C.TEST).index(b)] = "-"
for i in range(6):
    for j in range(17):
        ax.add_patch(plt.Rectangle((j, 5 - i), 1, 1, color=colmap[grid[i, j]], ec="white"))
for j in range(6, 17): ax.add_patch(plt.Rectangle((j, -1.2), 1, 1, color=colmap["test"], ec="white"))
ax.set_xlim(0, 17); ax.set_ylim(-1.4, 6); ax.set_xticks(np.arange(17) + 0.5); ax.set_xticklabels([b.replace("Bearing", "B") for b in C.LEARNING + C.TEST], rotation=60)
ax.set_yticks(list(np.arange(6) + 0.5) + [-0.7]); ax.set_yticklabels([f"fold {5-i}" for i in range(6)] + ["FINAL"]); ax.grid(False)
ax.set_title("Bearing-level protocol: blue = train, orange = validation (early stopping), red = held-out, green = final test (touched once)")
plt.tight_layout(); R.savefig(fig, "fig09_split_protocol.png"); plt.show()

# random-window split (WRONG protocol) vs bearing-level LOBO, same Random Forest, same learning bearings
sc_l = D.fit_scaler(feats, table, C.LEARNING); Xl, ml = D.make_windows(table, feats, C.LEARNING, sc_l); Tl = D.window_summary(Xl)
rng = np.random.default_rng(SEED); idx = rng.permutation(len(Tl)); cut = int(0.8 * len(idx)); tr_i, te_i = idx[:cut], idx[cut:]
rf = RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Tl[tr_i], ml.rul_cap_min.values[tr_i])
p_rand = rf.predict(Tl[te_i]); mae_rand = np.abs(p_rand - ml.rul_cap_min.values[te_i]).mean()
auc_rand = roc_auc_score(ml.rul_min.values[te_i] <= 60, -p_rand)
lobo_maes, lobo_aucs = [], []
for f in D.lobo_folds():
    Ttr, mtr, Tte, mte = tab_windows(f)
    p = RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Ttr, mtr.rul_cap_min).predict(Tte)
    lobo_maes.append(np.abs(p - mte.rul_cap_min).mean()); lobo_aucs.append(roc_auc_score(mte.rul_min <= 60, -p))
leak = pd.DataFrame({"protocol": ["RANDOM window split (wrong)", "Bearing-level LOBO (correct)"], "MAE (min)": [mae_rand, np.mean(lobo_maes)], "AUC (failure within 60 min)": [auc_rand, np.mean(lobo_aucs)]}).set_index("protocol")
display(leak.round(3))
R.show(f"**Figure 9 / table.** Same model, same bearings: the random window split reports an MAE of **{mae_rand:.1f} min** and AUC **{auc_rand:.2f}**, the correct bearing-level protocol **{np.mean(lobo_maes):.1f} min** and **{np.mean(lobo_aucs):.2f}**. "
       "The difference is pure leakage of near-duplicate windows. All results in this notebook use the bearing-level protocol.")
''')
