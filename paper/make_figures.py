"""Generates the manuscript figures from the frozen project results.
All plotted numbers are read from results/*.csv, the notebook training log, or the frozen
model's attention output (attn_b13.npy); nothing is re-trained or re-estimated."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "svg.fonttype": "path"})

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GREY = "#1f4e79", "#c55a11", "#548235", "#a50f15", "#595959"

preds = pd.read_csv(os.path.join(RES, "test_predictions_all_11_bearings.csv"))
perb = pd.read_csv(os.path.join(RES, "per_bearing_metrics_all_11_bearings.csv"))
chal = pd.read_csv(os.path.join(RES, "challenge_single_inspection_table.csv"))


# ================================================================ diagrams
# Boxes use pad=0, so the drawn outline is exactly the rectangle (cx +- w/2, cy +- h/2).
# Every connector is computed from those edges: arrows start at the centre of the source
# edge and their tips end at the centre of the destination edge (shrinkA = shrinkB = 0).
LW = 0.9


class Box:
    def __init__(self, ax, cx, cy, w, h, text, fc, fs=8.5, ls="-", ec="#222222", lsp=1.35):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=LW, ls=ls, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=lsp, zorder=3)

    top = property(lambda s: (s.cx, s.cy + s.h / 2))
    bottom = property(lambda s: (s.cx, s.cy - s.h / 2))
    left = property(lambda s: (s.cx - s.w / 2, s.cy))
    right = property(lambda s: (s.cx + s.w / 2, s.cy))


def connect(ax, p, q, color="#222222", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, shrinkA=0, shrinkB=0,
                                 color=color, lw=LW, ls=ls, zorder=4))


def line(ax, p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#222222", lw=LW, solid_capstyle="butt", zorder=4)


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save_diagram(fig, name):
    # drawn at final print size (1 figure inch = 1 printed inch); no tight cropping
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches=None)
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches=None)
    plt.close(fig)


def fig_architecture():
    W, H = 7.0, 4.2
    fig, ax = canvas(W, H)
    cx, bw, gap = 3.5, 3.3, 0.15
    rows = [  # (height, text, colour, output shape)
        (0.33, "Input window  $\\mathbf{X}\\in\\mathbb{R}^{16\\times 30}$\n16 snapshots (160 s) × 30 standardized features", "#e8eef7", "(16, 30)"),
        (0.33, "Conv1D-1: 64 filters, kernel 3, 'same', ReLU\nBatchNorm → Dropout (0.2)", "#dbe7f3", "(16, 64)"),
        (0.33, "Conv1D-2: 64 filters, kernel 3, 'same', ReLU\nBatchNorm → Dropout (0.2)", "#dbe7f3", "(16, 64)"),
        (0.33, "Bidirectional LSTM: 64 units per direction\n(return sequences) → Dropout (0.2)", "#e2efda", "(16, 128)"),
        (0.44, "Temporal attention\n$e_t=\\mathbf{u}^{\\top}\\tanh(\\mathbf{W}_a\\mathbf{h}_t+\\mathbf{b}_a)$,  "
               "$\\alpha_t=\\mathrm{softmax}_t(e_t)$,  $\\mathbf{c}=\\sum_t\\alpha_t\\mathbf{h}_t$", "#fff2cc",
         "c: (128)\nα: (16, 1)"),
        (0.26, "Shared dense: 64 units, ReLU → Dropout (0.2)", "#ededed", "(64)"),
    ]
    y = H - 0.06
    boxes = []
    for h, t, fc, shp in rows:
        b = Box(ax, cx, y - h / 2, bw, h, t, fc)
        ax.text(cx + bw / 2 + 0.12, b.cy, shp, fontsize=7.5, color=C_GREY, va="center", family="monospace")
        boxes.append(b)
        y -= h + gap
    for a, b in zip(boxes[:-1], boxes[1:]):
        connect(ax, a.bottom, b.top)
    att, shared = boxes[4], boxes[5]
    # attention weights exposed as a side output (explanation path)
    lab = Box(ax, 0.82, att.cy, 1.3, 0.44, "Attention weights\n$\\alpha_1,\\ldots,\\alpha_{16}$ (explanation)",
              "white", fs=7.8, ec=C_ORANGE, ls="--")
    connect(ax, att.left, lab.right, color=C_ORANGE, ls="--")
    # symmetric branch into the two heads
    hx, hw, hh = 1.45, 2.7, 0.36
    yj = shared.cy - shared.h / 2 - 0.13
    head_cy = yj - 0.19 - hh / 2
    line(ax, shared.bottom, (cx, yj))
    line(ax, (cx - hx, yj), (cx + hx, yj))
    rul = Box(ax, cx - hx, head_cy, hw, hh, "RUL head: Dense 32 (ReLU) → Dense 1 (linear)\n"
              "output $\\hat{y}_{rul}$ (normalized RUL)", "#dbe7f3")
    stg = Box(ax, cx + hx, head_cy, hw, hh, "Stage head: Dense 32 (ReLU) → Dense 3 (softmax)\n"
              "output $\\hat{\\mathbf{p}}$ = P(Normal, Warning, Critical)", "#fbe5d6")
    connect(ax, (cx - hx, yj), rul.top)
    connect(ax, (cx + hx, yj), stg.top)
    pc = head_cy - hh / 2 - gap - hh / 2
    p1 = Box(ax, cx - hx, pc, hw, hh, "clip($\\hat{y}_{rul}$, 0, 1) × $RUL_{cap}$ (16,812 s)\n→ predicted RUL (s, min)",
             "white", ec=C_BLUE, ls="--")
    p2 = Box(ax, cx + hx, pc, hw, hh, "arg max $\\hat{\\mathbf{p}}$ → predicted stage\n(0 Normal, 1 Warning, 2 Critical)",
             "white", ec=C_ORANGE, ls="--")
    connect(ax, rul.bottom, p1.top)
    connect(ax, stg.bottom, p2.top)
    assert pc - hh / 2 > 0.02, "diagram does not fit the canvas"
    save_diagram(fig, "fig_architecture")


def fig_flowchart():
    W, H = 6.4, 4.5
    fig, ax = canvas(W, H)
    bw, bh, gap = 2.8, 0.55, 0.16
    L, R = 1.6, 4.8
    left = [
        ("PRONOSTIA / IEEE PHM 2012 data\n17 bearings, 3 operating conditions\n2,560 samples at 25.6 kHz every 10 s", "#e8eef7"),
        ("Per-snapshot feature extraction\n15 time/frequency features × 2 axes\n= 30 features", "#dbe7f3"),
        ("Label generation\ncapped RUL = min($RUL_{cap}$, $T_{EOL}-t$)\n3 auxiliary stages from $y_{rul}$", "#dbe7f3"),
        ("Bearing-level split\nTrain: 1_1, 2_1, 3_1   Val: 1_2, 2_2, 3_2\nTest: 1_3–1_7, 2_3–2_7, 3_3", "#e2efda"),
        ("StandardScaler fitted on training\nbearings only; $RUL_{cap}$ = 0.6 × 28,020 s\n(longest training lifetime)", "#e2efda"),
        ("Sliding windows per bearing\nW = 16, stride 1\n(no window crosses bearings)", "#e2efda"),
    ]
    right = [  # listed bottom to top
        ("Train dual-head CNN–BiLSTM–attention\nAdam 1e-3, batch 64, ≤ 35 epochs\nEarlyStopping(val. loss, patience 7)", "#fff2cc"),
        ("Frozen model inference\non every test window", "#fff2cc"),
        ("De-normalize: $\\widehat{RUL}=\\mathrm{clip}(\\hat{y}_{rul},0,1)\\times RUL_{cap}$\n(no test-bearing lifetime used)", "#fff2cc"),
        ("Trajectory evaluation\nMAE, RMSE, normalized MAE, T-Score,\nstage accuracy / precision / recall / F1", "#fbe5d6"),
        ("Single-inspection evaluation\nlast Test_set snapshot per bearing\nofficial challenge score", "#fbe5d6"),
        ("Explainability\ntemporal attention weights $\\alpha_t$\nhealthy vs. near-failure windows", "#fbe5d6"),
    ]
    top = H - 0.3
    ys = [top - bh / 2 - i * (bh + gap) for i in range(6)]
    lb = [Box(ax, L, y, bw, bh, t, fc, fs=8.3, lsp=1.2) for y, (t, fc) in zip(ys, left)]
    rb = [Box(ax, R, y, bw, bh, t, fc, fs=8.3, lsp=1.2) for y, (t, fc) in zip(ys[::-1], right)]
    for a, b in zip(lb[:-1], lb[1:]):
        connect(ax, a.bottom, b.top)
    connect(ax, lb[-1].right, rb[0].left)
    for a, b in zip(rb[:-1], rb[1:]):
        connect(ax, a.top, b.bottom)
    ax.text(L, H - 0.13, "Stage A: data preparation (top to bottom)", ha="center", va="center", fontsize=8.5, style="italic")
    ax.text(R, H - 0.13, "Stage B: modelling and evaluation (bottom to top)", ha="center", va="center",
            fontsize=8.5, style="italic")
    assert ys[-1] - bh / 2 > 0.02
    save_diagram(fig, "fig_flowchart")


# ---------------------------------------------------------------- Target illustration
def fig_target():
    cap = 16812.0
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for T, c, lab in ((28020.0, C_BLUE, "Bearing1_1 (train, 28,020 s)"), (9100.0, C_GREEN, "Bearing2_1 (train, 9,100 s)")):
        t = np.linspace(0, T, 500)
        ax.plot(t / 3600, np.minimum(cap, T - t) / 60, color=c, lw=1.5, label=lab)
    # labels sit in the gap between the two curves (Bearing2_1 ends at 2.53 h; Bearing1_1 starts falling at 3.11 h)
    for v, s in ((0.40 * cap / 60, "0.40 cap = 112.1 min"), (0.15 * cap / 60, "0.15 cap = 42.0 min")):
        ax.axhline(v, color=C_GREY, ls=":", lw=1)
        ax.text(2.75, v + 4, s, ha="left", va="bottom", fontsize=7, color=C_GREY)
    ax.text(2.75, 185, "Normal", fontsize=7.5, color=C_GREEN)
    ax.text(2.75, 72, "Warning", fontsize=7.5, color=C_ORANGE)
    ax.text(2.75, 12, "Critical", fontsize=7.5, color=C_RED)
    ax.set_xlabel("Operating time (h)")
    ax.set_ylabel("Capped RUL target (min)")
    ax.set_ylim(0, 300)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, "fig_target.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Trajectories
def fig_trajectories():
    sel = ["Bearing1_3", "Bearing1_4", "Bearing2_5", "Bearing3_3"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3))
    for ax, b in zip(axes.flat, sel):
        d = preds[preds.bearing == b].sort_values("snapshot_idx")
        ax.plot(d.operating_time_m, d.capped_rul_m, color="black", ls="--", lw=1.3, label="Capped ground truth")
        ax.plot(d.operating_time_m, d.pred_rul_m, color=C_BLUE, lw=0.8, label="Prediction")
        r = perb[perb.Bearing == b].iloc[0]
        ax.set_title(f"{b}  (MAE {r.RUL_MAE_min:.2f} min, T-Score {r.PHM_T_Score:.4f})")
        ax.set_xlabel("Operating time (min)")
        ax.set_ylabel("RUL (min)")
        ax.set_ylim(-5, 300)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.savefig(os.path.join(OUT, "fig_trajectories.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Per-bearing bars
def fig_per_bearing():
    colors = {"Cond 1": C_BLUE, "Cond 2": C_ORANGE, "Cond 3": C_GREEN}
    c = [colors[k] for k in perb.Condition]
    labels = [b.replace("Bearing", "B") for b in perb.Bearing]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.15))
    a1.bar(labels, perb.RUL_MAE_min, color=c)
    a1.axhline(93.40, color="black", ls="--", lw=1)
    a1.text(10.4, 96, "aggregate 93.40", ha="right", fontsize=7)
    a1.set_ylabel("RUL MAE (min)")
    a1.set_title("(a) RUL MAE per test bearing")
    a2.bar(labels, perb.Risk_Acc_pct, color=c)
    a2.axhline(52.82, color="black", ls="--", lw=1)
    a2.text(10.4, 55, "aggregate 52.82%", ha="right", fontsize=7)
    a2.set_ylabel("Stage accuracy (%)")
    a2.set_title("(b) Auxiliary stage accuracy")
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=60)
        a.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    fig.legend(handles, ["Condition 1", "Condition 2", "Condition 3"], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_per_bearing.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Confusion matrix (11 bearings)
def fig_confusion():
    cm = confusion_matrix(preds.health_stage, preds.pred_risk_class, labels=[0, 1, 2])
    rn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    names = ["Normal", "Warning", "Critical"]
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({rn[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=8,
                    color="white" if rn[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    fig.savefig(os.path.join(OUT, "fig_confusion_11.png"))
    plt.close(fig)
    return cm


# ---------------------------------------------------------------- Single inspection
def fig_single_inspection():
    labels = [b.replace("Bearing", "B") for b in chal.Bearing]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.bar(x - 0.2, chal.Actual_RUL_Physical_m, 0.4, color="#7f7f7f", label="Actual RUL (Full_Test_Set files)")
    ax.bar(x + 0.2, chal.Predicted_RUL_m, 0.4, color=C_BLUE, label="Predicted RUL")
    for i, a in enumerate(chal.Score_A_Physical):
        ax.text(x[i], max(chal.Actual_RUL_Physical_m[i], chal.Predicted_RUL_m[i]) + 4, f"$A_i$ = {a:.4f}",
                ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RUL at inspection (min)")
    ax.set_ylim(0, 185)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(os.path.join(OUT, "fig_single_inspection.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Raw signals and feature trends (Bearing1_1)
def fig_raw_and_features():
    base = os.path.join(ROOT, "ieee-phm-2012-data-challenge-dataset-master", "Learning_set", "Bearing1_1")
    t_ms = np.arange(2560) / 25.6
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    for ax, f, title in ((axes[0], "acc_00010.csv", "(a) Snapshot 10 (early life)"),
                         (axes[1], "acc_02800.csv", "(b) Snapshot 2800 (end of test)")):
        d = pd.read_csv(os.path.join(base, f), header=None)
        ax.plot(t_ms, d[4], color=C_BLUE, lw=0.5, label="Horizontal")
        ax.plot(t_ms, d[5], color=C_ORANGE, lw=0.5, alpha=0.8, label="Vertical")
        ax.axhline(20, color="black", ls="--", lw=0.8)
        ax.axhline(-20, color="black", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Time within snapshot (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].legend(frameon=False, loc="lower left", ncol=2)
    axes[0].text(99, 21.5, "±20 g stop", ha="right", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_raw.png"))
    plt.close(fig)

    feats = pd.read_csv(os.path.join(RES, "extracted_bearing_features.csv"))
    b = feats[feats.bearing == "Bearing1_1"].sort_values("snapshot_idx")
    th = b.operating_time_s / 3600
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    axes[0].plot(th, b.horiz_rms, color=C_BLUE, lw=0.8, label="Horizontal")
    axes[0].plot(th, b.vert_rms, color=C_ORANGE, lw=0.8, label="Vertical")
    axes[0].set_ylabel("RMS (g)")
    axes[0].legend(frameon=False)
    axes[1].plot(th, b.horiz_kurtosis, color=C_BLUE, lw=0.8)
    axes[1].set_ylabel("Excess kurtosis (horiz.)")
    axes[2].plot(th, b.horiz_spec_centroid, color=C_BLUE, lw=0.8)
    axes[2].set_ylabel("Spectral centroid (Hz)")
    for ax, t in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_xlabel("Operating time (h)")
        ax.set_title(t)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_features_b11.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Training history (values logged in the notebook, Table VI)
HIST = {
    "loss": [0.2339, 0.0874, 0.0636, 0.0518, 0.0317, 0.0268, 0.0265, 0.0202],
    "val_loss": [0.7702, 1.1371, 1.4890, 1.6492, 1.8276, 1.8701, 1.7724, 1.8565],
    "mae": [0.1894, 0.1246, 0.1039, 0.0922, 0.0851, 0.0813, 0.0794, 0.0754],
    "val_mae": [0.2326, 0.2002, 0.2373, 0.3053, 0.3027, 0.2614, 0.2645, 0.2486],
    "acc": [82.89, 94.12, 95.84, 96.44, 98.18, 98.42, 98.47, 98.97],
    "val_acc": [37.45, 34.97, 38.56, 35.49, 35.46, 39.20, 37.82, 37.73],
}


def fig_training():
    ep = np.arange(1, 9)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for ax, (k, lab, t) in zip(axes, (("loss", "Total loss", "(a) Multi-task loss"),
                                      ("mae", "Normalized RUL MAE", "(b) RUL MAE"),
                                      ("acc", "Stage accuracy (%)", "(c) Stage accuracy"))):
        ax.plot(ep, HIST[k], "o-", color=C_BLUE, lw=1.2, ms=3.5, label="Training")
        ax.plot(ep, HIST["val_" + k], "s--", color=C_ORANGE, lw=1.2, ms=3.5, label="Validation")
        ax.axvline(1, color=C_GREY, ls=":", lw=0.9)
        ax.set_xticks(ep)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)
        ax.set_title(t)
        ax.grid(alpha=0.25)
    axes[0].text(1.2, 1.45, "restored\nweights\n(epoch 1)", fontsize=7, color=C_GREY)
    axes[0].legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_training.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Attention weights (frozen model, Bearing1_3)
def fig_attention():
    h, c = np.load(os.path.join(ROOT, "paper", "attn_b13.npy"))
    # must reproduce the values printed in the executed notebook
    assert np.argmax(h) == 11 and round(float(h.max()), 4) == 0.0813
    assert np.argmax(c) == 15 and round(float(c.max()), 4) == 0.1120
    steps = np.arange(1, 17)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    for ax, v, col, t in ((axes[0], h, C_BLUE, "(a) Early window (snapshots 50–65)"),
                          (axes[1], c, C_RED, "(b) Final window of the record")):
        ax.bar(steps, v, color=col, width=0.75)
        ax.axhline(1 / 16, color="black", ls="--", lw=0.9)
        ax.set_xticks(steps)
        ax.set_xlabel("Time step in window (16 = most recent)")
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"Attention weight $\alpha_t$")
    axes[1].text(0.6, 1 / 16 + 0.003, "uniform 1/16", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_attention.png"))
    plt.close(fig)
    return h, c


if __name__ == "__main__":
    fig_raw_and_features()
    fig_architecture()
    fig_flowchart()
    fig_target()
    fig_trajectories()
    fig_per_bearing()
    cm = fig_confusion()
    fig_single_inspection()
    fig_training()
    h, _ = fig_attention()
    # self-check: plotted data must match the frozen metrics
    assert len(preds) == 17190 and cm.sum() == 17190
    assert abs(np.trace(cm) / cm.sum() - 0.5282) < 5e-5
    assert abs(chal.Score_A_Physical.mean() - 0.0718) < 5e-5 and abs(chal.Score_A_Literal.mean() - 0.0234) < 5e-5
    print("figures written; checks passed; early-window attention min =", round(float(h.min()), 4))
