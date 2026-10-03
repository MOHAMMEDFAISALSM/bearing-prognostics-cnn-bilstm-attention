"""Report-only figure: high-level system block diagram. Reuses the exact-edge Box/connect
helpers from paper/make_figures.py (arrows run edge-centre to edge-centre)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper"))
import make_figures as mf  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def fig_system_block():
    W, H = 6.0, 1.25
    fig, ax = mf.canvas(W, H)
    bw, bh, gap = 1.0, 0.95, 0.2
    texts = [
        ("Vibration data\n2 accelerometers\n25.6 kHz, 0.1 s\nevery 10 s", "#e8eef7"),
        ("Feature\nextraction\n30 features per\nsnapshot", "#dbe7f3"),
        ("Preprocessing\ntrain-only scaling\ncapped RUL target\n16-step windows", "#e2efda"),
        ("Dual-head\nCNN–BiLSTM–\nattention model", "#fff2cc"),
        ("Outputs\nRUL (s, min)\nrisk stage\nattention weights", "#fbe5d6"),
    ]
    x0 = (W - (5 * bw + 4 * gap)) / 2 + bw / 2
    boxes = [mf.Box(ax, x0 + i * (bw + gap), H / 2, bw, bh, t, fc, fs=8, lsp=1.25) for i, (t, fc) in enumerate(texts)]
    for a, b in zip(boxes[:-1], boxes[1:]):
        mf.connect(ax, a.right, b.left)
    fig.savefig(os.path.join(OUT, "fig_system_block.png"), dpi=600, bbox_inches=None)
    fig.savefig(os.path.join(OUT, "fig_system_block.svg"), bbox_inches=None)


if __name__ == "__main__":
    fig_system_block()
    print("report figures written")
