"""Small reporting helpers used by the notebook (dynamic markdown, paired statistics, figure saving)."""
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from . import config as C


def show(text):
    display(Markdown(text))


def savefig(fig, name):
    C.FIGURES.mkdir(exist_ok=True)
    path = C.FIGURES / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def bootstrap_ci(values, n=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean over independent units (bearings/folds)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_difference(a, b, label_a, label_b, higher_is_better=False):
    """Mean of (b - a) over independent units (bearings/folds) with bootstrap 95% CI.
    higher_is_better=False: negative difference = b better (errors). True: positive difference = b better (AUC, F1)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if higher_is_better:
        d = -d                      # internally: negative = improvement
    m, lo, hi = bootstrap_ci(d)
    better = int((d < 0).sum())
    if hi < 0:
        verdict = "measurable improvement"
    elif lo > 0:
        verdict = "measurable degradation"
    else:
        verdict = "no measurable difference (CI includes 0)"
    if higher_is_better:
        m, lo, hi = -m, -hi, -lo    # report in the natural sign (positive = higher metric)
    return {"from": label_a, "to": label_b, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "units_improved": f"{better}/{len(d)}", "verdict": verdict}


def fmt_ms(x):
    """mean +/- std string for a Series."""
    x = pd.Series(x).dropna()
    return f"{x.mean():.3f} +/- {x.std():.3f}" if len(x) > 1 else f"{x.mean():.3f}"
