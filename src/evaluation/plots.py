from __future__ import annotations

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


PREFERRED_MODEL_ORDER = [
    "Static_Rules",
    "RandomForest",
    "MLP_Softmax",
    "LogisticRegression",
    "IsolationForest",
    "DEQZT",
]

DISPLAY_MODEL_NAMES = {
    "Static_Rules": "Static-Ruled-ZT\n[31]",
    "RandomForest": "RandomForest-ZT\n[32]",
    "MLP_Softmax": "SoftmaxRuled-ZT\n[33]",
    "LogisticRegression": "LogisticRegression-ZT\n[34]",
    "IsolationForest": "IsolationForest-ZT\n[35]",
    "DEQZT": "DEQ-ZT",
}


def _ordered_metric_rows(metric_rows):
    rows = list(metric_rows)
    order_map = {m: i for i, m in enumerate(PREFERRED_MODEL_ORDER)}
    rows.sort(key=lambda r: (order_map.get(str(r.get("model")), 999), str(r.get("model"))))
    return rows


def _display_name(name: str) -> str:
    return DISPLAY_MODEL_NAMES.get(str(name), str(name))


def _is_percent_metric(values: List[float]) -> bool:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return True
    return min(finite) >= 0.0 and max(finite) <= 1.05


def plot_bar(metric_rows, key, out_png, title, *, higher_is_better: Optional[bool] = None, sort_desc: bool = False):
    rows = _ordered_metric_rows(metric_rows)
    names_raw = [str(r["model"]) for r in rows]
    names = [_display_name(n) for n in names_raw]
    vals = [float(r.get(key, np.nan)) for r in rows]

    fig = plt.figure(figsize=(max(8, len(names) * 1.2), 6))
    ax = fig.add_subplot(111)

    bars = ax.bar(names, vals)
    ax.set_title(title)
    ax.set_ylabel(key)
    ax.tick_params(axis="x", rotation=30)

    finite_vals = [v for v in vals if np.isfinite(v)]
    if not finite_vals:
        finite_vals = [1.0]
    percent_like = _is_percent_metric(vals)
    if percent_like:
        ymax = min(1.0, max(finite_vals) * 1.20 + 0.05)
        ax.set_ylim(0.0, max(0.05, ymax))
    else:
        ymax = max(finite_vals) * 1.20 + (0.05 * max(finite_vals) if max(finite_vals) > 0 else 1.0)
        ax.set_ylim(0.0, max(1.0, ymax))

    if higher_is_better is None:
        higher_is_better = True
    best_val = (max(finite_vals) if higher_is_better else min(finite_vals))
    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02

    for b, v in zip(bars, vals):
        if np.isfinite(v) and np.isclose(v, best_val, rtol=0.0, atol=1e-12):
            b.set_hatch("//")
            b.set_linewidth(1.5)
        if not np.isfinite(v):
            label = "NA"
            y = 0.0
        elif percent_like:
            label = f"{v:.6f}\n({v * 100:.2f}%)"
            y = v
        else:
            label = f"{v:.3f}"
            y = v
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def selective_accuracy_curve(uncertainty, correct):
    """Returns arrays coverage and accuracy after rejecting most-uncertain samples."""
    u = np.asarray(uncertainty, dtype=np.float32)
    c = np.asarray(correct, dtype=np.float32)
    order = np.argsort(-u)
    c = c[order]
    n = len(c)

    coverages = []
    accs = []
    for k in range(0, 20):
        reject_frac = k / 20.0
        cut = int(reject_frac * n)
        kept = c[cut:]
        coverage = 1.0 - reject_frac
        acc = float(kept.mean()) if len(kept) else np.nan
        coverages.append(coverage)
        accs.append(acc)

    return np.array(coverages, dtype=np.float32), np.array(accs, dtype=np.float32)


def plot_selective_accuracy_vs_coverage(model_curves, out_png, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for mc in model_curves:
        ax.plot(mc["coverage"], mc["acc"], marker="o", label=_display_name(mc["model"]))

    ax.set_xlabel("Coverage (fraction kept)")
    ax.set_ylabel("Accuracy on kept samples")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
