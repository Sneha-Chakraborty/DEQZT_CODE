import numpy as np
import matplotlib.pyplot as plt

def plot_bar(metric_rows, key, out_png, title):
    names = [r["model"] for r in metric_rows]
    vals = [r.get(key, np.nan) for r in metric_rows]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(names, vals)
    ax.set_title(title)
    ax.set_ylabel(key)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def selective_accuracy_curve(uncertainty, correct):
    """
    Returns arrays:
      coverage: fraction kept (1 -> 0)
      acc: accuracy among kept points when rejecting most-uncertain first
    """
    u = np.asarray(uncertainty, dtype=np.float32)
    c = np.asarray(correct, dtype=np.float32)

    # sort by uncertainty descending (reject highest u first)
    order = np.argsort(-u)
    c = c[order]
    n = len(c)

    coverages = []
    accs = []
    # evaluate at 0%,5%,...,95% rejected
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
    """
    model_curves: list of dicts with keys: model, coverage, acc
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for mc in model_curves:
        ax.plot(mc["coverage"], mc["acc"], marker="o", label=mc["model"])

    ax.set_xlabel("Coverage (fraction kept)")
    ax.set_ylabel("Accuracy on kept samples")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
