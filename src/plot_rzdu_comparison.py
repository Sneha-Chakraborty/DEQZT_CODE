from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to metrics_event_level.csv")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Risk-Aware Zero-Trust Decision Utility (RZDU) Comparison")
    args = ap.parse_args()

    metrics_path = Path(args.metrics).resolve()
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if "model" not in df.columns or "rzdu" not in df.columns:
        raise ValueError("metrics_event_level.csv must contain 'model' and 'rzdu' columns.")

    order_map = {m: i for i, m in enumerate(PREFERRED_MODEL_ORDER)}
    df["__ord"] = df["model"].astype(str).map(lambda x: order_map.get(x, 999))
    df = df.sort_values(["__ord", "model"]).drop(columns=["__ord"]).reset_index(drop=True)

    names = [DISPLAY_MODEL_NAMES.get(str(x), str(x)) for x in df["model"].astype(str).tolist()]
    vals = df["rzdu"].astype(float).to_numpy()

    fig = plt.figure(figsize=(max(8, len(names) * 1.2), 6))
    ax = fig.add_subplot(111)
    bars = ax.bar(names, vals)
    ax.set_ylim(0.0, min(1.0, float(vals.max()) * 1.20 + 0.05))
    ax.set_ylabel("RZDU")
    ax.set_title(args.title)
    ax.tick_params(axis="x", rotation=25)

    best = float(vals.max()) if len(vals) else 0.0
    off = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    for b, v in zip(bars, vals):
        if abs(v - best) < 1e-12:
            b.set_hatch("//")
            b.set_linewidth(1.5)
        ax.text(b.get_x() + b.get_width()/2.0, v + off, f"{v:.6f}\n({v*100:.2f}%)", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()
