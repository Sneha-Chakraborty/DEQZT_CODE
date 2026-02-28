# plot_macro_f1.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0


def macro_f1_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    # Positive class (Attack)
    p_pos = safe_div(tp, tp + fp)
    r_pos = safe_div(tp, tp + fn)
    f1_pos = f1(p_pos, r_pos)

    # Negative class (Benign) treated as "positive" for its own one-vs-rest
    # For benign class:
    #   TP_benign = TN
    #   FP_benign = FN  (predicted benign but actually attack)
    #   FN_benign = FP  (predicted attack but actually benign)
    tp_b = tn
    fp_b = fn
    fn_b = fp
    p_b = safe_div(tp_b, tp_b + fp_b)
    r_b = safe_div(tp_b, tp_b + fn_b)
    f1_b = f1(p_b, r_b)

    return (f1_pos + f1_b) / 2.0


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize model names (optional)
    if "model" not in df.columns:
        raise ValueError("CSV must contain a 'model' column.")

    if "macro_f1" not in df.columns:
        required = {"tp", "fp", "tn", "fn"}
        if not required.issubset(set(df.columns)):
            raise ValueError(
                "CSV must contain 'macro_f1' OR the confusion columns: tp, fp, tn, fn."
            )
        df["macro_f1"] = df.apply(
            lambda r: macro_f1_from_confusion(int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])),
            axis=1,
        )

    # Ensure numeric
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce").fillna(0.0)

    return df


def plot_macro_f1(df: pd.DataFrame, out_path: Path | None, title: str):
    # Sort descending
    df_sorted = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    models = df_sorted["model"].astype(str).tolist()
    scores = df_sorted["macro_f1"].to_numpy(dtype=float)

    best = np.max(scores) if len(scores) else 0.0
    is_best = np.isclose(scores, best, rtol=0, atol=1e-12)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores)

    # Highlight best (ties allowed) without changing colors
    for b, flag in zip(bars, is_best):
        if flag:
            b.set_hatch("//")
            b.set_linewidth(1.5)

    # Labels & layout
    plt.title(title)
    plt.ylabel("macro_f1")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=25, ha="right")

    # Annotate values on bars
    for b, v in zip(bars, scores):
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.01,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"[OK] Saved plot to: {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="Path to metrics_event_level.csv")
    ap.add_argument("--out", default="", help="Output PNG path (e.g., results/figures/macro_f1.png). If empty, shows window.")
    ap.add_argument("--title", default="Macro-F1 Comparison", help="Plot title")
    args = ap.parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = load_metrics(metrics_path)
    out_path = Path(args.out).expanduser().resolve() if args.out else None
    plot_macro_f1(df, out_path, args.title)


if __name__ == "__main__":
    main()