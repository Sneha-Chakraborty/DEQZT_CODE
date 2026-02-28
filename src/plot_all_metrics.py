import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def norm_model_name(s: str) -> str:
    """Normalize model names across csv/json."""
    s = str(s).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in {"isolationforest", "isolation_forest", "iforest", "iso_forest"}:
        return "IsolationForest"
    if s in {"static_rules", "rules", "staticrules"}:
        return "Static_Rules"
    if s in {"deqzt", "deq_zt", "deq-zt"}:
        return "DEQZT"
    # fallback: title-ish
    return str(s)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def f1(p: float, r: float) -> float:
    return (2 * p * r) / (p + r) if (p + r) else 0.0


def macro_f1_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    # Attack as positive
    p_att = safe_div(tp, tp + fp)
    r_att = safe_div(tp, tp + fn)
    f1_att = f1(p_att, r_att)

    # Benign as positive (one-vs-rest)
    tp_b = tn
    fp_b = fn
    fn_b = fp
    p_b = safe_div(tp_b, tp_b + fp_b)
    r_b = safe_div(tp_b, tp_b + fn_b)
    f1_b = f1(p_b, r_b)

    return (f1_att + f1_b) / 2.0


def load_thresholds(thresholds_path: Path) -> pd.DataFrame:
    data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    rows = []
    for k, v in data.items():
        model = norm_model_name(k)
        fpr = None
        f1_attack = None

        # preferred keys
        if isinstance(v, dict):
            if "rates" in v and isinstance(v["rates"], dict):
                fpr = v["rates"].get("fpr", None)
            f1_attack = v.get("f1_attack", None)

        if fpr is None or f1_attack is None:
            # try to compute from rates if present
            rates = (v.get("rates") if isinstance(v, dict) else None) or {}
            tp = rates.get("tp"); fp = rates.get("fp"); tn = rates.get("tn"); fn = rates.get("fn")
            if f1_attack is None and all(x is not None for x in [tp, fp, fn]):
                p = safe_div(tp, tp + fp)
                r = safe_div(tp, tp + fn)
                f1_attack = f1(p, r)
            if fpr is None and all(x is not None for x in [fp, tn]):
                fpr = safe_div(fp, fp + tn)

        rows.append({"model": model, "fpr": float(fpr), "f1_attack": float(f1_attack)})

    df = pd.DataFrame(rows)
    return df


def load_macro_f1(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if "model" not in df.columns:
        raise ValueError(f"{metrics_csv} must contain a 'model' column")

    df["model"] = df["model"].apply(norm_model_name)

    if "macro_f1" in df.columns:
        df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    else:
        # recompute if missing
        needed = {"tp", "fp", "tn", "fn"}
        if not needed.issubset(set(df.columns)):
            raise ValueError(f"{metrics_csv} must contain macro_f1 OR tp/fp/tn/fn columns")
        df["macro_f1"] = df.apply(
            lambda r: macro_f1_from_confusion(int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])),
            axis=1,
        )

    return df[["model", "macro_f1"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", required=True, help="Path to thresholds.json (binary fpr + f1_attack)")
    ap.add_argument("--event-metrics", required=True, help="Path to metrics_event_level.csv (macro_f1)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="DEQ-ZT vs Baselines (Binary + Macro Metrics)", help="Figure title")
    args = ap.parse_args()

    thresholds_path = Path(args.thresholds).resolve()
    metrics_path = Path(args.event_metrics).resolve()

    df_thr = load_thresholds(thresholds_path)
    df_mac = load_macro_f1(metrics_path)

    df = df_thr.merge(df_mac, on="model", how="inner")

    # enforce preferred order if present
    preferred = ["IsolationForest", "Static_Rules", "DEQZT"]
    df["order"] = df["model"].apply(lambda m: preferred.index(m) if m in preferred else 999)
    df = df.sort_values(["order", "model"]).drop(columns=["order"]).reset_index(drop=True)

    models = df["model"].tolist()

    # One figure, three panels (same x-axis)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(args.title, fontsize=16)

    # 1) Attack-F1
    axes[0].bar(models, df["f1_attack"])
    axes[0].set_ylabel("F1_attack")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Binary Attack-F1 (higher is better)")
    for i, v in enumerate(df["f1_attack"]):
        axes[0].text(i, float(v) + 0.01, f"{float(v):.4f}", ha="center", va="bottom", fontsize=10)

    # 2) FPR
    axes[1].bar(models, df["fpr"])
    axes[1].set_ylabel("FPR")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Binary False Positive Rate (lower is better)")
    for i, v in enumerate(df["fpr"]):
        axes[1].text(i, float(v) + 0.01, f"{float(v):.4f}", ha="center", va="bottom", fontsize=10)

    # 3) Macro-F1
    axes[2].bar(models, df["macro_f1"])
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Macro-F1 (higher is better)")
    for i, v in enumerate(df["macro_f1"]):
        axes[2].text(i, float(v) + 0.01, f"{float(v):.4f}", ha="center", va="bottom", fontsize=10)

    axes[2].set_xticklabels(models, rotation=20, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()