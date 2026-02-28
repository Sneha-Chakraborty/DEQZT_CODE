from __future__ import annotations

import argparse, json, os, time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from utils.io import read_parquet, time_aware_split, cross_cloud_split, resolve_data_path
from evaluation.metrics import multiclass_metrics, binary_rates, save_confusion_matrix_plot
from evaluation.plots import plot_bar

from baselines.iforest import train_isolation_forest, anomaly_risk
from baselines.static_rules import static_rule_risk
from deqzt.deqzt import dirichlet_risk, deqzt_decide, simulate_sessions


# PQC (ML-KEM / ML-DSA) is optional at runtime.
# If pqcrypto isn't installed you can still run the pipeline with --skip-pqc.


@dataclass
class SplitInfo:
    mode: str
    train_clouds: Optional[List[str]] = None
    test_cloud: Optional[str] = None
    n_total: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0


def plot_selective_curve(uncertainty: np.ndarray, correct01: np.ndarray, out_path: str, title: str):
    """Standalone selective accuracy plot to avoid dependency drift in evaluation.plots."""
    import matplotlib.pyplot as plt

    u = np.asarray(uncertainty, dtype=np.float32)
    c = np.asarray(correct01, dtype=np.int32)
    order = np.argsort(-u)  # most uncertain first
    c_sorted = c[order]
    n = len(c_sorted)
    if n == 0:
        return

    fracs = np.linspace(0.0, 1.0, 21)
    accs = []
    for f in fracs:
        k = int(round(f * n))
        keep = c_sorted[k:]
        accs.append(float(keep.mean()) if len(keep) else 0.0)

    plt.figure()
    plt.plot(fracs, accs)
    plt.xlabel("Reject fraction (most uncertain removed)")
    plt.ylabel("Accuracy on remaining")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _precision_recall_f1_from_rates(rates: Dict[str, Any]) -> Dict[str, float]:
    tp = float(rates["tp"]); fp = float(rates["fp"]); fn = float(rates["fn"]); tn = float(rates["tn"])
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return {"precision_attack": float(prec), "recall_attack": float(rec), "f1_attack": float(f1), "accuracy": float(acc)}


def tune_threshold_binary(y: np.ndarray, risk: np.ndarray, grid: List[float], fpr_max: Optional[float] = None):
    """
    Tune threshold to maximize attack-F1 on validation.
    Tie-breaker: lower FPR.
    Optional constraint: fpr <= fpr_max
    """
    best = None
    y = y.astype(int)
    risk = np.asarray(risk, dtype=np.float32)

    for tau in grid:
        pred = (risk >= float(tau)).astype(int)
        rates = binary_rates(y, pred)
        prf = _precision_recall_f1_from_rates(rates)
        if fpr_max is not None and rates["fpr"] > float(fpr_max):
            continue

        score = (prf["f1_attack"], -rates["fpr"])  # maximize f1, minimize fpr
        if best is None or score > best["score"]:
            best = {"tau": float(tau), "rates": rates, **prf, "score": score}

    # If constraint filtered everything, fall back to unconstrained
    if best is None:
        for tau in grid:
            pred = (risk >= float(tau)).astype(int)
            rates = binary_rates(y, pred)
            prf = _precision_recall_f1_from_rates(rates)
            score = (prf["f1_attack"], -rates["fpr"])
            if best is None or score > best["score"]:
                best = {"tau": float(tau), "rates": rates, **prf, "score": score}

    return best


def tune_deqzt(val_df: pd.DataFrame,
              hypotheses: List[str],
              weights: Dict[str, float],
              tau_low_grid: List[float],
              tau_high_grid: List[float],
              u_quantiles: List[float],
              fpr_max: Optional[float] = None):
    """
    Tune DEQZT decision thresholds to maximize attack-F1 on validation.
    """
    y = val_df["is_attack"].astype(int).to_numpy()
    u = val_df["uncertainty_u"].to_numpy(dtype=np.float32)
    risk = dirichlet_risk(val_df, hypotheses, weights)

    best = None
    for q in u_quantiles:
        u_thr = float(np.quantile(u, q))
        for tl in tau_low_grid:
            for th in tau_high_grid:
                if float(th) <= float(tl):
                    continue
                dec = deqzt_decide(risk, u, float(tl), float(th), u_thr)
                pred = (dec == 2).astype(int)  # deny==alert
                rates = binary_rates(y, pred)
                prf = _precision_recall_f1_from_rates(rates)

                if fpr_max is not None and rates["fpr"] > float(fpr_max):
                    continue

                score = (prf["f1_attack"], -rates["fpr"])
                if best is None or score > best["score"]:
                    best = {
                        "tau_low": float(tl),
                        "tau_high": float(th),
                        "u_thr": float(u_thr),
                        "rates": rates,
                        **prf,
                        "score": score,
                    }

    # fallback unconstrained
    if best is None:
        for q in u_quantiles:
            u_thr = float(np.quantile(u, q))
            for tl in tau_low_grid:
                for th in tau_high_grid:
                    if float(th) <= float(tl):
                        continue
                    dec = deqzt_decide(risk, u, float(tl), float(th), u_thr)
                    pred = (dec == 2).astype(int)
                    rates = binary_rates(y, pred)
                    prf = _precision_recall_f1_from_rates(rates)
                    score = (prf["f1_attack"], -rates["fpr"])
                    if best is None or score > best["score"]:
                        best = {
                            "tau_low": float(tl),
                            "tau_high": float(th),
                            "u_thr": float(u_thr),
                            "rates": rates,
                            **prf,
                            "score": score,
                        }

    return best


def _ensure_session_cols(df: pd.DataFrame, session_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in session_cols:
        if c not in df.columns:
            df[c] = "NA"
    df[session_cols] = df[session_cols].fillna("NA").astype(str)
    return df


def main(data_path: str,
         config_path: str,
         skip_pqc: bool = False,
         split_mode: str = "time",
         train_clouds: str = "",
         test_cloud: str = "",
         outdir: str = "results"):

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    hypotheses = cfg["dirichlet"]["hypotheses"]
    weights = cfg["risk_weights"]
    time_col = cfg["data"]["time_column"]
    label_col = cfg["data"]["label_column"]
    bin_col = cfg["data"]["binary_label_column"]
    session_cols = cfg["data"]["session_keys"]
    cloud_col = cfg["data"].get("cloud_column", "cloud")

    ratios = cfg["split"]
    tune = cfg["tuning"]
    contain_n = int(cfg["session_eval"]["contain_within_n_events"])
    fpr_max = tune.get("fpr_max", None)

    out_tables = os.path.join(outdir, "tables")
    out_figs = os.path.join(outdir, "figures")
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)

    print("[Load] Reading dataset...")
    resolved_data = resolve_data_path(data_path)
    print(f"[Load] Data path: {resolved_data}")
    df = read_parquet(resolved_data)

    df = _ensure_session_cols(df, session_cols)

    # -------- Split --------
    split_info = SplitInfo(mode=str(split_mode), n_total=int(len(df)))

    if split_mode == "crosscloud":
        tc = [c.strip() for c in str(train_clouds).split(",") if c.strip()]
        if not tc:
            raise ValueError("--train-clouds must be provided for --split-mode crosscloud (e.g., CIC_NET,AZURE)")
        if not str(test_cloud).strip():
            raise ValueError("--test-cloud must be provided for --split-mode crosscloud (e.g., GCP)")

        print(f"[Split] cross-cloud: train={tc} | test={test_cloud}")
        train, val, test = cross_cloud_split(df, cloud_col=cloud_col, time_col=time_col, ratios=ratios,
                                            train_clouds=tc, test_cloud=str(test_cloud))
        split_info.train_clouds = tc
        split_info.test_cloud = str(test_cloud)
    else:
        print("[Split] time-aware split...")
        train, val, test = time_aware_split(df, time_col, ratios)

    split_info.n_train = int(len(train))
    split_info.n_val = int(len(val))
    split_info.n_test = int(len(test))
    with open(os.path.join(out_tables, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(split_info), f, indent=2)

    y_true_mc = test[label_col].astype(str).to_numpy()
    y_true_bin = test[bin_col].astype(int).to_numpy()

    metric_rows = []
    thresholds = {}

    # ------------------ Isolation Forest ------------------
    print("[Train] Isolation Forest...")
    t0 = time.time()
    if_model, if_meta = train_isolation_forest(train)
    train_time = time.time() - t0

    val_risk = anomaly_risk(if_model, if_meta, val)
    best = tune_threshold_binary(val[bin_col].to_numpy(), val_risk, tune["tau_grid"], fpr_max=fpr_max)
    thresholds["isolation_forest"] = best

    test_risk_if = anomaly_risk(if_model, if_meta, test)
    y_pred_bin = (test_risk_if >= best["tau"]).astype(int)
    rates = binary_rates(y_true_bin, y_pred_bin)
    prf = _precision_recall_f1_from_rates(rates)

    y_pred_mc = np.where(y_pred_bin == 1, "NETWORK_INTRUSION", "BENIGN")
    macro_f1, recalls, cm = multiclass_metrics(y_true_mc, y_pred_mc, hypotheses)
    save_confusion_matrix_plot(cm, hypotheses, os.path.join(out_figs, "cm_iforest.png"))

    metric_rows.append({
        "model": "IsolationForest",
        "macro_f1": float(macro_f1),
        **rates,
        **prf,
        "train_time_s": float(train_time),
    })

    # ------------------ Static Rules ------------------
    print("[Eval] Static Rules...")
    val_risk = static_rule_risk(val)
    best = tune_threshold_binary(val[bin_col].to_numpy(), val_risk, tune["tau_grid"], fpr_max=fpr_max)
    thresholds["static_rules"] = best

    test_risk_sr = static_rule_risk(test)
    y_pred_bin = (test_risk_sr >= best["tau"]).astype(int)
    rates = binary_rates(y_true_bin, y_pred_bin)
    prf = _precision_recall_f1_from_rates(rates)

    y_pred_mc = np.where(y_pred_bin == 1, "NETWORK_INTRUSION", "BENIGN")
    macro_f1, recalls, cm = multiclass_metrics(y_true_mc, y_pred_mc, hypotheses)
    save_confusion_matrix_plot(cm, hypotheses, os.path.join(out_figs, "cm_static.png"))

    metric_rows.append({
        "model": "Static_Rules",
        "macro_f1": float(macro_f1),
        **rates,
        **prf,
        "train_time_s": 0.0,
    })

    # ------------------ DEQZT (Proposed) ------------------
    print("[Tune] DEQZT...")
    best_deqzt = tune_deqzt(val, hypotheses, weights,
                            tune["tau_low_grid"], tune["tau_high_grid"], tune["u_quantiles"],
                            fpr_max=fpr_max)
    thresholds["deqzt"] = best_deqzt

    print("[Eval] DEQZT...")
    risk = dirichlet_risk(test, hypotheses, weights)
    u = test["uncertainty_u"].to_numpy(dtype=np.float32)
    decisions = deqzt_decide(risk, u, best_deqzt["tau_low"], best_deqzt["tau_high"], best_deqzt["u_thr"])

    # ------------------ Save evaluation trace (for PQC-vs-classical control/security metrics) ------------------
    # This trace enables computing rotation coverage, selectivity, delay, and overhead-per-containment metrics
    # without modifying the dataset. It is safe to include in an appendix for reproducibility.
    try:
        trace = test.copy().reset_index(drop=True)
        trace["_row_index"] = np.arange(len(trace), dtype=np.int64)
        trace["_risk"] = risk.astype(np.float32)
        trace["_uncertainty_u"] = u.astype(np.float32)
        trace["_decision"] = decisions.astype(np.int8)
    
        # Ensure an attack indicator for session/control metrics.
        if "is_attack" not in trace.columns and bin_col in trace.columns:
            trace["is_attack"] = trace[bin_col].astype(int)
    
        # Build a stable session id string from configured session keys (handles NaN/float).
        sid_df = trace[session_cols].copy()
        for c in session_cols:
            sid_df[c] = sid_df[c].fillna("NA")
            sid_df[c] = sid_df[c].apply(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))
        trace["_sid"] = sid_df.astype(str).agg("|".join, axis=1)
    
        # Best-effort timestamp column for delay metrics.
        if "ts" in trace.columns:
            trace["_ts"] = pd.to_datetime(trace["ts"], errors="coerce", utc=True)
        elif time_col in trace.columns:
            trace["_ts"] = pd.to_datetime(trace[time_col], errors="coerce", utc=True)
    
        keep = [
            "_row_index", "_sid", "_ts",
            cloud_col,
            "is_attack",
            label_col if label_col in trace.columns else None,
            bin_col if bin_col in trace.columns else None,
            "principal_id" if "principal_id" in trace.columns else None,
            "_risk", "_uncertainty_u", "_decision",
        ]
        keep = [c for c in keep if c is not None and c in trace.columns]
        trace[keep].to_csv(os.path.join(out_tables, "eval_trace.csv"), index=False)
        print(f"[Trace] Wrote {os.path.join(out_tables, 'eval_trace.csv')}")
    except Exception as e:
        print(f"[WARN] Could not write eval_trace.csv: {e}")
    
    
    # Event-level DEQZT binary metrics
    y_pred_bin = (decisions == 2).astype(int)
    rates = binary_rates(y_true_bin, y_pred_bin)
    prf = _precision_recall_f1_from_rates(rates)

    # Multi-class prediction from probability columns if available; otherwise fallback to binary mapping
    p_cols = [f"p__{h}" for h in hypotheses if f"p__{h}" in test.columns]
    if len(p_cols) == len(hypotheses):
        probs = test[p_cols].to_numpy(dtype=np.float32)
        y_pred_mc = np.array([hypotheses[i] for i in np.argmax(probs, axis=1)])
    else:
        y_pred_mc = np.where(y_pred_bin == 1, "NETWORK_INTRUSION", "BENIGN")

    macro_f1, recalls, cm = multiclass_metrics(y_true_mc, y_pred_mc, hypotheses)
    save_confusion_matrix_plot(cm, hypotheses, os.path.join(out_figs, "cm_deqzt.png"))

    metric_rows.append({
        "model": "DEQZT",
        "macro_f1": float(macro_f1),
        **rates,
        **prf,
        "train_time_s": 0.0,
    })

    correct = (y_pred_mc == y_true_mc).astype(int)
    plot_selective_curve(u, correct, os.path.join(out_figs, "deqzt_selective_accuracy.png"),
                         "DEQZT selective accuracy (by uncertainty)")

    # ------------------ PQC Session Rotation (STEP-UP) ------------------
    if skip_pqc:
        print("[Security] --skip-pqc enabled: skipping PQC session rotation.")
        pqc_df = pd.DataFrame([])
        pqc_summary = {
            "rotations_tested": 0,
            "avg_rotation_time_ms": 0.0,
            "avg_ciphertext_bytes": 0.0,
            "avg_signature_bytes": 0.0,
            "verify_ok_rate": 0.0,
            "skipped": True,
        }
    else:
        print("[Security] Applying PQC session rotations...")
        try:
            from pqc.zt_session import rotate_session
        except ModuleNotFoundError as e:
            print(f"[WARN] PQC dependencies missing ({e}). Install with: pip install -r requirements.txt, or rerun with --skip-pqc.")
            rotate_session = None

        stepup_idx = np.where(decisions == 1)[0]
        max_rotations = min(int(os.environ.get("DEQZT_PQC_MAX", "200")), len(stepup_idx))
        sample_idx = stepup_idx[:max_rotations]

        pqc_events = []
        if rotate_session is not None:
            for i in sample_idx:
                row = test.iloc[int(i)]
                subject = str(row.get("principal_id", "NA"))
                cloud = str(row.get(cloud_col, "NA"))
                ctx_hash = f"{subject}|{cloud}|{int(i)}"
                pqc_info = rotate_session(subject=subject, cloud=cloud, decision="STEP_UP", context_hash=ctx_hash)
                pqc_info["row_index"] = int(i)
                pqc_events.append(pqc_info)

        pqc_df = pd.DataFrame(pqc_events)
        pqc_summary = {
            "rotations_tested": int(len(pqc_df)),
            "avg_rotation_time_ms": float(pqc_df["rotation_time_ms"].mean()) if len(pqc_df) else 0.0,
            "avg_ciphertext_bytes": float(pqc_df["ciphertext_len"].mean()) if len(pqc_df) else 0.0,
            "avg_signature_bytes": float(pqc_df["signature_len"].mean()) if len(pqc_df) else 0.0,
            "verify_ok_rate": float(pqc_df["verify_ok"].mean()) if len(pqc_df) else 0.0,
            "skipped": bool(rotate_session is None),
        }

    pqc_df.to_csv(os.path.join(out_tables, "pqc_rotations.csv"), index=False)
    with open(os.path.join(out_tables, "pqc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(pqc_summary, f, indent=2)

    # ------------------ Session-level evaluation ------------------
    sess_rows = []

    def as_dec_from_alert(alert_bin):
        return np.where(alert_bin == 1, 2, 0).astype(np.int8)

    if_alert = (test_risk_if >= thresholds["isolation_forest"]["tau"]).astype(int)
    sess_rows.append({"model": "IsolationForest",
                      **simulate_sessions(test, session_cols, as_dec_from_alert(if_alert), contain_n)})

    sr_alert = (test_risk_sr >= thresholds["static_rules"]["tau"]).astype(int)
    sess_rows.append({"model": "Static_Rules",
                      **simulate_sessions(test, session_cols, as_dec_from_alert(sr_alert), contain_n)})

    sess_rows.append({"model": "DEQZT",
                      **simulate_sessions(test, session_cols, decisions, contain_n)})

    pd.DataFrame(sess_rows).to_csv(os.path.join(out_tables, "metrics_session_level.csv"), index=False)

    # ------------------ Save plots + tables ------------------
    pd.DataFrame(metric_rows).to_csv(os.path.join(out_tables, "metrics_event_level.csv"), index=False)
    plot_bar(metric_rows, "macro_f1", os.path.join(out_figs, "macro_f1_bar.png"), "Macro-F1 Comparison")
    plot_bar(metric_rows, "f1_attack", os.path.join(out_figs, "f1_attack_bar.png"), "Binary Attack-F1 Comparison")
    plot_bar(metric_rows, "fpr", os.path.join(out_figs, "fpr_bar.png"), "Binary FPR Comparison")
    plot_bar(metric_rows, "tpr", os.path.join(out_figs, "tpr_bar.png"), "Binary TPR Comparison")

    with open(os.path.join(out_tables, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print("[DONE] DEQZT pipeline executed successfully (baselines: IsolationForest, Static_Rules)!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-pqc", action="store_true", help="Skip PQC session-rotation step (allows running without pqcrypto installed).")
    ap.add_argument("--split-mode", choices=["time", "crosscloud"], default="time")
    ap.add_argument("--train-clouds", default="", help="Comma-separated train clouds for crosscloud split, e.g. CIC_NET,AZURE")
    ap.add_argument("--test-cloud", default="", help="Test cloud for crosscloud split, e.g. GCP")
    ap.add_argument("--outdir", default="results", help="Output folder (default: results)")
    args = ap.parse_args()

    main(
        args.data,
        args.config,
        skip_pqc=bool(args.skip_pqc),
        split_mode=str(args.split_mode),
        train_clouds=str(args.train_clouds),
        test_cloud=str(args.test_cloud),
        outdir=str(args.outdir),
    )
