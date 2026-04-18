from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from baselines.iforest import anomaly_risk, train_isolation_forest
from baselines.sklearn_models import (
    predict_proba_df,
    select_feature_cols,
    train_logistic_regression,
    train_mlp_softmax,
    train_random_forest,
)
from baselines.static_rules import static_rule_risk
from deqzt.deqzt import (
    deqzt_decide,
    dirichlet_risk,
    scalar_decide,
    simulate_sessions,
    temporal_dirichlet_update,
    uncertainty_weighted_risk,
)
from deqzt.edl_pipeline import EDLArtifacts, predict_edl, train_edl
from evaluation.metrics import (
    binary_summary,
    confusion_matrix_df,
    multiclass_metrics,
    multiclass_summary,
    rzdu_score,
    save_confusion_matrix_plot,
)
from evaluation.plots import plot_bar
from pqc.benchmark_pqc import bench_pqc
from utils.io import (
    cross_cloud_split,
    drop_exact_duplicates,
    enforce_min_label_support,
    read_parquet,
    resolve_data_path,
    stratified_group_time_aware_split,
    stratified_time_aware_split,
    summarize_label_distribution,
    time_aware_split,
)
from utils.quality import leakage_report


@dataclass
class SplitInfo:
    mode: str
    train_clouds: Optional[List[str]] = None
    test_cloud: Optional[str] = None
    n_total: int = 0
    n_after_dedup: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0


def plot_selective_curve(uncertainty: np.ndarray, correct01: np.ndarray, out_path: str, title: str):
    import matplotlib.pyplot as plt

    u = np.asarray(uncertainty, dtype=np.float32)
    c = np.asarray(correct01, dtype=np.int32)
    order = np.argsort(-u)
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
    plt.figure(figsize=(8, 5))
    plt.plot(fracs, accs, marker="o")
    plt.xlabel("Reject fraction (most uncertain removed)")
    plt.ylabel("Accuracy on remaining")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _ensure_session_cols(df: pd.DataFrame, session_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in session_cols:
        if c not in out.columns:
            out[c] = "NA"
    return out


def _probs_to_pred_labels(probs: np.ndarray, hypotheses: Sequence[str]) -> np.ndarray:
    idx = np.argmax(probs, axis=1).astype(np.int64)
    return np.array([hypotheses[int(i)] for i in idx])


def _attach_prob_columns(df: pd.DataFrame, probs: np.ndarray, hypotheses: List[str], uncertainty_fill: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for k, h in enumerate(hypotheses):
        out[f"p__{h}"] = probs[:, k].astype(np.float32)
    out["uncertainty_u"] = float(uncertainty_fill)
    return out


def _build_trace(
    test: pd.DataFrame,
    base_risk: np.ndarray,
    effective_risk: np.ndarray,
    u: np.ndarray,
    decisions: np.ndarray,
    session_cols: List[str],
    *,
    label_col: str,
    bin_col: str,
    cloud_col: str,
    time_col: str,
) -> pd.DataFrame:
    trace = test.copy().reset_index(drop=True)
    trace["_row_index"] = np.arange(len(trace), dtype=np.int64)
    trace["_base_risk"] = base_risk.astype(np.float32)
    trace["_risk"] = effective_risk.astype(np.float32)
    trace["_uncertainty_u"] = u.astype(np.float32)
    trace["_decision"] = decisions.astype(np.int8)
    if "is_attack" not in trace.columns and bin_col in trace.columns:
        trace["is_attack"] = trace[bin_col].astype(int)
    sid_df = trace[session_cols].copy()
    for c in session_cols:
        sid_df[c] = sid_df[c].fillna("NA").astype(str)
    trace["_sid"] = sid_df.astype(str).agg("|".join, axis=1)
    if time_col in trace.columns:
        trace["_ts"] = pd.to_datetime(trace[time_col], errors="coerce", utc=True)
    keep = [
        "_row_index", "_sid", "_ts", cloud_col, "is_attack", label_col if label_col in trace.columns else None,
        bin_col if bin_col in trace.columns else None, "principal_id" if "principal_id" in trace.columns else None,
        "_base_risk", "_risk", "_uncertainty_u", "_decision",
    ]
    keep = [c for c in keep if c is not None and c in trace.columns]
    return trace[keep]


def _prepare_tuning_context(y_true_bin: np.ndarray, y_true_labels: np.ndarray) -> Dict[str, np.ndarray]:
    labels = np.asarray(y_true_labels).astype(str)
    yb = np.asarray(y_true_bin).astype(np.int8)
    ideal = np.zeros(len(labels), dtype=np.int8)
    severity_weight = np.ones(len(labels), dtype=np.float32)
    under_beta = np.ones(len(labels), dtype=np.float32)
    for i, lab in enumerate(labels):
        ul = str(lab).upper()
        if ul == "BENIGN":
            ideal[i] = 0
            severity_weight[i] = 1.0
            under_beta[i] = 0.0
        elif ul in {"NETWORK_INTRUSION", "CREDENTIAL_MISUSE"}:
            ideal[i] = 1
            severity_weight[i] = 2.0 if ul == "NETWORK_INTRUSION" else 3.0
            under_beta[i] = 1.0 if ul == "NETWORK_INTRUSION" else 1.2
        else:
            ideal[i] = 2
            severity_weight[i] = 4.0 if ul != "DATA_EXFILTRATION" else 5.0
            under_beta[i] = 2.0 if ul == "DATA_EXFILTRATION" else 1.5
    return {"y_true_bin": yb, "ideal": ideal, "severity_weight": severity_weight, "under_beta": under_beta}


def _fast_binary_summary(y_true_bin: np.ndarray, pred_attack: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true_bin).astype(np.int8)
    p = np.asarray(pred_attack).astype(np.int8)
    tp = float(np.sum((p == 1) & (y == 1)))
    fp = float(np.sum((p == 1) & (y == 0)))
    fn = float(np.sum((p == 0) & (y == 1)))
    tn = float(np.sum((p == 0) & (y == 0)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    tpr = recall
    return {"f1_attack": float(f1), "fpr": float(fpr), "tpr": float(tpr)}


def _fast_rzdu(ideal: np.ndarray, decisions: np.ndarray, severity_weight: np.ndarray, under_beta: np.ndarray) -> float:
    d = np.asarray(decisions, dtype=np.int8)
    dist = np.abs(d - ideal).astype(np.float32)
    over_mask = d > ideal
    penalty = np.where(over_mask, 0.5 * dist, under_beta * dist)
    penalty = np.where(d == ideal, 0.0, penalty)
    weighted_penalty = float(np.sum(severity_weight * penalty))
    max_penalty = float(np.sum(severity_weight * np.maximum(under_beta, 0.5) * 2.0))
    score = 1.0 - (weighted_penalty / (max_penalty + 1e-9))
    return float(np.clip(score, 0.0, 1.0))


def _tune_score(ctx: Dict[str, np.ndarray], decisions: np.ndarray) -> tuple[float, float, float, float, float]:
    pred_attack = (np.asarray(decisions) >= 1).astype(np.int8)
    b = _fast_binary_summary(ctx["y_true_bin"], pred_attack)
    rz = _fast_rzdu(ctx["ideal"], np.asarray(decisions, dtype=np.int8), ctx["severity_weight"], ctx["under_beta"])
    intervention_rate = float(np.mean(np.asarray(decisions) >= 1))
    deny_rate = float(np.mean(np.asarray(decisions) == 2))
    composite = 0.45 * float(b["f1_attack"]) + 0.30 * float(rz) + 0.15 * float(b["tpr"]) - 0.20 * float(b["fpr"]) - 0.05 * intervention_rate
    return (float(composite), float(b["f1_attack"]), float(rz), -float(b["fpr"]), -float(intervention_rate - deny_rate))


def tune_scalar_zt(
    y_true_bin: np.ndarray,
    y_true_labels: np.ndarray,
    risk: np.ndarray,
    tau_low_grid: Sequence[float],
    tau_high_grid: Sequence[float],
    *,
    fpr_max: Optional[float] = None,
    detect_on_stepup: bool = True,
):
    best: Optional[Dict[str, Any]] = None
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_true_labels = np.asarray(y_true_labels).astype(str)
    risk = np.asarray(risk, dtype=np.float32)
    ctx = _prepare_tuning_context(y_true_bin, y_true_labels)
    for tl in tau_low_grid:
        for th in tau_high_grid:
            if float(th) <= float(tl):
                continue
            dec = scalar_decide(risk, float(tl), float(th))
            pred = (dec >= 1).astype(int) if detect_on_stepup else (dec == 2).astype(int)
            b = binary_summary(y_true_bin, pred)
            if fpr_max is not None and float(b["fpr"]) > float(fpr_max):
                continue
            score = _tune_score(ctx, dec)
            cand = {"tau_low": float(tl), "tau_high": float(th), "binary": b, "rzdu": rzdu_score(y_true_labels, dec), "score": score}
            if best is None or score > best["score"]:
                best = cand
    if best is None:
        for tl in tau_low_grid:
            for th in tau_high_grid:
                if float(th) <= float(tl):
                    continue
                dec = scalar_decide(risk, float(tl), float(th))
                score = _tune_score(ctx, dec)
                cand = {"tau_low": float(tl), "tau_high": float(th), "binary": binary_summary(y_true_bin, (dec >= 1).astype(int)), "rzdu": rzdu_score(y_true_labels, dec), "score": score}
                if best is None or score > best["score"]:
                    best = cand
    return best


def tune_deqzt(
    val_df: pd.DataFrame,
    hypotheses: List[str],
    weights: Dict[str, float],
    tau_low_grid: Sequence[float],
    tau_high_grid: Sequence[float],
    u_quantiles: Sequence[float],
    beta_u_grid: Sequence[float],
    min_stepup_risk_grid: Sequence[float],
    *,
    label_col: str,
    bin_col: str,
    fpr_max: Optional[float] = None,
    detect_on_stepup: bool = True,
):
    y_true_bin = val_df[bin_col].astype(int).to_numpy()
    y_true_labels = val_df[label_col].astype(str).to_numpy()
    ctx = _prepare_tuning_context(y_true_bin, y_true_labels)
    u = val_df["uncertainty_u"].to_numpy(dtype=np.float32)
    base_risk = dirichlet_risk(val_df, hypotheses, weights)
    unique_u_thr = sorted(set(float(np.quantile(u, q)) for q in u_quantiles))
    best: Optional[Dict[str, Any]] = None
    for beta_u in beta_u_grid:
        eff_risk = uncertainty_weighted_risk(base_risk, u, float(beta_u))
        for u_thr in unique_u_thr:
            for min_stepup_risk in min_stepup_risk_grid:
                for tl in tau_low_grid:
                    for th in tau_high_grid:
                        if float(th) <= float(tl):
                            continue
                        dec = deqzt_decide(eff_risk, u, float(tl), float(th), u_thr, min_stepup_risk=float(min_stepup_risk))
                        pred = (dec >= 1).astype(int) if detect_on_stepup else (dec == 2).astype(int)
                        b = binary_summary(y_true_bin, pred)
                        if fpr_max is not None and float(b["fpr"]) > float(fpr_max):
                            continue
                        score = _tune_score(ctx, dec)
                        cand = {
                            "beta_u": float(beta_u),
                            "tau_low": float(tl),
                            "tau_high": float(th),
                            "u_thr": float(u_thr),
                            "min_stepup_risk": float(min_stepup_risk),
                            "binary": b,
                            "rzdu": rzdu_score(y_true_labels, dec),
                            "score": score,
                        }
                        if best is None or score > best["score"]:
                            best = cand
    if best is None:
        raise RuntimeError("No valid DEQZT threshold configuration found.")
    return best


def _save_per_cloud_confusions(
    name: str,
    test_df: pd.DataFrame,
    y_pred_mc: np.ndarray,
    hypotheses: List[str],
    cloud_col: str,
    out_tables: str,
    out_figs: str,
    label_col: str,
) -> None:
    if cloud_col not in test_df.columns or label_col not in test_df.columns:
        return
    tmp = test_df.copy().reset_index(drop=True)
    tmp["_pred_mc"] = np.asarray(y_pred_mc).astype(str)
    for cloud, g in tmp.groupby(cloud_col, sort=True):
        if len(g) == 0:
            continue
        y_true = g[label_col].astype(str).to_numpy()
        present = [lab for lab in hypotheses if np.any(y_true == str(lab))]
        if len(present) < 2:
            continue
        _, _, cm, labels_used = multiclass_metrics(y_true, g["_pred_mc"].to_numpy(), hypotheses)
        safe_cloud = str(cloud).replace("/", "_").replace("\\", "_").replace(" ", "_")
        confusion_matrix_df(cm, labels_used).to_csv(os.path.join(out_tables, f"cm_{name.lower()}__cloud_{safe_cloud}.csv"))
        save_confusion_matrix_plot(cm, labels_used, os.path.join(out_figs, f"cm_{name.lower()}__cloud_{safe_cloud}.png"), title=f"Confusion Matrix - {name} - {cloud}")


def _evaluate_model(
    *,
    name: str,
    y_true_mc: np.ndarray,
    y_true_bin: np.ndarray,
    y_pred_mc: np.ndarray,
    decisions: np.ndarray,
    hypotheses: List[str],
    train_time_s: float,
    out_figs: str,
    out_tables: str,
) -> Dict[str, Dict[str, Any]]:
    label_attack_pred = (np.asarray(y_pred_mc).astype(str) != str(hypotheses[0])).astype(int)
    decision_attack_pred = (np.asarray(decisions) >= 1).astype(int)
    deny_pred = (np.asarray(decisions) == 2).astype(int)

    binary_label = binary_summary(y_true_bin, label_attack_pred)
    binary_policy = binary_summary(y_true_bin, decision_attack_pred)
    binary_deny = binary_summary(y_true_bin, deny_pred)

    macro_f1, recalls, cm, labels_used = multiclass_metrics(y_true_mc, y_pred_mc, hypotheses)
    mc_summary, _ = multiclass_summary(y_true_mc, y_pred_mc, hypotheses)
    rzdu = rzdu_score(y_true_mc, decisions)

    cm_df = confusion_matrix_df(cm, labels_used)
    cm_df.to_csv(os.path.join(out_tables, f"cm_{name.lower()}.csv"))
    save_confusion_matrix_plot(cm, labels_used, os.path.join(out_figs, f"cm_{name.lower()}.png"), title=f"Confusion Matrix - {name}")

    multiclass_row: Dict[str, Any] = {
        "model": name,
        "macro_f1": float(macro_f1),
        **mc_summary,
        "train_time_s": float(train_time_s),
    }
    for lab, val in recalls.items():
        multiclass_row[f"recall_present_{str(lab).lower()}"] = float(val)

    binary_policy_row = {"model": name, **binary_policy, **rzdu, "train_time_s": float(train_time_s)}
    binary_label_row = {"model": name, **binary_label, "train_time_s": float(train_time_s)}
    binary_deny_row = {"model": name, **binary_deny, "train_time_s": float(train_time_s)}
    overview_row = {
        "model": name,
        "macro_f1": float(macro_f1),
        "macro_precision": float(mc_summary["macro_precision"]),
        "macro_recall": float(mc_summary["macro_recall"]),
        "balanced_accuracy": float(mc_summary["balanced_accuracy"]),
        "label_f1_attack": float(binary_label["f1_attack"]),
        "decision_f1_attack": float(binary_policy["f1_attack"]),
        "decision_precision_attack": float(binary_policy["precision_attack"]),
        "decision_recall_attack": float(binary_policy["recall_attack"]),
        "decision_specificity": float(binary_policy["specificity"]),
        "decision_mcc": float(binary_policy["mcc"]),
        "decision_fpr": float(binary_policy["fpr"]),
        "decision_tpr": float(binary_policy["tpr"]),
        "deny_only_f1_attack": float(binary_deny["f1_attack"]),
        "rzdu": float(rzdu["rzdu"]),
        "train_time_s": float(train_time_s),
    }
    return {
        "multiclass": multiclass_row,
        "binary_policy": binary_policy_row,
        "binary_label": binary_label_row,
        "binary_deny": binary_deny_row,
        "overview": overview_row,
    }


def main(
    data_path: str,
    config_path: str,
    skip_pqc: bool = False,
    split_mode: str = "time",
    train_clouds: str = "",
    test_cloud: str = "",
    outdir: str = "results",
    model: str = "edl",
    edl_artifacts_dir: Optional[str] = None,
    edl_retrain: bool = False,
    edl_device: Optional[str] = None,
):
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
    detect_on_stepup = bool(tune.get("detect_on_stepup", True))
    beta_u_grid = tune.get("beta_u_grid", [cfg.get("temporal", {}).get("beta_u", 0.35)])
    min_stepup_risk_grid = tune.get("min_stepup_risk_grid", [0.00, 0.05, 0.10])
    lambda_grid = cfg.get("temporal", {}).get("lambda_grid", [float(cfg.get("temporal", {}).get("lambda", 0.85))])
    min_test_count = int(cfg.get("split_quality", {}).get("min_test_count_per_label", 1))
    min_val_count = int(cfg.get("split_quality", {}).get("min_val_count_per_label", 1))
    use_group_split = bool(cfg.get("split_quality", {}).get("use_group_split", True))
    deduplicate = bool(cfg.get("split_quality", {}).get("deduplicate_before_split", True))
    edl_cfg = cfg.get("edl", {})
    out_tables = os.path.join(outdir, "tables")
    out_figs = os.path.join(outdir, "figures")
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)

    print("[Load] Reading dataset...")
    resolved_data = resolve_data_path(data_path)
    print(f"[Load] Data path: {resolved_data}")
    raw_df = _ensure_session_cols(read_parquet(resolved_data), session_cols)
    missing_required = [c for c in [label_col, bin_col] if c not in raw_df.columns]
    if missing_required:
        preview = ", ".join(list(map(str, raw_df.columns[:40])))
        raise ValueError(
            "Dataset is missing required label columns: " + ", ".join(missing_required) + "\n\n"
            + f"Config expects label_column={label_col}, binary_label_column={bin_col}.\n"
            + "Columns preview: " + preview
        )
    raw_df[label_col] = raw_df[label_col].fillna(hypotheses[0]).astype(str)
    raw_df[bin_col] = raw_df[bin_col].astype(int)

    split_info = SplitInfo(mode=str(split_mode), n_total=int(len(raw_df)))
    if deduplicate:
        dedup_cols = [c for c in raw_df.columns if c not in {time_col, label_col, bin_col}]
        df, dedup_stats = drop_exact_duplicates(raw_df, subset=dedup_cols)
        pd.DataFrame([dedup_stats]).to_csv(os.path.join(out_tables, "dedup_stats.csv"), index=False)
    else:
        df = raw_df.copy()
        dedup_stats = {"rows_before": int(len(raw_df)), "rows_after": int(len(raw_df)), "rows_dropped": 0}
    split_info.n_after_dedup = int(len(df))

    print("[Split] building train/val/test...")
    if split_mode == "crosscloud":
        tc = [c.strip() for c in str(train_clouds).split(",") if c.strip()]
        if not tc:
            raise ValueError("--train-clouds must be provided for --split-mode crosscloud")
        if not str(test_cloud).strip():
            raise ValueError("--test-cloud must be provided for --split-mode crosscloud")
        train, val, test = cross_cloud_split(
            df,
            cloud_col=cloud_col,
            time_col=time_col,
            ratios=ratios,
            train_clouds=tc,
            test_cloud=str(test_cloud),
            label_col=label_col,
            group_cols=session_cols if use_group_split else None,
        )
        split_info.train_clouds = tc
        split_info.test_cloud = str(test_cloud)
    else:
        if use_group_split:
            train, val, test = stratified_group_time_aware_split(df, time_col, ratios, label_col, session_cols)
        else:
            train, val, test = stratified_time_aware_split(df, time_col, ratios, label_col)
    train, val, test = enforce_min_label_support(train, val, test, label_col, min_test_count=min_test_count, min_val_count=min_val_count)

    split_info.n_train = int(len(train))
    split_info.n_val = int(len(val))
    split_info.n_test = int(len(test))
    with open(os.path.join(out_tables, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(split_info), f, indent=2)
    summarize_label_distribution(train, val, test, label_col).to_csv(os.path.join(out_tables, "split_label_distribution.csv"), index=False)

    model = str(model).lower().strip()
    if model not in ("edl", "legacy"):
        raise ValueError(f"--model must be one of: edl, legacy (got: {model})")

    baseline_feature_cols = select_feature_cols(pd.concat([train, val], axis=0, ignore_index=True), prefixes=tuple(edl_cfg.get("feature_prefixes", ["e__", "f__"])), max_features=int(edl_cfg.get("max_features", 256)))
    pd.DataFrame({"feature_col": baseline_feature_cols}).to_csv(os.path.join(out_tables, "selected_feature_columns.csv"), index=False)
    leakage_report(train, val, test, label_col=label_col, bin_col=bin_col, feature_cols=baseline_feature_cols, session_cols=session_cols, out_dir=out_tables)

    edl_train_time = 0.0
    if model == "edl":
        art_dir = edl_artifacts_dir or os.path.join(outdir, "models", "edl")
        device = edl_device or str(edl_cfg.get("device", "cpu"))
        meta_path = os.path.join(art_dir, "meta.json")
        need_train = bool(edl_retrain) or (not os.path.exists(meta_path))
        if need_train:
            print(f"[EDL] Training evidential model (artifacts -> {art_dir}) ...")
            t0 = time.time()
            artifacts, _ = train_edl(
                train,
                val,
                label_col=label_col,
                hypotheses=hypotheses,
                prefixes=tuple(edl_cfg.get("feature_prefixes", ["e__", "f__"])),
                max_features=int(edl_cfg.get("max_features", 256)),
                hidden_sizes=tuple(edl_cfg.get("hidden_sizes", [256, 128])),
                dropout=float(edl_cfg.get("dropout", 0.2)),
                lr=float(edl_cfg.get("lr", 1e-3)),
                epochs=int(edl_cfg.get("epochs", 15)),
                batch_size=int(edl_cfg.get("batch_size", 1024)),
                anneal_epochs=int(edl_cfg.get("anneal_epochs", 10)),
                weight_decay=float(edl_cfg.get("weight_decay", 1e-4)),
                loss_type=str(edl_cfg.get("loss_type", "log")),
                seed=int(edl_cfg.get("seed", 42)),
                device=str(device),
                verbose=True,
            )
            edl_train_time = time.time() - t0
            artifacts.save(art_dir)
        else:
            print(f"[EDL] Loading evidential artifacts from: {art_dir}")
            artifacts = EDLArtifacts.load(art_dir)
        infer_bs = int(edl_cfg.get("infer_batch_size", 4096))

        def _attach_edl(df_in: pd.DataFrame) -> pd.DataFrame:
            probs, u, pred_idx = predict_edl(df_in, artifacts, device=str(device), batch_size=infer_bs)
            df_out = df_in.copy()
            for k, h in enumerate(hypotheses):
                df_out[f"p__{h}"] = probs[:, k]
            df_out["uncertainty_u"] = u
            df_out["hypothesis_pred"] = [hypotheses[int(i)] for i in pred_idx]
            return df_out

        train = _attach_edl(train)
        val = _attach_edl(val)
        test = _attach_edl(test)
    else:
        required = [f"p__{h}" for h in hypotheses] + ["uncertainty_u"]
        missing = [c for c in required if c not in train.columns]
        if missing:
            raise ValueError("Legacy mode requires the dataset to already contain Dirichlet columns. Missing: " + ", ".join(missing))

    y_true_mc = test[label_col].astype(str).to_numpy()
    y_true_bin = test[bin_col].astype(int).to_numpy()

    overview_rows: List[Dict[str, Any]] = []
    multiclass_rows: List[Dict[str, Any]] = []
    binary_policy_rows: List[Dict[str, Any]] = []
    binary_label_rows: List[Dict[str, Any]] = []
    binary_deny_rows: List[Dict[str, Any]] = []
    thresholds: Dict[str, Any] = {}
    session_rows: List[Dict[str, Any]] = []

    def record_model(name: str, y_pred_mc: np.ndarray, decisions: np.ndarray, train_time_s: float, test_df_for_sessions: pd.DataFrame) -> None:
        evals = _evaluate_model(
            name=name,
            y_true_mc=y_true_mc,
            y_true_bin=y_true_bin,
            y_pred_mc=y_pred_mc,
            decisions=decisions,
            hypotheses=hypotheses,
            train_time_s=train_time_s,
            out_figs=out_figs,
            out_tables=out_tables,
        )
        overview_rows.append(evals["overview"])
        multiclass_rows.append(evals["multiclass"])
        binary_policy_rows.append(evals["binary_policy"])
        binary_label_rows.append(evals["binary_label"])
        binary_deny_rows.append(evals["binary_deny"])
        session_rows.append({"model": name, **simulate_sessions(test_df_for_sessions, session_cols, decisions, contain_n)})
        _save_per_cloud_confusions(name, test_df_for_sessions, y_pred_mc, hypotheses, cloud_col, out_tables, out_figs, label_col)

    print("[Train] Isolation Forest...")
    t0 = time.time()
    if_model, if_meta = train_isolation_forest(train)
    if_train_time = time.time() - t0
    val_risk_if = anomaly_risk(if_model, if_meta, val)
    best_if = tune_scalar_zt(val[bin_col].to_numpy(), val[label_col].to_numpy(), val_risk_if, tune["tau_low_grid"], tune["tau_high_grid"], fpr_max=fpr_max, detect_on_stepup=detect_on_stepup)
    thresholds["isolation_forest"] = best_if
    test_risk_if = anomaly_risk(if_model, if_meta, test)
    decisions_if = scalar_decide(test_risk_if, best_if["tau_low"], best_if["tau_high"])
    record_model("IsolationForest", np.where(test_risk_if >= best_if["tau_low"], hypotheses[1] if len(hypotheses) > 1 else hypotheses[0], hypotheses[0]), decisions_if, if_train_time, test)

    print("[Eval] Static Rules...")
    risk_static_val = static_rule_risk(val)
    best_static = tune_scalar_zt(val[bin_col].to_numpy(), val[label_col].to_numpy(), risk_static_val, tune["tau_low_grid"], tune["tau_high_grid"], fpr_max=fpr_max, detect_on_stepup=detect_on_stepup)
    thresholds["static_rules"] = best_static
    risk_static_test = static_rule_risk(test)
    decisions_static = scalar_decide(risk_static_test, best_static["tau_low"], best_static["tau_high"])
    record_model("Static_Rules", np.where(risk_static_test >= best_static["tau_low"], hypotheses[1] if len(hypotheses) > 1 else hypotheses[0], hypotheses[0]), decisions_static, 0.0, test)

    print("[Train] Logistic Regression baseline...")
    t0 = time.time()
    lr_art = train_logistic_regression(train, label_col=label_col, hypotheses=hypotheses)
    lr_train_time = time.time() - t0
    val_lr_probs = predict_proba_df(val, lr_art)
    val_lr = _attach_prob_columns(val, val_lr_probs, hypotheses)
    best_lr = tune_scalar_zt(val[bin_col].to_numpy(), val[label_col].to_numpy(), dirichlet_risk(val_lr, hypotheses, weights), tune["tau_low_grid"], tune["tau_high_grid"], fpr_max=fpr_max, detect_on_stepup=detect_on_stepup)
    thresholds["logistic_regression"] = best_lr
    test_lr_probs = predict_proba_df(test, lr_art)
    test_lr = _attach_prob_columns(test, test_lr_probs, hypotheses)
    decisions_lr = scalar_decide(dirichlet_risk(test_lr, hypotheses, weights), best_lr["tau_low"], best_lr["tau_high"])
    record_model("LogisticRegression", _probs_to_pred_labels(test_lr_probs, hypotheses), decisions_lr, lr_train_time, test_lr)

    print("[Train] Random Forest baseline...")
    t0 = time.time()
    rf_art = train_random_forest(train, label_col=label_col, hypotheses=hypotheses)
    rf_train_time = time.time() - t0
    val_rf_probs = predict_proba_df(val, rf_art)
    val_rf = _attach_prob_columns(val, val_rf_probs, hypotheses)
    best_rf = tune_scalar_zt(val[bin_col].to_numpy(), val[label_col].to_numpy(), dirichlet_risk(val_rf, hypotheses, weights), tune["tau_low_grid"], tune["tau_high_grid"], fpr_max=fpr_max, detect_on_stepup=detect_on_stepup)
    thresholds["random_forest"] = best_rf
    test_rf_probs = predict_proba_df(test, rf_art)
    test_rf = _attach_prob_columns(test, test_rf_probs, hypotheses)
    decisions_rf = scalar_decide(dirichlet_risk(test_rf, hypotheses, weights), best_rf["tau_low"], best_rf["tau_high"])
    record_model("RandomForest", _probs_to_pred_labels(test_rf_probs, hypotheses), decisions_rf, rf_train_time, test_rf)

    print("[Train] Softmax MLP baseline...")
    t0 = time.time()
    mlp_art = train_mlp_softmax(train, label_col=label_col, hypotheses=hypotheses)
    mlp_train_time = time.time() - t0
    val_mlp_probs = predict_proba_df(val, mlp_art)
    val_mlp = _attach_prob_columns(val, val_mlp_probs, hypotheses)
    best_mlp = tune_scalar_zt(val[bin_col].to_numpy(), val[label_col].to_numpy(), dirichlet_risk(val_mlp, hypotheses, weights), tune["tau_low_grid"], tune["tau_high_grid"], fpr_max=fpr_max, detect_on_stepup=detect_on_stepup)
    thresholds["mlp_softmax"] = best_mlp
    test_mlp_probs = predict_proba_df(test, mlp_art)
    test_mlp = _attach_prob_columns(test, test_mlp_probs, hypotheses)
    decisions_mlp = scalar_decide(dirichlet_risk(test_mlp, hypotheses, weights), best_mlp["tau_low"], best_mlp["tau_high"])
    record_model("MLP_Softmax", _probs_to_pred_labels(test_mlp_probs, hypotheses), decisions_mlp, mlp_train_time, test_mlp)

    print("[Tune] Temporal variants + DEQZT...")
    best_temp_variant: Optional[Dict[str, Any]] = None
    best_test_temp: Optional[pd.DataFrame] = None
    for lam in lambda_grid:
        val_temp = temporal_dirichlet_update(val, hypotheses, session_cols, time_col=time_col, lam=float(lam))
        cand = tune_deqzt(val_temp, hypotheses, weights, tune["tau_low_grid"], tune["tau_high_grid"], tune["u_quantiles"], beta_u_grid, min_stepup_risk_grid, label_col=label_col, bin_col=bin_col, fpr_max=fpr_max, detect_on_stepup=detect_on_stepup)
        cand["lambda"] = float(lam)
        if best_temp_variant is None or cand["score"] > best_temp_variant["score"]:
            best_temp_variant = cand
            best_test_temp = temporal_dirichlet_update(test, hypotheses, session_cols, time_col=time_col, lam=float(lam))
    assert best_temp_variant is not None and best_test_temp is not None

    thresholds["deqzt"] = best_temp_variant
    test_temp = best_test_temp
    probs_temp = test_temp[[f"p__{h}" for h in hypotheses]].to_numpy(dtype=np.float32)
    base_risk_temp = dirichlet_risk(test_temp, hypotheses, weights)
    u_temp = test_temp["uncertainty_u"].to_numpy(dtype=np.float32)

    print("[Tune] DEQZT (Temporal + Uncertainty)...")
    risk_deqzt = uncertainty_weighted_risk(base_risk_temp, u_temp, best_temp_variant["beta_u"])
    decisions_deqzt = deqzt_decide(risk_deqzt, u_temp, best_temp_variant["tau_low"], best_temp_variant["tau_high"], best_temp_variant["u_thr"], min_stepup_risk=best_temp_variant["min_stepup_risk"])
    record_model("DEQZT", _probs_to_pred_labels(probs_temp, hypotheses), decisions_deqzt, edl_train_time, test_temp)

    correct = (_probs_to_pred_labels(probs_temp, hypotheses) == y_true_mc).astype(int)
    plot_selective_curve(u_temp, correct, os.path.join(out_figs, "deqzt_selective_accuracy.png"), "DEQZT selective accuracy (by uncertainty)")

    trace = _build_trace(test_temp, base_risk_temp, risk_deqzt, u_temp, decisions_deqzt, session_cols, label_col=label_col, bin_col=bin_col, cloud_col=cloud_col, time_col=time_col)
    trace.to_csv(os.path.join(out_tables, "eval_trace.csv"), index=False)
    print(f"[Trace] Wrote {os.path.join(out_tables, 'eval_trace.csv')}")

    if skip_pqc:
        print("[Security] --skip-pqc enabled: skipping PQC benchmark and rotations.")
        pqc_df = pd.DataFrame([])
        pqc_summary = {"rotations_tested": 0, "avg_rotation_time_ms": 0.0, "p95_rotation_time_ms": 0.0, "avg_ciphertext_bytes": 0.0, "avg_signature_bytes": 0.0, "avg_total_wire_bytes": 0.0, "verify_ok_rate": 0.0, "skipped": True}
    else:
        print("[Security] Benchmarking PQC primitives...")
        try:
            pqc_summary = bench_pqc(int(os.environ.get("DEQZT_PQC_BENCH_N", "25")))
        except Exception as e:
            print(f"[WARN] PQC benchmark unavailable: {e}")
            pqc_summary = {"rotations_tested": 0, "avg_rotation_time_ms": 0.0, "p95_rotation_time_ms": 0.0, "avg_ciphertext_bytes": 0.0, "avg_signature_bytes": 0.0, "avg_total_wire_bytes": 0.0, "verify_ok_rate": 0.0, "skipped": True, "error": str(e)}
        pqc_events: List[Dict[str, Any]] = []
        try:
            from pqc.zt_session import rotate_session
            stepup_idx = np.where(decisions_deqzt == 1)[0]
            max_rotations = min(int(os.environ.get("DEQZT_PQC_MAX", "200")), len(stepup_idx))
            for i in stepup_idx[:max_rotations]:
                row_evt = test_temp.iloc[int(i)]
                subject = str(row_evt.get("principal_id", "NA"))
                cloud = str(row_evt.get(cloud_col, "NA"))
                ctx_hash = f"{subject}|{cloud}|{int(i)}"
                evt = rotate_session(subject=subject, cloud=cloud, decision="STEP_UP", context_hash=ctx_hash)
                evt["row_index"] = int(i)
                pqc_events.append(evt)
        except Exception as e:
            print(f"[WARN] PQC event-driven rotations unavailable: {e}")
        pqc_df = pd.DataFrame(pqc_events)

    pqc_df.to_csv(os.path.join(out_tables, "pqc_rotations.csv"), index=False)
    with open(os.path.join(out_tables, "pqc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(pqc_summary, f, indent=2)

    overview_df = pd.DataFrame(overview_rows)
    multiclass_df = pd.DataFrame(multiclass_rows)
    binary_policy_df = pd.DataFrame(binary_policy_rows)
    binary_label_df = pd.DataFrame(binary_label_rows)
    binary_deny_df = pd.DataFrame(binary_deny_rows)
    session_df = pd.DataFrame(session_rows)

    overview_df.to_csv(os.path.join(out_tables, "metrics_event_level.csv"), index=False)
    overview_df.to_csv(os.path.join(out_tables, "metrics_event_overview.csv"), index=False)
    multiclass_df.to_csv(os.path.join(out_tables, "metrics_event_multiclass.csv"), index=False)
    binary_policy_df.to_csv(os.path.join(out_tables, "metrics_event_binary_policy.csv"), index=False)
    binary_label_df.to_csv(os.path.join(out_tables, "metrics_event_binary_label.csv"), index=False)
    binary_deny_df.to_csv(os.path.join(out_tables, "metrics_event_binary_deny.csv"), index=False)
    session_df.to_csv(os.path.join(out_tables, "metrics_session_level.csv"), index=False)

    plot_bar(overview_rows, "macro_f1", os.path.join(out_figs, "macro_f1_bar.png"), "Multiclass Macro-F1 Comparison", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "macro_precision", os.path.join(out_figs, "macro_precision_bar.png"), "Multiclass Macro-Precision Comparison", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "macro_recall", os.path.join(out_figs, "macro_recall_bar.png"), "Multiclass Macro-Recall Comparison", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "label_f1_attack", os.path.join(out_figs, "label_attack_f1_bar.png"), "Attack-vs-Benign F1 from Predicted Labels", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "decision_f1_attack", os.path.join(out_figs, "decision_f1_attack_bar.png"), "Attack Intervention F1 from Policy Decisions", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "decision_precision_attack", os.path.join(out_figs, "decision_precision_attack_bar.png"), "Attack Intervention Precision", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "decision_recall_attack", os.path.join(out_figs, "decision_recall_attack_bar.png"), "Attack Intervention Recall", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "decision_specificity", os.path.join(out_figs, "decision_specificity_bar.png"), "Attack Intervention Specificity", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "decision_mcc", os.path.join(out_figs, "decision_mcc_bar.png"), "Attack Intervention MCC", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "decision_fpr", os.path.join(out_figs, "decision_fpr_bar.png"), "Attack Intervention FPR from Policy Decisions", higher_is_better=False, sort_desc=False)
    plot_bar(overview_rows, "decision_tpr", os.path.join(out_figs, "decision_tpr_bar.png"), "Attack Intervention TPR from Policy Decisions", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "rzdu", os.path.join(out_figs, "rzdu_bar.png"), "Risk-aware ZT Decision Utility", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "balanced_accuracy", os.path.join(out_figs, "balanced_accuracy_bar.png"), "Balanced Accuracy Comparison", higher_is_better=True, sort_desc=True)
    plot_bar(overview_rows, "train_time_s", os.path.join(out_figs, "train_time_s_bar.png"), "Train Time Comparison (s)", higher_is_better=False, sort_desc=False)
    plot_bar(session_rows, "intervention_at_n", os.path.join(out_figs, "intervention_at_n_bar.png"), f"Session Intervention@{contain_n}", higher_is_better=True, sort_desc=True)
    plot_bar(session_rows, "containment_at_n", os.path.join(out_figs, "containment_at_n_bar.png"), f"Session Deny-Containment@{contain_n}", higher_is_better=True, sort_desc=True)
    plot_bar(session_rows, "benign_disruption_rate", os.path.join(out_figs, "benign_disruption_rate_bar.png"), "Benign Disruption Rate", higher_is_better=False, sort_desc=False)
    plot_bar(session_rows, "stepup_rate", os.path.join(out_figs, "stepup_rate_bar.png"), "Step-up Rate", higher_is_better=False, sort_desc=False)
    plot_bar(session_rows, "deny_rate", os.path.join(out_figs, "deny_rate_bar.png"), "Deny Rate", higher_is_better=False, sort_desc=False)

    with open(os.path.join(out_tables, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print("[DONE] DEQZT pipeline executed successfully with leakage checks, deduplication, group-aware splits, stronger baselines, richer metrics, and PQC benchmarking.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["edl", "legacy"], default="edl")
    ap.add_argument("--edl-artifacts-dir", default="")
    ap.add_argument("--edl-retrain", action="store_true")
    ap.add_argument("--edl-device", default="")
    ap.add_argument("--skip-pqc", action="store_true")
    ap.add_argument("--split-mode", choices=["time", "crosscloud"], default="time")
    ap.add_argument("--train-clouds", default="")
    ap.add_argument("--test-cloud", default="")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    main(
        args.data,
        args.config,
        skip_pqc=bool(args.skip_pqc),
        split_mode=str(args.split_mode),
        train_clouds=str(args.train_clouds),
        test_cloud=str(args.test_cloud),
        outdir=str(args.outdir),
        model=str(args.model),
        edl_artifacts_dir=(str(args.edl_artifacts_dir) if str(args.edl_artifacts_dir).strip() else None),
        edl_retrain=bool(args.edl_retrain),
        edl_device=(str(args.edl_device) if str(args.edl_device).strip() else None),
    )
