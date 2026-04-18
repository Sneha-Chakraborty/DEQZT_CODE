# src/crypto/compare_crypto.py
"""
Produce "PQC vs Classical" comparison tables for DEQ-ZT.

Tables:
1) Raw crypto overhead (latency/bytes):
   - crypto_comparison.csv
   - crypto_comparison.tex

2) Security/control-loop metrics (10-year scenario + ZT control metrics):
   - crypto_security_control_metrics.csv
   - crypto_security_control_metrics_econ.tex
   - crypto_security_control_metrics_control.tex

This script expects:
- results/tables/pqc_summary.json (from src/run_all.py)
- results/tables/eval_trace.csv (from src/run_all.py)  [used for rotation selectivity/coverage/delay]
- results/tables/metrics_session_level.csv (from src/run_all.py) [used for containment@N]

Classical baselines benchmarked (microbench):
  - ECDHE X25519 + Ed25519
  - ECDHE P-256 + ECDSA P-256
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from crypto.benchmark_classical import bench_classical
from pqc.benchmark_pqc import bench_pqc


CRYPTO_SCHEME_ORDER = [
    "Classical: ECDHE-X25519+ED25519\n[36]",
    "Classical: ECDHE-P256+ECDSA-P256\n[37][38]",
    "PQC: ML-KEM-768 + ML-DSA-65",
]


def _order_crypto_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Scheme" not in df.columns or df.empty:
        return df
    order_map = {name: i for i, name in enumerate(CRYPTO_SCHEME_ORDER)}
    tmp = df.copy()
    tmp["__order"] = tmp["Scheme"].astype(str).map(lambda x: order_map.get(x, len(order_map)))
    tmp = tmp.sort_values(["__order", "Scheme"], kind="mergesort").drop(columns=["__order"])
    return tmp


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _pqc_total_wire_bytes(pqc: Dict) -> float:
    # Approximation for on-wire payload per rotation:
    # KEM ciphertext + signature. (Public keys assumed provisioned.)
    return _safe(pqc.get("avg_ciphertext_bytes", 0.0)) + _safe(pqc.get("avg_signature_bytes", 0.0))


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _write_latex_table_overhead(df: pd.DataFrame, out_path: str) -> None:
    cols = ["Scheme", "Avg ms", "P95 ms", "Total bytes", "Latency vs PQC", "Bytes vs PQC"]
    lines: List[str] = []
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\hline")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for _, r in df.iterrows():
        row = [
            _latex_escape(str(r["Scheme"])),
            f"{_safe(r['Avg ms']):.2f}",
            f"{_safe(r['P95 ms']):.2f}",
            f"{_safe(r['Total bytes']):.0f}",
            f"{_safe(r['Latency vs PQC']):.2f}x",
            f"{_safe(r['Bytes vs PQC']):.2f}x",
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _protect_mask_per_session(trace: pd.DataFrame, ttl_steps: int) -> np.ndarray:
    """
    Returns boolean mask where event is within ttl_steps after a rotation (decision==1) inside the same session.
    Includes the rotation event itself.
    """
    ttl_steps = max(int(ttl_steps), 1)
    protected = np.zeros(len(trace), dtype=bool)

    # groupby preserves original order of indices, so we sort inside session by _row_index then apply ttl marking
    for _, g in trace.groupby("_sid", sort=False):
        g = g.sort_values("_row_index", kind="mergesort")
        idx = g.index.to_numpy()
        rot_pos = np.where(g["_decision"].to_numpy(dtype=int) == 1)[0]
        if len(rot_pos) == 0:
            continue
        for rp in rot_pos:
            end = min(rp + ttl_steps, len(idx) - 1)
            protected[idx[rp : end + 1]] = True
    return protected


def _compute_control_metrics(trace: pd.DataFrame, ttl_steps: int) -> Dict[str, float]:
    """
    Compute rotation coverage/selectivity/delay from eval_trace.
    High-risk is defined as decision != 0 (step-up or deny).
    """
    required = {"_sid", "_row_index", "_decision", "_risk", "is_attack"}
    missing = required.difference(trace.columns)
    if missing:
        raise ValueError(f"eval_trace.csv missing columns: {sorted(missing)}")

    # Basic masks
    high_risk = trace["_decision"].to_numpy(dtype=int) != 0
    rotations = trace["_decision"].to_numpy(dtype=int) == 1  # step-up triggers rotation
    is_attack = trace["is_attack"].astype(int).to_numpy()

    protected = _protect_mask_per_session(trace, ttl_steps=ttl_steps)

    # Coverage on "high-risk" events
    denom_hr = float(np.sum(high_risk))
    coverage = float(np.sum(high_risk & protected) / (denom_hr + 1e-9))

    # Risk-weighted (continuous)
    risk = trace["_risk"].to_numpy(dtype=float)
    denom_risk = float(np.sum(risk))
    rw_coverage = float(np.sum(risk * protected.astype(float)) / (denom_risk + 1e-9))

    # Rotation selectivity
    n_rot = int(np.sum(rotations))
    rot_attack = int(np.sum(rotations & (is_attack == 1)))
    attack_events = int(np.sum(is_attack == 1))

    rot_precision = float(rot_attack / (n_rot + 1e-9))
    rot_recall = float(rot_attack / (attack_events + 1e-9))

    # Detection-to-rotation delay per session:
    # detection = first non-allow decision (step-up or deny)
    # rotation = first step-up decision
    # Compute both in "steps" and in seconds if _ts exists.
    delays_steps = []
    delays_seconds = []
    has_ts = "_ts" in trace.columns

    for _, g in trace.groupby("_sid", sort=False):
        g = g.sort_values("_row_index", kind="mergesort")
        dec = g["_decision"].to_numpy(dtype=int)

        det_pos = np.where(dec != 0)[0]
        rot_pos = np.where(dec == 1)[0]
        if len(det_pos) == 0 or len(rot_pos) == 0:
            continue

        d = int(rot_pos[0] - det_pos[0])
        if d < 0:
            d = 0
        delays_steps.append(d)

        if has_ts:
            ts = pd.to_datetime(g["_ts"], errors="coerce", utc=True)
            t_det = ts.iloc[int(det_pos[0])]
            t_rot = ts.iloc[int(rot_pos[0])]
            if pd.notna(t_det) and pd.notna(t_rot):
                delays_seconds.append(float((t_rot - t_det).total_seconds()))

    def _p95(arr: List[float]) -> float:
        return float(np.quantile(arr, 0.95)) if len(arr) else 0.0

    out = {
        "n_rotations": float(n_rot),
        "rotation_precision": float(rot_precision),
        "rotation_recall": float(rot_recall),
        "high_risk_coverage": float(coverage),
        "risk_weighted_coverage": float(rw_coverage),
        "det_to_rot_mean_steps": float(np.mean(delays_steps)) if len(delays_steps) else 0.0,
        "det_to_rot_p95_steps": _p95(delays_steps),
    }

    if has_ts:
        out["det_to_rot_mean_s"] = float(np.mean(delays_seconds)) if len(delays_seconds) else 0.0
        out["det_to_rot_p95_s"] = _p95(delays_seconds)

    return out


def _write_latex_econ(df: pd.DataFrame, out_path: str) -> None:
    cols = ["Scheme", "P(break,10y)", "E[L](10y)", "LossRem", "LossRed\\%", "RotCovGain", "RW-SecGain"]
    lines = []
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\hline")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for _, r in df.iterrows():
        row = [
            _latex_escape(str(r["Scheme"])),
            f"{_safe(r['p_break_10y']):.2f}",
            f"{_safe(r['expected_loss_10y']):.2f}",
            f"{_safe(r['loss_remaining']):.2f}",
            f"{_safe(r['loss_reduction_pct']):.1f}",
            f"{_safe(r['rotation_coverage_gain']):.3f}",
            f"{_safe(r['risk_weighted_security_gain']):.3f}",
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_latex_control(df: pd.DataFrame, out_path: str) -> None:
    # Use seconds if available; otherwise steps.
    use_seconds = "det_to_rot_p95_s" in df.columns
    delay_col = "det_to_rot_p95_s" if use_seconds else "det_to_rot_p95_steps"
    delay_hdr = "Det\\rightarrow Rot (P95 s)" if use_seconds else "Det\\rightarrow Rot (P95 steps)"

    cols = ["Scheme", "Rotations", "RotPrec", "RotRecall", delay_hdr, "Bytes/Contain", "Ms/Contain"]
    lines = []
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\hline")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\hline")
    for _, r in df.iterrows():
        row = [
            _latex_escape(str(r["Scheme"])),
            f"{_safe(r['n_rotations']):.0f}",
            f"{_safe(r['rotation_precision']):.2f}",
            f"{_safe(r['rotation_recall']):.2f}",
            f"{_safe(r[delay_col]):.2f}",
            f"{_safe(r['bytes_per_attack_session_contained']):.0f}",
            f"{_safe(r['ms_per_attack_session_contained']):.1f}",
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")




def _plot_scheme_bars(df: pd.DataFrame, col: str, out_path: str, title: str, *, higher_is_better: bool = True) -> None:
    import matplotlib.pyplot as plt
    vals = df[col].astype(float).to_numpy()
    names = df["Scheme"].astype(str).tolist()
    fig = plt.figure(figsize=(max(8, len(names) * 1.4), 6))
    ax = fig.add_subplot(111)
    bars = ax.bar(names, vals)
    finite = vals[np.isfinite(vals)]
    percent_like = bool(len(finite)) and float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) <= 1.05
    if percent_like:
        ax.set_ylim(0.0, min(1.0, float(np.nanmax(finite)) * 1.20 + 0.05))
    else:
        top = float(np.nanmax(finite)) if len(finite) else 1.0
        ax.set_ylim(0.0, max(1.0, top * 1.20 + 0.05 * top))
    best = (float(np.nanmax(finite)) if higher_is_better else float(np.nanmin(finite))) if len(finite) else 0.0
    off = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    for b, v in zip(bars, vals):
        if np.isfinite(v) and np.isclose(v, best, rtol=0.0, atol=1e-12):
            b.set_hatch("//")
            b.set_linewidth(1.5)
        label = f"{v:.2f}%" if percent_like else f"{v:.3f}"
        if percent_like:
            label = f"{v*100:.2f}%"
        ax.text(b.get_x() + b.get_width()/2.0, v + off, label, ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(col)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_crypto_overhead_grouped(df: pd.DataFrame, out_path: str) -> None:
    """Grouped chart for Avg ms, P95 ms, Total bytes, and Verify ok.
    Uses two y-axes so verification success remains readable.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        return

    plot_df = _order_crypto_df(df.copy())
    names = plot_df["Scheme"].astype(str).tolist()
    x = np.arange(len(names), dtype=float)
    width = 0.18

    avg_ms = plot_df["Avg ms"].astype(float).to_numpy()
    p95_ms = plot_df["P95 ms"].astype(float).to_numpy()
    total_bytes = plot_df["Total bytes"].astype(float).to_numpy()
    verify_ok = plot_df["Verify ok"].astype(float).to_numpy()

    fig, ax1 = plt.subplots(figsize=(max(10, len(names) * 2.2), 6.5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - 1.5 * width, avg_ms, width, label="Avg latency (ms)")
    b2 = ax1.bar(x - 0.5 * width, p95_ms, width, label="P95 latency (ms)")
    b3 = ax1.bar(x + 0.5 * width, total_bytes, width, label="Total byte overhead")
    b4 = ax2.bar(x + 1.5 * width, verify_ok, width, label="Verification success")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20)
    ax1.set_ylabel("Latency / Bytes")
    ax2.set_ylabel("Verification success")
    ax2.set_ylim(0.0, 1.05)
    ax1.set_title("Grouped PQC/Classical Overhead Comparison")

    def _label(ax, bars, percent_like=False):
        y0, y1 = ax.get_ylim()
        off = (y1 - y0) * 0.02
        for bar in bars:
            v = float(bar.get_height())
            label = f"{v*100:.2f}%" if percent_like else (f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}")
            ax.text(bar.get_x() + bar.get_width()/2.0, v + off, label, ha="center", va="bottom", fontsize=8)

    if np.isfinite(np.nanmax(np.concatenate([avg_ms, p95_ms, total_bytes]))) and np.nanmax(np.concatenate([avg_ms, p95_ms, total_bytes])) > 0:
        ax1.set_ylim(0.0, np.nanmax(np.concatenate([avg_ms, p95_ms, total_bytes])) * 1.18)
    _label(ax1, b1)
    _label(ax1, b2)
    _label(ax1, b3)
    _label(ax2, b4, percent_like=True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqc-summary", default="results/tables/pqc_summary.json")
    ap.add_argument("--eval-trace", default="results/tables/eval_trace.csv")
    ap.add_argument("--session-metrics", default="results/tables/metrics_session_level.csv")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--n", type=int, default=int(os.environ.get("DEQZT_CLASSICAL_N", "200")))
    ap.add_argument("--include", nargs="*", default=["x25519+ed25519", "p256+ecdsa_p256"])

    # 10-year scenario / security economics
    ap.add_argument("--horizon-years", type=int, default=10)
    ap.add_argument("--impact", type=float, default=1.0)
    ap.add_argument("--p-break-classical", type=float, default=0.30,
                    help="Scenario P(crypto broken within horizon) for classical schemes.")
    ap.add_argument("--p-break-pqc", type=float, default=0.05,
                    help="Scenario P(crypto broken within horizon) for PQC schemes.")
    ap.add_argument("--ttl-steps", type=int, default=10,
                    help="Within-session TTL in number of events after a rotation considered 'protected' for coverage metrics.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.pqc_summary):
        raise FileNotFoundError(
            f"Could not find {args.pqc_summary}. Run the pipeline first (src/run_all.py) "
            "to generate pqc_summary.json, or point --pqc-summary to the correct file."
        )

    pqc = _load_json(args.pqc_summary)
    pqc_needs_bench = bool(pqc.get("skipped", False)) or int(_safe(pqc.get("rotations_tested", 0))) <= 0 or _safe(pqc.get("avg_rotation_time_ms", 0.0)) <= 0.0
    if pqc_needs_bench:
        try:
            pqc = bench_pqc(max(10, int(args.n // 4)))
            with open(args.pqc_summary, "w", encoding="utf-8") as f:
                json.dump(pqc, f, indent=2)
            print(f"[PQC] Re-benchmarked PQC and refreshed {args.pqc_summary}")
        except Exception as e:
            print(f"[WARN] Could not benchmark PQC directly: {e}")
    pqc_avg_ms = _safe(pqc.get("avg_rotation_time_ms", 0.0))
    pqc_p95_ms = _safe(pqc.get("p95_rotation_time_ms", pqc.get("avg_rotation_time_ms", 0.0)))
    pqc_total_bytes = _safe(pqc.get("avg_total_wire_bytes", 0.0)) or _pqc_total_wire_bytes(pqc)
    pqc_verify = _safe(pqc.get("verify_ok_rate", 0.0))

    # -------- Raw overhead table --------
    rows_overhead: List[Dict] = []
    rows_overhead.append(
        {
            "Scheme": "PQC: ML-KEM-768 + ML-DSA-65",
            "Avg ms": pqc_avg_ms,
            "P95 ms": pqc_p95_ms,
            "KEX bytes": _safe(pqc.get("avg_ciphertext_bytes", 0.0)),
            "Sig bytes": _safe(pqc.get("avg_signature_bytes", 0.0)),
            "Total bytes": pqc_total_bytes,
            "Verify ok": pqc_verify,
            "Latency vs PQC": 1.0,
            "Bytes vs PQC": 1.0,
        }
    )

    mapping = {
        "x25519+ed25519": ("x25519", "ed25519"),
        "p256+ecdsa_p256": ("p256", "ecdsa_p256"),
    }

    classical_summaries: Dict[str, Dict] = {}

    for key in args.include:
        if key not in mapping:
            continue
        kex, sig = mapping[key]
        summary = bench_classical(args.n, kex=kex, sig=sig)
        classical_summaries[key] = summary

        out_summary = os.path.join(args.out, f"classical_summary_{kex}_{sig}.json")
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        avg_ms = _safe(summary.get("avg_rotation_time_ms", 0.0))
        p95_ms = _safe(summary.get("p95_rotation_time_ms", avg_ms))
        total_bytes = _safe(summary.get("avg_total_wire_bytes", 0.0))

        rows_overhead.append(
            {
                "Scheme": f"Classical: ECDHE-{kex.upper()} + {sig.upper()}",
                "Avg ms": avg_ms,
                "P95 ms": p95_ms,
                "KEX bytes": _safe(summary.get("avg_kex_wire_bytes", 0.0)),
                "Sig bytes": _safe(summary.get("avg_signature_bytes", 0.0)),
                "Total bytes": total_bytes,
                "Verify ok": _safe(summary.get("verify_ok_rate", 0.0)),
                # Relative overhead (PQC / classical):
                "Latency vs PQC": (pqc_avg_ms / avg_ms) if avg_ms > 0 else 0.0,
                "Bytes vs PQC": (pqc_total_bytes / total_bytes) if total_bytes > 0 else 0.0,
            }
        )

    df_over = _order_crypto_df(pd.DataFrame(rows_overhead))

    csv_path = os.path.join(args.out, "crypto_comparison.csv")
    df_over.to_csv(csv_path, index=False)

    tex_path = os.path.join(args.out, "crypto_comparison.tex")
    _write_latex_table_overhead(df_over, tex_path)

    # -------- Security/control-loop metrics table --------
    # Load eval_trace and session metrics from DEQZT run.
    if not os.path.exists(args.eval_trace):
        raise FileNotFoundError(
            f"Could not find {args.eval_trace}. Please re-run src/run_all.py "
            "with this updated codebase to generate eval_trace.csv."
        )
    trace = pd.read_csv(args.eval_trace)

    # Normalize optional timestamp column
    if "_ts" in trace.columns:
        trace["_ts"] = pd.to_datetime(trace["_ts"], errors="coerce", utc=True)

    control = _compute_control_metrics(trace, ttl_steps=int(args.ttl_steps))

    # Containment@N from session metrics for DEQZT
    contained_sessions = 0.0
    if os.path.exists(args.session_metrics):
        sm = pd.read_csv(args.session_metrics)
        if "model" in sm.columns:
            row = sm[sm["model"].astype(str).str.upper() == "DEQZT"]
            if len(row):
                attack_sess = _safe(row.iloc[0].get("attack_sessions", 0.0))
                contain_at_n = _safe(row.iloc[0].get("containment_at_n", 0.0))
                contained_sessions = float(attack_sess * contain_at_n)

    n_rot_total = float(control.get("n_rotations", 0.0))

    # Economics scenario (10y)
    p_break_classical = float(args.p_break_classical)
    p_break_pqc = float(args.p_break_pqc)
    impact = float(args.impact)

    expected_loss_classical = p_break_classical * impact

    def scheme_row(name: str, avg_ms: float, total_bytes: float, p_break: float) -> Dict[str, float]:
        expected_loss = p_break * impact
        loss_remaining = (expected_loss / expected_loss_classical) if expected_loss_classical > 0 else 0.0
        loss_reduction_pct = 100.0 * (1.0 - loss_remaining)

        survival = max(0.0, 1.0 - p_break)

        # Gains incorporate both coverage and survival probability under the horizon scenario.
        rot_cov_gain = float(control["high_risk_coverage"]) * survival
        rw_sec_gain = float(control["risk_weighted_coverage"]) * survival

        total_overhead_bytes = n_rot_total * float(total_bytes)
        total_overhead_ms = n_rot_total * float(avg_ms)

        bytes_per_contained = total_overhead_bytes / (contained_sessions + 1e-9)
        ms_per_contained = total_overhead_ms / (contained_sessions + 1e-9)

        row = {
            "Scheme": name,
            "horizon_years": float(args.horizon_years),
            "p_break_10y": float(p_break),
            "survival_prob_10y": float(survival),
            "expected_loss_10y": float(expected_loss),
            "loss_remaining": float(loss_remaining),
            "loss_reduction_pct": float(loss_reduction_pct),

            "rotation_coverage_gain": float(rot_cov_gain),
            "risk_weighted_security_gain": float(rw_sec_gain),

            # Control metrics (scheme-independent, derived from policy decisions)
            "n_rotations": float(control["n_rotations"]),
            "rotation_precision": float(control["rotation_precision"]),
            "rotation_recall": float(control["rotation_recall"]),
            "high_risk_coverage": float(control["high_risk_coverage"]),
            "risk_weighted_coverage": float(control["risk_weighted_coverage"]),
            "det_to_rot_mean_steps": float(control["det_to_rot_mean_steps"]),
            "det_to_rot_p95_steps": float(control["det_to_rot_p95_steps"]),
        }

        if "det_to_rot_mean_s" in control:
            row["det_to_rot_mean_s"] = float(control["det_to_rot_mean_s"])
            row["det_to_rot_p95_s"] = float(control["det_to_rot_p95_s"])

        row.update({
            # Overhead per contained attack session
            "bytes_per_attack_session_contained": float(bytes_per_contained),
            "ms_per_attack_session_contained": float(ms_per_contained),
        })
        return row

    rows_sec: List[Dict] = []

    rows_sec.append(
        scheme_row(
            "PQC: ML-KEM-768 + ML-DSA-65",
            avg_ms=pqc_avg_ms,
            total_bytes=pqc_total_bytes,
            p_break=p_break_pqc,
        )
    )

    # Classical rows (use measured avg/p95 bytes)
    for key in args.include:
        if key not in mapping:
            continue
        kex, sig = mapping[key]
        summary = classical_summaries.get(key, {})
        avg_ms = _safe(summary.get("avg_rotation_time_ms", 0.0))
        total_bytes = _safe(summary.get("avg_total_wire_bytes", 0.0))
        rows_sec.append(
            scheme_row(
                f"Classical: ECDHE-{kex.upper()} + {sig.upper()}",
                avg_ms=avg_ms,
                total_bytes=total_bytes,
                p_break=p_break_classical,
            )
        )

    df_sec = _order_crypto_df(pd.DataFrame(rows_sec))

    out_sec_csv = os.path.join(args.out, "crypto_security_control_metrics.csv")
    df_sec.to_csv(out_sec_csv, index=False)

    out_econ_tex = os.path.join(args.out, "crypto_security_control_metrics_econ.tex")
    _write_latex_econ(df_sec, out_econ_tex)

    out_ctrl_tex = os.path.join(args.out, "crypto_security_control_metrics_control.tex")
    _write_latex_control(df_sec, out_ctrl_tex)

    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {tex_path}")
    print(f"[OK] Wrote {out_sec_csv}")
    print(f"[OK] Wrote {out_econ_tex}")
    print(f"[OK] Wrote {out_ctrl_tex}")

    out_figs = os.path.join(os.path.dirname(args.out), "figures") if os.path.basename(args.out) == "tables" else os.path.join(args.out, "figures")
    os.makedirs(out_figs, exist_ok=True)
    _plot_scheme_bars(df_over, "Verify ok", os.path.join(out_figs, "crypto_verify_ok_bar.png"), "Signature Verification Success", higher_is_better=True)
    _plot_scheme_bars(df_sec, "p_break_10y", os.path.join(out_figs, "crypto_break_probability_bar.png"), "10-Year Break Probability", higher_is_better=False)
    _plot_scheme_bars(df_sec, "survival_prob_10y", os.path.join(out_figs, "crypto_survival_probability_bar.png"), "10-Year Survival Probability", higher_is_better=True)
    _plot_scheme_bars(df_sec, "loss_reduction_pct", os.path.join(out_figs, "crypto_loss_reduction_bar.png"), "Expected Loss Reduction vs Classical", higher_is_better=True)
    _plot_crypto_overhead_grouped(df_over, os.path.join(out_figs, "crypto_overhead_grouped_bar.png"))

    # Print brief summary to stdout
    disp_cols = [
        "Scheme", "p_break_10y", "expected_loss_10y", "loss_reduction_pct",
        "rotation_coverage_gain", "risk_weighted_security_gain",
        "rotation_precision", "rotation_recall",
        "bytes_per_attack_session_contained", "ms_per_attack_session_contained",
    ]
    print(df_sec[disp_cols])


if __name__ == "__main__":
    main()
