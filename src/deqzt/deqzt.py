from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def dirichlet_risk(df: pd.DataFrame, hypotheses: List[str], weights: Dict[str, float]) -> np.ndarray:
    r = np.zeros(len(df), dtype=np.float32)
    for h in hypotheses:
        col = f"p__{h}"
        if col in df.columns:
            r += float(weights.get(h, 0.0)) * df[col].fillna(0.0).to_numpy(dtype=np.float32)
    return np.clip(r, 0.0, 1.0).astype(np.float32)


def uncertainty_weighted_risk(base_risk: np.ndarray, u: np.ndarray, beta_u: float) -> np.ndarray:
    """Implements the manuscript-style risk fusion.

    Risk_t = (1 - u_t) * base_risk_t + beta_u * u_t
    where u is epistemic uncertainty in [0,1].
    """
    base_risk = np.asarray(base_risk, dtype=np.float32)
    u = np.clip(np.asarray(u, dtype=np.float32), 0.0, 1.0)
    risk = (1.0 - u) * base_risk + float(beta_u) * u
    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def scalar_decide(risk: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    d = np.zeros(len(risk), dtype=np.int8)
    d[risk >= float(tau_high)] = 2
    d[(risk >= float(tau_low)) & (risk < float(tau_high))] = 1
    return d


def deqzt_decide(
    risk: np.ndarray,
    u: np.ndarray,
    tau_low: float,
    tau_high: float,
    u_thr: float,
    *,
    min_stepup_risk: float = 0.0,
) -> np.ndarray:
    """Decision policy with uncertainty-aware step-up gate.

    0 allow, 1 step-up, 2 deny
    Deny is driven by risk. Uncertainty can escalate ALLOW to STEP_UP, but only
    when the base effective risk is not trivially small.
    """
    d = scalar_decide(risk, tau_low, tau_high)
    d[(d == 0) & (u >= float(u_thr)) & (risk >= float(min_stepup_risk))] = 1
    return d


def attach_probs_uncertainty(
    df: pd.DataFrame,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    hypotheses: List[str],
    *,
    prefix: str = "",
) -> pd.DataFrame:
    out = df.copy()
    for k, h in enumerate(hypotheses):
        out[f"{prefix}p__{h}"] = probs[:, k]
    out[f"{prefix}uncertainty_u"] = uncertainty.astype(np.float32)
    return out


def recover_alpha_from_probs_uncertainty(
    df: pd.DataFrame,
    hypotheses: List[str],
    *,
    uncertainty_col: str = "uncertainty_u",
) -> np.ndarray:
    K = max(1, len(hypotheses))
    probs = np.stack(
        [df[f"p__{h}"].fillna(0.0).to_numpy(dtype=np.float32) for h in hypotheses],
        axis=1,
    )
    u = df[uncertainty_col].fillna(1.0).to_numpy(dtype=np.float32)
    u = np.clip(u, 1e-6, 1e6)
    S = float(K) / u
    alpha = probs * S[:, None]
    alpha = np.clip(alpha, 1.0, None)
    return alpha.astype(np.float32)


def temporal_dirichlet_update(
    df: pd.DataFrame,
    hypotheses: List[str],
    session_cols: List[str],
    *,
    time_col: str = "ts",
    lam: float = 0.85,
    uncertainty_col: str = "uncertainty_u",
) -> pd.DataFrame:
    out = df.copy()
    for c in session_cols:
        if c not in out.columns:
            out[c] = "NA"
    sid_df = out[session_cols].copy()
    for c in session_cols:
        sid_df[c] = sid_df[c].fillna("NA")
        sid_df[c] = sid_df[c].apply(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))
    out["_sid_tmp"] = sid_df.astype(str).agg("|".join, axis=1)
    if time_col in out.columns:
        out["_ts_tmp"] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
        out = out.sort_values(["_sid_tmp", "_ts_tmp"], kind="mergesort")
    else:
        out = out.sort_values(["_sid_tmp"], kind="mergesort")

    alpha = recover_alpha_from_probs_uncertainty(out, hypotheses, uncertainty_col=uncertainty_col)
    evidence = np.clip(alpha - 1.0, 0.0, None)
    updated_alpha = np.zeros_like(alpha, dtype=np.float32)

    sid_values = out["_sid_tmp"].to_numpy()
    current_sid = None
    E_prev = None
    for i, sid in enumerate(sid_values):
        e_t = evidence[i]
        if sid != current_sid or E_prev is None:
            E_prev = e_t.copy()
            current_sid = sid
        else:
            E_prev = float(lam) * E_prev + e_t
        updated_alpha[i] = E_prev + 1.0

    S = np.clip(updated_alpha.sum(axis=1, keepdims=True), 1e-9, None)
    probs = updated_alpha / S
    K = float(len(hypotheses))
    u = (K / S.squeeze(1)).astype(np.float32)

    for k, h in enumerate(hypotheses):
        out[f"p__{h}"] = probs[:, k].astype(np.float32)
    out[uncertainty_col] = np.clip(u, 0.0, 1.0)

    out = out.sort_index()
    return out.drop(columns=["_sid_tmp", "_ts_tmp"], errors="ignore")


def simulate_sessions(df: pd.DataFrame, session_cols: List[str], decisions: np.ndarray, contain_within_n: int):
    df = df.copy()
    df["_decision"] = decisions
    for c in session_cols:
        if c not in df.columns:
            df[c] = "NA"

    sid_df = df[session_cols].copy()
    for c in session_cols:
        sid_df[c] = sid_df[c].fillna("NA")
        sid_df[c] = sid_df[c].apply(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))
    df["_sid"] = sid_df.astype(str).agg("|".join, axis=1)

    if "ts" in df.columns:
        df["_ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.sort_values(["_sid", "_ts"], kind="mergesort")
    else:
        df = df.sort_values(["_sid"], kind="mergesort")

    sess_attack = df.groupby("_sid")["is_attack"].max() if "is_attack" in df.columns else None

    contain_hits = 0
    attack_sessions = 0
    benign_disrupt = 0
    stepup_sessions = 0
    deny_sessions = 0
    n_sessions = int(df["_sid"].nunique())

    for sid, g in df.groupby("_sid", sort=False):
        is_attack_sess = int(sess_attack.loc[sid]) if sess_attack is not None else 0
        dec = g["_decision"].to_numpy()
        denied = np.any(dec == 2)
        stepped = np.any(dec == 1)
        if stepped:
            stepup_sessions += 1
        if denied:
            deny_sessions += 1

        if is_attack_sess == 1:
            attack_sessions += 1
            if np.any(dec[:contain_within_n] == 2):
                contain_hits += 1
        else:
            if denied:
                benign_disrupt += 1

    benign_sessions = n_sessions - attack_sessions
    intervention_hits = 0
    for _, g in df.groupby("_sid", sort=False):
        if int(g["is_attack"].max()) == 1 and np.any(g["_decision"].to_numpy()[:contain_within_n] != 0):
            intervention_hits += 1
    return {
        "n_sessions": n_sessions,
        "attack_sessions": int(attack_sessions),
        "containment_at_n": float(contain_hits / (attack_sessions + 1e-9)),
        "intervention_at_n": float(intervention_hits / (attack_sessions + 1e-9)),
        "benign_sessions": int(benign_sessions),
        "benign_disruption_rate": float(benign_disrupt / (benign_sessions + 1e-9)),
        "stepup_rate": float(stepup_sessions / (n_sessions + 1e-9)),
        "deny_rate": float(deny_sessions / (n_sessions + 1e-9)),
    }
