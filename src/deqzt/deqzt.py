from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

def dirichlet_risk(df: pd.DataFrame, hypotheses: List[str], weights: Dict[str, float]) -> np.ndarray:
    r = np.zeros(len(df), dtype=np.float32)
    for h in hypotheses:
        col = f"p__{h}"
        if col in df.columns:
            r += float(weights.get(h, 0.0)) * df[col].fillna(0.0).to_numpy(dtype=np.float32)
    return r

def deqzt_decide(risk: np.ndarray, u: np.ndarray, tau_low: float, tau_high: float, u_thr: float):
    # 0 allow, 1 step-up, 2 deny
    d = np.zeros(len(risk), dtype=np.int8)
    d[risk >= tau_high] = 2
    d[((risk >= tau_low) & (risk < tau_high)) | (u >= u_thr)] = 1
    return d

def simulate_sessions(df: pd.DataFrame, session_cols: List[str], decisions: np.ndarray, contain_within_n: int):
    df = df.copy()
    df["_decision"] = decisions
    for c in session_cols:
        if c not in df.columns:
            df[c] = "NA"
    # df["_sid"] = df[session_cols].astype(str).agg("|".join, axis=1)
    # Build a safe session id string (handles NaN/float)
    sid_df = df[session_cols].copy()

    for c in session_cols:
        # Replace NaN with a sentinel, then stringify
        sid_df[c] = sid_df[c].fillna("NA")
        # Convert floats like 123.0 -> "123" to avoid ugly ids
        sid_df[c] = sid_df[c].apply(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))

    df["_sid"] = sid_df.astype(str).agg("|".join, axis=1)

    if "ts" in df.columns:
        df["_ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.sort_values(["_sid","_ts"], kind="mergesort")
    else:
        df = df.sort_values(["_sid"], kind="mergesort")

    sess_attack = df.groupby("_sid")["is_attack"].max() if "is_attack" in df.columns else None

    contain_hits = 0; attack_sessions = 0; benign_disrupt = 0; stepup_sessions = 0
    n_sessions = int(df["_sid"].nunique())

    for sid, g in df.groupby("_sid", sort=False):
        is_attack_sess = int(sess_attack.loc[sid]) if sess_attack is not None else 0
        dec = g["_decision"].to_numpy()
        denied = np.any(dec == 2)
        stepped = np.any(dec == 1)
        if stepped: stepup_sessions += 1

        if is_attack_sess == 1:
            attack_sessions += 1
            if np.any(dec[:contain_within_n] == 2):
                contain_hits += 1
        else:
            if denied:
                benign_disrupt += 1

    benign_sessions = n_sessions - attack_sessions
    return {
        "n_sessions": n_sessions,
        "attack_sessions": int(attack_sessions),
        "containment_at_n": float(contain_hits / (attack_sessions + 1e-9)),
        "benign_sessions": int(benign_sessions),
        "benign_disruption_rate": float(benign_disrupt / (benign_sessions + 1e-9)),
        "stepup_rate": float(stepup_sessions / (n_sessions + 1e-9)),
    }
