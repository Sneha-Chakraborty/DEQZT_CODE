from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def train_isolation_forest(train_df: pd.DataFrame):
    """
    Isolation Forest baseline over numeric telemetry features.

    Returns:
      model: sklearn Pipeline
      meta: dict with num_cols + scaling stats to map anomaly risk into [0,1]
    """
    num_cols = [c for c in train_df.columns if c.startswith("e__")]
    f_cols = [c for c in train_df.columns if c.startswith("f__")]
    num_cols = num_cols + f_cols[:50]
    if not num_cols:
        raise ValueError("No numeric columns found for Isolation Forest (need e__* or f__*).")

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=True)),
        ("iforest", IsolationForest(
            n_estimators=200,
            max_samples=256,
            contamination="auto",
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipe.fit(train_df[num_cols])

    # Calibrate risk scaling on training distribution so tau_grid in [0,1] works reliably.
    train_scores = pipe.score_samples(train_df[num_cols])
    risk_raw = (-train_scores).astype(np.float32)

    p1 = float(np.quantile(risk_raw, 0.01))
    p99 = float(np.quantile(risk_raw, 0.99))
    if p99 <= p1:
        p1 = float(risk_raw.min())
        p99 = float(risk_raw.max()) if float(risk_raw.max()) > p1 else p1 + 1.0

    return pipe, {"num_cols": num_cols, "risk_p1": p1, "risk_p99": p99}

def anomaly_risk(model, meta, df: pd.DataFrame):
    """
    Returns anomaly risk in [0,1] (approximately) using train-calibrated scaling.
    """
    X = df.copy()
    for c in meta["num_cols"]:
        if c not in X.columns:
            X[c] = np.nan

    scores = model.score_samples(X[meta["num_cols"]])
    risk_raw = (-scores).astype(np.float32)

    p1 = float(meta.get("risk_p1", np.quantile(risk_raw, 0.01)))
    p99 = float(meta.get("risk_p99", np.quantile(risk_raw, 0.99)))
    denom = (p99 - p1) if (p99 - p1) != 0 else 1.0

    risk = (risk_raw - p1) / denom
    return np.clip(risk, 0.0, 1.0).astype(np.float32)
