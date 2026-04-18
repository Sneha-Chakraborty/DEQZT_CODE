from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.io import build_group_id


SUSPICIOUS_NAME_TOKENS = {
    "label", "target", "attack", "hypothesis", "decision", "risk", "groundtruth", "y_true", "class",
}


def _hash_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return pd.Series([], dtype="uint64")
    tmp = df[list(cols)].copy()
    for c in tmp.columns:
        if pd.api.types.is_datetime64_any_dtype(tmp[c]):
            tmp[c] = pd.to_datetime(tmp[c], errors="coerce", utc=True).astype(str)
        else:
            tmp[c] = tmp[c].fillna("NA").astype(str)
    return pd.util.hash_pandas_object(tmp, index=False)


def detect_suspicious_feature_names(columns: Iterable[str], label_col: str, bin_col: str) -> List[str]:
    out: List[str] = []
    for c in columns:
        low = str(c).lower()
        if low in {str(label_col).lower(), str(bin_col).lower()}:
            continue
        if any(tok in low for tok in SUSPICIOUS_NAME_TOKENS):
            out.append(str(c))
    return sorted(set(out))


def leakage_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    label_col: str,
    bin_col: str,
    feature_cols: Sequence[str],
    session_cols: Sequence[str],
    out_dir: str,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = [c for c in feature_cols if c in train.columns and c in val.columns and c in test.columns]

    hashes_train = set(_hash_frame(train, feature_cols).astype(str).tolist())
    hashes_val = set(_hash_frame(val, feature_cols).astype(str).tolist())
    hashes_test = set(_hash_frame(test, feature_cols).astype(str).tolist())

    sess_train = set(build_group_id(train, session_cols).astype(str).tolist()) if session_cols else set()
    sess_val = set(build_group_id(val, session_cols).astype(str).tolist()) if session_cols else set()
    sess_test = set(build_group_id(test, session_cols).astype(str).tolist()) if session_cols else set()

    suspicious = detect_suspicious_feature_names(feature_cols, label_col=label_col, bin_col=bin_col)

    report = {
        "feature_overlap": {
            "train_val": int(len(hashes_train & hashes_val)),
            "train_test": int(len(hashes_train & hashes_test)),
            "val_test": int(len(hashes_val & hashes_test)),
        },
        "session_overlap": {
            "train_val": int(len(sess_train & sess_val)),
            "train_test": int(len(sess_train & sess_test)),
            "val_test": int(len(sess_val & sess_test)),
        },
        "n_rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "suspicious_feature_names": suspicious,
    }

    with open(os.path.join(out_dir, "leakage_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame([
        {"check": "feature_hash_overlap_train_val", "value": report["feature_overlap"]["train_val"]},
        {"check": "feature_hash_overlap_train_test", "value": report["feature_overlap"]["train_test"]},
        {"check": "feature_hash_overlap_val_test", "value": report["feature_overlap"]["val_test"]},
        {"check": "session_overlap_train_val", "value": report["session_overlap"]["train_val"]},
        {"check": "session_overlap_train_test", "value": report["session_overlap"]["train_test"]},
        {"check": "session_overlap_val_test", "value": report["session_overlap"]["val_test"]},
        {"check": "n_suspicious_feature_names", "value": len(suspicious)},
    ]).to_csv(os.path.join(out_dir, "leakage_checks.csv"), index=False)

    if suspicious:
        pd.DataFrame({"suspicious_feature_name": suspicious}).to_csv(os.path.join(out_dir, "suspicious_feature_names.csv"), index=False)
    return report
