from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd


def resolve_data_path(path: str) -> str:
    """
    Resolve the user-supplied --data path.

    Supports:
    - Direct file path (CSV or Parquet)
    - Directory path containing a single .parquet file
    - Directory path containing 'dirichlet_training.parquet'
    """
    p = Path(str(path)).expanduser()

    # If user passed a directory, try to pick a sensible parquet inside.
    if p.exists() and p.is_dir():
        preferred = p / "dirichlet_training.parquet"
        if preferred.exists():
            return str(preferred)

        parquet_files = sorted(p.glob("*.parquet"))
        if len(parquet_files) == 1:
            return str(parquet_files[0])
        if len(parquet_files) > 1:
            sample = "\n".join([f"  - {x.name}" for x in parquet_files[:10]])
            raise FileNotFoundError(
                f"Provided --data is a directory with multiple parquet files. "
                f"Please pass the exact file path.\nFound (sample):\n{sample}"
            )

        raise FileNotFoundError(
            f"Provided --data is a directory but no .parquet files were found: {p}"
        )

    # If it's a file path but doesn't exist, fail with a helpful message.
    if not p.exists():
        raise FileNotFoundError(
            f"Data file not found: {p}\n"
            f"You passed: '{path}'\n"
            f"Fix: pass the correct path to your parquet, e.g.\n"
            f"  python src\\run_all.py --data D:\\research\\datasets\\output\\dirichlet_training.parquet --config configs\\config.yaml"
        )

    return str(p)


def read_parquet(path: str) -> pd.DataFrame:
    # Convenience: allow CSV inputs for quick debugging.
    if str(path).lower().endswith(".csv"):
        resolved = resolve_data_path(path)
        return pd.read_csv(resolved)

    resolved = resolve_data_path(path)

    try:
        return pd.read_parquet(resolved, engine="pyarrow")
    except ImportError as e:
        raise ImportError(
            "Parquet support requires 'pyarrow'. Install dependencies with: pip install -r requirements.txt"
        ) from e


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def time_aware_split(df: pd.DataFrame, time_col: str, ratios: Dict[str, float]):
    df = df.copy()
    if time_col in df.columns:
        df["_ts_parsed"] = safe_to_datetime(df[time_col])
        nat_ratio = df["_ts_parsed"].isna().mean()
        if nat_ratio < 0.25:
            df = df.sort_values("_ts_parsed", kind="mergesort")
        else:
            df = df.sample(frac=1.0, random_state=42)
    else:
        df = df.sample(frac=1.0, random_state=42)

    n = len(df)
    n_train = int(ratios["train"] * n)
    n_val = int(ratios["val"] * n)

    train = df.iloc[:n_train].drop(columns=["_ts_parsed"], errors="ignore")
    val = df.iloc[n_train : n_train + n_val].drop(columns=["_ts_parsed"], errors="ignore")
    test = df.iloc[n_train + n_val :].drop(columns=["_ts_parsed"], errors="ignore")
    return train, val, test


def cross_cloud_split(df: pd.DataFrame, cloud_col: str, time_col: str, ratios: Dict[str, float],
                      train_clouds: list[str], test_cloud: str):
    """Cross-cloud split:
    - Train/Val are drawn ONLY from train_clouds (time-aware split within that pool).
    - Test is ALL rows from test_cloud.
    """
    d = df.copy()
    if cloud_col not in d.columns:
        raise ValueError(f"cloud column '{cloud_col}' not found in dataframe")
    d[cloud_col] = d[cloud_col].fillna("NA").astype(str)

    train_pool = d[d[cloud_col].isin([str(x) for x in train_clouds])].copy()
    test = d[d[cloud_col] == str(test_cloud)].copy()

    if len(train_pool) == 0:
        raise ValueError(f"No rows for train_clouds={train_clouds} in column '{cloud_col}'")
    if len(test) == 0:
        raise ValueError(f"No rows for test_cloud='{test_cloud}' in column '{cloud_col}'")

    train, val, _ = time_aware_split(train_pool, time_col, ratios)
    return train, val, test

