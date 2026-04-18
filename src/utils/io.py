from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _looks_like_git_lfs_pointer(head: bytes) -> bool:
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _is_valid_parquet_file(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_file() or p.stat().st_size < 8:
            return False
        with p.open("rb") as f:
            head = f.read(64)
            if _looks_like_git_lfs_pointer(head):
                return False
            if len(head) < 4 or head[:4] != b"PAR1":
                return False
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def _choose_best_parquet(parquets: Iterable[Path]) -> Optional[Path]:
    parquets = list(parquets)
    if not parquets:
        return None
    valid = [p for p in parquets if _is_valid_parquet_file(p)]
    if not valid:
        return None
    by_name = {p.name.lower(): p for p in valid}
    if "dirichlet_training.parquet" in by_name:
        return by_name["dirichlet_training.parquet"]
    if "unified_normalized.parquet" in by_name:
        return by_name["unified_normalized.parquet"]
    return sorted(valid, key=lambda x: x.stat().st_size, reverse=True)[0]


def resolve_data_path(path: str) -> str:
    p = Path(str(path)).expanduser()
    if p.exists() and p.is_dir():
        parquet_files = sorted(p.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Provided --data is a directory but no .parquet files were found: {p}")
        best = _choose_best_parquet(parquet_files)
        if best is not None:
            return str(best)
        sample = "\n".join([f"  - {x.name} ({x.stat().st_size} bytes)" for x in parquet_files[:10]])
        raise FileNotFoundError(
            "Found .parquet files, but none look like valid parquet (missing PAR1 magic bytes). "
            "This often means the files are corrupted OR are Git-LFS pointer files.\n\n"
            "Fix options:\n"
            "  1) Run `git lfs install` then `git lfs pull` inside the dataset repo.\n"
            "  2) Or pass a different parquet that is real.\n\n"
            f"Files found (sample):\n{sample}"
        )
    if not p.exists():
        raise FileNotFoundError(
            f"Data file not found: {p}\n"
            f"You passed: '{path}'\n"
            "Fix: pass the correct path to your parquet, e.g.\n"
            "  python src\\run_all.py --data D:\\DEQZT_Dataset\\output\\normalized_synth.parquet --config configs\\config.yaml"
        )
    return str(p)


def read_parquet(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(resolve_data_path(path))
    resolved = resolve_data_path(path)
    try:
        return pd.read_parquet(resolved, engine="pyarrow")
    except ImportError as e:
        raise ImportError("Parquet support requires 'pyarrow'. Install dependencies with: pip install -r requirements.txt") from e
    except Exception as e:
        msg = str(e)
        if "magic bytes" in msg.lower():
            raise ValueError(
                "Failed to read the file as parquet. The file does not appear to be a real parquet file (missing PAR1 magic bytes).\n\n"
                "Common causes:\n"
                "  • The file is corrupted/truncated.\n"
                "  • The file is a Git-LFS pointer instead of the real parquet data.\n\n"
                f"Verify (PowerShell): Get-Content -TotalCount 3 '{resolved}'\n"
                "If you see `version https://git-lfs.github.com/spec/v1`, you need Git LFS."
            ) from e
        raise


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def build_group_id(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(np.arange(len(df)), index=df.index, dtype="object")
    work = df.copy()
    for c in group_cols:
        if c not in work.columns:
            work[c] = "NA"
        work[c] = work[c].fillna("NA").astype(str)
    return work[list(group_cols)].astype(str).agg("|".join, axis=1)


def _ordered_group(df: pd.DataFrame, time_col: str, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    if time_col in out.columns:
        out["_ts_parsed"] = safe_to_datetime(out[time_col])
        nat_ratio = out["_ts_parsed"].isna().mean()
        if nat_ratio < 0.25:
            out = out.sort_values("_ts_parsed", kind="mergesort")
        else:
            out = out.sample(frac=1.0, random_state=int(seed))
    else:
        out = out.sample(frac=1.0, random_state=int(seed))
    return out


def _split_counts(n: int, ratios: Dict[str, float]) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    n_train = int(round(float(ratios["train"]) * n))
    n_val = int(round(float(ratios["val"]) * n))
    n_test = n - n_train - n_val

    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        n_train = n - n_val - n_test
        if n_train <= 0:
            n_train = 1
            if n_val > n_test:
                n_val -= 1
            else:
                n_test -= 1
    elif n == 2:
        n_train, n_val, n_test = 1, 0, 1
    else:
        n_train, n_val, n_test = 1, 0, 0

    while n_train + n_val + n_test > n:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            n_train -= 1
    while n_train + n_val + n_test < n:
        n_train += 1
    return int(n_train), int(n_val), int(n_test)


def time_aware_split(df: pd.DataFrame, time_col: str, ratios: Dict[str, float], seed: int = 42):
    ordered = _ordered_group(df, time_col=time_col, seed=seed)
    n_train, n_val, _ = _split_counts(len(ordered), ratios)
    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train:n_train + n_val]
    test = ordered.iloc[n_train + n_val:]
    return (
        train.drop(columns=["_ts_parsed"], errors="ignore"),
        val.drop(columns=["_ts_parsed"], errors="ignore"),
        test.drop(columns=["_ts_parsed"], errors="ignore"),
    )


def stratified_time_aware_split(
    df: pd.DataFrame,
    time_col: str,
    ratios: Dict[str, float],
    label_col: str,
    seed: int = 42,
):
    if label_col not in df.columns:
        return time_aware_split(df, time_col, ratios, seed=seed)

    parts_train = []
    parts_val = []
    parts_test = []
    working = df.copy()
    working[label_col] = working[label_col].fillna("BENIGN").astype(str)

    for _, g in working.groupby(label_col, sort=True):
        ordered = _ordered_group(g, time_col=time_col, seed=seed)
        n_train, n_val, _ = _split_counts(len(ordered), ratios)
        parts_train.append(ordered.iloc[:n_train])
        parts_val.append(ordered.iloc[n_train:n_train + n_val])
        parts_test.append(ordered.iloc[n_train + n_val:])

    train = pd.concat(parts_train, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(parts_val, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(parts_test, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return (
        train.drop(columns=["_ts_parsed"], errors="ignore"),
        val.drop(columns=["_ts_parsed"], errors="ignore"),
        test.drop(columns=["_ts_parsed"], errors="ignore"),
    )


def stratified_group_time_aware_split(
    df: pd.DataFrame,
    time_col: str,
    ratios: Dict[str, float],
    label_col: str,
    group_cols: Sequence[str],
    seed: int = 42,
):
    if not group_cols:
        return stratified_time_aware_split(df, time_col, ratios, label_col, seed=seed)
    work = df.copy()
    work[label_col] = work[label_col].fillna("BENIGN").astype(str)
    work["_group_id"] = build_group_id(work, group_cols)
    if time_col in work.columns:
        work["_ts_parsed"] = safe_to_datetime(work[time_col])
    else:
        work["_ts_parsed"] = pd.NaT

    # summarize each group by earliest timestamp + dominant label
    def _group_label(g: pd.Series) -> str:
        counts = g.value_counts(dropna=False)
        if len(counts) == 0:
            return "BENIGN"
        if len(counts) == 1:
            return str(counts.index[0])
        non_benign = counts[counts.index.astype(str) != "BENIGN"]
        if len(non_benign) > 0:
            return str(non_benign.index[0])
        return str(counts.index[0])

    groups = (
        work.groupby("_group_id", sort=False)
        .agg(
            _split_label=(label_col, _group_label),
            _min_ts=("_ts_parsed", "min"),
        )
        .reset_index()
    )

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    for _, g in groups.groupby("_split_label", sort=True):
        ordered = g.sort_values("_min_ts", kind="mergesort") if g["_min_ts"].notna().any() else g.sample(frac=1.0, random_state=seed)
        n_train, n_val, _ = _split_counts(len(ordered), ratios)
        train_ids.extend(ordered.iloc[:n_train]["_group_id"].astype(str).tolist())
        val_ids.extend(ordered.iloc[n_train:n_train + n_val]["_group_id"].astype(str).tolist())
        test_ids.extend(ordered.iloc[n_train + n_val:]["_group_id"].astype(str).tolist())

    train = work[work["_group_id"].astype(str).isin(set(train_ids))].copy()
    val = work[work["_group_id"].astype(str).isin(set(val_ids))].copy()
    test = work[work["_group_id"].astype(str).isin(set(test_ids))].copy()

    train = train.drop(columns=["_group_id", "_ts_parsed"], errors="ignore").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = val.drop(columns=["_group_id", "_ts_parsed"], errors="ignore").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = test.drop(columns=["_group_id", "_ts_parsed"], errors="ignore").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test


def enforce_min_label_support(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str,
    *,
    min_test_count: int = 1,
    min_val_count: int = 1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if label_col not in train.columns or label_col not in val.columns or label_col not in test.columns:
        return train, val, test
    train = train.copy().reset_index(drop=True)
    val = val.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)
    labels = sorted({*train[label_col].astype(str).unique().tolist(), *val[label_col].astype(str).unique().tolist(), *test[label_col].astype(str).unique().tolist()})

    def _move_rows(src: pd.DataFrame, dst: pd.DataFrame, lab: str, need: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if need <= 0:
            return src, dst
        idx = src.index[src[label_col].astype(str) == str(lab)].tolist()
        if not idx:
            return src, dst
        take = idx[-need:]
        moved = src.loc[take].copy()
        src = src.drop(index=take).reset_index(drop=True)
        dst = pd.concat([dst, moved], axis=0).reset_index(drop=True)
        return src, dst

    for lab in labels:
        te = int((test[label_col].astype(str) == lab).sum())
        va = int((val[label_col].astype(str) == lab).sum())
        if te < min_test_count:
            need = min_test_count - te
            val, test = _move_rows(val, test, lab, need)
            te = int((test[label_col].astype(str) == lab).sum())
            if te < min_test_count:
                train, test = _move_rows(train, test, lab, min_test_count - te)
        if va < min_val_count:
            need = min_val_count - va
            train, val = _move_rows(train, val, lab, need)
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = val.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test


def drop_exact_duplicates(df: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    work = df.copy()
    if subset is None:
        subset = [c for c in work.columns if c not in {"ts", "timestamp", "event_time"}]
    subset = [c for c in subset if c in work.columns]
    before = len(work)
    work = work.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(work)
    return work, {"rows_before": int(before), "rows_after": int(after), "rows_dropped": int(before - after)}


def summarize_label_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    labels = sorted({*train.get(label_col, pd.Series(dtype=str)).astype(str).unique().tolist(), *val.get(label_col, pd.Series(dtype=str)).astype(str).unique().tolist(), *test.get(label_col, pd.Series(dtype=str)).astype(str).unique().tolist()})
    rows = []
    for lab in labels:
        tr = int((train[label_col].astype(str) == lab).sum()) if label_col in train.columns else 0
        va = int((val[label_col].astype(str) == lab).sum()) if label_col in val.columns else 0
        te = int((test[label_col].astype(str) == lab).sum()) if label_col in test.columns else 0
        rows.append({"label": lab, "train": tr, "val": va, "test": te, "test_present": int(te > 0), "val_present": int(va > 0)})
    return pd.DataFrame(rows)


def cross_cloud_split(
    df: pd.DataFrame,
    cloud_col: str,
    time_col: str,
    ratios: Dict[str, float],
    train_clouds: list[str],
    test_cloud: str,
    *,
    label_col: Optional[str] = None,
    group_cols: Optional[Sequence[str]] = None,
    seed: int = 42,
):
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

    if label_col and label_col in train_pool.columns:
        if group_cols:
            train, val, _ = stratified_group_time_aware_split(train_pool, time_col, ratios, label_col, group_cols, seed=seed)
        else:
            train, val, _ = stratified_time_aware_split(train_pool, time_col, ratios, label_col, seed=seed)
    else:
        train, val, _ = time_aware_split(train_pool, time_col, ratios, seed=seed)
    return train, val, test.reset_index(drop=True)
