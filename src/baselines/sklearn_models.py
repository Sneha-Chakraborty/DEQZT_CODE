from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass
class SklearnBaselineArtifacts:
    model_name: str
    model: Any
    feature_cols: List[str]
    preprocessor: Any
    hypotheses: List[str]


def _configure_torch_threads() -> None:
    threads = int(__import__('os').environ.get("DEQZT_TORCH_THREADS", "1"))
    threads = max(1, threads)
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def select_feature_cols(df: pd.DataFrame, prefixes: Sequence[str] = ("e__", "f__"), max_features: int = 256) -> List[str]:
    banned_tokens = {"label", "target", "attack", "hypothesis", "decision", "risk", "groundtruth", "y_true", "class"}

    def allowed(col: str) -> bool:
        s = str(col).lower()
        return not str(col).startswith("p__") and str(col) not in {"uncertainty_u", "S"} and not any(tok in s for tok in banned_tokens)

    cols: List[str] = []
    for pref in prefixes:
        cols.extend([c for c in df.columns if str(c).startswith(str(pref)) and allowed(c)])

    seen = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    if not ordered:
        ordered = [c for c in df.columns if allowed(c)]

    filtered: List[str] = []
    sample = df.head(min(len(df), 5000)).copy()
    for c in ordered:
        s = sample[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            if s.notna().any():
                filtered.append(c)
            continue
        s2 = s.astype(str).str.strip()
        nonempty = s2.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan}).notna().mean()
        if float(nonempty) >= 0.05:
            filtered.append(c)

    ordered = filtered
    if max_features and len(ordered) > int(max_features):
        ordered = ordered[: int(max_features)]
    if not ordered:
        raise ValueError(
            "No usable feature columns found for classical baselines. "
            "Checked prefixed features and general non-label columns, but all were empty/unusable."
        )
    return ordered


def _prepare_mixed_feature_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    Xdf = df.copy()
    out = pd.DataFrame(index=Xdf.index)
    for c in feature_cols:
        if c not in Xdf.columns:
            out[c] = np.nan
            continue
        s = Xdf[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(float)
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
            continue
        stripped = s.astype(str).str.strip()
        coerced = pd.to_numeric(stripped, errors="coerce")
        if float(coerced.notna().mean()) >= 0.90:
            out[c] = coerced
        else:
            out[c] = stripped.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    return out


def _filter_categorical_columns(Xdf: pd.DataFrame, cat_cols: Sequence[str], *, max_unique: int = 64, max_ratio: float = 0.02) -> List[str]:
    keep: List[str] = []
    n = max(1, len(Xdf))
    for c in cat_cols:
        s = Xdf[c]
        nunique = int(s.nunique(dropna=True))
        if nunique <= 1:
            continue
        if nunique <= int(max_unique) and (nunique / n) <= float(max_ratio):
            keep.append(c)
    return keep


def _build_preprocessor_from_df(
    Xdf: pd.DataFrame,
    *,
    sparse_cat: bool = False,
    max_cat_unique: int = 64,
    max_cat_ratio: float = 0.02,
):
    num_cols = [c for c in Xdf.columns if pd.api.types.is_numeric_dtype(Xdf[c]) or pd.api.types.is_bool_dtype(Xdf[c])]
    raw_cat_cols = [c for c in Xdf.columns if c not in num_cols]
    cat_cols = _filter_categorical_columns(Xdf, raw_cat_cols, max_unique=max_cat_unique, max_ratio=max_cat_ratio)

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(with_centering=True)),
            ]),
            num_cols,
        ))
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_cat, min_frequency=2)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=sparse_cat)
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe),
            ]),
            cat_cols,
        ))
    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns available for classical baselines after filtering high-cardinality columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3 if sparse_cat else 0.0), num_cols, cat_cols


def _label_series(df: pd.DataFrame, label_col: str, hypotheses: Sequence[str]) -> pd.Series:
    default_label = str(hypotheses[0]) if hypotheses else "BENIGN"
    return df[label_col].fillna(default_label).astype(str)


def _maybe_downsample(train_df: pd.DataFrame, label_col: str, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(train_df) <= max_rows:
        return train_df
    parts = []
    frac = max_rows / max(1, len(train_df))
    for _, g in train_df.groupby(label_col, dropna=False, sort=False):
        k = max(1, int(round(len(g) * frac)))
        parts.append(g.sample(n=min(k, len(g)), random_state=random_state))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if len(out) > max_rows:
        out = out.iloc[:max_rows].reset_index(drop=True)
    return out


def _fit_common(
    train_df: pd.DataFrame,
    label_col: str,
    hypotheses: Sequence[str],
    *,
    model_name: str,
    estimator: Any,
    prefixes: Sequence[str] = ("e__", "f__"),
    max_features: int = 256,
    encode_numeric_labels: bool = False,
    max_train_rows: int = 0,
    random_state: int = 42,
    sparse_cat: bool = False,
    max_cat_unique: int = 64,
    max_cat_ratio: float = 0.02,
) -> SklearnBaselineArtifacts:
    sampled = _maybe_downsample(train_df, label_col=label_col, max_rows=int(max_train_rows), random_state=int(random_state))
    feature_cols = select_feature_cols(sampled, prefixes=prefixes, max_features=max_features)
    X_train_df = _prepare_mixed_feature_frame(sampled, feature_cols)
    pre, _, _ = _build_preprocessor_from_df(X_train_df, sparse_cat=sparse_cat, max_cat_unique=max_cat_unique, max_cat_ratio=max_cat_ratio)
    X_train = pre.fit_transform(X_train_df)
    y_train = _label_series(sampled, label_col, hypotheses)
    if encode_numeric_labels:
        hyp2idx = {str(h): i for i, h in enumerate(hypotheses)}
        y_train = y_train.map(lambda x: hyp2idx.get(str(x), 0)).astype(int)
    estimator.fit(X_train, y_train)
    return SklearnBaselineArtifacts(model_name=model_name, model=estimator, feature_cols=list(feature_cols), preprocessor=pre, hypotheses=list(hypotheses))


class _TorchSoftmaxWrapper:
    def __init__(self, input_dim: int, num_classes: int, hidden_sizes: Sequence[int], lr: float, epochs: int, batch_size: int, random_state: int):
        _configure_torch_threads()
        torch.manual_seed(int(random_state))
        layers: List[nn.Module] = []
        d = int(input_dim)
        for h in hidden_sizes:
            layers.extend([nn.Linear(d, int(h)), nn.ReLU()])
            d = int(h)
        layers.append(nn.Linear(d, int(num_classes)))
        self.net = nn.Sequential(*layers)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.classes_: List[int] = list(range(int(num_classes)))

    def fit(self, X, y):
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        y = np.asarray(y, dtype=np.int64)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        counts = np.bincount(y, minlength=len(self.classes_)).astype(np.float32)
        counts[counts <= 0] = 1.0
        weights = counts.sum() / (len(self.classes_) * counts)
        sampler = WeightedRandomSampler(torch.tensor(weights[y], dtype=torch.double), num_samples=len(y), replacement=True)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=min(self.batch_size, len(y_t)), sampler=sampler, drop_last=False)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
        self.net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


def train_logistic_regression(train_df: pd.DataFrame, label_col: str, hypotheses: Sequence[str], *, prefixes: Sequence[str] = ("e__", "f__"), max_features: int = 256, random_state: int = 42, max_train_rows: int = 100000) -> SklearnBaselineArtifacts:
    estimator = LogisticRegression(max_iter=300, class_weight="balanced", solver="saga", random_state=int(random_state), n_jobs=1)
    return _fit_common(train_df, label_col, hypotheses, model_name="LogisticRegression", estimator=estimator, prefixes=prefixes, max_features=max_features, encode_numeric_labels=False, max_train_rows=max_train_rows, random_state=random_state, sparse_cat=True, max_cat_unique=32, max_cat_ratio=0.005)


def train_random_forest(train_df: pd.DataFrame, label_col: str, hypotheses: Sequence[str], *, prefixes: Sequence[str] = ("e__", "f__"), max_features: int = 256, random_state: int = 42, max_train_rows: int = 75000) -> SklearnBaselineArtifacts:
    estimator = RandomForestClassifier(n_estimators=120, max_depth=None, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=1, random_state=int(random_state))
    return _fit_common(train_df, label_col, hypotheses, model_name="RandomForest", estimator=estimator, prefixes=prefixes, max_features=max_features, encode_numeric_labels=False, max_train_rows=max_train_rows, random_state=random_state, sparse_cat=False, max_cat_unique=16, max_cat_ratio=0.002)


def train_mlp_softmax(train_df: pd.DataFrame, label_col: str, hypotheses: Sequence[str], *, prefixes: Sequence[str] = ("e__", "f__"), max_features: int = 256, hidden_sizes: Sequence[int] = (128, 64), random_state: int = 42, max_train_rows: int = 50000) -> SklearnBaselineArtifacts:
    sampled = _maybe_downsample(train_df, label_col=label_col, max_rows=int(max_train_rows), random_state=int(random_state))
    feature_cols = select_feature_cols(sampled, prefixes=prefixes, max_features=max_features)
    X_train_df = _prepare_mixed_feature_frame(sampled, feature_cols)
    pre, _, _ = _build_preprocessor_from_df(X_train_df, sparse_cat=False, max_cat_unique=16, max_cat_ratio=0.002)
    X_train = pre.fit_transform(X_train_df)
    hyp2idx = {str(h): i for i, h in enumerate(hypotheses)}
    y_train = _label_series(sampled, label_col, hypotheses).map(lambda x: hyp2idx.get(str(x), 0)).astype(int).to_numpy()
    model = _TorchSoftmaxWrapper(input_dim=X_train.shape[1], num_classes=len(hypotheses), hidden_sizes=hidden_sizes, lr=1e-3, epochs=8, batch_size=2048, random_state=random_state)
    model.fit(X_train, y_train)
    return SklearnBaselineArtifacts(model_name="MLP_Softmax", model=model, feature_cols=list(feature_cols), preprocessor=pre, hypotheses=list(hypotheses))


def predict_proba_df(df: pd.DataFrame, artifacts: SklearnBaselineArtifacts) -> np.ndarray:
    Xdf = df.copy()
    for c in artifacts.feature_cols:
        if c not in Xdf.columns:
            Xdf[c] = np.nan
    X = artifacts.preprocessor.transform(_prepare_mixed_feature_frame(Xdf, artifacts.feature_cols))
    if not hasattr(artifacts.model, "predict_proba"):
        raise AttributeError(f"Model {artifacts.model_name} does not support predict_proba().")
    probs_native = artifacts.model.predict_proba(X)
    out = np.zeros((len(Xdf), len(artifacts.hypotheses)), dtype=np.float32)
    classes = list(getattr(artifacts.model, "classes_", []))
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    numeric_direct = all(isinstance(c, (int, np.integer)) for c in classes)
    for k, h in enumerate(artifacts.hypotheses):
        idx = class_to_idx.get(str(h))
        if idx is None and numeric_direct and k in classes:
            idx = classes.index(k)
        if idx is not None:
            out[:, k] = probs_native[:, idx]
    row_sums = out.sum(axis=1, keepdims=True)
    missing_rows = (row_sums.squeeze(1) <= 1e-9)
    if np.any(missing_rows):
        out[missing_rows, 0] = 1.0
        row_sums = out.sum(axis=1, keepdims=True)
    return (out / np.clip(row_sums, 1e-9, None)).astype(np.float32)


def predict_label_df(df: pd.DataFrame, artifacts: SklearnBaselineArtifacts) -> np.ndarray:
    probs = predict_proba_df(df, artifacts)
    idx = np.argmax(probs, axis=1)
    return np.array([artifacts.hypotheses[int(i)] for i in idx])
