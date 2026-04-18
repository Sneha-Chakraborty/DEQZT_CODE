from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from utils.seed import seed_everything

# Torch is required for EDL.
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from deqzt.edl_model import EDLMLP, EDLForward
from deqzt.edl_loss import edl_loss


def _configure_torch_threads() -> None:
    threads = int(os.environ.get("DEQZT_TORCH_THREADS", "1"))
    threads = max(1, threads)
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


@dataclass
class EDLArtifacts:
    hypotheses: List[str]
    feature_cols: List[str]
    preprocessor: Pipeline
    state_dict: Dict
    config: Dict

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # Save preprocessor with joblib (scikit-learn dependency)
        import joblib
        joblib.dump(self.preprocessor, os.path.join(out_dir, "preprocessor.joblib"))
        torch.save(self.state_dict, os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hypotheses": self.hypotheses,
                    "feature_cols": self.feature_cols,
                    "config": self.config,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load(out_dir: str) -> "EDLArtifacts":
        import joblib
        pre = joblib.load(os.path.join(out_dir, "preprocessor.joblib"))
        state = torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu")
        meta = json.load(open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8"))
        return EDLArtifacts(
            hypotheses=list(meta["hypotheses"]),
            feature_cols=list(meta["feature_cols"]),
            preprocessor=pre,
            state_dict=state,
            config=dict(meta.get("config", {})),
        )


def _select_feature_cols(df: pd.DataFrame, prefixes: Sequence[str], max_features: int) -> List[str]:
    cols: List[str] = []
    for pref in prefixes:
        cols.extend([c for c in df.columns if str(c).startswith(str(pref))])
    # Deduplicate preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    # Fallback: if prefixes don't exist in the dataset, use numeric columns.
    # This makes the repo runnable across different normalized parquet schemas.
    if not out:
        numeric = [
            c
            for c in df.select_dtypes(include=["number", "bool"]).columns
            if not str(c).startswith("p__") and str(c) not in {"uncertainty_u", "S"}
        ]
        out = list(numeric)

    if max_features and len(out) > int(max_features):
        out = out[: int(max_features)]

    if not out:
        raise ValueError(
            "No usable numeric feature columns found for EDL. "
            "Tried prefixes "
            f"{list(prefixes)} (e.g., e__*, f__*). "
            "Fallback also found 0 numeric columns. "
            "Fix: ensure your parquet has numeric telemetry features, or update config.edl.feature_prefixes."
        )
    return out


def _build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=True)),
        ]
    )


def train_edl(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    hypotheses: List[str],
    *,
    prefixes: Sequence[str] = ("e__", "f__"),
    max_features: int = 256,
    hidden_sizes: Sequence[int] = (256, 128),
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 15,
    batch_size: int = 1024,
    anneal_epochs: int = 10,
    weight_decay: float = 1e-4,
    loss_type: str = "log",
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[EDLArtifacts, EDLMLP]:
    """
    Train an Evidential Deep Learning (EDL) classifier and return artifacts.
    """
    seed_everything(seed)
    _configure_torch_threads()
    device = str(device)

    feature_cols = _select_feature_cols(pd.concat([train_df, val_df], axis=0, ignore_index=True), prefixes, max_features)
    pre = _build_preprocessor()

    # Prepare X
    X_train = pre.fit_transform(train_df[feature_cols])
    X_val = pre.transform(val_df[feature_cols])

    # Labels -> indices (must match hypotheses list)
    hyp2idx = {h: i for i, h in enumerate(hypotheses)}
    y_train_raw = train_df[label_col].fillna("BENIGN").astype(str).values
    y_val_raw = val_df[label_col].fillna("BENIGN").astype(str).values

    def map_y(arr):
        out = np.zeros(len(arr), dtype=np.int64)
        for i, s in enumerate(arr):
            out[i] = hyp2idx.get(str(s), hyp2idx.get("BENIGN", 0))
        return out

    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)

    class_counts = np.bincount(y_train, minlength=len(hypotheses)).astype(np.float32)
    class_counts[class_counts <= 0] = 1.0
    class_weights = class_counts.sum() / (len(hypotheses) * class_counts)
    sample_weights = class_weights[y_train].astype(np.float32)

    # Torch tensors
    Xtr = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
    Xva = torch.tensor(X_val.toarray() if hasattr(X_val, "toarray") else X_val, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    yva = torch.tensor(y_val, dtype=torch.long)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=int(batch_size), sampler=sampler, drop_last=False)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=int(batch_size), shuffle=False, drop_last=False)

    K = int(len(hypotheses))
    model = EDLMLP(input_dim=int(Xtr.shape[1]), num_classes=K, hidden_sizes=hidden_sizes, dropout=float(dropout))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val = float("inf")
    best_state = None
    patience = 5
    bad = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        n = 0
        anneal = min(1.0, float(epoch) / max(1, int(anneal_epochs)))

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            alpha = model(xb)
            y_onehot = torch.zeros((len(yb), K), device=device, dtype=torch.float32)
            y_onehot.scatter_(1, yb.view(-1, 1), 1.0)
            loss = edl_loss(alpha, y_onehot, num_classes=K, anneal=anneal, loss_type=str(loss_type))
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)

        # Validate
        model.eval()
        vtotal = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                alpha = model(xb)
                y_onehot = torch.zeros((len(yb), K), device=device, dtype=torch.float32)
                y_onehot.scatter_(1, yb.view(-1, 1), 1.0)
                vloss = edl_loss(alpha, y_onehot, num_classes=K, anneal=1.0, loss_type=str(loss_type)).mean()
                vtotal += float(vloss.item()) * len(yb)
                vn += len(yb)
        vavg = vtotal / max(1, vn)
        if verbose:
            print(f"[EDL] epoch {epoch:02d}/{epochs}  train_loss={total/max(1,n):.4f}  val_loss={vavg:.4f}")

        if vavg + 1e-6 < best_val:
            best_val = vavg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[EDL] early stop (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    artifacts = EDLArtifacts(
        hypotheses=list(hypotheses),
        feature_cols=list(feature_cols),
        preprocessor=pre,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        config={
            "prefixes": list(prefixes),
            "max_features": int(max_features),
            "hidden_sizes": list(hidden_sizes),
            "dropout": float(dropout),
            "lr": float(lr),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "anneal_epochs": int(anneal_epochs),
            "weight_decay": float(weight_decay),
            "loss_type": str(loss_type),
            "seed": int(seed),
            "device": str(device),
        },
    )
    return artifacts, model


@torch.no_grad()
def predict_edl(
    df: pd.DataFrame,
    artifacts: EDLArtifacts,
    *,
    device: str = "cpu",
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict with trained EDL model.

    Returns:
      probs: (N,K)
      uncertainty: (N,)
      pred_idx: (N,)
    """
    _configure_torch_threads()
    feature_cols = artifacts.feature_cols
    pre = artifacts.preprocessor

    # Ensure missing cols exist
    Xdf = df.copy()
    for c in feature_cols:
        if c not in Xdf.columns:
            Xdf[c] = np.nan

    X = pre.transform(Xdf[feature_cols])
    X = X.toarray() if hasattr(X, "toarray") else X
    X = torch.tensor(X, dtype=torch.float32)

    K = len(artifacts.hypotheses)
    model = EDLMLP(
        input_dim=int(X.shape[1]),
        num_classes=int(K),
        hidden_sizes=artifacts.config.get("hidden_sizes", [256, 128]),
        dropout=float(artifacts.config.get("dropout", 0.2)),
    )
    model.load_state_dict(artifacts.state_dict)
    model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(X), batch_size=int(batch_size), shuffle=False, drop_last=False)
    probs_all = []
    u_all = []
    pred_all = []

    for (xb,) in loader:
        xb = xb.to(device)
        alpha = model(xb)
        fw = EDLForward.from_alpha(alpha)
        probs = fw.probs.detach().cpu().numpy()
        u = fw.uncertainty.detach().cpu().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int64)
        probs_all.append(probs)
        u_all.append(u)
        pred_all.append(pred)

    probs_all = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, K), dtype=np.float32)
    u_all = np.concatenate(u_all, axis=0) if u_all else np.zeros((0,), dtype=np.float32)
    pred_all = np.concatenate(pred_all, axis=0) if pred_all else np.zeros((0,), dtype=np.int64)
    return probs_all.astype(np.float32), u_all.astype(np.float32), pred_all
