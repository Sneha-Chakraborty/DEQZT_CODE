from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support


def binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    fpr = fp / (fp + tn + 1e-9)
    tpr = tp / (tp + fn + 1e-9)
    return {"fpr": float(fpr), "tpr": float(tpr), "fp": fp, "tn": tn, "tp": tp, "fn": fn}


def binary_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rates = binary_rates(y_true, y_pred)
    tp = float(rates["tp"])
    fp = float(rates["fp"])
    fn = float(rates["fn"])
    tn = float(rates["tn"])
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    npv = tn / (tn + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return {
        **rates,
        "precision_attack": float(precision),
        "recall_attack": float(recall),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1_attack": float(f1),
        "accuracy": float(acc),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(y_true) else 0.0,
    }


def _present_labels(y_true: Sequence[str], labels: Sequence[str]) -> List[str]:
    present = [str(lab) for lab in labels if np.any(np.asarray(y_true).astype(str) == str(lab))]
    return present if present else [str(lab) for lab in labels]


def multiclass_metrics(y_true, y_pred, labels):
    labels_present = _present_labels(y_true, labels)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    recalls = {}
    for i, lab in enumerate(labels_present):
        denom = cm[i, :].sum()
        recalls[lab] = float(cm[i, i] / (denom + 1e-9))
    return float(macro_f1), recalls, cm, labels_present


def multiclass_summary(y_true, y_pred, labels):
    labels_present = _present_labels(y_true, labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_present,
        average=None,
        zero_division=0,
    )
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)) if len(precision) else 0.0,
        "macro_recall": float(np.mean(recall)) if len(recall) else 0.0,
    }
    for lab in labels:
        slug = str(lab).lower()
        out[f"precision_{slug}"] = 0.0
        out[f"recall_{slug}"] = 0.0
        out[f"f1_{slug}"] = 0.0
        out[f"support_{slug}"] = 0
    for lab, p, r, f, s in zip(labels_present, precision, recall, f1, support):
        slug = str(lab).lower()
        out[f"precision_{slug}"] = float(p)
        out[f"recall_{slug}"] = float(r)
        out[f"f1_{slug}"] = float(f)
        out[f"support_{slug}"] = int(s)
    return out, labels_present


def confusion_matrix_df(cm: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(cm, index=[f"true::{x}" for x in labels], columns=[f"pred::{x}" for x in labels])


def _default_ideal_decision(label: str) -> int:
    label = str(label).upper()
    if label == "BENIGN":
        return 0
    if label in {"NETWORK_INTRUSION", "CREDENTIAL_MISUSE"}:
        return 1
    return 2


def rzdu_score(y_true_labels: Iterable[str], decisions: np.ndarray) -> Dict[str, float]:
    severity_weight = {
        "BENIGN": 1.0,
        "NETWORK_INTRUSION": 2.0,
        "CREDENTIAL_MISUSE": 3.0,
        "PRIV_ESC": 4.0,
        "LATERAL_MOVEMENT": 4.0,
        "DEFENSE_EVASION": 4.0,
        "DATA_EXFILTRATION": 5.0,
    }
    under_beta = {
        "BENIGN": 0.0,
        "NETWORK_INTRUSION": 1.0,
        "CREDENTIAL_MISUSE": 1.2,
        "PRIV_ESC": 1.5,
        "LATERAL_MOVEMENT": 1.5,
        "DEFENSE_EVASION": 1.5,
        "DATA_EXFILTRATION": 2.0,
    }
    over_alpha = 0.5
    labels = [str(x).upper() for x in y_true_labels]
    pred = np.asarray(decisions, dtype=np.int8)
    if len(labels) != len(pred):
        raise ValueError("rzdu_score: y_true_labels and decisions must have same length.")
    total_weight = 0.0
    weighted_penalty = 0.0
    max_penalty = 0.0
    for lab, d in zip(labels, pred):
        w = float(severity_weight.get(lab, 1.0))
        ideal = int(_default_ideal_decision(lab))
        dist = abs(int(d) - ideal)
        if int(d) > ideal:
            penalty = over_alpha * dist
        elif int(d) < ideal:
            penalty = float(under_beta.get(lab, 1.0)) * dist
        else:
            penalty = 0.0
        total_weight += w
        weighted_penalty += w * penalty
        max_penalty += w * max(float(under_beta.get(lab, 1.0)), over_alpha) * 2.0
    score = 1.0 - (weighted_penalty / (max_penalty + 1e-9))
    return {
        "rzdu": float(np.clip(score, 0.0, 1.0)),
        "rzdu_weighted_penalty": float(weighted_penalty),
        "rzdu_max_penalty": float(max_penalty),
        "rzdu_total_weight": float(total_weight),
    }


def save_confusion_matrix_plot(cm, labels, out_png, title: str = "Confusion Matrix"):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
