from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def binary_rates(y_true: np.ndarray, y_pred: np.ndarray):
    fp = np.sum((y_pred==1) & (y_true==0))
    tn = np.sum((y_pred==0) & (y_true==0))
    tp = np.sum((y_pred==1) & (y_true==1))
    fn = np.sum((y_pred==0) & (y_true==1))
    fpr = fp / (fp + tn + 1e-9)
    tpr = tp / (tp + fn + 1e-9)
    return dict(fpr=float(fpr), tpr=float(tpr), fp=int(fp), tn=int(tn), tp=int(tp), fn=int(fn))

def multiclass_metrics(y_true, y_pred, labels):
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = {}
    for i, lab in enumerate(labels):
        denom = cm[i,:].sum()
        recalls[lab] = float(cm[i,i] / (denom + 1e-9))
    return float(macro_f1), recalls, cm

def save_confusion_matrix_plot(cm, labels, out_png):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
