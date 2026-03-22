"""
Evaluation metrics for binary sarcasm classification.
Accuracy, Macro/Weighted/Binary F1, Precision, Recall, ROC-AUC, PR-AUC.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    y_true, y_pred: 0/1 labels.
    y_prob: predicted probabilities for positive class (optional, for AUC metrics).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = 0.0
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics
