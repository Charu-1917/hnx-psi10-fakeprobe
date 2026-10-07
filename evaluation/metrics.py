"""
Forensic Evaluation Metrics Engine for FakeProbe-X.
Calculates Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, FPR, FNR, Brier Score, and Confusion Matrix.
Handles edge cases with zero division gracefully and clearly outputs NOT AVAILABLE where appropriate.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)


def calculate_forensic_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive binary forensic metrics.
    Label convention: 0 = REAL / Bonafide, 1 = FAKE / Spoof.
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {
            "status": "DATASET REQUIRED",
            "sample_count": 0,
            "real_count": 0,
            "fake_count": 0,
            "accuracy": "NOT AVAILABLE",
            "precision": "NOT AVAILABLE",
            "recall": "NOT AVAILABLE",
            "specificity": "NOT AVAILABLE",
            "f1_score": "NOT AVAILABLE",
            "roc_auc": "NOT AVAILABLE",
            "fpr": "NOT AVAILABLE",
            "fnr": "NOT AVAILABLE",
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "brier_score": "NOT AVAILABLE",
        }

    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    n_samples = len(y_t)
    n_real = int(np.sum(y_t == 0))
    n_fake = int(np.sum(y_t == 1))

    # Confusion matrix elements: TN, FP, FN, TP
    # [[TN, FP], [FN, TP]]
    try:
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = float(accuracy_score(y_t, y_p))

    # Precision & Recall
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))

    # Specificity: TN / (TN + FP)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # F1 Score
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # FPR & FNR
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC & Brier Score
    if y_prob is not None and len(y_prob) == n_samples and n_real > 0 and n_fake > 0:
        try:
            auc = float(roc_auc_score(y_t, y_prob))
        except Exception:
            auc = "NOT AVAILABLE"

        try:
            brier = float(brier_score_loss(y_t, y_prob))
        except Exception:
            brier = "NOT AVAILABLE"
    else:
        auc = "NOT AVAILABLE"
        brier = "NOT AVAILABLE"

    return {
        "status": "COMPUTED",
        "sample_count": n_samples,
        "real_count": n_real,
        "fake_count": n_fake,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "brier_score": brier,
    }
