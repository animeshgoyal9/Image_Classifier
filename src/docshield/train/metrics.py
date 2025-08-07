from __future__ import annotations

from typing import Dict, List

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError("scikit-learn is required for metric computation") from exc


def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and macro-F1 scores.

    Parameters
    ----------
    preds:
        Predicted class indices.
    labels:
        Ground-truth class indices.

    Returns
    -------
    Dict[str, float]
        Dictionary containing ``accuracy``, ``precision``, ``recall``, and
        ``f1_macro`` metrics.
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
    }
