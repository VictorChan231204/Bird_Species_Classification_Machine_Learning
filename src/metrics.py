"""Metrics: the two scores the brief asks for.

train.py and svm.py call score(y_true, y_pred) on the validation photos and on the test photos.
    Top-1 accuracy      - share of photos whose first guess is right (W5 s70)
    average per-class   - the accuracy of each species, then the mean of the 200 values (brief)
The two differ because the test set has 4-8 photos per species: Top-1 gives every photo equal weight,
the per-class average gives every species equal weight.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data import NUM_CLASSES


def per_class_accuracy(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    """Accuracy of each species: correct guesses for that species / photos of that species."""
    # Confusion matrix: row = true species, column = guessed species, cell = number of photos.
    matrix = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    # The diagonal holds the correct guesses; each row sum is the number of photos of that species.
    return matrix.diagonal() / matrix.sum(axis=1)


def score(y_true: list[int], y_pred: list[int]) -> dict:
    """Both scores as fractions from 0 to 1, plus the 200 per-species accuracies."""
    per_class = per_class_accuracy(y_true, y_pred)
    return {
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_per_class": float(per_class.mean()),
        "per_class": [round(float(a), 4) for a in per_class],
    }
