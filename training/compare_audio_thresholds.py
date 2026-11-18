
import csv
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = (
    ROOT / "training" / "outputs"
    / "audio_final_validation.csv"
)

rows = []

with open(CSV_FILE, newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        if row["status"] == "success":
            rows.append(row)

actual = np.array([
    1 if r["actual"] == "Fake" else 0
    for r in rows
])

scores = np.array([
    float(r["fake_score"]) / 100
    for r in rows
])

print("\nROBUSTGUARD AUDIO THRESHOLD COMPARISON")
print("Samples:", len(rows))

thresholds = [
    0.50, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95
]

print(
    "\nThreshold | Accuracy | Real Recall | "
    "Fake Recall | False Positives | False Negatives"
)

print("-" * 85)

for threshold in thresholds:

    pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        actual, pred, labels=[1, 0]
    )

    fake_correct = cm[0, 0]
    fake_missed = cm[0, 1]
    real_wrong = cm[1, 0]
    real_correct = cm[1, 1]

    accuracy = accuracy_score(actual, pred) * 100

    real_recall = (
        real_correct /
        max(real_correct + real_wrong, 1)
    ) * 100

    fake_recall = recall_score(
        actual, pred
    ) * 100

    print(
        f"{threshold * 100:>7.0f}%   | "
        f"{accuracy:>7.2f}% | "
        f"{real_recall:>9.2f}% | "
        f"{fake_recall:>9.2f}% | "
        f"{real_wrong:>15} | "
        f"{fake_missed:>15}"
    )

print("\nCOMPARISON COMPLETE!")
