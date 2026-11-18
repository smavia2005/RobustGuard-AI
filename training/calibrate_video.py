
# RobustGuard AI - Video Threshold Calibration
# Uses saved SDFVD predictions
# No model retraining

import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "outputs"

CSV_PATH = OUT / "face_video_predictions.csv"
CONFIG_PATH = OUT / "video_threshold.json"

SEED = 42


def pair_id_from_path(path):
    name = Path(path).stem.lower()

    match = re.fullmatch(r"vs?(\d+)", name)

    if not match:
        raise ValueError(f"Unknown video name: {name}")

    return int(match.group(1))


def load_predictions():
    rows = []

    with open(CSV_PATH, encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:

            if row["predicted"] == "FAILED":
                continue

            if not row["fake_score_percent"]:
                continue

            actual = (
                1 if row["actual"].strip().lower() == "fake"
                else 0
            )

            rows.append({
                "pair_id": pair_id_from_path(row["video"]),
                "actual": actual,
                "fake_score": float(
                    row["fake_score_percent"]
                ) / 100.0
            })

    return rows


def get_metrics(rows, threshold):
    y_true = np.array(
        [row["actual"] for row in rows]
    )

    scores = np.array(
        [row["fake_score"] for row in rows]
    )

    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        y_true, y_pred, labels=[1, 0]
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "fake_correct": int(cm[0, 0]),
        "fake_missed": int(cm[0, 1]),
        "real_wrong": int(cm[1, 0]),
        "real_correct": int(cm[1, 1])
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * 35)

    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1: {metrics['f1'] * 100:.2f}%")

    print("\nConfusion matrix:")
    print("                 Pred Fake  Pred Real")
    print(
        f"Actual Fake      "
        f"{metrics['fake_correct']:>9}  "
        f"{metrics['fake_missed']:>9}"
    )
    print(
        f"Actual Real      "
        f"{metrics['real_wrong']:>9}  "
        f"{metrics['real_correct']:>9}"
    )


def main():
    print("\nROBUSTGUARD VIDEO THRESHOLD CALIBRATION")

    rows = load_predictions()

    grouped = {}

    for row in rows:
        grouped.setdefault(row["pair_id"], []).append(row)

    # Only complete Real/Fake video pairs.
    valid_ids = sorted([
        pair_id for pair_id, items in grouped.items()
        if len(items) == 2
        and {item["actual"] for item in items} == {0, 1}
    ])

    if len(valid_ids) < 10:
        raise RuntimeError(
            "Not enough complete Real/Fake video pairs."
        )

    print("Complete pairs:", len(valid_ids))

    calibration_ids, test_ids = train_test_split(
        valid_ids,
        test_size=0.30,
        random_state=SEED
    )

    calibration_set = set(calibration_ids)
    test_set = set(test_ids)

    calibration_rows = [
        item for item in rows
        if item["pair_id"] in calibration_set
    ]

    test_rows = [
        item for item in rows
        if item["pair_id"] in test_set
    ]

    print("Calibration videos:", len(calibration_rows))
    print("Comparison videos:", len(test_rows))

    thresholds = np.arange(0.30, 0.951, 0.025)

    # Select using calibration data ONLY.
    # Prefer better balanced accuracy, then
    # fewer false positives on Real videos.
    candidates = []

    for threshold in thresholds:
        m = get_metrics(calibration_rows, threshold)

        candidates.append((
            m["balanced_accuracy"],
            -m["real_wrong"],
            -abs(threshold - 0.5),
            float(threshold)
        ))

    best_threshold = max(candidates)[3]

    print(
        f"\nSelected threshold: "
        f"{best_threshold * 100:.2f}%"
    )

    before = get_metrics(test_rows, 0.50)
    after = get_metrics(test_rows, best_threshold)

    print_metrics(
        "BASELINE ON COMPARISON SET (50%)",
        before
    )

    print_metrics(
        "CALIBRATED ON COMPARISON SET",
        after
    )

    improved = (
        after["balanced_accuracy"]
        > before["balanced_accuracy"]
    )

    config = {
        "selected_threshold": round(best_threshold, 4),
        "default_threshold": 0.5,
        "calibration_pairs": len(calibration_ids),
        "comparison_pairs": len(test_ids),
        "comparison_balanced_accuracy_before": float(
            before["balanced_accuracy"]
        ),
        "comparison_balanced_accuracy_after": float(
            after["balanced_accuracy"]
        ),
        "comparison_false_positives_before": int(
            before["real_wrong"]
        ),
        "comparison_false_positives_after": int(
            after["real_wrong"]
        ),
        "improved_on_comparison_set": bool(improved),
        "note": (
            "Exploratory threshold selection on SDFVD. "
            "This dataset was previously evaluated. "
            "Independent external validation is still needed."
        )
    }

    with open(
        CONFIG_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=4)

    print("\nSaved:", CONFIG_PATH)

    if improved:
        print("\nCalibration improved balanced accuracy.")
        print("Review false positives before deployment.")
    else:
        print("\nNo balanced-accuracy improvement observed.")
        print("Keep the existing 50% threshold for now.")


if __name__ == "__main__":
    main()
