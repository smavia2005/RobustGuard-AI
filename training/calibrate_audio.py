
# RobustGuard AI - Audio Threshold Calibration
# Uses existing 400 saved predictions
# No model download or audio processing

import csv
import json
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "outputs"

INPUT_FILE = OUT / "audio_local_predictions.csv"
OUTPUT_FILE = OUT / "audio_threshold.json"

BASELINE_THRESHOLD = 0.50
RANDOM_STATE = 42

# Minimum Fake recall on calibration data.
MIN_FAKE_RECALL = 0.90


def load_predictions():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing predictions: {INPUT_FILE}"
        )

    records = []
    seen_ids = set()

    with open(
        INPUT_FILE,
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("status", "").lower() != "success":
                continue

            actual = row.get(
                "actual", ""
            ).strip().lower()

            if actual not in ("real", "fake"):
                continue

            audio_id = row.get("audio_id", "").strip()

            if not audio_id:
                continue

            if audio_id in seen_ids:
                raise ValueError(
                    f"Duplicate audio ID: {audio_id}"
                )

            seen_ids.add(audio_id)

            try:
                score = float(row["fake_score"])

                # Existing CSV stores scores as percentages.
                if not 0 <= score <= 100:
                    raise ValueError(
                        "Score outside 0-100."
                    )

                records.append({
                    "audio_id": audio_id,
                    "actual": 1 if actual == "fake" else 0,
                    "fake_score": score / 100.0
                })

            except (ValueError, KeyError, TypeError):
                continue

    if len(records) < 40:
        raise RuntimeError(
            "Not enough valid predictions."
        )

    return records


def calculate_metrics(y_true, scores, threshold):

    predictions = (
        scores >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[1, 0]
    )

    fake_correct = int(cm[0, 0])
    fake_missed = int(cm[0, 1])

    real_wrong = int(cm[1, 0])
    real_correct = int(cm[1, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision_fake": float(
            precision_score(
                y_true, predictions,
                zero_division=0
            )
        ),
        "recall_fake": float(
            recall_score(
                y_true, predictions,
                zero_division=0
            )
        ),
        "recall_real": (
            real_correct /
            max(real_correct + real_wrong, 1)
        ),
        "f1_fake": float(
            f1_score(
                y_true, predictions,
                zero_division=0
            )
        ),
        "false_positives": real_wrong,
        "false_negatives": fake_missed,
        "real_correct": real_correct,
        "fake_correct": fake_correct,
        "confusion_matrix": cm.tolist()
    }


def print_results(title, metrics):

    print("\n" + "=" * 44)
    print(title)
    print("=" * 44)

    print(
        f"Threshold: "
        f"{metrics['threshold'] * 100:.2f}%"
    )

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "precision_fake",
        "recall_fake",
        "recall_real",
        "f1_fake"
    ]:
        print(
            f"{metric}: "
            f"{metrics[metric] * 100:.2f}%"
        )

    print(
        "False positives:",
        metrics["false_positives"]
    )

    print(
        "False negatives:",
        metrics["false_negatives"]
    )

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def main():

    print("\nROBUSTGUARD AUDIO THRESHOLD CALIBRATION")

    records = load_predictions()

    actual = np.array([
        r["actual"]
        for r in records
    ], dtype=int)

    scores = np.array([
        r["fake_score"]
        for r in records
    ], dtype=float)

    # Separate calibration and comparison subsets.
    indices = np.arange(len(records))

    calibration_idx, comparison_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=actual,
        random_state=RANDOM_STATE
    )

    y_cal = actual[calibration_idx]
    scores_cal = scores[calibration_idx]

    y_test = actual[comparison_idx]
    scores_test = scores[comparison_idx]

    print("Total samples:", len(records))
    print("Calibration samples:", len(y_cal))
    print("Comparison samples:", len(y_test))

    # Select threshold using ONLY calibration data.
    candidates = np.arange(
        0.50,
        0.951,
        0.01
    )

    candidates = np.round(candidates, 2)

    eligible = []

    for threshold in candidates:

        metrics = calculate_metrics(
            y_cal,
            scores_cal,
            threshold
        )

        if metrics["recall_fake"] >= MIN_FAKE_RECALL:
            eligible.append(metrics)

    if not eligible:
        selected_threshold = BASELINE_THRESHOLD

    else:
        # Prioritise balanced accuracy,
        # then fewer false positives.
        best = max(
            eligible,
            key=lambda m: (
                m["balanced_accuracy"],
                -m["false_positives"],
                -abs(m["threshold"] - 0.5)
            )
        )

        selected_threshold = best["threshold"]

    print(
        "\nSelected threshold:",
        f"{selected_threshold * 100:.2f}%"
    )

    baseline = calculate_metrics(
        y_test,
        scores_test,
        BASELINE_THRESHOLD
    )

    calibrated = calculate_metrics(
        y_test,
        scores_test,
        selected_threshold
    )

    print_results(
        "BASELINE ON COMPARISON SET",
        baseline
    )

    print_results(
        "CALIBRATED ON COMPARISON SET",
        calibrated
    )

    balanced_improved = (
        calibrated["balanced_accuracy"]
        > baseline["balanced_accuracy"]
    )

    false_positives_reduced = (
        calibrated["false_positives"]
        < baseline["false_positives"]
    )

    fake_recall_acceptable = (
        calibrated["recall_fake"]
        >= MIN_FAKE_RECALL
    )

    improvement = (
        balanced_improved
        and false_positives_reduced
        and fake_recall_acceptable
    )

    report = {
        "dataset": "ASVspoof 2019 LA",
        "total_samples": len(records),
        "calibration_samples": len(y_cal),
        "comparison_samples": len(y_test),
        "baseline_threshold": BASELINE_THRESHOLD,
        "selected_threshold": selected_threshold,
        "minimum_fake_recall": MIN_FAKE_RECALL,
        "baseline_comparison": baseline,
        "calibrated_comparison": calibrated,
        "improved_on_comparison_set": bool(improvement),
        "note": (
            "Exploratory calibration using existing "
            "ASVspoof evaluation predictions. "
            "This is not independent validation. "
            "Do not repeatedly tune on the "
            "comparison subset."
        )
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print("\n====================================")
    print("FINAL DECISION")
    print("====================================")

    if improvement:

        print(
            "Promising improvement on comparison set."
        )

        print(
            "Independent validation is recommended "
            "before changing the website threshold."
        )

    else:

        print(
            "No acceptable improvement demonstrated."
        )

        print(
            "Keep the original 50% threshold."
        )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()