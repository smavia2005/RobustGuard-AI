
# RobustGuard AI - Final Audio Validation
# Tests previously unused ASVspoof recordings.
# Compares 50% vs 90% thresholds.
# CPU | Python 3.15

import sys
import csv
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from audio_web_detector import predict_audio_path

OUT = ROOT / "training" / "outputs"

DATA = (
    ROOT / "datasets" / "audio"
    / "ASVspoof2019_LA"
    / "test-00000-of-00009.parquet"
)

OLD_CSV = OUT / "audio_local_predictions.csv"
NEW_CSV = OUT / "audio_final_validation.csv"
REPORT = OUT / "audio_final_validation.json"

TARGET_PER_CLASS = 50

FIELDS = [
    "audio_id",
    "actual",
    "fake_score",
    "status",
    "error"
]


def read_csv(path):
    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_rows(rows):
    with open(NEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(rows, threshold):
    actual = np.array([
        1 if r["actual"] == "Fake" else 0
        for r in rows
    ])

    scores = np.array([
        float(r["fake_score"]) / 100.0
        for r in rows
    ])

    predicted = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        actual, predicted, labels=[1, 0]
    )

    real_total = int(cm[1].sum())
    fake_total = int(cm[0].sum())

    return {
        "threshold": threshold,
        "samples": len(rows),
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "precision_fake": float(
            precision_score(
                actual, predicted, zero_division=0
            )
        ),
        "recall_fake": float(
            recall_score(
                actual, predicted, zero_division=0
            )
        ),
        "recall_real": float(
            cm[1, 1] / max(real_total, 1)
        ),
        "f1_fake": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "false_positives": int(cm[1, 0]),
        "false_negatives": int(cm[0, 1]),
        "real_correct": int(cm[1, 1]),
        "fake_correct": int(cm[0, 0]),
        "real_total": real_total,
        "fake_total": fake_total,
        "confusion_matrix": cm.tolist()
    }


def show_metrics(result):
    print("\n" + "=" * 40)
    print(
        f"THRESHOLD: {result['threshold'] * 100:.0f}%"
    )
    print("=" * 40)

    for name in [
        "accuracy",
        "balanced_accuracy",
        "precision_fake",
        "recall_fake",
        "recall_real",
        "f1_fake"
    ]:
        print(
            f"{name}: {result[name] * 100:.2f}%"
        )

    print(
        "False positives:",
        result["false_positives"]
    )
    print(
        "False negatives:",
        result["false_negatives"]
    )
    print(
        "Confusion matrix:",
        result["confusion_matrix"]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA}"
        )

    if not OLD_CSV.exists():
        raise FileNotFoundError(
            f"Previous evaluation missing: {OLD_CSV}"
        )

    # Exclude ALL previously attempted audio IDs.
    old_ids = {
        r["audio_id"]
        for r in read_csv(OLD_CSV)
    }

    new_rows = read_csv(NEW_CSV)

    new_ids = {
        r["audio_id"]
        for r in new_rows
    }

    counts = {
        "Real": sum(
            r["status"] == "success"
            and r["actual"] == "Real"
            for r in new_rows
        ),
        "Fake": sum(
            r["status"] == "success"
            and r["actual"] == "Fake"
            for r in new_rows
        )
    }

    print("\nROBUSTGUARD FINAL AUDIO VALIDATION")
    print("Previously used IDs:", len(old_ids))
    print("New validation progress:", counts)

    parquet = pq.ParquetFile(DATA)

    for batch in parquet.iter_batches(
        batch_size=32,
        columns=["path", "audio", "label"]
    ):

        for item in batch.to_pylist():

            audio_id = str(item["path"])

            if audio_id in old_ids:
                continue

            if audio_id in new_ids:
                continue

            label = item["label"]

            if label in (0, "0", "bonafide"):
                actual = "Real"
            elif label in (1, "1", "spoof"):
                actual = "Fake"
            else:
                continue

            if counts[actual] >= TARGET_PER_CLASS:
                continue

            row = {
                "audio_id": audio_id,
                "actual": actual,
                "fake_score": "",
                "status": "failed",
                "error": ""
            }

            print(
                f"\n{actual} "
                f"{counts[actual] + 1}/{TARGET_PER_CLASS}: "
                f"{audio_id}"
            )

            try:
                audio_bytes = item["audio"]["bytes"]

                if not audio_bytes:
                    raise ValueError("Audio bytes missing.")

                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "sample.flac"
                    path.write_bytes(audio_bytes)

                    result = predict_audio_path(path)

                row["fake_score"] = result[
                    "class_scores"
                ]["Fake"]

                row["status"] = "success"
                counts[actual] += 1

                print(
                    "Fake score:",
                    row["fake_score"],
                    "%"
                )

            except Exception as error:
                row["error"] = str(error)
                print("ERROR:", error)

            new_rows.append(row)
            new_ids.add(audio_id)

            # Save after every sample.
            save_rows(new_rows)

            if all(
                counts[c] >= TARGET_PER_CLASS
                for c in counts
            ):
                break

        if all(
            counts[c] >= TARGET_PER_CLASS
            for c in counts
        ):
            break

    successful = [
        r for r in new_rows
        if r["status"] == "success"
    ]

    print("\nValidation counts:", counts)

    if min(counts.values()) < TARGET_PER_CLASS:
        print(
            "Warning: target not fully reached. "
            "Results are based on available samples."
        )

    if min(counts.values()) == 0:
        print("Both classes are required.")
        return

    baseline = evaluate(successful, 0.50)
    calibrated = evaluate(successful, 0.90)

    show_metrics(baseline)
    show_metrics(calibrated)

    improved = (
        calibrated["balanced_accuracy"]
        > baseline["balanced_accuracy"]
        and calibrated["false_positives"]
        < baseline["false_positives"]
        and calibrated["recall_fake"] >= 0.90
    )

    selected = 0.90 if improved else 0.50

    print("\n" + "=" * 40)
    print("FINAL AUDIO THRESHOLD DECISION")
    print("=" * 40)

    print(
        "Recommended threshold:",
        f"{selected * 100:.0f}%"
    )

    if improved:
        print(
            "90% improved balanced accuracy "
            "and reduced false positives."
        )
    else:
        print(
            "90% did not meet the improvement "
            "criteria. Retain 50% for now."
        )

    report = {
        "dataset": "ASVspoof2019 LA - local shard 0",
        "unused_samples": len(successful),
        "counts": counts,
        "baseline": baseline,
        "calibrated": calibrated,
        "recommended_threshold": selected,
        "improved": improved,
        "notes": (
            "Previously evaluated sample IDs were excluded. "
            "This is an unused-sample holdout within the "
            "same dataset shard, not a separate external "
            "dataset. Model scores are not calibrated "
            "probabilities."
        )
    }

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:", NEW_CSV)
    print("Saved:", REPORT)
    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()
