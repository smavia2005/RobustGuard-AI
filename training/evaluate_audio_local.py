
# RobustGuard AI - Local Audio Dataset Evaluation
# ASVspoof2019 LA Parquet | Python 3.15 | CPU

import sys
import csv
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from audio_web_detector import predict_audio_path

DATA_FILE = (
    ROOT / "datasets" / "audio" / "ASVspoof2019_LA"
    / "test-00000-of-00009.parquet"
)

OUT = ROOT / "training" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUT / "audio_local_predictions.csv"
METRICS_FILE = OUT / "audio_local_metrics.json"
GRAPH_FILE = OUT / "audio_local_confusion_matrix.png"

TARGET = 200


def load_results():
    if not CSV_FILE.exists():
        return {}

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        return {
            row["audio_id"]: row
            for row in csv.DictReader(f)
        }


def save_results(records):
    with open(
        CSV_FILE, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_id", "actual", "predicted",
                "real_score", "fake_score",
                "status", "error"
            ]
        )
        writer.writeheader()
        writer.writerows(records.values())


def label_name(value):
    if value in (0, "bonafide", "real", "0"):
        return "Real"
    if value in (1, "spoof", "fake", "1"):
        return "Fake"
    raise ValueError(f"Unknown label: {value}")


def main():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_FILE}"
        )

    print("Opening local dataset:", DATA_FILE.name)

    parquet = pq.ParquetFile(DATA_FILE)

    records = load_results()

    counts = {"Real": 0, "Fake": 0}

    for row in records.values():
        if row["status"] == "success":
            counts[row["actual"]] += 1

    for batch in parquet.iter_batches(
        batch_size=32,
        columns=["path", "audio", "label"]
    ):
        data = batch.to_pylist()

        for item in data:
            actual = label_name(item["label"])
            audio_id = str(item["path"])

            if counts[actual] >= TARGET:
                continue

            if (
                audio_id in records
                and records[audio_id]["status"] == "success"
            ):
                continue

            print(
                f"\n{actual}: "
                f"{counts[actual] + 1}/{TARGET} "
                f"| {audio_id}"
            )

            record = {
                "audio_id": audio_id,
                "actual": actual,
                "predicted": "",
                "real_score": "",
                "fake_score": "",
                "status": "failed",
                "error": ""
            }

            try:
                audio = item["audio"]

                if not isinstance(audio, dict):
                    raise ValueError(
                        "Unexpected Parquet audio format"
                    )

                audio_bytes = audio.get("bytes")

                if not audio_bytes:
                    raise ValueError("Audio bytes missing")

                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "sample.flac"
                    path.write_bytes(audio_bytes)

                    result = predict_audio_path(path)

                record["predicted"] = result["prediction"]
                record["real_score"] = result["class_scores"]["Real"]
                record["fake_score"] = result["class_scores"]["Fake"]
                record["status"] = "success"

                counts[actual] += 1

                print("Prediction:", record["predicted"])

            except Exception as error:
                record["error"] = str(error)
                print("FAILED:", error)

            records[audio_id] = record
            save_results(records)

        if all(counts[x] >= TARGET for x in counts):
            break

    successful = [
        row for row in records.values()
        if row["status"] == "success"
    ]

    y_true = [
        1 if row["actual"] == "Fake" else 0
        for row in successful
    ]

    y_pred = [
        1 if row["predicted"] == "Fake" else 0
        for row in successful
    ]

    if len(set(y_true)) != 2:
        print("Not enough Real and Fake data for evaluation.")
        print("Current counts:", counts)
        return

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    metrics = {
        "dataset": "ASVspoof 2019 LA, local shard 0",
        "samples": len(y_true),
        "real_samples": y_true.count(0),
        "fake_samples": y_true.count(1),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision_fake": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_fake": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_real": (
            int(cm[1, 1]) / max(int(cm[1].sum()), 1)
        ),
        "f1_fake": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": cm.tolist()
    }

    print("\n==================================")
    print("ROBUSTGUARD AUDIO DATASET RESULTS")
    print("==================================")

    print("Samples:", len(y_true))

    for key in [
        "accuracy", "balanced_accuracy",
        "precision_fake", "recall_fake",
        "recall_real", "f1_fake"
    ]:
        print(f"{key}: {metrics[key] * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    ).plot(ax=ax, colorbar=False)

    ax.set_title("RobustGuard Audio Evaluation")
    fig.tight_layout()
    fig.savefig(GRAPH_FILE, dpi=200)
    plt.close(fig)

    print("\nSaved:", CSV_FILE)
    print("Saved:", METRICS_FILE)
    print("Saved:", GRAPH_FILE)


if __name__ == "__main__":
    main()
