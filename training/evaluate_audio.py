
# RobustGuard AI - Audio Dataset Evaluation
# ASVspoof 2019 LA
# Existing pretrained Wav2Vec2 detector
# CPU | Python 3.15
# No model training

import sys
import csv
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from datasets import load_dataset, Audio

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from audio_web_detector import predict_audio_path


DATASET_ID = (
    "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
)

OUTPUT = ROOT / "training" / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT / "audio_predictions.csv"
METRICS_PATH = OUTPUT / "audio_metrics.json"
GRAPH_PATH = OUTPUT / "audio_confusion_matrix.png"

# First batch: 50 Real + 50 Fake
TARGET_PER_CLASS = 50
MAX_SCANNED = 6000


def load_existing():

    records = {}

    if not CSV_PATH.exists():
        return records

    with open(
        CSV_PATH,
        newline="",
        encoding="utf-8"
    ) as file:

        for row in csv.DictReader(file):
            records[row["audio_id"]] = row

    return records


def save_records(records):

    fields = [
        "audio_id",
        "actual",
        "predicted",
        "real_score",
        "fake_score",
        "status",
        "error"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for row in records.values():
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })


def predict_dataset_audio(audio_data):

    audio_bytes = audio_data.get("bytes")

    if not audio_bytes:
        raise ValueError(
            "Audio bytes unavailable from dataset."
        )

    with tempfile.TemporaryDirectory() as folder:

        path = Path(folder) / "sample.flac"
        path.write_bytes(audio_bytes)

        # Use EXACT same detector as website.
        result = predict_audio_path(path)

    return result


def evaluate(records):

    successful = [
        row for row in records.values()
        if row["status"] == "success"
    ]

    if not successful:
        print("No successful predictions.")
        return

    actual = [
        1 if row["actual"] == "Fake" else 0
        for row in successful
    ]

    predicted = [
        1 if row["predicted"] == "Fake" else 0
        for row in successful
    ]

    if len(set(actual)) < 2:
        print(
            "Both Real and Fake classes are required."
        )
        return

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[1, 0]
    )

    metrics = {
        "model": "Wav2Vec2 Audio Deepfake Detector",
        "dataset": "ASVspoof 2019 LA",
        "successful_samples": len(successful),
        "real_samples": actual.count(0),
        "fake_samples": actual.count(1),
        "accuracy": accuracy_score(
            actual, predicted
        ),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "precision_fake": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall_fake": recall_score(
            actual, predicted, zero_division=0
        ),
        "recall_real": (
            int(cm[1, 1]) / max(int(sum(cm[1])), 1)
        ),
        "f1_fake": f1_score(
            actual, predicted, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "note": (
            "Exploratory balanced streaming subset. "
            "Samples are not guaranteed independent "
            "of pretrained model training data."
        )
    }

    print("\n===================================")
    print("ROBUSTGUARD AUDIO EVALUATION")
    print("===================================")

    print("Samples:", len(successful))

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision_fake",
        "recall_fake",
        "recall_real",
        "f1_fake"
    ]:
        print(
            f"{key}: {metrics[key] * 100:.2f}%"
        )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        actual,
        predicted,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    ).plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "RobustGuard Audio Confusion Matrix"
    )

    fig.tight_layout()
    fig.savefig(
        GRAPH_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved:")
    print(CSV_PATH)
    print(METRICS_PATH)
    print(GRAPH_PATH)


def main():

    print("\nROBUSTGUARD AUDIO DATASET EVALUATION")
    print("Loading ASVspoof streaming dataset...")

    dataset = load_dataset(
        DATASET_ID,
        split="test",
        streaming=True
    )

    # We want encoded FLAC bytes, not automatic
    # decoding that may require torchcodec.
    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False)
    )

    dataset = dataset.shuffle(
        seed=42,
        buffer_size=256
    )

    records = load_existing()

    counts = {"Real": 0, "Fake": 0}

    for row in records.values():
        if row["status"] == "success":
            counts[row["actual"]] += 1

    print("Previously completed:", counts)

    scanned = 0

    for sample in dataset:

        scanned += 1

        if scanned > MAX_SCANNED:
            print("Reached scanning limit.")
            break

        label = int(sample["label"])

        # Dataset labels:
        # 0 = bonafide (Real)
        # 1 = spoof (Fake)
        actual = "Real" if label == 0 else "Fake"

        if counts[actual] >= TARGET_PER_CLASS:
            continue

        audio_id = str(sample["path"])

        if audio_id in records:
            if records[audio_id]["status"] == "success":
                continue

        print(
            f"\n{actual} sample "
            f"{counts[actual] + 1}/{TARGET_PER_CLASS}"
        )

        result_row = {
            "audio_id": audio_id,
            "actual": actual,
            "predicted": "",
            "real_score": "",
            "fake_score": "",
            "status": "failed",
            "error": ""
        }

        try:
            prediction = predict_dataset_audio(
                sample["audio"]
            )

            result_row.update({
                "predicted": prediction["prediction"],
                "real_score": prediction[
                    "class_scores"
                ]["Real"],
                "fake_score": prediction[
                    "class_scores"
                ]["Fake"],
                "status": "success"
            })

            counts[actual] += 1

            print(
                "Prediction:",
                prediction["prediction"]
            )

        except Exception as error:
            result_row["error"] = str(error)
            print("FAILED:", str(error))

        records[audio_id] = result_row

        # Checkpoint after every sample.
        save_records(records)

        if all(
            count >= TARGET_PER_CLASS
            for count in counts.values()
        ):
            break

    print("\nCompleted counts:", counts)
    evaluate(records)

    print("\nAUDIO EVALUATION FINISHED!")


if __name__ == "__main__":
    main()
