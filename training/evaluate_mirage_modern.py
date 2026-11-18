
# RobustGuard AI - Modern Image Benchmark
# Local dataset evaluation | Python 3.15 | CPU
# Evaluation only - no retraining

import sys
import csv
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------
# PATH CONFIGURATION
# --------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from mirage_detector import load_detector

DATASET_ID = (
    "julienlucas/"
    "midjourney-dalle-sd-nanobananapro-dataset"
)

DATASET_DIR = ROOT / "datasets" / "modern_images" / "data"

OUT = ROOT / "training" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PER_CLASS = 1000
BATCH_SIZE = 8
SEED = 42

torch.set_num_threads(2)


# --------------------------------------
# LABEL CONVERSION
# --------------------------------------

def decode_label(value, class_names=None):

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, str):

        name = value.strip().lower()

        if name in ("fake", "ai", "generated", "ai-generated"):
            return 1

        if name in ("real", "human", "authentic"):
            return 0

    if isinstance(value, int):

        if class_names is not None:
            return decode_label(class_names[value])

        # Dataset label mapping must be verified
        # against the downloaded dataset metadata.
        # Currently assumed:
        # 0 = Fake, 1 = Real

        if value == 0:
            return 1

        if value == 1:
            return 0

    raise ValueError(f"Unexpected dataset label: {value}")


# --------------------------------------
# BATCH EVALUATION
# --------------------------------------

def evaluate_batch(
    images,
    labels,
    model,
    processor,
    truths,
    predictions,
    records
):

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    with torch.inference_mode():

        logits = model(**inputs).logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

    for i, actual in enumerate(labels):

        # Mirage model:
        # Class index 0 = Real
        # Class index 1 = Fake

        fake_score = float(
            probabilities[i, 1].item()
        )

        predicted = int(fake_score >= 0.5)

        truths.append(actual)
        predictions.append(predicted)

        records.append({
            "actual": "Fake" if actual == 1 else "Real",
            "predicted": "Fake" if predicted == 1 else "Real",
            "fake_score_percent": round(
                fake_score * 100, 4
            ),
            "correct": actual == predicted
        })


# --------------------------------------
# MAIN EVALUATION
# --------------------------------------

def main():

    print("\n======================================")
    print("ROBUSTGUARD - MODERN IMAGE EVALUATION")
    print("======================================")

    print("Dataset:", DATASET_ID)

    # ----------------------------------
    # LOAD LOCAL DATASET
    # ----------------------------------

    print("\nChecking local dataset files...")

    local_files = sorted(
        DATASET_DIR.glob("test-*.parquet")
    )

    if not local_files:

        raise FileNotFoundError(
            f"No test parquet files found in: {DATASET_DIR}"
        )

    print("Files found:", len(local_files))

    for file in local_files:
        print(" -", file.name)

    print("\nLoading local test dataset...")

    ds = load_dataset(
        "parquet",
        data_files={
            "test": [
                str(file) for file in local_files
            ]
        },
        split="test",
        streaming=False
    )

    print("Total test images:", len(ds))
    print("Dataset columns:", ds.column_names)

    if "image" not in ds.column_names:
        raise ValueError(
            "Dataset does not contain an 'image' column."
        )

    if "label" not in ds.column_names:
        raise ValueError(
            "Dataset does not contain a 'label' column."
        )

    feature = ds.features.get("label")
    class_names = getattr(feature, "names", None)

    print("Dataset label names:", class_names)

    print("\nSelecting balanced evaluation subset...")

    # Non-streaming Hugging Face Dataset:
    # shuffle() does NOT accept buffer_size.

    ds = ds.shuffle(seed=SEED)

    images = []
    labels = []

    # 0 = Real, 1 = Fake
    counts = {
        0: 0,
        1: 0
    }

    # ----------------------------------
    # LOAD MIRAGE
    # ----------------------------------

    print("\nLoading Mirage model...")

    processor, model = load_detector()
    model.eval()

    if model.config.num_labels != 2:
        raise ValueError(
            "Expected Mirage model with 2 output classes."
        )

    print("Model labels:", model.config.id2label)

    truths = []
    predictions = []
    records = []

    # ----------------------------------
    # MAKE PREDICTIONS
    # ----------------------------------

    print("\nStarting image evaluation...")

    for row in ds:

        actual = decode_label(
            row["label"],
            class_names
        )

        if counts[actual] >= PER_CLASS:
            continue

        image = row["image"]

        if not hasattr(image, "convert"):
            raise ValueError(
                "Image could not be decoded as a PIL image."
            )

        images.append(
            image.convert("RGB")
        )

        labels.append(actual)

        counts[actual] += 1

        if len(images) >= BATCH_SIZE:

            evaluate_batch(
                images,
                labels,
                model,
                processor,
                truths,
                predictions,
                records
            )

            images = []
            labels = []

            print(
                f"\rEvaluated: {len(truths)}/{PER_CLASS * 2}",
                end="",
                flush=True
            )

        if all(
            count >= PER_CLASS
            for count in counts.values()
        ):
            break

    # Process remaining images

    if images:

        evaluate_batch(
            images,
            labels,
            model,
            processor,
            truths,
            predictions,
            records
        )

    if counts != {
        0: PER_CLASS,
        1: PER_CLASS
    }:

        raise RuntimeError(
            f"Not enough balanced samples: {counts}"
        )

    # ----------------------------------
    # CALCULATE METRICS
    # ----------------------------------

    print("\n\nCalculating metrics...")

    accuracy = accuracy_score(
        truths,
        predictions
    )

    precision = precision_score(
        truths,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        truths,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        truths,
        predictions,
        zero_division=0
    )

    # Rows: Actual Fake, Real
    # Columns: Predicted Fake, Real

    cm = confusion_matrix(
        truths,
        predictions,
        labels=[1, 0]
    )

    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    # ----------------------------------
    # PRINT RESULTS
    # ----------------------------------

    print("\n================================")
    print("MIRAGE MODERN IMAGE TEST RESULTS")
    print("================================")

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")

    print("                 Pred Fake  Pred Real")
    print(f"Actual Fake      {tp:>9}  {fn:>9}")
    print(f"Actual Real      {fp:>9}  {tn:>9}")

    print("\nClassification Report:")

    print(
        classification_report(
            truths,
            predictions,
            labels=[1, 0],
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    # ----------------------------------
    # SAVE JSON METRICS
    # ----------------------------------

    metrics = {
        "model": "Mirage SigLIP2",
        "dataset": DATASET_ID,
        "split": "test",
        "samples": len(truths),
        "fake_samples": counts[1],
        "real_samples": counts[0],
        "accuracy": float(accuracy),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn
        },
        "limitations": [
            "Evaluation uses a balanced 200-image subset.",
            "Dataset label mapping must be verified.",
            "Training-data overlap has not been verified.",
            "Results may differ on other AI generators."
        ]
    }

    metrics_path = OUT / "mirage_modern_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # ----------------------------------
    # SAVE CSV PREDICTIONS
    # ----------------------------------

    csv_path = OUT / "mirage_modern_predictions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "actual",
                "predicted",
                "fake_score_percent",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    # ----------------------------------
    # SAVE CONFUSION MATRIX
    # ----------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "Mirage - Modern Image Dataset"
    )

    fig.tight_layout()

    plot_path = (
        OUT / "mirage_modern_confusion_matrix.png"
    )

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n================================")
    print("FILES SAVED SUCCESSFULLY")
    print("================================")

    print(metrics_path)
    print(csv_path)
    print(plot_path)

    print("\nEVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
