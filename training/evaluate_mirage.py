
# RobustGuard AI - Mirage Image Model Evaluation
# CIFAKE official test dataset
# Python 3.15 | CPU

import sys
import json
import csv
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"

sys.path.insert(0, str(BACKEND))

from mirage_detector import load_detector

DATASET = (
    ROOT / "datasets" / "image"
    / "test-00000-of-00001.parquet"
)

OUTPUT = ROOT / "training" / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

# 500 Fake + 500 Real = 1000 images
SAMPLES_PER_CLASS = 500
BATCH_SIZE = 8
SEED = 42

# CIFAKE: 0 = Fake, 1 = Real
# Evaluation: 1 = Fake (positive class)
#             0 = Real

random.seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


def decode_image(image_data):
    if isinstance(image_data, dict):
        content = image_data.get("bytes")

        if content is not None:
            image = Image.open(BytesIO(content))
        elif image_data.get("path"):
            image = Image.open(image_data["path"])
        else:
            raise ValueError("Missing image data")

    elif isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))

    else:
        raise ValueError("Unknown image format")

    with image:
        return image.convert("RGB")


def main():
    print("\n===================================")
    print("ROBUSTGUARD AI - MIRAGE EVALUATION")
    print("===================================")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"CIFAKE dataset not found: {DATASET}"
        )

    print("\nLoading CIFAKE test dataset...")
    df = pd.read_parquet(DATASET)

    if not {"image", "label"}.issubset(df.columns):
        raise ValueError("Required columns missing")

    print("Total test images:", len(df))

    fake_df = df[df["label"] == 0]
    real_df = df[df["label"] == 1]

    if len(fake_df) < SAMPLES_PER_CLASS:
        raise ValueError("Not enough Fake samples")

    if len(real_df) < SAMPLES_PER_CLASS:
        raise ValueError("Not enough Real samples")

    fake_df = fake_df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=SEED
    )

    real_df = real_df.sample(
        n=SAMPLES_PER_CLASS,
        random_state=SEED
    )

    selected = pd.concat(
        [fake_df, real_df],
        ignore_index=True
    )

    selected = selected.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    print("Selected images:", len(selected))
    print("Fake:", SAMPLES_PER_CLASS)
    print("Real:", SAMPLES_PER_CLASS)

    print("\nLoading pretrained Mirage...")
    processor, model = load_detector()

    model.eval()

    if model.config.num_labels != 2:
        raise ValueError(
            "Expected 2 output classes"
        )

    print("Model labels:", model.config.id2label)
    print("Starting predictions...")

    y_true = []
    y_pred = []
    results = []

    total = len(selected)

    for start in range(0, total, BATCH_SIZE):
        batch = selected.iloc[
            start:start + BATCH_SIZE
        ]

        images = [
            decode_image(item)
            for item in batch["image"]
        ]

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        with torch.inference_mode():
            logits = model(**inputs).logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            ).cpu().numpy()

        # Mirage mapping used in our working detector:
        # index 0 = Real
        # index 1 = Fake
        fake_scores = probabilities[:, 1]

        for local_idx, (_, row) in enumerate(
            batch.iterrows()
        ):
            actual = (
                1 if int(row["label"]) == 0
                else 0
            )

            fake_score = float(
                fake_scores[local_idx]
            )

            predicted = (
                1 if fake_score >= 0.5
                else 0
            )

            y_true.append(actual)
            y_pred.append(predicted)

            results.append({
                "sample_index": start + local_idx,
                "actual": (
                    "Fake" if actual == 1
                    else "Real"
                ),
                "predicted": (
                    "Fake" if predicted == 1
                    else "Real"
                ),
                "fake_score_percent": round(
                    fake_score * 100, 4
                ),
                "correct": actual == predicted
            })

        processed = min(
            start + BATCH_SIZE,
            total
        )

        print(
            f"\rProcessed: {processed}/{total}",
            end="",
            flush=True
        )

    print("\n\nCalculating evaluation metrics...")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    # Order: Fake, Real
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, 0]
    )

    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    print("\n===================================")
    print("MIRAGE - TEST RESULTS")
    print("===================================")

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print("                Pred Fake  Pred Real")
    print(f"Actual Fake     {tp:>9}  {fn:>9}")
    print(f"Actual Real     {fp:>9}  {tn:>9}")

    report = classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    metrics = {
        "model": "Mirage SigLIP2",
        "dataset": "CIFAKE official test subset",
        "seed": SEED,
        "samples": len(y_true),
        "fake_samples": SAMPLES_PER_CLASS,
        "real_samples": SAMPLES_PER_CLASS,
        "accuracy": round(float(accuracy), 6),
        "precision_fake": round(
            float(precision), 6
        ),
        "recall_fake": round(float(recall), 6),
        "f1_fake": round(float(f1), 6),
        "confusion_matrix": {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn
        },
        "limitations": [
            "CIFAKE images are low resolution.",
            "Results may not represent modern generators.",
            "Training overlap with Mirage is unverified."
        ]
    }

    metrics_path = OUTPUT / "mirage_metrics.json"

    with open(
        metrics_path, "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    csv_path = OUTPUT / "mirage_predictions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_index",
                "actual",
                "predicted",
                "fake_score_percent",
                "correct"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "RobustGuard AI - Mirage Confusion Matrix"
    )

    fig.tight_layout()

    plot_path = OUTPUT / "mirage_confusion_matrix.png"

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nFILES SAVED:")
    print(metrics_path)
    print(csv_path)
    print(plot_path)

    print("\nEVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
