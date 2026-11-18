
# RobustGuard AI - Final Three-Model Video Comparison
# Uses the same held-out test videos for all models.
# No model training or video processing required.

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "outputs"

TRANSFER_CSV = OUT / "video_transfer_predictions.csv"
FACE_CSV = OUT / "face_video_predictions.csv"
VIDEOMAE_CSV = OUT / "videomae_predictions.csv"

RESULT_JSON = OUT / "video_three_model_comparison.json"
RESULT_CSV = OUT / "video_three_model_comparison.csv"
RESULT_GRAPH = OUT / "video_three_model_comparison.png"

MODELS = [
    "Face ViT + YuNet",
    "Transfer Learning",
    "VideoMAE + YuNet"
]


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {path}"
        )

    with open(
        path,
        newline="",
        encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def normalise_label(value):
    value = str(value).strip().lower()

    if value == "fake":
        return 1

    if value == "real":
        return 0

    return None


def video_key(row):
    """
    Retain actual class in the key, to prevent
    accidental matching of unrelated filenames.
    """
    path = str(row.get("video", ""))
    name = path.replace("\\", "/").split("/")[-1].lower()
    actual = normalise_label(row.get("actual", ""))

    if not name or actual is None:
        return None

    return (actual, name)


def load_predictions(path):
    predictions = {}

    for row in read_rows(path):
        key = video_key(row)

        if key is None:
            continue

        predicted = normalise_label(
            row.get("predicted", "")
        )

        if predicted is None:
            continue

        if row.get("status", "success").lower() != "success":
            continue

        if key in predictions:
            raise ValueError(
                f"Duplicate prediction in {path.name}: {key}"
            )

        predictions[key] = predicted

    return predictions


def evaluate(name, y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, 0]
    )

    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    fake_correct = int(cm[0, 0])
    fake_missed = int(cm[0, 1])
    real_wrong = int(cm[1, 0])
    real_correct = int(cm[1, 1])

    print("\n" + "=" * 48)
    print(name)
    print("=" * 48)

    print(f"Accuracy:          {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced * 100:.2f}%")
    print(f"Precision:         {precision * 100:.2f}%")
    print(f"Fake Recall:       {recall * 100:.2f}%")
    print(f"F1 Score:          {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print(
        f"\nReal correctly detected: "
        f"{real_correct}/{real_correct + real_wrong}"
    )

    print(
        f"Fake correctly detected: "
        f"{fake_correct}/{fake_correct + fake_missed}"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    return {
        "model": name,
        "sample_count": len(y_true),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "real_correct": real_correct,
        "real_wrong": real_wrong,
        "fake_correct": fake_correct,
        "fake_missed": fake_missed,
        "confusion_matrix": cm.tolist()
    }


def main():

    print("\nROBUSTGUARD FINAL VIDEO MODEL COMPARISON")

    face = load_predictions(FACE_CSV)
    transfer = load_predictions(TRANSFER_CSV)
    videomae = load_predictions(VIDEOMAE_CSV)

    common = sorted(
        set(face)
        & set(transfer)
        & set(videomae)
    )

    print("\nFace ViT predictions:", len(face))
    print("Transfer predictions:", len(transfer))
    print("VideoMAE predictions:", len(videomae))
    print("Common test videos:", len(common))

    missing = sorted(
        set(transfer) - set(common)
    )

    if missing:
        print("\nTransfer test videos without all predictions:")
        for key in missing:
            print(" -", key)

    if len(common) < 10:
        raise RuntimeError(
            "Too few common videos for comparison."
        )

    y_true = np.array([
        key[0] for key in common
    ], dtype=int)

    if len(set(y_true)) < 2:
        raise RuntimeError(
            "Both Real and Fake examples are required."
        )

    predictions = {
        "Face ViT + YuNet": np.array([
            face[key] for key in common
        ], dtype=int),

        "Transfer Learning": np.array([
            transfer[key] for key in common
        ], dtype=int),

        "VideoMAE + YuNet": np.array([
            videomae[key] for key in common
        ], dtype=int)
    }

    metrics = []

    for name in MODELS:
        metrics.append(
            evaluate(
                name,
                y_true,
                predictions[name]
            )
        )

    ranked = sorted(
        metrics,
        key=lambda item: (
            item["balanced_accuracy"],
            item["real_correct"],
            item["f1_fake"]
        ),
        reverse=True
    )

    print("\n" + "=" * 48)
    print("FINAL RANKING ON COMMON TEST VIDEOS")
    print("=" * 48)

    for rank, item in enumerate(ranked, 1):
        print(
            f"{rank}. {item['model']} | "
            f"Balanced Accuracy: "
            f"{item['balanced_accuracy'] * 100:.2f}% | "
            f"Real Correct: {item['real_correct']}"
        )

    winner = ranked[0]["model"]

    print("\nTOP MODEL ON THIS TEST SET:", winner)

    with open(
        RESULT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "video",
            "actual",
            "face_vit_prediction",
            "transfer_prediction",
            "videomae_prediction"
        ])

        for i, key in enumerate(common):
            writer.writerow([
                key[1],
                "Fake" if y_true[i] else "Real",
                "Fake" if predictions[MODELS[0]][i] else "Real",
                "Fake" if predictions[MODELS[1]][i] else "Real",
                "Fake" if predictions[MODELS[2]][i] else "Real"
            ])

    report = {
        "dataset": "SDFVD",
        "common_test_videos": len(common),
        "real_test_videos": int(
            np.sum(y_true == 0)
        ),
        "fake_test_videos": int(
            np.sum(y_true == 1)
        ),
        "ranking": ranked,
        "top_model_on_this_set": winner,
        "note": (
            "Exploratory comparison using the transfer "
            "model's existing held-out test videos. "
            "Selecting a final model based on this "
            "test set introduces selection bias. "
            "Independent external validation is required."
        )
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    # Show both balanced accuracy and Real recall.
    names = [item["model"] for item in metrics]

    balanced_values = [
        item["balanced_accuracy"] * 100
        for item in metrics
    ]

    real_values = [
        (
            item["real_correct"]
            / max(
                item["real_correct"] + item["real_wrong"],
                1
            )
        ) * 100
        for item in metrics
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2,
        balanced_values,
        width,
        label="Balanced Accuracy"
    )

    bars2 = ax.bar(
        x + width / 2,
        real_values,
        width,
        label="Real Recall"
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title(
        "RobustGuard: Video Model Comparison"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.legend()

    ax.bar_label(
        bars1,
        fmt="%.1f%%",
        padding=3
    )

    ax.bar_label(
        bars2,
        fmt="%.1f%%",
        padding=3
    )

    fig.tight_layout()

    fig.savefig(
        RESULT_GRAPH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved:")
    print(RESULT_JSON)
    print(RESULT_CSV)
    print(RESULT_GRAPH)

    print("\nCOMPARISON COMPLETE!")


if __name__ == "__main__":
    main()
