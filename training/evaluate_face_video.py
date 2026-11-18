# RobustGuard AI - Full Video Evaluation
# SDFVD | Python 3.15 | CPU

import sys
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
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
sys.path.insert(0, str(ROOT / "backend"))

from face_video_detector import detect_face_video_path

DATASET = ROOT / "datasets" / "video" / "SDFVD"
OUT = ROOT / "training" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    videos = []

    for label, folder in [(1, "Fake"), (0, "Real")]:
        for path in sorted((DATASET / folder).glob("*.mp4")):
            videos.append((path, label))

    print("\n=== VIDEO EVALUATION STARTED ===")
    print("Total videos found:", len(videos))

    if not videos:
        raise RuntimeError("No videos found.")

    records = []
    y_true = []
    y_pred = []
    failed = 0

    for index, (path, actual) in enumerate(videos, 1):
        print(f"\n[{index}/{len(videos)}] {path.name}")

        try:
            result = detect_face_video_path(path)

            predicted = 1 if result["prediction"] == "Fake" else 0
            fake_score = result["class_scores"]["Fake"]

            y_true.append(actual)
            y_pred.append(predicted)

            records.append({
                "video": str(path),
                "actual": "Fake" if actual == 1 else "Real",
                "predicted": result["prediction"],
                "fake_score_percent": fake_score,
                "frames_analysed": result["frames_analysed"],
                "error": ""
            })

            print(
                "Actual:", "Fake" if actual == 1 else "Real",
                "| Predicted:", result["prediction"],
                "| Fake score:", fake_score
            )

        except Exception as error:
            failed += 1

            records.append({
                "video": str(path),
                "actual": "Fake" if actual == 1 else "Real",
                "predicted": "FAILED",
                "fake_score_percent": "",
                "frames_analysed": 0,
                "error": str(error)
            })

            print("FAILED:", error)

    with open(
        OUT / "face_video_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "video",
                "actual",
                "predicted",
                "fake_score_percent",
                "frames_analysed",
                "error"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    if not y_true:
        raise RuntimeError("No successful predictions.")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    print("\n================================")
    print("FACE VIDEO MODEL TEST RESULTS")
    print("================================")
    print("Successfully evaluated:", len(y_true))
    print("Failed:", failed)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    metrics = {
        "model": "Face Deepfake ViT",
        "dataset": "SDFVD",
        "total_videos": len(videos),
        "evaluated_videos": len(y_true),
        "failed_videos": failed,
        "accuracy": float(accuracy),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": cm.tolist(),
        "note": "Experimental frame-based video deepfake evaluation"
    }

    with open(
        OUT / "face_video_metrics.json",
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

    ax.set_title("RobustGuard AI - Video Confusion Matrix")
    fig.tight_layout()

    fig.savefig(
        OUT / "face_video_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved files:")
    print(OUT / "face_video_predictions.csv")
    print(OUT / "face_video_metrics.json")
    print(OUT / "face_video_confusion_matrix.png")
    print("\nEVALUATION COMPLETE!")


if __name__ == "__main__":
    main()