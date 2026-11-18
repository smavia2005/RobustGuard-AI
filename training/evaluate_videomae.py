
# RobustGuard AI - Full VideoMAE Evaluation
# SDFVD: 53 Real + 53 Fake videos
# YuNet face crops + pretrained VideoMAE
# Python 3.15 | CPU

import sys
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

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

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "backend"))

from test_videomae_faces import (
    extract_frames,
    detect_face,
    load_models
)


DATASET = ROOT / "datasets" / "video" / "SDFVD"
OUTPUT = ROOT / "training" / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT / "videomae_predictions.csv"
METRICS_PATH = OUTPUT / "videomae_metrics.json"
MATRIX_PATH = OUTPUT / "videomae_confusion_matrix.png"

torch.set_num_threads(2)


def get_videos():

    videos = []

    for pair_id in range(1, 54):

        for folder, prefix, label in [
            ("Real", "v", 0),
            ("Fake", "vs", 1)
        ]:

            video_path = (
                DATASET / folder /
                f"{prefix}{pair_id}.mp4"
            )

            videos.append({
                "path": video_path,
                "label": label,
                "pair_id": pair_id
            })

    return videos


def load_previous_results():

    results = {}

    if not CSV_PATH.exists():
        return results

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            key = row["video"].replace(
                "\\", "/"
            ).split("/")[-1].lower()

            results[key] = row

    return results


def save_results(results):

    fields = [
        "video",
        "pair_id",
        "actual",
        "predicted",
        "real_score",
        "fake_score",
        "faces_detected",
        "frames_analysed",
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

        for row in results.values():
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })


def predict_video(video_path, detector, processor, model):

    frames = extract_frames(video_path)

    crops = []
    faces_detected = 0

    for frame in frames:

        face = detect_face(frame, detector)

        if face is not None:

            crops.append(face)
            faces_detected += 1

        else:

            if crops:
                crops.append(crops[-1].copy())
            else:
                crops.append(None)

    valid_faces = [
        face for face in crops
        if face is not None
    ]

    if not valid_faces:

        raise ValueError(
            "No faces detected in this video."
        )

    first_face = valid_faces[0]

    crops = [
        face if face is not None
        else first_face.copy()
        for face in crops
    ]

    inputs = processor(
        list(crops),
        return_tensors="pt"
    )

    with torch.inference_mode():

        output = model(**inputs)

        probabilities = torch.softmax(
            output.logits,
            dim=-1
        )[0]

    real_score = float(
        probabilities[0].item()
    )

    fake_score = float(
        probabilities[1].item()
    )

    prediction = (
        "Fake"
        if fake_score >= 0.5
        else "Real"
    )

    return {
        "prediction": prediction,
        "real_score": round(
            real_score * 100, 4
        ),
        "fake_score": round(
            fake_score * 100, 4
        ),
        "faces_detected": faces_detected,
        "frames_analysed": len(crops)
    }


def evaluate(results):

    successful = [
        row for row in results.values()
        if row.get("status") == "success"
        and row.get("predicted") in ["Real", "Fake"]
    ]

    failed = [
        row for row in results.values()
        if row.get("status") != "success"
    ]

    if not successful:
        print("No successful predictions.")
        return

    y_true = np.array([
        1 if row["actual"] == "Fake" else 0
        for row in successful
    ])

    y_pred = np.array([
        1 if row["predicted"] == "Fake" else 0
        for row in successful
    ])

    accuracy = accuracy_score(
        y_true, y_pred
    )

    balanced = balanced_accuracy_score(
        y_true, y_pred
    )

    precision = precision_score(
        y_true, y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        zero_division=0
    )

    # Row order: Actual Fake, Actual Real
    # Column order: Pred Fake, Pred Real
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, 0]
    )

    fake_correct = int(cm[0, 0])
    fake_missed = int(cm[0, 1])

    real_wrong = int(cm[1, 0])
    real_correct = int(cm[1, 1])

    print("\n===================================")
    print("ROBUSTGUARD VIDEOMAE RESULTS")
    print("===================================")

    print("Successful videos:", len(successful))
    print("Failed videos:", len(failed))

    print(f"\nAccuracy:  {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nReal correctly detected:", real_correct)
    print("Real incorrectly called Fake:", real_wrong)

    print("Fake correctly detected:", fake_correct)
    print("Fake incorrectly called Real:", fake_missed)

    print("\nClassification Report:")

    print(classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    metrics = {
        "model": "VideoMAE + YuNet",
        "dataset": "SDFVD",
        "total_expected": 106,
        "successful_videos": len(successful),
        "failed_videos": len(failed),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "real_correct": real_correct,
        "real_wrong": real_wrong,
        "fake_correct": fake_correct,
        "fake_missed": fake_missed,
        "limitations": [
            "Evaluation on SDFVD only",
            "Pretrained model may have different training data",
            "Missing-face frames reuse detected crops",
            "Not a general-purpose authenticity guarantee"
        ]
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    ).plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "VideoMAE + YuNet Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        MATRIX_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved:")
    print(CSV_PATH)
    print(METRICS_PATH)
    print(MATRIX_PATH)


def main():

    print("\nROBUSTGUARD VIDEOMAE FULL EVALUATION")

    videos = get_videos()

    previous = load_previous_results()

    print("Expected videos:", len(videos))
    print("Previously recorded:", len(previous))

    detector, processor, model = load_models()

    start = time.time()

    for index, item in enumerate(videos, start=1):

        path = item["path"]
        key = path.name.lower()

        old = previous.get(key)

        if (
            old is not None
            and old.get("status") == "success"
        ):

            print(
                f"[{index}/106] SKIPPED: {path.name}"
            )

            continue

        print(
            f"\n[{index}/106] Processing: {path.name}"
        )

        actual = (
            "Fake" if item["label"] == 1
            else "Real"
        )

        row = {
            "video": str(path),
            "pair_id": item["pair_id"],
            "actual": actual,
            "predicted": "",
            "real_score": "",
            "fake_score": "",
            "faces_detected": 0,
            "frames_analysed": 0,
            "status": "failed",
            "error": ""
        }

        try:

            result = predict_video(
                path,
                detector,
                processor,
                model
            )

            row.update({
                "predicted": result["prediction"],
                "real_score": result["real_score"],
                "fake_score": result["fake_score"],
                "faces_detected": result[
                    "faces_detected"
                ],
                "frames_analysed": result[
                    "frames_analysed"
                ],
                "status": "success"
            })

            print(
                f"Actual: {actual} | "
                f"Predicted: {result['prediction']} | "
                f"Fake score: {result['fake_score']:.2f}%"
            )

        except Exception as error:

            row["error"] = str(error)

            print(
                f"FAILED: {path.name}: {error}"
            )

        previous[key] = row

        # Save after every video.
        save_results(previous)

    elapsed = time.time() - start

    print(
        f"\nEvaluation runtime: {elapsed / 60:.2f} minutes"
    )

    evaluate(previous)

    print("\nEVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
