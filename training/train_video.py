
# RobustGuard AI - SDFVD Video Training and Testing
# CPU | Paired video-level split

import sys
import json
import csv
from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

from video_detector import extract_video_features


DATASET = ROOT / "datasets" / "video" / "SDFVD"
OUTPUT = ROOT / "training" / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

CACHE = OUTPUT / "video_features_cache.json"
MODEL_FILE = OUTPUT / "video_classifier.joblib"

SEED = 42


def get_video_pairs():
    pairs = []

    for number in range(1, 54):
        real_path = (
            DATASET / "Real" / f"v{number}.mp4"
        )

        fake_path = (
            DATASET / "Fake" / f"vs{number}.mp4"
        )

        if real_path.is_file() and fake_path.is_file():
            pairs.append((
                number,
                real_path,
                fake_path
            ))

    return pairs


def collect_features(pairs):
    if CACHE.exists():
        with open(CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    records = []

    total = len(pairs) * 2
    done = 0

    for pair_id, real_path, fake_path in pairs:
        for label, path in [
            (0, real_path),
            (1, fake_path)
        ]:
            key = str(path.relative_to(DATASET))

            if key not in cache:
                try:
                    result = extract_video_features(path)
                    cache[key] = result["features"]

                    with open(
                        CACHE,
                        "w",
                        encoding="utf-8"
                    ) as f:
                        json.dump(cache, f, indent=2)

                except Exception as error:
                    print(
                        "\nFailed:", path.name,
                        str(error)
                    )
                    continue

            records.append({
                "pair_id": pair_id,
                "path": str(path),
                "label": label,
                "features": cache[key]
            })

            done += 1
            print(
                f"\rExtracted {done}/{total} videos",
                end="",
                flush=True
            )

    return records


def main():
    print("\n=== ROBUSTGUARD VIDEO TRAINING ===")

    pairs = get_video_pairs()

    print("Complete Real/Fake pairs:", len(pairs))

    if len(pairs) < 10:
        raise RuntimeError(
            "Not enough paired videos found."
        )

    print("\nExtracting frame features...")
    records = collect_features(pairs)

    # Only retain complete Real/Fake pairs.
    grouped = {}

    for record in records:
        grouped.setdefault(
            record["pair_id"], []
        ).append(record)

    valid_pairs = sorted(
        pair_id
        for pair_id, items in grouped.items()
        if len(items) == 2
        and {x["label"] for x in items} == {0, 1}
    )

    records = [
        item
        for pair_id in valid_pairs
        for item in grouped[pair_id]
    ]

    print("\nUsable pairs:", len(valid_pairs))
    print("Usable videos:", len(records))

    if len(valid_pairs) < 10:
        raise RuntimeError(
            "Not enough decoded video pairs."
        )

    # Both videos from the same numbered pair
    # stay in the SAME split.
    train_pairs, test_pairs = train_test_split(
        valid_pairs,
        test_size=0.25,
        random_state=SEED
    )

    train_set = set(train_pairs)
    test_set = set(test_pairs)

    train_rows = [
        r for r in records
        if r["pair_id"] in train_set
    ]

    test_rows = [
        r for r in records
        if r["pair_id"] in test_set
    ]

    X_train = np.array(
        [r["features"] for r in train_rows],
        dtype=np.float64
    )

    y_train = np.array(
        [r["label"] for r in train_rows]
    )

    X_test = np.array(
        [r["features"] for r in test_rows],
        dtype=np.float64
    )

    y_test = np.array(
        [r["label"] for r in test_rows]
    )

    print("\nTraining videos:", len(y_train))
    print("Testing videos:", len(y_test))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=SEED
        )
    )

    print("\nTraining video classifier...")
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    probs = model.predict_proba(X_test)
    fake_index = list(model.classes_).index(1)

    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(
        y_test, predicted, zero_division=0
    )
    recall = recall_score(
        y_test, predicted, zero_division=0
    )
    f1 = f1_score(
        y_test, predicted, zero_division=0
    )

    cm = confusion_matrix(
        y_test, predicted, labels=[1, 0]
    )

    print("\n=== VIDEO TEST RESULTS ===")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predicted,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    joblib.dump(model, MODEL_FILE)

    metrics = {
        "model": "Mirage frame features + Logistic Regression",
        "dataset": "SDFVD",
        "train_video_count": len(y_train),
        "test_video_count": len(y_test),
        "train_pair_ids": sorted(train_pairs),
        "test_pair_ids": sorted(test_pairs),
        "accuracy": float(accuracy),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": cm.tolist(),
        "limitations": [
            "Small dataset with 53 Real/Fake pairs.",
            "Evaluation is based on one paired split.",
            "Mirage was developed for image classification.",
            "Frame scores do not capture full temporal dynamics.",
            "Generalisation to unseen video sources is unknown."
        ]
    }

    with open(
        OUTPUT / "video_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    with open(
        OUTPUT / "video_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "actual", "predicted", "fake_score"
        ])

        for i, row in enumerate(test_rows):
            writer.writerow([
                row["path"],
                "Fake" if y_test[i] else "Real",
                "Fake" if predicted[i] else "Real",
                float(probs[i, fake_index])
            ])

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
        "RobustGuard AI - Video Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "video_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved model:", MODEL_FILE)
    print("Saved metrics and confusion matrix.")
    print("VIDEO TRAINING AND TESTING COMPLETE!")


if __name__ == "__main__":
    main()
