
# RobustGuard AI - Video Transfer Learning
# Frozen pretrained ViT + Logistic Regression
# Python 3.15 | CPU | SDFVD

import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

from face_video_detector import (
    extract_frames,
    detect_face_crop,
    load_models
)

DATA = ROOT / "datasets" / "video" / "SDFVD"
OUT = ROOT / "training" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CACHE_PATH = OUT / "video_transfer_features.json"
MODEL_PATH = OUT / "video_transfer_classifier.joblib"

SEED = 42
torch.set_num_threads(2)


# --------------------------------------
# LOAD EXISTING PRETRAINED ViT
# --------------------------------------

processor, vit, face_detector, fake_index = load_models()

vit.eval()


# --------------------------------------
# EXTRACT VIDEO FEATURES
# --------------------------------------

def extract_features(video_path):

    frames = extract_frames(video_path)

    embeddings = []
    fake_scores = []

    for frame in frames:

        face = detect_face_crop(
            frame,
            face_detector
        )

        if face is None:
            continue

        inputs = processor(
            images=face,
            return_tensors="pt"
        )

        with torch.inference_mode():

            output = vit(
                **inputs,
                output_hidden_states=True
            )

            # Last hidden layer's CLS token.
            embedding = output.hidden_states[-1][
                0, 0, :
            ].cpu().numpy()

            probabilities = torch.softmax(
                output.logits,
                dim=-1
            )

            fake_score = float(
                probabilities[0, fake_index].item()
            )

        embeddings.append(embedding)
        fake_scores.append(fake_score)

    if not embeddings:
        raise ValueError(
            "No usable face frames found."
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    fake_scores = np.asarray(
        fake_scores,
        dtype=np.float32
    )

    # Average face embedding across frames.
    mean_embedding = embeddings.mean(axis=0)

    # Add frame-score summary.
    summary = np.array([
        fake_scores.mean(),
        fake_scores.std(),
        fake_scores.min(),
        fake_scores.max()
    ], dtype=np.float32)

    return np.concatenate([
        mean_embedding,
        summary
    ]).tolist()


# --------------------------------------
# COLLECT DATA WITH LOCAL CACHE
# --------------------------------------

def collect_data():

    if CACHE_PATH.exists():
        with open(
            CACHE_PATH,
            encoding="utf-8"
        ) as file:
            cache = json.load(file)
    else:
        cache = {}

    records = []
    failed = []

    for pair_id in range(1, 54):

        pair_items = []
        pair_failed = False

        for folder, prefix, label in [
            ("Real", "v", 0),
            ("Fake", "vs", 1)
        ]:

            path = (
                DATA / folder /
                f"{prefix}{pair_id}.mp4"
            )

            if not path.exists():
                pair_failed = True
                failed.append(str(path))
                continue

            key = f"{folder}/{path.name}"

            try:
                if key not in cache:

                    cache[key] = extract_features(
                        path
                    )

                    # Save progress after every video.
                    with open(
                        CACHE_PATH,
                        "w",
                        encoding="utf-8"
                    ) as file:
                        json.dump(cache, file)

                pair_items.append({
                    "path": str(path),
                    "pair_id": pair_id,
                    "label": label,
                    "features": cache[key]
                })

                print(
                    f"Processed: {folder}/{path.name}"
                )

            except Exception as error:
                pair_failed = True
                failed.append(
                    f"{path}: {error}"
                )
                print("FAILED:", path.name, error)

        # Only keep complete Real/Fake pairs.
        if not pair_failed and len(pair_items) == 2:
            records.extend(pair_items)

    return records, failed


# --------------------------------------
# TRAIN + EVALUATE
# --------------------------------------

def main():

    print("\nROBUSTGUARD VIDEO TRANSFER LEARNING")

    records, failed = collect_data()

    pair_ids = sorted(
        set(r["pair_id"] for r in records)
    )

    print("\nValid pairs:", len(pair_ids))
    print("Usable videos:", len(records))
    print("Failed videos:", len(failed))

    if len(pair_ids) < 20:
        raise RuntimeError(
            "Not enough paired videos."
        )

    train_ids, test_ids = train_test_split(
        pair_ids,
        test_size=0.25,
        random_state=SEED
    )

    train_set = set(train_ids)
    test_set = set(test_ids)

    train_rows = [
        r for r in records
        if r["pair_id"] in train_set
    ]

    test_rows = [
        r for r in records
        if r["pair_id"] in test_set
    ]

    X_train = np.asarray(
        [r["features"] for r in train_rows],
        dtype=np.float64
    )

    y_train = np.asarray(
        [r["label"] for r in train_rows]
    )

    groups = np.asarray(
        [r["pair_id"] for r in train_rows]
    )

    X_test = np.asarray(
        [r["features"] for r in test_rows],
        dtype=np.float64
    )

    y_test = np.asarray(
        [r["label"] for r in test_rows]
    )

    print("\nTraining videos:", len(y_train))
    print("Testing videos:", len(y_test))

    # PCA reduces pretrained embedding dimensions.
    # Every transform is fitted within training folds.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=12,
            random_state=SEED
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced"
        ))
    ])

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": [
                0.01, 0.1, 1.0, 10.0
            ]
        },
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=1,
        refit=True
    )

    print("\nTraining classifier...")
    search.fit(
        X_train,
        y_train,
        groups=groups
    )

    model = search.best_estimator_

    print("Best C:", search.best_params_)
    print(
        "Training CV balanced accuracy:",
        round(search.best_score_ * 100, 2), "%"
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )

    fake_index_model = list(
        model.classes_
    ).index(1)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[1, 0]
    )

    print("\n================================")
    print("VIDEO TRANSFER LEARNING RESULTS")
    print("================================")

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[1, 0],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    # Save trained model.
    joblib.dump(
        model,
        MODEL_PATH
    )

    metrics = {
        "model": (
            "Frozen Face ViT embeddings "
            "+ PCA + Logistic Regression"
        ),
        "dataset": "SDFVD",
        "training_videos": len(y_train),
        "testing_videos": len(y_test),
        "valid_pairs": len(pair_ids),
        "failed_count": len(failed),
        "test_pair_ids": sorted(test_ids),
        "accuracy": float(accuracy),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": cm.tolist(),
        "cv_balanced_accuracy": float(
            search.best_score_
        ),
        "limitations": [
            "Small paired dataset.",
            "Frozen image-model features.",
            "One held-out test split.",
            "No explicit temporal modelling.",
            "Not validated on independent video sources."
        ]
    }

    with open(
        OUT / "video_transfer_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2
        )

    with open(
        OUT / "video_transfer_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "video",
            "actual",
            "predicted",
            "fake_score"
        ])

        for i, row in enumerate(test_rows):

            writer.writerow([
                row["path"],
                "Fake" if y_test[i] else "Real",
                "Fake" if predictions[i] else "Real",
                float(
                    probabilities[
                        i, fake_index_model
                    ]
                )
            ])

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
        "Video Transfer Learning Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "video_transfer_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSAVED:")
    print(MODEL_PATH)
    print(OUT / "video_transfer_metrics.json")
    print(OUT / "video_transfer_predictions.csv")
    print(OUT / "video_transfer_confusion_matrix.png")

    print("\nTRAINING AND TESTING COMPLETE!")


if __name__ == "__main__":
    main()
