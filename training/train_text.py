
# RobustGuard AI - HC3 Text Training
# Model: TF-IDF + Logistic Regression
# Python 3.15 / CPU

import json
import re
from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "datasets" / "text" / "all.jsonl"
OUTPUT = ROOT / "training" / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_QUESTIONS = 8000

# 0 = Human, 1 = AI Generated
LABELS = ["Human", "AI Generated"]


def clean_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def load_hc3():
    texts = []
    labels = []
    groups = []

    seen_questions = set()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Text dataset not found: {DATA_PATH}"
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            question = clean_text(row.get("question", ""))
            human = row.get("human_answers") or []
            ai = row.get("chatgpt_answers") or []

            # Use one response from each class per question
            if not question or not human or not ai:
                continue

            group = question.casefold()

            if group in seen_questions:
                continue

            human_text = clean_text(human[0])
            ai_text = clean_text(ai[0])

            if len(human_text) < 30 or len(ai_text) < 30:
                continue

            # Remove exact identical responses
            if human_text.casefold() == ai_text.casefold():
                continue

            seen_questions.add(group)

            texts.extend([human_text, ai_text])
            labels.extend([0, 1])
            groups.extend([group, group])

            if len(seen_questions) >= MAX_QUESTIONS:
                break

    if len(seen_questions) < 100:
        raise ValueError(
            "Not enough usable HC3 question groups."
        )

    return (
        np.array(texts, dtype=object),
        np.array(labels),
        np.array(groups)
    )


def get_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ai": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_ai": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_ai": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist()
    }


def main():
    print("Loading HC3 text dataset...")

    texts, labels, groups = load_hc3()

    print("Total samples:", len(texts))
    print("Unique question groups:", len(set(groups)))

    # Split by question, not individual answer.
    # This keeps paired human/AI answers together.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED
    )

    train_val_idx, test_idx = next(
        splitter.split(texts, labels, groups)
    )

    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=SEED
    )

    inner_train, inner_val = next(
        inner_splitter.split(
            texts[train_val_idx],
            labels[train_val_idx],
            groups[train_val_idx]
        )
    )

    train_idx = train_val_idx[inner_train]
    val_idx = train_val_idx[inner_val]

    print("Training:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Testing:", len(test_idx))

    # Check group-level leakage
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    print("\nTraining TF-IDF + Logistic Regression...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=SEED
            )
        )
    ])

    model.fit(texts[train_idx], labels[train_idx])

    val_predictions = model.predict(texts[val_idx])
    val_metrics = get_metrics(
        labels[val_idx], val_predictions
    )

    print(
        "Validation Accuracy:",
        f"{val_metrics['accuracy'] * 100:.2f}%"
    )

    # Evaluate once on the untouched test set
    test_predictions = model.predict(texts[test_idx])
    metrics = get_metrics(
        labels[test_idx], test_predictions
    )

    metrics["training_samples"] = int(len(train_idx))
    metrics["validation_samples"] = int(len(val_idx))
    metrics["test_samples"] = int(len(test_idx))
    metrics["question_groups"] = int(len(set(groups)))
    metrics["seed"] = SEED
    metrics["model"] = "TF-IDF + Logistic Regression"
    metrics["labels"] = LABELS

    model_path = OUTPUT / "hc3_text_model.joblib"

    joblib.dump(model, model_path)

    with open(
        OUTPUT / "text_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nTEXT TRAINING AND TESTING COMPLETE")
    print(
        f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"AI Precision: {metrics['precision_ai']:.4f}"
    )
    print(
        f"AI Recall: {metrics['recall_ai']:.4f}"
    )
    print(
        f"AI F1 Score: {metrics['f1_ai']:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            labels[test_idx],
            test_predictions,
            target_names=LABELS,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nSaved files:")
    print(model_path)
    print(OUTPUT / "text_metrics.json")


if __name__ == "__main__":
    main()
