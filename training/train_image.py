
import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "image"
OUTPUT = ROOT / "training" / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA / "train-00000-of-00001.parquet"
TEST_FILE = DATA / "test-00000-of-00001.parquet"

SEED = 42
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


class CIFAKEDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_data = row["image"]

        if isinstance(image_data, dict):
            raw = image_data.get("bytes")

            if raw is not None:
                image = Image.open(BytesIO(raw))
            else:
                image = Image.open(image_data["path"])

        elif isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data))

        else:
            raise ValueError(
                "Unsupported image format in parquet."
            )

        with image:
            image = image.convert("RGB")
            image = image.resize((32, 32))

            array = np.asarray(
                image, dtype=np.float32
            ).copy() / 255.0

        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)

        # CIFAKE: 0 = FAKE, 1 = REAL
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return tensor, label


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def balanced_sample(df, n_per_class):
    return (
        df.groupby("label", group_keys=False)
          .sample(n=n_per_class, random_state=SEED)
          .sample(frac=1, random_state=SEED)
          .reset_index(drop=True)
    )


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    actual = []
    predicted = []

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            correct += (predictions == labels).sum().item()
            total += len(labels)

            actual.extend(labels.tolist())
            predicted.extend(predictions.tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "actual": actual,
        "predicted": predicted
    }


def calculate_metrics(actual, predicted):
    # Positive class: FAKE (label 0)
    tp = sum(
        a == 0 and p == 0
        for a, p in zip(actual, predicted)
    )
    tn = sum(
        a == 1 and p == 1
        for a, p in zip(actual, predicted)
    )
    fp = sum(
        a == 1 and p == 0
        for a, p in zip(actual, predicted)
    )
    fn = sum(
        a == 0 and p == 1
        for a, p in zip(actual, predicted)
    )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "accuracy": (tp + tn) / len(actual),
        "fake_precision": precision,
        "fake_recall": recall,
        "fake_f1": f1,
        "confusion_matrix": {
            "actual_fake_pred_fake": tp,
            "actual_fake_pred_real": fn,
            "actual_real_pred_fake": fp,
            "actual_real_pred_real": tn
        }
    }


def main():
    print("Loading CIFAKE dataset...")

    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    print("Full train samples:", len(train_df))
    print("Full test samples:", len(test_df))

    # First CPU-friendly experiment:
    # 2000 FAKE + 2000 REAL training-pool samples.
    train_pool = balanced_sample(train_df, 2000)

    # Separate validation: 400 images from the
    # training pool, with no overlap.
    val_indices = (
        train_pool.groupby("label")
                  .sample(n=200, random_state=SEED)
                  .index
    )

    val_df = train_pool.loc[val_indices].copy()
    train_subset = train_pool.drop(val_indices).copy()

    # Final test: 400 FAKE + 400 REAL
    test_subset = balanced_sample(test_df, 400)

    train_loader = DataLoader(
        CIFAKEDataset(train_subset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        CIFAKEDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        CIFAKEDataset(test_subset),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = -1

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        validation = evaluate(
            model, val_loader, criterion
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Training loss: "
            f"{running_loss / len(train_loader):.4f} | "
            f"Validation accuracy: "
            f"{validation['accuracy'] * 100:.2f}%"
        )

        if validation["accuracy"] > best_val_accuracy:
            best_val_accuracy = validation["accuracy"]

            torch.save(
                model.state_dict(),
                OUTPUT / "cifake_cnn_best.pth"
            )

    # Load best validation checkpoint
    model.load_state_dict(
        torch.load(
            OUTPUT / "cifake_cnn_best.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    # Final evaluation happens only once,
    # after selecting the best checkpoint.
    test_results = evaluate(
        model, test_loader, criterion
    )

    metrics = calculate_metrics(
        test_results["actual"],
        test_results["predicted"]
    )

    metrics["train_samples"] = len(train_subset)
    metrics["validation_samples"] = len(val_df)
    metrics["test_samples"] = len(test_subset)
    metrics["epochs"] = EPOCHS
    metrics["seed"] = SEED

    with open(
        OUTPUT / "image_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nTRAINING & TESTING COMPLETE")
    print(
        f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%"
    )
    print(
        f"Fake Precision: "
        f"{metrics['fake_precision']:.4f}"
    )
    print(
        f"Fake Recall: {metrics['fake_recall']:.4f}"
    )
    print(
        f"Fake F1 Score: {metrics['fake_f1']:.4f}"
    )
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nSaved files:")
    print(OUTPUT / "cifake_cnn_best.pth")
    print(OUTPUT / "image_metrics.json")


if __name__ == "__main__":
    main()
