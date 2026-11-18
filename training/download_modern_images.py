
from pathlib import Path
from datasets import load_dataset
import random

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "datasets" / "image" / "Modern_Test"

REAL_DIR = OUTPUT / "Real"
FAKE_DIR = OUTPUT / "Fake"

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 100

counts = {
    "Real": 0,
    "Fake": 0
}

print("Connecting to Mirage-Test dataset...")

dataset = load_dataset(
    "Yunncheng/Mirage-Test",
    split="test",
    streaming=True
)

# Shuffle the order so images are not all
# selected from one consecutive dataset block.
dataset = dataset.shuffle(
    seed=42,
    buffer_size=1000
)

for item in dataset:

    label = str(item["is_real"]).strip().lower()

    if label == "0_real":
        category = "Real"

    elif label == "1_fake":
        category = "Fake"

    else:
        continue

    if counts[category] >= TARGET:
        continue

    image = item["image"].convert("RGB")

    index = counts[category] + 1

    folder = (
        REAL_DIR if category == "Real"
        else FAKE_DIR
    )

    filename = folder / f"{category}_{index:03d}.jpg"

    image.save(
        filename,
        format="JPEG",
        quality=95
    )

    counts[category] += 1

    print(
        f"{category}: "
        f"{counts[category]}/{TARGET}"
    )

    if all(
        count == TARGET
        for count in counts.values()
    ):
        break

print("\n================================")
print("DATASET DOWNLOAD COMPLETE")
print("================================")

print("Real images:", counts["Real"])
print("Fake images:", counts["Fake"])

print("Saved to:", OUTPUT)
