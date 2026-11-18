
from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

dataset_path = (
    ROOT / "datasets" / "image"
    / "train-00000-of-00001.parquet"
)

output_dir = ROOT / "training" / "label_check"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(dataset_path)

print("Dataset columns:", df.columns.tolist())
print("Label counts:")
print(df["label"].value_counts())

for label in sorted(df["label"].unique()):
    row = df[df["label"] == label].iloc[0]
    image_data = row["image"]

    if isinstance(image_data, dict):
        if image_data.get("bytes") is not None:
            image = Image.open(
                BytesIO(image_data["bytes"])
            )
        else:
            image = Image.open(image_data["path"])

    elif isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data))

    else:
        raise ValueError("Unknown image format")

    with image:
        image = image.convert("RGB")

        filename = output_dir / f"label_{label}.png"
        image.save(filename)

        print(f"Label {label}: saved {filename}")

print("\nCheck the dataset documentation to confirm")
print("which numeric label means FAKE and REAL.")
