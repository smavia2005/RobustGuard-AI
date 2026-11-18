
# RobustGuard AI - Video Frame AI Test
# Python 3.15 | CPU | No OpenCV

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

ROOT = Path(__file__).resolve().parent.parent

FRAMES_FOLDER = (
    ROOT / "datasets" / "video" / "test_frames"
)

MODEL_ID = "shunda012/vit-deepfake-detector"

torch.set_num_threads(2)

print("\n=== ROBUSTGUARD VIDEO AI TEST ===")

# Find extracted frames
frame_files = sorted(
    FRAMES_FOLDER.glob("frame_*.jpg")
)

if not frame_files:
    raise FileNotFoundError(
        "No frames found. Run test_video_frames.py first."
    )

print("Frames found:", len(frame_files))

# Load pretrained model
print("\nLoading face deepfake AI model...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID
)

model.to("cpu")
model.eval()

print("Model loaded successfully!")

print("\nModel labels:", model.config.id2label)

# Identify labels using model configuration
label_map = {
    int(index): str(name).strip().lower()
    for index, name in model.config.id2label.items()
}

fake_indices = [
    index for index, name in label_map.items()
    if name == "fake"
]

real_indices = [
    index for index, name in label_map.items()
    if name == "real"
]

if len(fake_indices) != 1 or len(real_indices) != 1:
    raise ValueError(
        "Unexpected model labels. "
        "Expected one Real and one Fake class."
    )

fake_index = fake_indices[0]
real_index = real_indices[0]

scores = []

# Analyse frames one at a time
for frame_path in frame_files:

    with Image.open(frame_path) as source:
        image = source.convert("RGB")

        # Approximate central subject crop.
        # This is NOT actual face detection.
        width, height = image.size

        crop_size = int(
            min(width, height) * 0.85
        )

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2

        image = image.crop((
            left,
            top,
            left + crop_size,
            top + crop_size
        ))

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():

        logits = model(**inputs).logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

    fake_score = float(
        probabilities[fake_index].item()
    )

    real_score = float(
        probabilities[real_index].item()
    )

    scores.append(fake_score)

    prediction = (
        "FAKE" if fake_score >= 0.5
        else "REAL"
    )

    print(
        f"{frame_path.name} | "
        f"Fake: {fake_score * 100:.2f}% | "
        f"Real: {real_score * 100:.2f}% | "
        f"{prediction}"
    )

# Combine frame scores
average_fake = float(np.mean(scores))
average_real = 1.0 - average_fake

video_prediction = (
    "FAKE" if average_fake >= 0.5
    else "REAL"
)

print("\n================================")
print("FINAL VIDEO PREDICTION")
print("================================")

print("Frames analysed:", len(scores))
print(f"Average Fake score: {average_fake * 100:.2f}%")
print(f"Average Real score: {average_real * 100:.2f}%")
print("Video prediction:", video_prediction)

print("\nNote:")
print(
    "This is an experimental frame-based "
    "prediction, not proof that a video "
    "is authentic or manipulated."
)

print("\nVIDEO AI TEST COMPLETE!")
