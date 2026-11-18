
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor
)

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

torch.set_num_threads(2)

print("Loading CommunityForensics...")

processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()

SIZE = 384
RESIZE = 440


def predict(images, already_cropped=False):
    inputs = processor(
        images=images,
        return_tensors="pt",
        do_resize=not already_cropped,
        do_center_crop=not already_cropped
    )

    with torch.inference_mode():
        logits = model(**inputs).logits.flatten()
        scores = torch.sigmoid(logits)

    return scores.tolist()


def get_five_crops(image):
    width, height = image.size

    scale = RESIZE / min(width, height)

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC
    )

    max_x = new_width - SIZE
    max_y = new_height - SIZE

    positions = {
        "Top Left": (0, 0),
        "Top Right": (max_x, 0),
        "Centre": (max_x // 2, max_y // 2),
        "Bottom Left": (0, max_y),
        "Bottom Right": (max_x, max_y)
    }

    crops = []

    for x, y in positions.values():
        crops.append(
            resized.crop(
                (x, y, x + SIZE, y + SIZE)
            )
        )

    return positions, crops


def test_image(path):
    with Image.open(path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")

    print("\n" + "=" * 55)
    print("IMAGE:", Path(path).name)
    print("Original size:", image.size)

    original_score = predict([image])[0]

    positions, crops = get_five_crops(image)
    crop_scores = predict(crops, already_cropped=True)

    mean_score = sum(crop_scores) / len(crop_scores)

    print("\n--- ORIGINAL MODEL ---")
    print(f"Fake score: {original_score * 100:.2f}%")
    print(
        "Prediction:",
        "FAKE" if original_score >= 0.5 else "REAL"
    )

    print("\n--- FIVE CROP SCORES ---")

    for name, score in zip(positions, crop_scores):
        print(f"{name}: {score * 100:.2f}% Fake")

    print("\n--- MULTI-CROP AVERAGE ---")
    print(f"Average Fake score: {mean_score * 100:.2f}%")
    print(
        "Prediction:",
        "FAKE" if mean_score >= 0.5 else "REAL"
    )

    print(
        "\nNote: Multi-crop averaging is experimental."
        "\nThe average is not a calibrated probability."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python multicrop_test.py '
            '"image1.png" "image2.jpg"'
        )
        sys.exit(1)

    for path in sys.argv[1:]:
        if not Path(path).is_file():
            print("File not found:", path)
            continue

        try:
            test_image(path)
        except Exception as error:
            print("Error:", error)
