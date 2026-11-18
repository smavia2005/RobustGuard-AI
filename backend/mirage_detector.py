
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

MODEL_ID = "prithivMLmods/Mirage-Photo-Classifier"

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_detector():

    print("Loading Mirage AI Image Detector...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.to("cpu")
    model.eval()

    print("Mirage detector loaded successfully!")

    return processor, model


def detect_trained_image(image_bytes):

    processor, model = load_detector()

    with Image.open(BytesIO(image_bytes)) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    # Official Mirage label mapping:
    # 0 = Real, 1 = Fake
    real_score = float(probabilities[0].item())
    fake_score = float(probabilities[1].item())

    prediction = "Fake" if fake_score >= real_score else "Real"
    selected_score = max(fake_score, real_score)

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": "Mirage SigLIP2 Image Detector",
        "prediction": prediction,
        "confidence": f"{selected_score * 100:.2f}%",
        "class_scores": {
            "Fake": round(fake_score * 100, 2),
            "Real": round(real_score * 100, 2)
        },
        "explanation": (
            f"Mirage SigLIP2 classified this image as {prediction}. "
            f"Fake score: {fake_score * 100:.2f}%. "
            f"Real score: {real_score * 100:.2f}%. "
            "This model estimates whether an image is AI-generated. "
            "Its prediction is not proof of authenticity."
        ),
        "message": "Image detection completed."
    }


if __name__ == "__main__":

    load_detector()

    if len(sys.argv) > 1:

        image_path = Path(sys.argv[1])

        if not image_path.is_file():
            print("Image not found:", image_path)
            sys.exit(1)

        result = detect_trained_image(
            image_path.read_bytes()
        )

        print("\n=== MIRAGE RESULT ===")
        print("Image:", image_path.name)
        print("Prediction:", result["prediction"])
        print("Confidence:", result["confidence"])
        print("Scores:", result["class_scores"])
