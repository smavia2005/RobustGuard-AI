
from functools import lru_cache
from io import BytesIO

import torch
from PIL import Image, ImageOps
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor
)

MODEL_ID = (
    "buildborderless/"
    "CommunityForensics-DeepfakeDet-ViT"
)

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_detector():
    print("Loading CommunityForensics detector...")

    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.to("cpu")
    model.eval()

    print("New image detector ready!")
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

    fake_probability = float(
        torch.sigmoid(logits.flatten()[0]).item()
    )

    real_probability = 1.0 - fake_probability

    if fake_probability >= 0.5:
        prediction = "Fake"
        score = fake_probability
    else:
        prediction = "Real"
        score = real_probability

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": "CommunityForensics ViT-Small",
        "prediction": prediction,
        "confidence": f"{score * 100:.2f}%",
        "class_scores": {
            "Fake": round(fake_probability * 100, 2),
            "Real": round(real_probability * 100, 2)
        },
        "explanation": (
            f"CommunityForensics assigned the higher "
            f"classification score to {prediction}. "
            "The image was analysed using a "
            "384x384 Vision Transformer. "
            "This is an AI classification, not "
            "definitive proof of authenticity."
        ),
        "message": "Image AI analysis completed."
    }


if __name__ == "__main__":
    load_detector()
