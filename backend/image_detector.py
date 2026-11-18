
# image_detector.py
# Real AI-powered image classification

from functools import lru_cache
from io import BytesIO

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

# Reduce unnecessary CPU usage
torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_model():
    """
    Download and load the model once.
    Future requests reuse the same model.
    """

    print("Loading RobustGuard AI image detector...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        use_safetensors=True
    )

    model.to("cpu")
    model.eval()

    print("Image detection model loaded successfully!")

    return processor, model


def predict_image(image_bytes: bytes):
    """
    Perform real inference on image bytes.
    """

    processor, model = load_model()

    # Open the uploaded image
    with Image.open(BytesIO(image_bytes)) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")

    # Preprocess image for the model
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Disable training gradients for faster inference
    with torch.inference_mode():
        outputs = model(**inputs)

    # Convert model logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    prediction_id = int(
        torch.argmax(probabilities).item()
    )

    prediction_label = model.config.id2label[
        prediction_id
    ]

    prediction_score = float(
        probabilities[prediction_id].item()
    )

    # Read both class probabilities
    class_scores = {}

    for index, probability in enumerate(probabilities):
        label = model.config.id2label[index]
        class_scores[label] = round(
            float(probability.item()) * 100, 2
        )

    # Explain what the output means
    if prediction_label.lower() == "fake":
        explanation = (
            "The image classifier assigned a higher "
            "probability to the Fake category. "
            "This is a model-based classification, "
            "not proof of manipulation. The model "
            "does not identify which pixels were "
            "edited or how the image was generated."
        )
    else:
        explanation = (
            "The image classifier assigned a higher "
            "probability to the Real category. "
            "This does not prove that the image is "
            "authentic. Manipulated images can still "
            "be incorrectly classified as real."
        )

    return {
        "status": "success",
        "type": "image",
        "model_status": "ready",
        "model_name": MODEL_NAME,
        "prediction": prediction_label,
        "confidence": f"{prediction_score * 100:.2f}%",
        "class_scores": class_scores,
        "explanation": explanation,
        "message": "Real AI image classification completed."
    }


if __name__ == "__main__":
    print("Testing model download and loading...")
    load_model()
    print("RobustGuard AI image detector is ready.")
