
# RobustGuard AI - Dataset Trained Models
# Python 3.15 | CPU Inference

from pathlib import Path
from functools import lru_cache
from io import BytesIO

import numpy as np
import torch
import joblib

from PIL import Image, ImageOps
from torch import nn


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "training" / "outputs"

IMAGE_PATH = OUTPUT / "cifake_cnn_best.pth"
TEXT_PATH = OUTPUT / "hc3_text_model.joblib"

torch.set_num_threads(2)


# --------------------------------------
# IMAGE MODEL - SAME AS TRAINING
# --------------------------------------

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


@lru_cache(maxsize=1)
def load_image_model():

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Trained image model not found: {IMAGE_PATH}"
        )

    model = SmallCNN()

    weights = torch.load(
        IMAGE_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    print("CIFAKE trained CNN loaded successfully.")

    return model


@lru_cache(maxsize=1)
def load_text_model():

    if not TEXT_PATH.is_file():
        raise FileNotFoundError(
            f"Trained text model not found: {TEXT_PATH}"
        )

    # Load only your own trusted joblib model file.
    model = joblib.load(TEXT_PATH)

    print("HC3 trained text model loaded successfully.")

    return model


# --------------------------------------
# IMAGE DETECTION
# --------------------------------------

def detect_trained_image(image_bytes: bytes):

    model = load_image_model()

    with Image.open(BytesIO(image_bytes)) as img:
        image = ImageOps.exif_transpose(img)
        image = image.convert("RGB")
        image = image.resize((32, 32))

        array = np.asarray(
            image,
            dtype=np.float32
        ).copy() / 255.0

    array = np.transpose(array, (2, 0, 1))

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits, dim=1
        )[0]

    # CIFAKE training labels:
    # 0 = FAKE, 1 = REAL

    fake_score = float(probabilities[0].item())
    real_score = float(probabilities[1].item())

    if fake_score >= real_score:
        prediction = "Fake"
        selected_score = fake_score
    else:
        prediction = "Real"
        selected_score = real_score

    explanation = (
        f"The trained CIFAKE CNN assigned a higher "
        f"classification score to the {prediction} class. "
        f"The image was resized to 32x32 pixels and "
        f"analysed using learned visual patterns. "
        f"This model was evaluated on CIFAKE images, "
        f"so performance may differ on real-world images. "
        f"The prediction does not prove authenticity "
        f"or identify specific manipulated regions."
    )

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": "RobustGuard CIFAKE SmallCNN",
        "prediction": prediction,
        "confidence": f"{selected_score * 100:.2f}%",
        "class_scores": {
            "Fake": round(fake_score * 100, 2),
            "Real": round(real_score * 100, 2)
        },
        "explanation": explanation,
        "message": "Dataset-trained image analysis completed."
    }


# --------------------------------------
# TEXT DETECTION
# --------------------------------------

def detect_trained_text(text: str):

    model = load_text_model()

    cleaned = text.strip()

    if not cleaned:
        raise ValueError("Text cannot be empty.")

    predicted_id = int(
        model.predict([cleaned])[0]
    )

    probabilities = model.predict_proba(
        [cleaned]
    )[0]

    classes = list(model.classes_)

    human_score = float(
        probabilities[classes.index(0)]
    )

    ai_score = float(
        probabilities[classes.index(1)]
    )

    if predicted_id == 1:
        prediction = "AI Generated"
        selected_score = ai_score
    else:
        prediction = "Human Written"
        selected_score = human_score

    explanation = (
        f"The HC3-trained text classifier assigned "
        f"a higher score to the {prediction} category. "
        f"It uses TF-IDF word patterns and Logistic "
        f"Regression, not direct knowledge of the "
        f"original author. Its HC3 evaluation results "
        f"do not guarantee reliability for different "
        f"languages, writing styles, or AI tools. "
        f"This prediction is not proof of authorship."
    )

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": "RobustGuard HC3 Text Classifier",
        "prediction": prediction,
        "confidence": f"{selected_score * 100:.2f}%",
        "class_scores": {
            "Human Written": round(human_score * 100, 2),
            "AI Generated": round(ai_score * 100, 2)
        },
        "explanation": explanation,
        "message": "Dataset-trained text analysis completed."
    }


if __name__ == "__main__":
    load_image_model()
    load_text_model()
    print("Both trained RobustGuard AI models are ready.")
