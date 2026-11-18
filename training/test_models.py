
# RobustGuard AI
# Test trained Image CNN and Text ML model
# Python 3.15 | CPU

from pathlib import Path
import numpy as np
import torch
import joblib

from PIL import Image
from torch import nn


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

IMAGE_MODEL_PATH = OUTPUT / "cifake_cnn_best.pth"
TEXT_MODEL_PATH = OUTPUT / "hc3_text_model.joblib"

torch.set_num_threads(2)


# -------------------------------------
# IMAGE CNN ARCHITECTURE
# Must match training/train_image.py
# -------------------------------------

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


# -------------------------------------
# LOAD TRAINED IMAGE MODEL
# -------------------------------------

def load_image_model():

    model = SmallCNN()

    weights = torch.load(
        IMAGE_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    print("Image CNN loaded successfully.")

    return model


# -------------------------------------
# IMAGE PREDICTION
# -------------------------------------

def test_image(model):

    path = input(
        "\nEnter full image path: "
    ).strip().strip('"').strip("'")

    if not Path(path).is_file():
        print("ERROR: Image file not found.")
        return

    try:
        with Image.open(path) as img:
            image = img.convert("RGB")
            image = image.resize((32, 32))

            array = np.asarray(
                image,
                dtype=np.float32
            ).copy() / 255.0

        array = np.transpose(
            array, (2, 0, 1)
        )

        tensor = torch.from_numpy(
            array
        ).unsqueeze(0)

        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(
                output, dim=1
            )[0]

        fake_score = float(
            probabilities[0].item()
        ) * 100

        real_score = float(
            probabilities[1].item()
        ) * 100

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        prediction = (
            "FAKE"
            if predicted_class == 0
            else "REAL"
        )

        confidence = max(
            fake_score, real_score
        )

        print("\n============================")
        print("ROBUSTGUARD AI - IMAGE TEST")
        print("============================")

        print("Prediction:", prediction)
        print(f"Confidence: {confidence:.2f}%")
        print(f"Fake score: {fake_score:.2f}%")
        print(f"Real score: {real_score:.2f}%")

        print("\nExplanation:")
        print(
            "The trained CIFAKE CNN classified "
            "this image using patterns learned "
            "from its training dataset. "
            "This prediction is not proof "
            "of authenticity or manipulation."
        )

    except Exception as error:
        print("Image error:", error)


# -------------------------------------
# TEXT PREDICTION
# -------------------------------------

def test_text(model):

    print("\nEnter text to analyse:")
    text = input("Text: ").strip()

    if not text:
        print("ERROR: Text cannot be empty.")
        return

    prediction_id = int(
        model.predict([text])[0]
    )

    probabilities = model.predict_proba(
        [text]
    )[0]

    classes = list(model.classes_)

    human_score = float(
        probabilities[classes.index(0)]
    ) * 100

    ai_score = float(
        probabilities[classes.index(1)]
    ) * 100

    prediction = (
        "AI GENERATED"
        if prediction_id == 1
        else "HUMAN WRITTEN"
    )

    confidence = max(
        human_score, ai_score
    )

    print("\n============================")
    print("ROBUSTGUARD AI - TEXT TEST")
    print("============================")

    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")
    print(f"AI score: {ai_score:.2f}%")
    print(f"Human score: {human_score:.2f}%")

    print("\nExplanation:")
    print(
        "The HC3-trained text classifier "
        "used TF-IDF features and Logistic "
        "Regression to classify the text. "
        "Its score is not a guarantee of "
        "human or AI authorship."
    )


# -------------------------------------
# MAIN PROGRAM
# -------------------------------------

def main():

    print("\n================================")
    print("   ROBUSTGUARD AI MODEL TESTER")
    print("================================")

    print("\nLoading trained models...")

    image_model = load_image_model()

    # Only load a joblib file you created or trust.
    text_model = joblib.load(
        TEXT_MODEL_PATH
    )

    print("Text classifier loaded successfully.")

    while True:

        print("\n----------------------------")
        print("1. Test Image Detection")
        print("2. Test Text Detection")
        print("3. Exit")
        print("----------------------------")

        choice = input(
            "\nSelect option (1/2/3): "
        ).strip()

        if choice == "1":
            test_image(image_model)

        elif choice == "2":
            test_text(text_model)

        elif choice == "3":
            print("RobustGuard AI tester closed.")
            break

        else:
            print("Please enter 1, 2 or 3.")


if __name__ == "__main__":
    main()
