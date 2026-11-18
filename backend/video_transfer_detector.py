
# RobustGuard AI - Trained Video Transfer Detector
# Frozen Face ViT + PCA + Logistic Regression
# Python 3.15 | CPU

from pathlib import Path
from functools import lru_cache
import tempfile
import os

import numpy as np
import torch
import joblib

from face_video_detector import (
    extract_frames,
    detect_face_crop,
    load_models
)


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "training" / "outputs"
    / "video_transfer_classifier.joblib"
)

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_trained_classifier():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained video classifier missing: {MODEL_PATH}"
        )

    # Load only the trusted model created by your project.
    classifier = joblib.load(MODEL_PATH)

    print("Video transfer classifier loaded!")

    return classifier


def extract_transfer_features(video_path):

    # Same extraction method used during training.
    frames = extract_frames(video_path)

    processor, vit, face_detector, fake_index = load_models()

    embeddings = []
    fake_scores = []

    for frame in frames:

        face = detect_face_crop(
            frame,
            face_detector
        )

        if face is None:
            continue

        inputs = processor(
            images=face,
            return_tensors="pt"
        )

        with torch.inference_mode():

            output = vit(
                **inputs,
                output_hidden_states=True
            )

            # Final hidden layer, CLS token.
            embedding = (
                output.hidden_states[-1][0, 0, :]
                .cpu()
                .numpy()
            )

            probabilities = torch.softmax(
                output.logits,
                dim=-1
            )

            fake_score = float(
                probabilities[0, fake_index].item()
            )

        embeddings.append(embedding)
        fake_scores.append(fake_score)

    if not embeddings:
        raise ValueError(
            "No usable faces detected in the video."
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    fake_scores = np.asarray(
        fake_scores,
        dtype=np.float32
    )

    mean_embedding = embeddings.mean(axis=0)

    summary = np.array([
        fake_scores.mean(),
        fake_scores.std(),
        fake_scores.min(),
        fake_scores.max()
    ], dtype=np.float32)

    features = np.concatenate([
        mean_embedding,
        summary
    ])

    return {
        "features": features,
        "frames_analysed": len(fake_scores),
        "frame_fake_scores": fake_scores.tolist()
    }


def predict_video_path(video_path):

    extracted = extract_transfer_features(video_path)

    classifier = load_trained_classifier()

    X = np.asarray(
        [extracted["features"]],
        dtype=np.float64
    )

    probabilities = classifier.predict_proba(X)[0]

    classes = list(classifier.classes_)

    # Training labels:
    # 0 = Real
    # 1 = Fake

    fake_score = float(
        probabilities[classes.index(1)]
    )

    real_score = float(
        probabilities[classes.index(0)]
    )

    prediction = (
        "Fake" if fake_score >= 0.5 else "Real"
    )

    confidence = max(
        fake_score,
        real_score
    )

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": (
            "RobustGuard Video Transfer Learning"
        ),
        "prediction": prediction,
        "confidence": f"{confidence * 100:.2f}%",
        "class_scores": {
            "Fake": round(fake_score * 100, 2),
            "Real": round(real_score * 100, 2)
        },
        "frames_analysed": extracted["frames_analysed"],
        "frame_fake_scores": [
            round(score * 100, 2)
            for score in extracted["frame_fake_scores"]
        ],
        "explanation": (
            f"The video was classified as {prediction}. "
            f"{extracted['frames_analysed']} face frames "
            "were processed using YuNet face detection "
            "and a pretrained Vision Transformer. "
            "Extracted visual features were analysed "
            "by a Logistic Regression classifier "
            "trained on the SDFVD dataset. "
            "This is an experimental classification "
            "and does not prove authenticity."
        ),
        "message": "Video analysis completed."
    }


def detect_video_bytes(content, filename="video.mp4"):

    extension = Path(filename).suffix.lower()

    if extension not in (".mp4", ".webm"):
        raise ValueError(
            "Only MP4 and WebM videos are supported."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp:

            temp.write(content)
            temp_path = temp.name

        return predict_video_path(temp_path)

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(
            'Usage: python video_transfer_detector.py '
            '"path_to_video.mp4"'
        )
        sys.exit(1)

    result = predict_video_path(sys.argv[1])

    print("\n================================")
    print("ROBUSTGUARD VIDEO RESULT")
    print("================================")

    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Class scores:", result["class_scores"])
    print("Frames analysed:", result["frames_analysed"])
