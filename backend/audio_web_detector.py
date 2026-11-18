
# RobustGuard AI - FINAL Audio Deepfake Detector
# Wav2Vec2 | 90% Fake-score threshold
# Python 3.15 | CPU
#
# Exploratory validation:
# ASVspoof2019 LA - 100 previously unused samples
# Accuracy: 92%
# Real Recall: 96%
# Fake Recall: 88%
#
# These results do not guarantee real-world accuracy.

import os
import tempfile

from pathlib import Path

import numpy as np
import torch

# Reuse the already working pretrained model.
from test_audio_model2 import (
    load_model,
    load_audio,
    SAMPLE_RATE
)


# -----------------------------------
# SETTINGS
# -----------------------------------

AUDIO_THRESHOLD = 0.90

CLIP_SECONDS = 10
MAX_CLIPS = 6

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".webm"
}

torch.set_num_threads(2)


# -----------------------------------
# AUDIO PREDICTION
# -----------------------------------

def predict_audio_path(audio_path):

    # Load and convert audio to 16 kHz.
    # Existing decoder processes up to 30 seconds.
    audio = load_audio(audio_path)

    # Load pretrained Wav2Vec2.
    processor, model = load_model()

    # Verify labels before making predictions.
    labels = {
        int(key): str(value).strip().lower()
        for key, value in model.config.id2label.items()
    }

    if labels != {0: "real", 1: "fake"}:
        raise ValueError(
            f"Unexpected audio model labels: {labels}"
        )

    clip_size = SAMPLE_RATE * CLIP_SECONDS

    segment_scores = []

    # Process up to 10-second clips.
    for start in range(0, len(audio), clip_size):

        if len(segment_scores) >= MAX_CLIPS:
            break

        segment = audio[
            start:start + clip_size
        ]

        # Skip segments shorter than 2 seconds.
        if len(segment) < SAMPLE_RATE * 2:
            continue

        inputs = processor(
            segment,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        with torch.inference_mode():

            output = model(**inputs)

            probabilities = torch.softmax(
                output.logits,
                dim=-1
            )[0]

        segment_scores.append(
            probabilities.cpu().numpy()
        )

    if not segment_scores:
        raise ValueError(
            "Audio must contain at least "
            "2 seconds of usable speech."
        )

    # Same score aggregation as validation.
    average_scores = np.mean(
        segment_scores,
        axis=0
    )

    real_score = float(average_scores[0])
    fake_score = float(average_scores[1])

    # -----------------------------------
    # FINAL VALIDATED THRESHOLD
    # -----------------------------------

    prediction = (
        "Fake"
        if fake_score >= AUDIO_THRESHOLD
        else "Real"
    )

    # This is a model class score, NOT
    # a calibrated probability or confidence.
    selected_score = (
        fake_score
        if prediction == "Fake"
        else real_score
    )

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": (
            "RobustGuard Wav2Vec2 "
            "Audio Deepfake Detector"
        ),
        "prediction": prediction,
        "confidence": (
            f"{selected_score * 100:.2f}%"
        ),
        "class_scores": {
            "Real": round(real_score * 100, 2),
            "Fake": round(fake_score * 100, 2)
        },
        "fake_score": round(fake_score * 100, 2),
        "detection_threshold": 90,
        "score_type": (
            "Uncalibrated model score"
        ),
        "segments_analysed": len(segment_scores),
        "audio_duration": round(
            len(audio) / SAMPLE_RATE,
            2
        ),
        "explanation": (
            f"The uploaded audio was classified as "
            f"{prediction} using a pretrained "
            "Wav2Vec2 audio detection model. "
            f"{len(segment_scores)} audio segments "
            "were analysed. "
            "A 90% Fake-score threshold was selected "
            "through experimental evaluation on "
            "ASVspoof2019 LA audio recordings. "
            "The displayed scores are not calibrated "
            "probabilities and do not guarantee "
            "the authenticity of the recording."
        ),
        "message": "Audio analysis completed."
    }


# -----------------------------------
# FASTAPI UPLOAD SUPPORT
# -----------------------------------

def detect_audio_bytes(
    content,
    filename="audio.wav"
):

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported audio format. "
            "Use WAV, MP3, M4A, FLAC, "
            "OGG, OPUS, AAC or WebM."
        )

    if not content:
        raise ValueError(
            "Uploaded audio is empty."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp:

            temp.write(content)
            temp_path = temp.name

        return predict_audio_path(temp_path)

    finally:

        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)


# -----------------------------------
# TERMINAL TEST
# -----------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(
            'Usage: python audio_web_detector.py '
            '"audio.wav"'
        )
        sys.exit(1)

    result = predict_audio_path(sys.argv[1])

    print("\n===================================")
    print("ROBUSTGUARD FINAL AUDIO DETECTION")
    print("===================================")

    print("Prediction:", result["prediction"])

    print(
        "Real score:",
        result["class_scores"]["Real"],
        "%"
    )

    print(
        "Fake score:",
        result["class_scores"]["Fake"],
        "%"
    )

    print(
        "Classification threshold:",
        result["detection_threshold"],
        "%"
    )

    print(
        "Segments analysed:",
        result["segments_analysed"]
    )

    print("\nAUDIO DETECTION COMPLETE!")
