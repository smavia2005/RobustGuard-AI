
# RobustGuard AI - Audio Deepfake Detector
# Pretrained Wav2Vec2 | CPU | Python 3.15
# Standalone testing before website integration

import sys
import os
import tempfile
import subprocess

from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import imageio_ffmpeg

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)


MODEL_ID = "Bgeorge/deepfake-audio-detector-wav2vec2"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
MAX_AUDIO_SECONDS = 60
MAX_WINDOWS = 12

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a",
    ".ogg", ".opus", ".aac", ".webm"
}

torch.set_num_threads(2)


# ------------------------------------------
# LOAD PRETRAINED MODEL
# ------------------------------------------

@lru_cache(maxsize=1)
def load_audio_model():

    print("\nLoading pretrained Audio Wav2Vec2 model...")

    processor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID
    )

    model.to("cpu")
    model.eval()

    if model.config.num_labels != 2:
        raise ValueError(
            "Expected a binary Real/Fake model."
        )

    print("Model labels:", model.config.id2label)
    print("Audio model successfully loaded!")

    # The model card specifies:
    # 0 = REAL, 1 = FAKE.
    return processor, model


# ------------------------------------------
# AUDIO DECODING WITH EXISTING FFMPEG
# ------------------------------------------

def load_audio(audio_path):

    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported audio file format."
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(path),
        "-t", str(MAX_AUDIO_SECONDS),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1"
    ]

    result = subprocess.run(
        command,
        capture_output=True
    )

    if result.returncode != 0:
        error = result.stderr.decode(
            "utf-8",
            errors="replace"
        )
        raise RuntimeError(
            f"FFmpeg audio decoding failed: {error}"
        )

    audio = np.frombuffer(
        result.stdout,
        dtype="<f4"
    ).copy()

    if len(audio) == 0:
        raise ValueError(
            "No audio samples were decoded."
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    duration = len(audio) / SAMPLE_RATE

    if duration < 0.25:
        raise ValueError(
            "Audio is too short for meaningful analysis."
        )

    print(f"Audio duration analysed: {duration:.2f}s")

    return audio, duration


# ------------------------------------------
# SELECT WINDOWS THROUGHOUT AUDIO
# ------------------------------------------

def create_audio_windows(audio):

    window_size = SAMPLE_RATE * WINDOW_SECONDS

    if len(audio) < window_size:
        padded = np.pad(
            audio,
            (0, window_size - len(audio))
        )
        return [padded.astype(np.float32)], [0.0]

    total_windows = len(audio) // window_size

    count = min(
        MAX_WINDOWS,
        total_windows
    )

    # Uniformly sample from across the recording.
    indices = np.linspace(
        0,
        total_windows - 1,
        num=count,
        dtype=int
    )

    windows = []
    positions = []

    for index in indices:

        start = index * window_size
        end = start + window_size

        segment = audio[start:end]

        # Ignore near-silent segments.
        rms = float(
            np.sqrt(np.mean(segment ** 2))
        )

        if rms < 0.003:
            continue

        windows.append(
            segment.astype(np.float32)
        )

        positions.append(
            round(start / SAMPLE_RATE, 2)
        )

    if not windows:
        raise ValueError(
            "Audio is silent or has no usable speech signal."
        )

    return windows, positions


# ------------------------------------------
# PRETRAINED AUDIO PREDICTION
# ------------------------------------------

def predict_audio(audio_path):

    audio, duration = load_audio(audio_path)

    windows, positions = create_audio_windows(audio)

    processor, model = load_audio_model()

    fake_scores = []
    real_scores = []

    print(f"Audio windows selected: {len(windows)}")

    for index, segment in enumerate(windows, start=1):

        inputs = processor(
            segment,
            sampling_rate=SAMPLE_RATE,
            max_length=SAMPLE_RATE,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        with torch.inference_mode():

            output = model(**inputs)

            probabilities = torch.softmax(
                output.logits,
                dim=-1
            )[0]

        real_score = float(
            probabilities[0].item()
        )

        fake_score = float(
            probabilities[1].item()
        )

        real_scores.append(real_score)
        fake_scores.append(fake_score)

        print(
            f"Window {index} "
            f"({positions[index - 1]:.1f}s): "
            f"Real={real_score * 100:.2f}% "
            f"Fake={fake_score * 100:.2f}%"
        )

    # Mean of window-level model scores.
    # Not a calibrated probability of authenticity.
    average_fake = float(
        np.mean(fake_scores)
    )

    average_real = float(
        np.mean(real_scores)
    )

    prediction = (
        "Fake" if average_fake >= 0.5
        else "Real"
    )

    selected_score = (
        average_fake
        if prediction == "Fake"
        else average_real
    )

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": "Wav2Vec2 Audio Deepfake Detector",
        "prediction": prediction,
        "confidence": f"{selected_score * 100:.2f}%",
        "class_scores": {
            "Fake": round(average_fake * 100, 2),
            "Real": round(average_real * 100, 2)
        },
        "audio_duration": round(duration, 2),
        "segments_analysed": len(windows),
        "segment_fake_scores": [
            round(x * 100, 2)
            for x in fake_scores
        ],
        "explanation": (
            f"The audio was classified as {prediction} "
            "by a pretrained Wav2Vec2 speech detector. "
            f"{len(windows)} audio segments were analysed. "
            "Scores were averaged across sampled segments. "
            "This is an experimental detection and does "
            "not establish whether a recording is authentic."
        )
    }


# ------------------------------------------
# OPTIONAL FASTAPI BYTE WRAPPER
# ------------------------------------------

def detect_audio_bytes(content, filename="audio.wav"):

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported audio file format."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp:

            temp.write(content)
            temp_path = temp.name

        return predict_audio(temp_path)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ------------------------------------------
# TERMINAL TEST
# ------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python audio_detector.py "audio.wav"'
        )
        sys.exit(1)

    result = predict_audio(sys.argv[1])

    print("\n======================================")
    print("ROBUSTGUARD AUDIO DETECTION RESULT")
    print("======================================")

    print("Prediction:", result["prediction"])
    print("Model score:", result["confidence"])
    print("Real score:", result["class_scores"]["Real"], "%")
    print("Fake score:", result["class_scores"]["Fake"], "%")
    print("Segments:", result["segments_analysed"])
    print("\nAUDIO ANALYSIS COMPLETE!")
