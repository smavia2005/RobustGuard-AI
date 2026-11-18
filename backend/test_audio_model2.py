
# RobustGuard AI - Alternative Audio Model Test
# CPU only | Python 3.15
# Existing audio_detector.py remains unchanged.

import sys
import subprocess
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import imageio_ffmpeg

from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification
)

MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE = 16000

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_model():

    print("Loading alternative audio model...")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID,
         use_safetensors=True
    )

    model.to("cpu")
    model.eval()

    print("Model labels:", model.config.id2label)

    labels = {
        int(k): str(v).strip().lower()
        for k, v in model.config.id2label.items()
    }

    if labels != {0: "real", 1: "fake"}:
        raise ValueError(
            f"Unexpected model labels: {labels}"
        )

    return processor, model


def load_audio(path):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-i", str(path),
            "-t", "30",
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-f", "f32le",
            "pipe:1"
        ],
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode(
                "utf-8",
                errors="replace"
            )
        )

    audio = np.frombuffer(
        result.stdout,
        dtype="<f4"
    ).copy()

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    return np.clip(
        np.nan_to_num(audio),
        -1,
        1
    )


def predict(path):

    audio = load_audio(path)

    processor, model = load_model()

    # Analyse clips of up to 10 seconds,
    # rather than the previous 1-second windows.
    clip_size = SAMPLE_RATE * 10

    scores = []

    for start in range(0, len(audio), clip_size):

        segment = audio[
            start:start + clip_size
        ]

        # Skip extremely short segments.
        if len(segment) < SAMPLE_RATE * 2:
            continue

        inputs = processor(
            segment,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        with torch.inference_mode():
            logits = model(**inputs).logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )[0]

        scores.append(
            probabilities.cpu().numpy()
        )

    if not scores:
        raise ValueError(
            "Please use audio containing at least "
            "2 seconds of speech."
        )

    average = np.mean(scores, axis=0)

    real_score = float(average[0])
    fake_score = float(average[1])

    prediction = (
        "FAKE" if fake_score >= 0.5
        else "REAL"
    )

    print("\n================================")
    print("ALTERNATIVE AUDIO MODEL RESULT")
    print("================================")

    print("File:", Path(path).name)
    print("Prediction:", prediction)
    print(f"Real score: {real_score * 100:.2f}%")
    print(f"Fake score: {fake_score * 100:.2f}%")
    print("Segments analysed:", len(scores))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python test_audio_model2.py '
            '"audio.wav"'
        )
        sys.exit(1)

    predict(sys.argv[1])
