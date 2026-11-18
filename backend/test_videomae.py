
# RobustGuard AI - VideoMAE Pretrained Detector
# Python 3.15 | CPU | FFmpeg
# Standalone testing - existing website unchanged

import sys
import re
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

import torch
import imageio_ffmpeg
from PIL import Image

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)

MODEL_ID = "SoraExplora/VideoMae"
NUM_FRAMES = 16

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_model():

    print("\nLoading pretrained VideoMAE...")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID
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
            f"Unexpected labels: {labels}"
        )

    if model.config.num_frames != NUM_FRAMES:
        raise ValueError(
            "Model frame count does not match."
        )

    print("VideoMAE loaded successfully!")
    return processor, model


def get_duration(video_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(video_path)
        ],
        capture_output=True,
        text=True
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr
    )

    if not match:
        raise ValueError(
            "Could not determine video duration."
        )

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    duration = (
        hours * 3600
        + minutes * 60
        + seconds
    )

    if duration <= 0:
        raise ValueError("Invalid video duration.")

    return duration


def extract_frames(video_path):

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    duration = get_duration(video_path)

    print(f"Video duration: {duration:.2f} seconds")

    # Sample approximately 16 frames over
    # the complete duration of the video.
    sampling_fps = NUM_FRAMES / duration

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as folder:

        output = Path(folder) / "frame_%03d.jpg"

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video_path),
            "-vf",
            f"fps={sampling_fps:.8f},scale=256:-2",
            "-frames:v", str(NUM_FRAMES),
            str(output)
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        frame_files = sorted(
            Path(folder).glob("frame_*.jpg")
        )

        if not frame_files:
            raise ValueError(
                "No frames extracted from video."
            )

        frames = []

        for path in frame_files:

            with Image.open(path) as image:
                frames.append(
                    image.convert("RGB").copy()
                )

    # Pad very short videos if FFmpeg
    # returns fewer than 16 frames.
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())

    return frames[:NUM_FRAMES]


def predict_video(video_path):

    frames = extract_frames(video_path)

    print("Frames extracted:", len(frames))

    processor, model = load_model()

    inputs = processor(
        list(frames),
        return_tensors="pt"
    )

    print(
        "Input tensor shape:",
        tuple(inputs["pixel_values"].shape)
    )

    with torch.inference_mode():
        logits = model(**inputs).logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

    real_score = float(
        probabilities[0].item()
    )

    fake_score = float(
        probabilities[1].item()
    )

    prediction = (
        "FAKE" if fake_score >= 0.5
        else "REAL"
    )

    print("\n================================")
    print("VIDEO MAE PREDICTION")
    print("================================")

    print("Video:", Path(video_path).name)
    print("Prediction:", prediction)
    print(f"Real score: {real_score * 100:.2f}%")
    print(f"Fake score: {fake_score * 100:.2f}%")

    print(
        "\nThis is a pretrained model prediction, "
        "not proof of video authenticity."
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python test_videomae.py '
            '"path_to_video.mp4"'
        )
        sys.exit(1)

    predict_video(sys.argv[1])
