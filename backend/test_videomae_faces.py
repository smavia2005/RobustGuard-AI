
# RobustGuard AI - VideoMAE + YuNet Face Detection
# Python 3.15 | CPU | FFmpeg
# Standalone experiment

import sys
import re
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache

import cv2
import torch
import numpy as np
import imageio_ffmpeg

from PIL import Image
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)


ROOT = Path(__file__).resolve().parent.parent

YUNET_PATH = (
    ROOT / "backend" / "models"
    / "face_detection_yunet"
    / "face_detection_yunet_2023mar.onnx"
)

MODEL_ID = "SoraExplora/VideoMae"
NUM_FRAMES = 16

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_models():

    if not YUNET_PATH.is_file():
        raise FileNotFoundError(
            f"YuNet model missing: {YUNET_PATH}"
        )

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "OpenCV FaceDetectorYN unavailable."
        )

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    print("Loading pretrained VideoMAE...")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID
    )

    model.to("cpu")
    model.eval()

    labels = {
        int(k): str(v).strip().lower()
        for k, v in model.config.id2label.items()
    }

    print("Model labels:", labels)

    if labels != {0: "real", 1: "fake"}:
        raise ValueError(
            f"Unexpected label mapping: {labels}"
        )

    if model.config.num_frames != NUM_FRAMES:
        raise ValueError(
            "VideoMAE expected frame count differs."
        )

    return detector, processor, model


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

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as folder:

        output = Path(folder) / "frame_%03d.jpg"

        sampling_fps = NUM_FRAMES / duration

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video_path),
            "-vf",
            f"fps={sampling_fps:.8f},scale=384:-2",
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

        frames = []

        for path in frame_files:
            with Image.open(path) as img:
                frames.append(
                    img.convert("RGB").copy()
                )

    if not frames:
        raise ValueError("No frames extracted.")

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].copy())

    return frames[:NUM_FRAMES]


def detect_face(image, detector):

    rgb = np.asarray(image.convert("RGB"))

    height, width = rgb.shape[:2]

    small = cv2.resize(
        rgb,
        (320, 320),
        interpolation=cv2.INTER_LINEAR
    )

    bgr = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2BGR
    )

    detector.setInputSize((320, 320))

    _, faces = detector.detect(bgr)

    if faces is None or len(faces) == 0:
        return None

    largest = max(
        faces,
        key=lambda item: item[2] * item[3]
    )

    x, y, fw, fh = largest[:4]

    scale_x = width / 320.0
    scale_y = height / 320.0

    x *= scale_x
    y *= scale_y
    fw *= scale_x
    fh *= scale_y

    # Include context around the face.
    margin_x = fw * 0.35
    margin_y = fh * 0.35

    left = max(0, int(x - margin_x))
    top = max(0, int(y - margin_y))

    right = min(
        width,
        int(x + fw + margin_x)
    )

    bottom = min(
        height,
        int(y + fh + margin_y)
    )

    if right <= left or bottom <= top:
        return None

    return image.crop((
        left,
        top,
        right,
        bottom
    ))


def predict_video(video_path):

    frames = extract_frames(video_path)

    detector, processor, model = load_models()

    cropped_frames = []
    detected_count = 0

    for frame in frames:

        face = detect_face(
            frame,
            detector
        )

        if face is not None:
            cropped_frames.append(face)
            detected_count += 1
        else:
            # Keep sequence length at 16.
            # Use previous detected crop when available.
            if cropped_frames:
                cropped_frames.append(
                    cropped_frames[-1].copy()
                )
            else:
                cropped_frames.append(None)

    # Fill missing initial crops with first detected face.
    valid_faces = [
        frame for frame in cropped_frames
        if frame is not None
    ]

    if not valid_faces:
        raise ValueError(
            "No faces detected in any sampled frame."
        )

    first_face = valid_faces[0]

    cropped_frames = [
        frame if frame is not None
        else first_face.copy()
        for frame in cropped_frames
    ]

    print("Total frames:", len(frames))
    print("Frames with detected faces:", detected_count)
    print("Frames supplied to VideoMAE:", len(cropped_frames))

    inputs = processor(
        list(cropped_frames),
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
    print("VIDEOMAE + YUNET RESULT")
    print("================================")

    print("Video:", Path(video_path).name)
    print("Prediction:", prediction)
    print(f"Real score: {real_score * 100:.2f}%")
    print(f"Fake score: {fake_score * 100:.2f}%")

    print(
        "\nExperimental result. "
        "This is not proof of video authenticity."
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python test_videomae_faces.py '
            '"video.mp4"'
        )
        sys.exit(1)

    predict_video(sys.argv[1])
