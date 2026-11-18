
# RobustGuard AI - Face-Based Video Deepfake Detection
# Python 3.15 | CPU | FFmpeg + YuNet + Pretrained ViT

import os
import sys
import tempfile
import subprocess

from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import torch
import imageio_ffmpeg

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


ROOT = Path(__file__).resolve().parent.parent

FACE_MODEL = (
    ROOT / "backend" / "models"
    / "face_detection_yunet"
    / "face_detection_yunet_2023mar.onnx"
)

DEEPFAKE_MODEL = "shunda012/vit-deepfake-detector"

NUM_FRAMES = 6

torch.set_num_threads(2)


@lru_cache(maxsize=1)
def load_models():

    if not FACE_MODEL.is_file():
        raise FileNotFoundError(
            f"YuNet model missing: {FACE_MODEL}"
        )

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "Installed OpenCV does not support FaceDetectorYN."
        )

    print("Loading YuNet face detector...")

    detector = cv2.FaceDetectorYN.create(
        str(FACE_MODEL),
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    print("Loading pretrained ViT deepfake detector...")

    processor = AutoImageProcessor.from_pretrained(
        DEEPFAKE_MODEL
    )

    model = AutoModelForImageClassification.from_pretrained(
        DEEPFAKE_MODEL
    )

    model.eval()
    model.to("cpu")

    labels = {
        int(k): str(v).strip().lower()
        for k, v in model.config.id2label.items()
    }

    fake_ids = [
        k for k, v in labels.items() if v == "fake"
    ]

    real_ids = [
        k for k, v in labels.items() if v == "real"
    ]

    if len(fake_ids) != 1 or len(real_ids) != 1:
        raise ValueError(f"Unexpected labels: {labels}")

    print("Models loaded successfully!")

    return processor, model, detector, fake_ids[0]


def extract_frames(video_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    frames = []

    with tempfile.TemporaryDirectory() as folder:

        output = Path(folder) / "frame_%02d.jpg"

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(video_path),
            "-vf", "fps=6,scale=384:-2",
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

        files = sorted(
            Path(folder).glob("frame_*.jpg")
        )

        for file in files:
            with Image.open(file) as img:
                frames.append(img.convert("RGB").copy())

    if not frames:
        raise ValueError("No video frames extracted.")

    return frames


def detect_face_crop(image, detector):

    rgb = np.asarray(image.convert("RGB"))

    height, width = rgb.shape[:2]

    # YuNet 2023 model uses fixed 320x320 input.
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

    _, detected = detector.detect(bgr)

    if detected is None or len(detected) == 0:
        return None

    # Select largest detected face.
    face = max(
        detected,
        key=lambda row: row[2] * row[3]
    )

    x, y, fw, fh = face[:4]

    # Convert face bounding box back to original size.
    scale_x = width / 320.0
    scale_y = height / 320.0

    x = x * scale_x
    y = y * scale_y

    fw = fw * scale_x
    fh = fh * scale_y

    # Include surrounding facial context.
    margin_x = fw * 0.25
    margin_y = fh * 0.25

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

    return image.crop(
        (left, top, right, bottom)
    )


def detect_face_video_path(video_path):

    frames = extract_frames(video_path)

    processor, model, detector, fake_index = load_models()

    face_crops = []

    for frame in frames:

        crop = detect_face_crop(
            frame,
            detector
        )

        if crop is not None:
            face_crops.append(crop)

    if not face_crops:
        raise ValueError(
            "No faces detected in the sampled frames."
        )

    frame_scores = []

    for face in face_crops:

        inputs = processor(
            images=face,
            return_tensors="pt"
        )

        with torch.inference_mode():

            logits = model(**inputs).logits

            probs = torch.softmax(
                logits, dim=-1
            )[0]

        fake_score = float(
            probs[fake_index].item()
        )

        frame_scores.append(fake_score)

    average_fake = float(
        np.mean(frame_scores)
    )

    average_real = 1.0 - average_fake

    prediction = (
        "Fake" if average_fake >= 0.5 else "Real"
    )

    confidence = max(
        average_fake,
        average_real
    )

    return {
        "status": "success",
        "model_status": "ready",
        "model_name": "YuNet + Face Deepfake ViT",
        "prediction": prediction,
        "confidence": f"{confidence * 100:.2f}%",
        "class_scores": {
            "Fake": round(average_fake * 100, 2),
            "Real": round(average_real * 100, 2)
        },
        "frames_analysed": len(face_crops),
        "frame_fake_scores": [
            round(score * 100, 2)
            for score in frame_scores
        ],
        "explanation": (
            f"The video was classified as {prediction}. "
            f"YuNet detected faces in {len(face_crops)} "
            "sampled frames. A pretrained ViT classifier "
            "analysed the cropped faces. "
            "Frame scores were averaged to obtain "
            "an experimental video-level result. "
            "This does not prove authenticity."
        )
    }


def detect_face_video_bytes(
    content,
    filename="video.mp4"
):

    extension = Path(filename).suffix.lower()

    if extension not in (".mp4", ".webm"):
        raise ValueError("Unsupported video format.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp:

            temp.write(content)
            temp_path = temp.name

        return detect_face_video_path(temp_path)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python face_video_detector.py "video.mp4"'
        )
        sys.exit(1)

    result = detect_face_video_path(
        sys.argv[1]
    )

    print("\n=== VIDEO RESULT ===")
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Fake score:", result["class_scores"]["Fake"])
    print("Frames:", result["frames_analysed"])
