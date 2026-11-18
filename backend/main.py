
# RobustGuard AI - Main FastAPI Application
# Multimodal Deepfake Detection
# Image + Text + Video + Audio
# Python 3.15 | CPU

from io import BytesIO
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from PIL import Image, UnidentifiedImageError
from guardbot_api import router as guardbot_router

# Existing working models
from trained_models import detect_trained_text
from mirage_detector import detect_trained_image
from video_transfer_detector import detect_video_bytes
from audio_web_detector import detect_audio_bytes


# -------------------------------------
# FASTAPI APPLICATION
# -------------------------------------

app = FastAPI(
    title="RobustGuard AI",
    description="Multimodal AI Deepfake Detection",
    version="2.1"
)

app.include_router(guardbot_router)


# -------------------------------------
# CORS CONFIGURATION
# -------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# -------------------------------------
# FILE SETTINGS
# -------------------------------------

MAX_SIZE = 20 * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

ALLOWED_TYPES = {
    "image": {
        "image/jpeg",
        "image/png",
        "image/webp"
    },
    "video": {
        "video/mp4",
        "video/webm"
    },
    "audio": {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/ogg",
        "audio/flac",
        "audio/x-flac",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
        "audio/webm",
        "audio/opus",
        "application/ogg",
        "application/octet-stream"
    }
}

ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp"},
    "video": {".mp4", ".webm"},
    "audio": {
        ".wav", ".mp3", ".m4a",
        ".flac", ".ogg", ".opus",
        ".aac", ".webm"
    }
}


# -------------------------------------
# HOME
# -------------------------------------

@app.get("/")
def home():
    return {
        "project": "RobustGuard AI",
        "version": "2.1",
        "image_model": "Mirage Photo Classifier",
        "text_model": "HC3 Text Classifier",
        "video_model": "Video Transfer Learning",
        "audio_model": "Wav2Vec2 Audio Detector",
        "modules": [
            "image",
            "text",
            "video",
            "audio"
        ]
    }


# -------------------------------------
# HEALTH CHECK
# -------------------------------------

@app.get("/health")
def health():
    return {
        "status": "success",
        "message": "RobustGuard AI backend is running",
        "version": "2.1"
    }


# -------------------------------------
# TEXT DETECTION
# -------------------------------------

@app.post("/detect/text")
async def detect_text(text: str = Form(...)):

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter text."
        )

    if len(text) > 100000:
        raise HTTPException(
            status_code=413,
            detail="Text input is too long."
        )

    try:
        result = await run_in_threadpool(
            detect_trained_text,
            text
        )

        return result

    except Exception as error:
        print("Text model error:", repr(error))

        raise HTTPException(
            status_code=503,
            detail="Text detection could not run."
        )


# -------------------------------------
# FILE DETECTION
# -------------------------------------

@app.post("/detect/file")
async def detect_file(
    detection_type: str = Form(...),
    file: UploadFile = File(...)
):

    # Check detection type
    if detection_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid detection type."
        )

    # Check file extension
    filename = file.filename or ""

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS[detection_type]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension."
        )

    # Validate file MIME type
    if file.content_type not in ALLOWED_TYPES[detection_type]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format."
        )

    # Read uploaded file
    content = await file.read(MAX_SIZE + 1)
    await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum supported file size is 20 MB."
        )

    # =====================================
    # IMAGE DETECTION
    # =====================================

    if detection_type == "image":

        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported image file."
            )

        try:
            result = await run_in_threadpool(
                detect_trained_image,
                content
            )

            result["filename"] = filename

            return result

        except Exception as error:
            print("Image model error:", repr(error))

            raise HTTPException(
                status_code=503,
                detail="Image detection could not run."
            )

    # =====================================
    # VIDEO DETECTION
    # =====================================

    if detection_type == "video":

        try:
            result = await run_in_threadpool(
                detect_video_bytes,
                content,
                filename
            )

            result["filename"] = filename

            return result

        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error)
            )

        except Exception as error:
            print("Video model error:", repr(error))

            raise HTTPException(
                status_code=503,
                detail="Video detection could not run."
            )

    # =====================================
    # AUDIO DETECTION
    # =====================================

    if detection_type == "audio":

        try:
            result = await run_in_threadpool(
                detect_audio_bytes,
                content,
                filename
            )

            result["filename"] = filename

            return result

        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error)
            )

        except Exception as error:
            print("Audio model error:", repr(error))

            raise HTTPException(
                status_code=503,
                detail="Audio detection could not run."
            )

    # Should not be reached for valid types
    raise HTTPException(
        status_code=400,
        detail="Unable to process detection type."
    )
