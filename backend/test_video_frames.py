
import subprocess
from pathlib import Path
from io import BytesIO

import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

VIDEO = (
    ROOT / "datasets" / "video" / "SDFVD"
    / "Fake" / "vs1.mp4"
)

OUTPUT = ROOT / "datasets" / "video" / "test_frames"
OUTPUT.mkdir(parents=True, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

if not VIDEO.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO}")

print("Reading video:", VIDEO.name)

# Extract 6 frames at regular time intervals
command = [
    FFMPEG,
    "-hide_banner",
    "-loglevel", "error",
    "-i", str(VIDEO),
    "-vf", "fps=1/2,scale=384:-2",
    "-frames:v", "6",
    "-f", "image2pipe",
    "-vcodec", "mjpeg",
    "-"
]

# For reliable extraction, save individual frames.
command = [
    FFMPEG,
    "-hide_banner",
    "-loglevel", "error",
    "-y",
    "-i", str(VIDEO),
    "-vf", "fps=6,scale=384:-2",
    "-frames:v", "6",
    str(OUTPUT / "frame_%02d.jpg")
]

result = subprocess.run(
    command,
    capture_output=True,
    text=True
)

if result.returncode != 0:
    raise RuntimeError(result.stderr)

frames = sorted(OUTPUT.glob("frame_*.jpg"))

print("Frames extracted:", len(frames))

for frame in frames:
    with Image.open(frame) as img:
        print(frame.name, img.size)

print("\nVIDEO FRAME EXTRACTION SUCCESSFUL!")
